"""
Multi-Tenant Agent Service — Production FastAPI application.

Key production features implemented here:
  - FastAPI lifespan for startup/shutdown resource management
  - LRU-cached AgentRunner per tenant (see factory.py)
  - Structured JSON logging with tenant_id + request_id on every line
  - TenantAuditMiddleware: request_id, latency_ms, status_code per request
  - /health/live and /health/ready endpoints for Kubernetes probes
  - Clean error boundaries (500 never leaks internal tracebacks to callers)

Run locally:
  uvicorn src.main:app --reload --port 8000

Environment:
  LLM_PROVIDER = ollama | bedrock   (default: ollama)
  OLLAMA_MODEL = llama3.2           (default)
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .factory import get_agent_runner
from .middleware import TenantAuditMiddleware, get_tenant_context, init_registry
from .models import TenantConfig
from .registry import TenantRegistry

load_dotenv()

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tenants.yaml")


def _setup_logging() -> None:
    """Configure structured JSON logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: runs on startup and shutdown.

    Startup:
      - Initialise registry (loads tenants.yaml; fails fast on bad config)
      - Pre-warm LLM client so first request isn't slow
      - Expose registry on app.state for health checks

    Shutdown:
      - Clear AgentRunner LRU cache (releases connections)
    """
    _setup_logging()
    logger.info("[lifespan] starting up...")

    registry = TenantRegistry(_CONFIG_PATH)
    init_registry(registry)
    app.state.registry = registry
    logger.info(f"[lifespan] registry loaded | tenants={registry.list_tenants()}")

    try:
        from .llm_provider import get_llm
        get_llm()
        logger.info("[lifespan] LLM client initialised")
    except Exception as exc:
        logger.warning(f"[lifespan] LLM pre-warm failed: {exc}")

    yield

    get_agent_runner.cache_clear()
    logger.info("[lifespan] shutdown complete")


app = FastAPI(
    title="Multi-Tenant Agent Service",
    description="Production-grade multi-tenant AI agent with Ollama/Bedrock LLM support.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(TenantAuditMiddleware)


# ---------------------------------------------------------------------------
# HEALTH ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health/live", tags=["Health"])
async def health_live():
    """Kubernetes liveness probe — is the process up?"""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def health_ready(request: Request):
    """
    Kubernetes readiness probe — can the service handle requests?
    Verifies: registry loaded + LLM client reachable.
    """
    errors = []

    registry: Optional[TenantRegistry] = getattr(request.app.state, "registry", None)
    if registry is None:
        errors.append("registry not initialised")

    try:
        from .llm_provider import get_llm
        get_llm()
    except Exception as exc:
        errors.append(f"llm_provider: {exc}")

    if errors:
        return JSONResponse(status_code=503, content={"status": "not_ready", "errors": errors})

    tenants = registry.list_tenants() if registry else []
    return {"status": "ready", "tenants": tenants}


# ---------------------------------------------------------------------------
# CHAT ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/chat", tags=["Agent"])
async def chat(
    request: Request,
    question: str,
    user_id: str = "anonymous",
    tenant: TenantConfig = Depends(get_tenant_context),
):
    """
    Single-turn chat with tenant-scoped agent.

    Headers required:
      X-Customer-ID: customer_a | customer_b | customer_c

    Query params:
      question  — the user's message
      user_id   — optional user identifier (for memory scoping)

    Returns:
      { tenant, response, status, request_id, blocked }
    """
    request_id = getattr(request.state, "request_id", "-")
    logger.info(
        f"[chat] tenant={tenant.customer_id} | user={user_id} "
        f"| request_id={request_id} | q={question[:80]!r}"
    )

    try:
        runner = get_agent_runner(tenant.customer_id)
        result = runner.invoke(question=question, user_id=user_id)
        return {
            "tenant": tenant.customer_id,
            "request_id": request_id,
            "response": result["response"],
            "status": result["status"],
            "blocked": result.get("blocked", False),
        }
    except Exception as exc:
        logger.error(
            f"[chat] error | tenant={tenant.customer_id} | request_id={request_id} | {exc}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal agent error. See server logs.")


# ---------------------------------------------------------------------------
# ADMIN / INTROSPECTION
# ---------------------------------------------------------------------------

@app.get("/tenants", tags=["Admin"])
async def list_tenants(request: Request):
    """List all registered tenant IDs."""
    registry: Optional[TenantRegistry] = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialised")
    return {"tenants": registry.list_tenants()}


@app.post("/admin/cache/clear", tags=["Admin"])
async def clear_cache():
    """
    Clears the AgentRunner LRU cache.
    Use after updating tenants.yaml to pick up new config without restarting.
    """
    get_agent_runner.cache_clear()
    logger.info("[admin] AgentRunner cache cleared")
    return {"detail": "AgentRunner cache cleared"}
