"""
Tenant Middleware & FastAPI Dependency.

Enforces:
  - Mandatory X-Customer-ID header (→ 401 if missing)
  - Known tenant lookup            (→ 403 if unknown)
  - Structured audit logging per request (tenant, request_id, route, latency_ms)
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .registry import TenantRegistry
from .models import TenantConfig

logger = logging.getLogger("middleware")

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")


def get_request_id() -> str:
    return _request_id_var.get()


def get_current_tenant_id() -> str:
    return _tenant_id_var.get()


async def get_tenant_context(
    x_customer_id: Optional[str] = Header(None, alias="X-Customer-ID"),
) -> TenantConfig:
    """
    FastAPI dependency: validates X-Customer-ID and resolves TenantConfig.

    Returns 401 when the header is missing.
    Returns 403 when the tenant is unknown.
    """
    if not x_customer_id:
        raise HTTPException(
            status_code=401,
            detail="Missing required header: X-Customer-ID",
        )

    if _global_registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialised")

    try:
        return _global_registry.get_tenant(x_customer_id)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Unknown tenant: '{x_customer_id}'",
        )


_global_registry: Optional[TenantRegistry] = None


def init_registry(registry: TenantRegistry) -> None:
    """Call this from FastAPI lifespan to inject the registry singleton."""
    global _global_registry
    _global_registry = registry


class TenantAuditMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that:
      - Generates a unique request_id per request
      - Binds tenant_id + request_id to ContextVars (for log injection)
      - Emits a structured audit log line after every response
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        _request_id_var.set(request_id)

        tenant_id = request.headers.get("X-Customer-ID", "unknown")
        _tenant_id_var.set(tenant_id)

        request.state.request_id = request_id
        request.state.tenant_id = tenant_id

        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "audit",
            extra={
                "request_id": request_id,
                "tenant_id": tenant_id,
                "method": request.method,
                "route": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 1),
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
