"""
AgentFactory — builds fully-wired, tenant-scoped agent components.

Design decisions for this environment:
  - LRU cache: AgentRunner is built once per tenant, reused on subsequent
    requests (avoids re-creating LLM clients and connection objects).
  - FileCheckpointer: local-file episodic memory; swap for S3Checkpointer
    in production by setting CHECKPOINTER=s3 + S3_BUCKET env vars.
  - No Solr dependency: long-term memory omitted here; add SolrAdapter
    when SOLR_URL env var is set.
  - llm_provider is a zero-arg lambda so se2_agent_shared can call it
    lazily (matching the library's expected interface).
"""

import logging
import os
from functools import lru_cache
from typing import Optional, Tuple

from se2_agent_shared.core import CoreConfig, SimpleThinker, SimpleSupervisor
from se2_agent_shared.memory import MemoryConfig
from se2_agent_shared.security import LLMGuardrail, SecurityConfig

from .models import TenantConfig

logger = logging.getLogger("factory")

STAGE_DEFINITIONS = {
    "Data": "Retrieve data from tenant-specific sources.",
    "Output": "Generate the final, grounded response for the user.",
}

def _checkpoints_dir() -> str:
    return os.getenv("CHECKPOINTS_DIR", "./checkpoints")


def _make_solr_adapter(tenant: TenantConfig):
    """Return a SolrAdapter if SOLR_URL is configured, else None."""
    solr_url = os.getenv("SOLR_URL")
    if not solr_url:
        return None
    try:
        from se2_agent_shared.tools.solr.solr_client import get_solr_connection
        from se2_agent_shared.tools.solr import SolrAdapter
        conn = get_solr_connection(base_url=solr_url)
        return SolrAdapter(
            collection_name="user_facts",
            embedding_model_dims=384,
            solr_connection=conn,
            core_name=tenant.solr_core,
        )
    except Exception as exc:
        logger.warning(f"[factory] Solr unavailable for {tenant.customer_id}: {exc}")
        return None


class AgentRunner:
    """
    Fully-wired, tenant-scoped agent.

    Holds references to pre-built se2_agent_shared components so they
    are not re-instantiated on every request.
    """

    def __init__(
        self,
        tenant: TenantConfig,
        guardrail: LLMGuardrail,
        thinker: SimpleThinker,
        supervisor: SimpleSupervisor,
        memory_cfg: MemoryConfig,
    ):
        self.tenant = tenant
        self.guardrail = guardrail
        self.thinker = thinker
        self.supervisor = supervisor
        self.memory_cfg = memory_cfg
        logger.info(f"[AgentRunner] built | tenant={tenant.customer_id}")

    def invoke(self, question: str, user_id: str = "anonymous") -> dict:
        """
        Run one chat turn:
          1. Input guardrail
          2. Thinker  (plan stages)
          3. Supervisor (execute stages, compose answer)
        """
        from langgraph.graph import StateGraph, END

        guardrail_result = self.guardrail.check(question)
        if guardrail_result.is_blocked:
            logger.info(
                f"[AgentRunner] blocked | tenant={self.tenant.customer_id} "
                f"| reason={guardrail_result.reason}"
            )
            return {
                "response": guardrail_result.response_message
                or self.tenant.out_of_scope_msg,
                "status": "blocked_by_guardrail",
                "blocked": True,
            }

        thinker = self.thinker
        supervisor = self.supervisor

        def thinker_node(state):
            return thinker.think(state)

        def supervisor_node(state):
            return supervisor.supervise(state)

        workflow = StateGraph(dict)
        workflow.add_node("thinker", thinker_node)
        workflow.add_node("supervisor", supervisor_node)
        workflow.set_entry_point("thinker")
        workflow.add_edge("thinker", "supervisor")
        workflow.add_edge("supervisor", END)

        compiled = workflow.compile()
        state = {
            "question": question,
            "user_id": user_id,
            "tenant_id": self.tenant.customer_id,
            "workflow": [],
            "stream_chunks": [],
        }
        result = compiled.invoke(state)

        response_text = _extract_response(result)
        logger.info(
            f"[AgentRunner] success | tenant={self.tenant.customer_id} | user={user_id}"
        )
        return {"response": response_text, "status": "success", "blocked": False}


def _extract_response(result: dict) -> str:
    """Pull the best available response text out of the workflow result."""
    if result.get("messages"):
        last = result["messages"][-1]
        return getattr(last, "content", str(last))
    if result.get("stream_chunks"):
        combined = "".join(result["stream_chunks"]).strip()
        if combined:
            return combined
    if result.get("workflow"):
        last_step = result["workflow"][-1]
        text = last_step.get("node_instructions") or last_step.get("user_message", "")
        if text:
            return text
    return "Process completed."


@lru_cache(maxsize=8)
def get_agent_runner(customer_id: str) -> AgentRunner:
    """
    Build (and LRU-cache) an AgentRunner for the given customer_id.

    The cache key is customer_id only — config changes require a process
    restart (or call get_agent_runner.cache_clear()).
    """
    from .middleware import _global_registry
    from .llm_provider import get_llm, get_llm_provider

    if _global_registry is None:
        from .registry import TenantRegistry
        _fallback = TenantRegistry(
            os.path.join(os.path.dirname(__file__), "..", "config", "tenants.yaml")
        )
        tenant = _fallback.get_tenant(customer_id)
    else:
        tenant = _global_registry.get_tenant(customer_id)

    # SecurityConfig.llm_provider is used as the LLM object directly (llm.invoke())
    # CoreConfig.llm_provider is called as a zero-arg factory (llm_provider() -> llm)
    llm = get_llm()
    llm_provider = get_llm_provider()

    security_cfg = SecurityConfig(
        llm_provider=llm,
        out_of_scope_message=tenant.out_of_scope_msg,
    ).with_llm_guardrail(
        allowed_topics=tenant.allowed_topics,
        blocked_topics=tenant.blocked_topics,
        custom_instructions=tenant.custom_instructions,
    )

    core_cfg = CoreConfig(llm_provider=llm_provider).with_thinker(
        stages=list(STAGE_DEFINITIONS.keys()),
        stage_definitions=STAGE_DEFINITIONS,
        custom_instructions=(
            f"You are the official assistant for {tenant.display_name}. "
            f"{tenant.custom_instructions}"
        ),
    ).with_supervisor(
        stage_definitions=STAGE_DEFINITIONS,
        custom_instructions=(
            f"Ensure all responses follow {tenant.display_name} guidelines."
        ),
    )

    memory_cfg = MemoryConfig(llm_provider=llm_provider)

    solr_adapter = _make_solr_adapter(tenant)
    if solr_adapter:
        memory_cfg = memory_cfg.with_long_term_memory(
            memory_store=solr_adapter,
            max_facts_per_user=100,
        )

    guardrail = LLMGuardrail(security_cfg)
    thinker = SimpleThinker(core_cfg)
    supervisor = SimpleSupervisor(core_cfg)

    logger.info(f"[factory] AgentRunner built | tenant={customer_id}")
    return AgentRunner(tenant, guardrail, thinker, supervisor, memory_cfg)


def create_tenant_agent_config(
    tenant: TenantConfig,
    llm_provider,
    region: str = "us-east-1",
) -> Tuple[CoreConfig, SecurityConfig, MemoryConfig]:
    """
    Legacy helper kept for backwards compatibility with older callers.
    Prefer `get_agent_runner(customer_id)` for new code.

    llm_provider must be the LLM object itself (has .invoke()), not a lambda.
    CoreConfig receives a zero-arg lambda wrapping it.
    """
    security = SecurityConfig(
        llm_provider=llm_provider,
        out_of_scope_message=tenant.out_of_scope_msg,
    ).with_llm_guardrail(
        allowed_topics=tenant.allowed_topics,
        blocked_topics=tenant.blocked_topics,
        custom_instructions=tenant.custom_instructions,
    )

    core = CoreConfig(llm_provider=llm_provider).with_thinker(
        stages=list(STAGE_DEFINITIONS.keys()),
        stage_definitions=STAGE_DEFINITIONS,
        custom_instructions=f"You are the official assistant for {tenant.display_name}.",
    ).with_supervisor(
        stage_definitions=STAGE_DEFINITIONS,
        custom_instructions=f"Ensure responses adhere to {tenant.display_name} guidelines.",
    )

    memory = MemoryConfig(llm_provider=llm_provider)
    return core, security, memory
