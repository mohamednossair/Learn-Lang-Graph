"""
LLM Provider — Ollama (primary) with AWS Bedrock fallback.

Priority:
  1. Ollama  (USE_OLLAMA=true OR LLM_PROVIDER=ollama)
  2. Bedrock (USE_BEDROCK=true OR LLM_PROVIDER=bedrock)
  3. Raises  RuntimeError if neither is configured

Environment variables:
  LLM_PROVIDER        = "ollama" | "bedrock"   (default: "ollama")
  OLLAMA_MODEL        = model name              (default: "llama3.2")
  OLLAMA_BASE_URL     = base url                (default: "http://localhost:11434")
  BEDROCK_MODEL_ID    = model id                (default: "anthropic.claude-3-haiku-20240307-v1:0")
  AWS_REGION          = region                  (default: "us-east-1")
"""

import logging
import os
from functools import lru_cache

logger = logging.getLogger("llm_provider")


def _build_ollama():
    """Instantiate ChatOllama from langchain-ollama."""
    from langchain_ollama import ChatOllama

    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    logger.info(f"[LLM] Ollama | model={model} | url={base_url}")
    return llm


def _build_bedrock():
    """Instantiate ChatBedrock from langchain-aws."""
    from langchain_aws import ChatBedrock

    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-haiku-20240307-v1:0",
    )
    region = os.getenv("AWS_REGION", "us-east-1")
    llm = ChatBedrock(model_id=model_id, region_name=region)
    logger.info(f"[LLM] Bedrock | model={model_id} | region={region}")
    return llm


@lru_cache(maxsize=1)
def _get_cached_llm():
    """Build and cache the LLM instance (process-scoped singleton)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        try:
            return _build_ollama()
        except Exception as exc:
            logger.warning(f"[LLM] Ollama unavailable ({exc}), trying Bedrock...")
            return _build_bedrock()

    if provider == "bedrock":
        try:
            return _build_bedrock()
        except Exception as exc:
            logger.warning(f"[LLM] Bedrock unavailable ({exc}), trying Ollama...")
            return _build_ollama()

    raise RuntimeError(
        f"Unknown LLM_PROVIDER='{provider}'. Use 'ollama' or 'bedrock'."
    )


def get_llm():
    """
    Return the cached LLM instance.

    This is what you pass as llm_provider to se2_agent_shared configs.
    The library calls llm_provider() — so we return the LLM directly;
    callers that need a factory lambda should use `get_llm_provider()`.
    """
    return _get_cached_llm()


def get_llm_provider():
    """
    Return a zero-arg callable that returns the LLM.

    Use this when se2_agent_shared config expects `llm_provider=callable`.
    Example:
        SecurityConfig(llm_provider=get_llm_provider())
    """
    llm = _get_cached_llm()
    return lambda: llm
