"""
Lesson 21: AWS Bedrock — Production LLMs Without Ollama
========================================================
Teaches:
  - Replacing ChatOllama with AWS Bedrock models (Claude, Titan, Llama)
  - boto3 session management and credential patterns (IAM roles, profiles, env vars)
  - Bedrock model IDs, regions, and inference profiles
  - Streaming responses from Bedrock inside LangGraph
  - Cost-aware model selection (Claude Haiku vs Sonnet vs Opus)
  - Bedrock Guardrails for content filtering
  - Migrating any existing LangGraph agent from Ollama to Bedrock
  - Running locally (with AWS credentials) or in AWS (EC2/Lambda with IAM role)

Prerequisites:
    pip install boto3 langchain-aws

AWS Setup:
    Option A (local dev):
        aws configure  # sets ~/.aws/credentials
        # or: export AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

    Option B (EC2/ECS/Lambda):
        Attach IAM role with policy:
        {
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            "Resource": "arn:aws:bedrock:*::foundation-model/*"
        }

    Enable model access in AWS Console:
        AWS Console → Bedrock → Model access → Enable Claude / Titan / Llama

Environment variables (see .env.example):
    AWS_REGION                  (default: us-east-1)
    BEDROCK_MODEL_ID            (default: anthropic.claude-3-haiku-20240307-v1:0)
    AWS_PROFILE                 (optional: named profile from ~/.aws/credentials)
    BEDROCK_GUARDRAIL_ID        (optional: content guardrail)
    BEDROCK_GUARDRAIL_VERSION   (optional: DRAFT or version number)
"""

import logging
import os
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_21")

# ===========================================================================
# SECTION 1 — MODEL CONFIGURATION
# ===========================================================================
#
# AWS Bedrock model IDs (as of 2024 — check AWS console for latest):
#
# ANTHROPIC CLAUDE (recommended for agents — best instruction following):
#   anthropic.claude-3-haiku-20240307-v1:0    ← fast + cheap  (~$0.00025/1k input)
#   anthropic.claude-3-sonnet-20240229-v1:0   ← balanced      (~$0.003/1k input)
#   anthropic.claude-3-opus-20240229-v1:0     ← most capable  (~$0.015/1k input)
#   anthropic.claude-instant-v1               ← legacy fast
#
# AMAZON TITAN (AWS-native, no separate vendor agreement):
#   amazon.titan-text-express-v1              ← general purpose
#   amazon.titan-text-lite-v1                 ← fast + cheap
#   amazon.titan-text-premier-v1:0            ← most capable Titan
#
# META LLAMA (open-weight, similar to Ollama models):
#   meta.llama3-8b-instruct-v1:0              ← small, fast (similar to local llama3)
#   meta.llama3-70b-instruct-v1:0             ← large, powerful
#
# MISTRAL:
#   mistral.mistral-7b-instruct-v0:2
#   mistral.mixtral-8x7b-instruct-v0:1
#
# MIGRATION FROM OLLAMA:
#   ChatOllama(model="llama3.2") → ChatBedrockConverse(model_id="meta.llama3-8b-instruct-v1:0")
#   ChatOllama(model="llama3.2")   → ChatBedrockConverse(model_id="anthropic.claude-3-haiku-20240307-v1:0")
#
# ===========================================================================

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")  # None = use default credential chain
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0",  # cheapest Claude, great for learning
)
BEDROCK_GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID")
BEDROCK_GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

# ===========================================================================
# SECTION 2 — BEDROCK CLIENT SETUP
# ===========================================================================
#
# AWS credential resolution order (boto3 default chain):
#   1. Explicit env vars: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
#   2. Named profile:     AWS_PROFILE or ~/.aws/credentials [profile]
#   3. IAM role (EC2/ECS/Lambda): instance metadata service (IMDS)
#   4. AWS SSO session
#
# Enterprise best practice:
#   - Local dev:    aws configure or AWS SSO login
#   - Production:   IAM role attached to EC2/ECS task — NO hardcoded keys
#
# ===========================================================================

def create_bedrock_llm(
    model_id: str = BEDROCK_MODEL_ID,
    region: str = AWS_REGION,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    streaming: bool = False,
) -> "ChatBedrockConverse":
    """
    Create a Bedrock LLM instance using ChatBedrockConverse.

    ChatBedrockConverse vs ChatBedrock:
      - ChatBedrockConverse: uses the newer Converse API (unified across all models)
        Supports: streaming, tool calling, system messages natively
        Recommended for: all new code
      - ChatBedrock: uses the older InvokeModel API (model-specific request format)
        Use only if: model not yet supported by Converse API

    boto3 session:
      If AWS_PROFILE is set: use named profile (for local dev with multiple accounts)
      If not set: use default credential chain (IAM role in production)
    """
    try:
        import boto3
        from langchain_aws import ChatBedrockConverse
    except ImportError:
        raise ImportError(
            "Install AWS dependencies: pip install boto3 langchain-aws"
        )

    # Build boto3 session (handles all credential sources)
    if AWS_PROFILE:
        logger.info(f"Using AWS profile: {AWS_PROFILE}")
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=region)
    else:
        logger.info(f"Using default AWS credential chain, region={region}")
        session = boto3.Session(region_name=region)

    # Build bedrock-runtime client from session
    bedrock_client = session.client("bedrock-runtime")

    # Optional guardrail config
    guardrail_config = None
    if BEDROCK_GUARDRAIL_ID:
        guardrail_config = {
            "guardrailIdentifier": BEDROCK_GUARDRAIL_ID,
            "guardrailVersion": BEDROCK_GUARDRAIL_VERSION,
        }
        logger.info(f"Bedrock guardrail enabled: {BEDROCK_GUARDRAIL_ID}")

    kwargs = dict(
        model_id=model_id,
        client=bedrock_client,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
    if guardrail_config:
        kwargs["guardrail_config"] = guardrail_config

    return ChatBedrockConverse(**kwargs)


# ===========================================================================
# SECTION 3 — BEDROCK AVAILABILITY CHECK
# ===========================================================================

def check_bedrock_available() -> bool:
    """
    Check if AWS credentials and Bedrock access are available.
    Used for graceful fallback to Ollama in demo mode.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION,
        )
        client = session.client("bedrock-runtime")
        # Minimal call: list foundation models (doesn't invoke a model)
        sts = session.client("sts")
        sts.get_caller_identity()
        return True
    except Exception as e:
        logger.debug(f"Bedrock not available: {e}")
        return False


BEDROCK_AVAILABLE = check_bedrock_available()


def get_llm(
    model_id: str = BEDROCK_MODEL_ID,
    temperature: float = 0.1,
    streaming: bool = False,
):
    """
    Get LLM: Bedrock if credentials available, else Ollama fallback.

    This pattern lets the lesson run on any machine:
      - With AWS credentials: uses real Bedrock
      - Without credentials: falls back to local Ollama (same as lessons 1-20)
    """
    if BEDROCK_AVAILABLE:
        logger.info(f"[LLM] Using AWS Bedrock: {model_id}")
        return create_bedrock_llm(
            model_id=model_id,
            temperature=temperature,
            streaming=streaming,
        )
    else:
        logger.info("[LLM] Bedrock not available — falling back to ChatOllama")
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=get_ollama_model(), temperature=temperature)
        except ImportError:
            raise RuntimeError(
                "Neither AWS credentials nor Ollama available. "
                "Run: aws configure  OR  ollama pull llama3.2"
            )


# ===========================================================================
# SECTION 4 — LANGGRAPH STATE
# ===========================================================================

class BedrockAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    model_id: str            # which Bedrock model was used
    input_tokens: int        # track cost
    output_tokens: int       # track cost
    estimated_cost_usd: float


# ===========================================================================
# SECTION 5 — COST ESTIMATION
# ===========================================================================
#
# Bedrock pricing (us-east-1, on-demand, approximate 2024 prices):
# Always check https://aws.amazon.com/bedrock/pricing/ for current rates.
#
MODEL_PRICING = {
    # model_id: (input_per_1k_tokens, output_per_1k_tokens) in USD
    "anthropic.claude-3-haiku-20240307-v1:0":  (0.00025,  0.00125),
    "anthropic.claude-3-sonnet-20240229-v1:0": (0.003,    0.015),
    "anthropic.claude-3-opus-20240229-v1:0":   (0.015,    0.075),
    "amazon.titan-text-express-v1":            (0.0002,   0.0006),
    "amazon.titan-text-lite-v1":               (0.00015,  0.0002),
    "meta.llama3-8b-instruct-v1:0":            (0.0003,   0.0006),
    "meta.llama3-70b-instruct-v1:0":           (0.00265,  0.0035),
    "mistral.mistral-7b-instruct-v0:2":        (0.00015,  0.0002),
    "mistral.mixtral-8x7b-instruct-v0:1":      (0.00045,  0.0007),
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a Bedrock invocation."""
    if model_id not in MODEL_PRICING:
        return 0.0
    in_rate, out_rate = MODEL_PRICING[model_id]
    return (input_tokens / 1000 * in_rate) + (output_tokens / 1000 * out_rate)


# ===========================================================================
# SECTION 6 — GRAPH NODES
# ===========================================================================

def chat_node(state: BedrockAgentState) -> dict:
    """
    Core chat node — calls Bedrock via LangChain.

    Token usage extraction:
      Bedrock returns usage in response.response_metadata["usage"]
      Keys vary slightly by model family:
        Claude:  {"input_tokens": N, "output_tokens": N}
        Titan:   {"inputTokenCount": N, "outputTokenCount": N}
        Llama:   {"prompt_token_count": N, "generation_token_count": N}
      ChatBedrockConverse normalises these to input_tokens/output_tokens.
    """
    llm = get_llm(model_id=state.get("model_id", BEDROCK_MODEL_ID))
    logger.info(f"[chat_node] Invoking model: {state.get('model_id', BEDROCK_MODEL_ID)}")

    response = llm.invoke(state["messages"])

    # Extract token usage from response metadata
    usage = getattr(response, "response_metadata", {}).get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    model_id = state.get("model_id", BEDROCK_MODEL_ID)
    cost = estimate_cost(model_id, input_tokens, output_tokens)

    logger.info(
        f"[chat_node] tokens: in={input_tokens} out={output_tokens} "
        f"cost=${cost:.6f}"
    )

    return {
        "messages": [response],
        "input_tokens": state.get("input_tokens", 0) + input_tokens,
        "output_tokens": state.get("output_tokens", 0) + output_tokens,
        "estimated_cost_usd": state.get("estimated_cost_usd", 0.0) + cost,
    }


def model_selector_node(state: BedrockAgentState) -> dict:
    """
    Selects the cheapest Bedrock model sufficient for the task.

    Strategy:
      - Short / simple question  → Haiku (cheapest, fastest)
      - Medium complexity        → Sonnet (balanced)
      - Complex / long context   → Opus (most capable)

    In production: replace keyword heuristic with a small classifier
    or use the complexity router from Lesson 20 patterns.
    """
    last_message = state["messages"][-1].content if state["messages"] else ""
    word_count = len(last_message.split())
    lower = last_message.lower()

    complex_keywords = {
        "analyze", "compare", "architecture", "design", "explain in detail",
        "write code", "implement", "debug", "refactor", "review",
    }

    if word_count < 15 and not any(k in lower for k in complex_keywords):
        selected = "anthropic.claude-3-haiku-20240307-v1:0"
        tier = "fast"
    elif any(k in lower for k in complex_keywords) or word_count > 100:
        selected = "anthropic.claude-3-sonnet-20240229-v1:0"
        tier = "balanced"
    else:
        selected = "anthropic.claude-3-haiku-20240307-v1:0"
        tier = "fast"

    logger.info(f"[model_selector] tier={tier} → {selected}")
    return {"model_id": selected}


def system_prompt_node(state: BedrockAgentState) -> dict:
    """
    Injects a system prompt if messages don't already have one.
    Bedrock Claude requires the system message to be separate from the
    human/assistant turn — ChatBedrockConverse handles this automatically.
    """
    has_system = any(
        isinstance(m, SystemMessage) for m in state.get("messages", [])
    )
    if not has_system:
        system = SystemMessage(
            content=(
                "You are a helpful enterprise AI assistant powered by AWS Bedrock. "
                "Be concise, accurate, and professional."
            )
        )
        return {"messages": [system]}
    return {}


# ===========================================================================
# SECTION 7 — STREAMING NODE
# ===========================================================================

def streaming_chat_node(state: BedrockAgentState):
    """
    Streaming variant: yields chunks as they arrive from Bedrock.

    Bedrock streaming flow:
      invoke_model_with_response_stream → AsyncIterator[chunk]
      ChatBedrockConverse.stream()      → Iterator[AIMessageChunk]

    Use case: FastAPI StreamingResponse or terminal typewriter effect.
    This node is NOT wired into the main graph (it's a standalone demo).
    """
    llm = get_llm(
        model_id=state.get("model_id", BEDROCK_MODEL_ID),
        streaming=True,
    )
    logger.info("[streaming_chat_node] Starting stream...")
    full_response = ""
    for chunk in llm.stream(state["messages"]):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content:
            print(content, end="", flush=True)
            full_response += content
    print()  # newline after streaming
    return {"messages": [AIMessage(content=full_response)]}


# ===========================================================================
# SECTION 8 — BUILD THE GRAPH
# ===========================================================================

def build_bedrock_graph(checkpointer=None):
    """
    Graph: START → system_prompt → model_selector → chat → END

    Nodes in order:
      1. system_prompt_node  — inject system instructions once
      2. model_selector_node — pick cheapest sufficient model
      3. chat_node           — call Bedrock, extract token usage
    """
    graph = StateGraph(BedrockAgentState)

    graph.add_node("system_prompt", system_prompt_node)
    graph.add_node("model_selector", model_selector_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "system_prompt")
    graph.add_edge("system_prompt", "model_selector")
    graph.add_edge("model_selector", "chat")
    graph.add_edge("chat", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# ===========================================================================
# SECTION 9 — MIGRATION GUIDE: OLLAMA → BEDROCK
# ===========================================================================
#
# ANY existing lesson can be migrated from Ollama to Bedrock by changing ONE line:
#
# BEFORE (Ollama — local, free):
#   from langchain_ollama import ChatOllama
#   llm = ChatOllama(model="llama3.2", temperature=0.1)
#
# AFTER (Bedrock — AWS, pay per token):
#   from langchain_aws import ChatBedrockConverse
#   llm = ChatBedrockConverse(
#       model_id="anthropic.claude-3-haiku-20240307-v1:0",
#       client=boto3.Session().client("bedrock-runtime"),
#   )
#
# EVERYTHING ELSE STAYS THE SAME:
#   llm.invoke(messages)       ← same
#   llm.stream(messages)       ← same
#   llm.bind_tools(tools)      ← same (Claude Haiku/Sonnet support tool use)
#   with_structured_output()   ← same
#   graph.invoke() / ainvoke() ← same
#
# WHY BEDROCK INSTEAD OF OLLAMA?
# ┌─────────────────┬──────────────────────────────┬────────────────────────────┐
# │ Feature         │ Ollama (local)               │ AWS Bedrock                │
# ├─────────────────┼──────────────────────────────┼────────────────────────────┤
# │ Cost            │ Free (GPU/CPU)               │ Pay per token              │
# │ Setup           │ Install + pull model         │ AWS account + IAM          │
# │ Model quality   │ Varies by model/GPU          │ Best-in-class (Claude)     │
# │ Production use  │ Limited (single machine)     │ Scalable, HA, multi-region │
# │ Context window  │ 8k-128k depends on model     │ 200k (Claude 3)            │
# │ Data privacy    │ Fully local                  │ AWS VPC + no training opt  │
# │ Tool calling    │ Model-dependent              │ Native (Claude, Titan)     │
# │ Compliance      │ Your responsibility          │ SOC2, HIPAA, GDPR-ready    │
# └─────────────────┴──────────────────────────────┴────────────────────────────┘
#
# WHEN TO USE BEDROCK:
#   - Production systems (not just local dev)
#   - Team collaboration (no GPU required on each developer machine)
#   - Compliance requirements (HIPAA, FedRAMP, SOC2)
#   - Context windows > 32k tokens
#   - Best-in-class models (Claude 3 > most local models)
#
# WHEN TO KEEP OLLAMA:
#   - Learning / experimentation (this curriculum lessons 1-20)
#   - Air-gapped environments (no internet)
#   - Cost-sensitive batch processing with open-weight models
#   - Data that must never leave your machine
#
# ===========================================================================


# ===========================================================================
# SECTION 10 — BEDROCK GUARDRAILS
# ===========================================================================
#
# AWS Bedrock Guardrails let you filter content at the infrastructure level,
# before your application code ever sees the response.
#
# Guardrail capabilities:
#   - Content filters: block harmful/hate/violence content
#   - Topic denial: block specific topics (e.g. competitor names)
#   - Word filters: block specific words/phrases
#   - PII detection: detect + mask SSN, credit card, email in responses
#   - Grounding: check if response is grounded in provided context (RAG)
#
# Setup (AWS Console):
#   Bedrock → Guardrails → Create guardrail
#   Copy the Guardrail ID → set BEDROCK_GUARDRAIL_ID env var
#
# How it works with ChatBedrockConverse:
#   The guardrail is applied by Bedrock BEFORE returning the response.
#   If triggered: response.response_metadata["amazon-bedrock-guardrailAction"] = "INTERVENED"
#   The content is replaced with a safe message automatically.
#
# Example:
#   guardrail_config = {
#       "guardrailIdentifier": "abc123",
#       "guardrailVersion": "DRAFT",
#   }
#   llm = ChatBedrockConverse(model_id=..., guardrail_config=guardrail_config)
#
# ===========================================================================


# ===========================================================================
# SECTION 11 — DEMO
# ===========================================================================

def run_demo():
    """
    Demonstrates Bedrock integration with LangGraph.

    Sections:
      1. Model info display
      2. Simple Q&A with token tracking
      3. Model selection routing (auto-selects Haiku vs Sonnet)
      4. Multi-turn conversation with memory (thread_id)
      5. Cost summary
    """
    print("\n" + "=" * 60)
    print("LESSON 21 — AWS Bedrock Integration")
    print("=" * 60)

    if BEDROCK_AVAILABLE:
        print(f"  Mode:    AWS Bedrock (real)")
        print(f"  Region:  {AWS_REGION}")
        print(f"  Model:   {BEDROCK_MODEL_ID}")
        if AWS_PROFILE:
            print(f"  Profile: {AWS_PROFILE}")
        else:
            print(f"  Creds:   default chain (env / IAM role)")
    else:
        print("  Mode:    FALLBACK (Ollama — AWS credentials not found)")
        print("  To use Bedrock: run 'aws configure' then re-run this lesson")

    if BEDROCK_GUARDRAIL_ID:
        print(f"  Guardrail: {BEDROCK_GUARDRAIL_ID} ({BEDROCK_GUARDRAIL_VERSION})")

    # -------------------------------------------------------------------
    # Demo 1: Simple Q&A
    # -------------------------------------------------------------------
    print("\n--- Demo 1: Simple Q&A with Token Tracking ---")
    graph = build_bedrock_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    question = "What is AWS Bedrock in one sentence?"
    print(f"  Q: {question}")

    state = {
        "messages": [HumanMessage(content=question)],
        "model_id": BEDROCK_MODEL_ID,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

    result = graph.invoke(state, config)
    answer = result["messages"][-1].content
    print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print(f"  Tokens: in={result['input_tokens']} out={result['output_tokens']}")
    print(f"  Cost:   ${result['estimated_cost_usd']:.6f}")

    # -------------------------------------------------------------------
    # Demo 2: Model Selection — auto-routes to cheapest sufficient model
    # -------------------------------------------------------------------
    print("\n--- Demo 2: Model Selection (Auto-Routes to Cheapest Model) ---")

    test_questions = [
        ("Hi!", "Simple greeting → should pick Haiku"),
        ("What is 2+2?", "Simple math → should pick Haiku"),
        (
            "Please analyze and compare the architecture of microservices "
            "vs monolithic systems in detail.",
            "Complex → should pick Sonnet",
        ),
    ]

    for question, expected in test_questions:
        state2 = {
            "messages": [HumanMessage(content=question)],
            "model_id": BEDROCK_MODEL_ID,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        config2 = {"configurable": {"thread_id": f"demo-2-{hash(question)}"}}
        result2 = graph.invoke(state2, config2)
        selected_model = result2.get("model_id", BEDROCK_MODEL_ID)
        short_model = selected_model.split(".")[1].split("-")[1] if "." in selected_model else selected_model
        print(f"  Q: {question[:50]:<50} → model: {short_model:10} | {expected}")

    # -------------------------------------------------------------------
    # Demo 3: Multi-turn conversation (memory via MemorySaver + thread_id)
    # -------------------------------------------------------------------
    print("\n--- Demo 3: Multi-Turn Conversation (Persistent Memory) ---")
    config3 = {"configurable": {"thread_id": "bedrock-session-001"}}
    turn_total_cost = 0.0

    turns = [
        "My name is Alice and I work with AWS services.",
        "What cloud service did I mention I work with?",
        "What is my name?",
    ]

    for i, question in enumerate(turns, 1):
        state3 = {
            "messages": [HumanMessage(content=question)],
            "model_id": BEDROCK_MODEL_ID,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        result3 = graph.invoke(state3, config3)
        answer3 = result3["messages"][-1].content
        turn_cost = result3["estimated_cost_usd"]
        turn_total_cost += turn_cost
        print(f"  Turn {i}: Q={question}")
        print(f"         A={answer3[:120]}{'...' if len(answer3) > 120 else ''}")
        print(f"         cost=${turn_cost:.6f}")

    print(f"\n  Session total cost: ${turn_total_cost:.6f}")

    # -------------------------------------------------------------------
    # Demo 4: Cost comparison across models
    # -------------------------------------------------------------------
    print("\n--- Demo 4: Cost Comparison Across Bedrock Models ---")
    sample_question = "Explain LangGraph in two sentences."
    # Simulated token counts for illustration (100 in, 80 out)
    sim_input, sim_output = 100, 80
    print(f"  Sample: '{sample_question}'")
    print(f"  Simulated usage: {sim_input} input + {sim_output} output tokens")
    print()
    print(f"  {'Model':<52} {'Cost per call':>15}")
    print("  " + "-" * 69)
    for model_id, (in_rate, out_rate) in MODEL_PRICING.items():
        cost = estimate_cost(model_id, sim_input, sim_output)
        short = model_id.replace("anthropic.", "").replace("amazon.", "").replace("meta.", "").replace("mistral.", "")
        print(f"  {short:<52} ${cost:.6f}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n--- Summary ---")
    print("  ✓ AWS Bedrock replaces Ollama with one line of code change")
    print("  ✓ ChatBedrockConverse is a drop-in for ChatOllama")
    print("  ✓ Token usage tracked automatically via response_metadata")
    print("  ✓ Model selection routes to cheapest sufficient model")
    print("  ✓ MemorySaver + thread_id: multi-turn memory works identically")
    if BEDROCK_AVAILABLE:
        print(f"  ✓ Real Bedrock invocations used — check AWS Cost Explorer")
    else:
        print("  ⚠  Ran in Ollama fallback mode — set AWS credentials to use Bedrock")
    print()


# ===========================================================================
# SECTION 12 — TOOL CALLING WITH BEDROCK
# ===========================================================================
#
# Claude 3 (Haiku/Sonnet/Opus) on Bedrock fully supports tool calling
# via the Converse API — identical to how bind_tools() works with Ollama.
#
# EXAMPLE (same pattern as Lesson 4):
#
#   from langchain_core.tools import tool
#   from langgraph.prebuilt import ToolNode
#
#   @tool
#   def get_weather(city: str) -> str:
#       """Get weather for a city."""
#       return f"Sunny, 22°C in {city}"
#
#   tools = [get_weather]
#   llm = create_bedrock_llm()
#   llm_with_tools = llm.bind_tools(tools)
#   tool_node = ToolNode(tools)
#
#   # Then build the ReAct graph exactly as in Lesson 4 — nothing changes.
#
# SUPPORTED MODELS FOR TOOL CALLING on Bedrock (Converse API):
#   ✓ anthropic.claude-3-haiku
#   ✓ anthropic.claude-3-sonnet
#   ✓ anthropic.claude-3-opus
#   ✓ amazon.titan-text-premier-v1:0
#   ✗ amazon.titan-text-express-v1 (no tool calling)
#   ✗ meta.llama3-8b-instruct-v1:0 (limited tool calling support)
#
# ===========================================================================


if __name__ == "__main__":
    run_demo()
