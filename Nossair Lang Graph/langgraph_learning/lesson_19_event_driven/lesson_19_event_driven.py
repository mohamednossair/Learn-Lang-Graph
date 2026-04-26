"""
Lesson 19: Event-Driven Agent Architecture
==========================================
Teaches:
  - Async job queue with Redis + Celery
  - Webhook-triggered agents (GitHub PR, Slack, etc.)
  - Producer/consumer pattern for agent tasks
  - Dead-letter queue for failed tasks
  - Idempotency — safe to retry without side-effects

Prerequisites: pip install celery redis
               (Redis must be running: docker run -p 6379:6379 redis)

Without Redis: demo runs in fallback sync mode to show the patterns.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_19")

REDIS_URL    = os.getenv("REDIS_URL",    "redis://localhost:6379/0")
USE_CELERY   = os.getenv("USE_CELERY",   "false").lower() == "true"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret-change-in-production")

# ---------------------------------------------------------------------------
# WEBHOOK HMAC SIGNATURE VERIFICATION
# ---------------------------------------------------------------------------
def verify_webhook_signature(payload_bytes: bytes, signature_header: str, secret: str = WEBHOOK_SECRET) -> bool:
    """
    Verify that a webhook payload was signed by the expected sender.

    How it works (GitHub/Stripe/Slack all use this pattern):
      1. Sender computes HMAC-SHA256(secret, payload)
      2. Sender sends signature in a header: X-Hub-Signature-256: sha256=<hex>
      3. Receiver recomputes HMAC and compares with constant-time comparison

    CRITICAL: use hmac.compare_digest() NOT == for comparison.
    String equality short-circuits on first mismatch -- timing attack vector.
    hmac.compare_digest() always takes the same time regardless of match.

    Example FastAPI endpoint:
        @app.post("/webhooks/github")
        async def github_webhook(request: Request):
            body = await request.body()
            sig  = request.headers.get("X-Hub-Signature-256", "")
            if not verify_webhook_signature(body, sig):
                raise HTTPException(403, "Invalid signature")
            # safe to process
    """
    if not signature_header:
        logger.warning("[webhook] REJECTED: no signature header")
        return False

    prefix = "sha256="
    if not signature_header.startswith(prefix):
        logger.warning(f"[webhook] REJECTED: unexpected signature format: {signature_header[:20]}")
        return False

    received_sig = signature_header[len(prefix):]
    expected_sig = hmac.new(
        secret.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(received_sig, expected_sig):
        logger.warning("[webhook] REJECTED: signature mismatch (possible replay/tampering)")
        return False

    logger.info("[webhook] signature verified OK")
    return True


def sign_webhook_payload(payload_bytes: bytes, secret: str = WEBHOOK_SECRET) -> str:
    """Generate the HMAC signature for a payload (used by test helpers / mock senders)."""
    sig = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


# ---------------------------------------------------------------------------
# DEAD LETTER QUEUE (in-memory; production: separate Redis queue or SQS DLQ)
# ---------------------------------------------------------------------------
class DeadLetterQueue:
    """
    Tasks that exhausted all retries are moved here instead of silently dropped.

    Production implementation:
      Redis: LPUSH dlq:agent_tasks <serialised task JSON>
      SQS:   send_message(QueueUrl=DLQ_URL, MessageBody=json.dumps(task))

    Monitoring:
      Alert if DLQ depth > 0 (every message in DLQ = a failed task needing manual review).
      Grafana: gauge dlq_depth, alert on dlq_depth > 0 for 5m.
    """

    def __init__(self, max_size: int = 1000):
        self._queue: deque[dict] = deque(maxlen=max_size)

    def put(self, task: 'AgentTask', error: str):
        entry = {
            "task":       task.to_dict(),
            "error":      error,
            "failed_at":  datetime.now(timezone.utc).isoformat(),
            "retry_count": task.retry_count,
        }
        self._queue.append(entry)
        logger.error(
            f"[DLQ] task moved to dead-letter queue | task_id={task.task_id} | "
            f"type={task.task_type} | error={error} | dlq_depth={len(self._queue)}"
        )

    def depth(self) -> int:
        return len(self._queue)

    def drain(self) -> list[dict]:
        """Return all DLQ entries and clear the queue (for manual reprocessing)."""
        entries = list(self._queue)
        self._queue.clear()
        return entries


dead_letter_queue = DeadLetterQueue()


# ---------------------------------------------------------------------------
# FASTAPI TASK STATUS POLLING PATTERN
# ---------------------------------------------------------------------------
# Full FastAPI app is in lesson 15. This shows the polling endpoint pattern.
# To use with Celery:
#
#   from fastapi import FastAPI, HTTPException
#   from lesson_19_event_driven import enqueue_task, dead_letter_queue
#
#   @app.post("/tasks/pr-review", status_code=202)
#   async def submit_pr_review(payload: dict):
       #   task = AgentTask(task_id=str(uuid.uuid4()), task_type="pr_review", ...)
#   task_id = enqueue_task(task)
#   return {"task_id": task_id, "status": "queued",
#           "poll_url": f"/tasks/{task_id}/status"}
#
#   @app.get("/tasks/{task_id}/status")
#   async def task_status(task_id: str):
#       if CELERY_AVAILABLE:
#           result = celery_app.AsyncResult(task_id)
#           return {"task_id": task_id, "status": result.state,
#                   "result": result.result if result.ready() else None}
#       # Sync fallback: check idempotency_store
#       cached = idempotency_store.get_result(task_id)
#       if cached:
#           return {"task_id": task_id, "status": "SUCCESS", "result": cached}
#       raise HTTPException(404, "Task not found")
#
#   @app.get("/tasks/dlq/depth")
#   async def dlq_depth():
#       return {"depth": dead_letter_queue.depth()}
#
#   @app.post("/tasks/dlq/drain")  # admin endpoint
#   async def drain_dlq():
#       entries = dead_letter_queue.drain()
#       return {"drained": len(entries), "entries": entries}
#

# ---------------------------------------------------------------------------
# CELERY SETUP (with fallback)
# ---------------------------------------------------------------------------
try:
    from celery import Celery

    celery_app = Celery(
        "langgraph_tasks",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,           # enterprise: ack only after success
        task_reject_on_worker_lost=True,  # re-queue if worker dies mid-task
        task_max_retries=3,
        task_default_retry_delay=60,   # seconds between retries
    )
    CELERY_AVAILABLE = True
    logger.info("Celery initialized")
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning("Celery not installed — running in sync fallback mode")


# ---------------------------------------------------------------------------
# EVENT TYPES (webhook payloads)
# ---------------------------------------------------------------------------
@dataclass
class AgentTask:
    """Serializable task envelope — what gets put on the queue."""
    task_id: str
    task_type: str              # "pr_review" | "slack_message" | "scheduled_report"
    tenant_id: str
    user_id: str
    payload: dict[str, Any]
    idempotency_key: str        # prevents duplicate processing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "payload": self.payload,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentTask":
        return cls(**d)


# ---------------------------------------------------------------------------
# IDEMPOTENCY STORE (prevents duplicate task execution)
# ---------------------------------------------------------------------------
class IdempotencyStore:
    """
    In-memory store for demo. In production: Redis with TTL.

    Pattern: before processing, check idempotency_key.
    If already processed → return cached result, skip re-execution.
    Critical for: webhooks (GitHub may retry), payment events, email triggers.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def is_processed(self, key: str) -> bool:
        return key in self._store

    def mark_processed(self, key: str, result: Any):
        self._store[key] = {"result": result, "processed_at": datetime.now(timezone.utc).isoformat()}

    def get_result(self, key: str) -> Any:
        return self._store.get(key, {}).get("result")


idempotency_store = IdempotencyStore()


# ---------------------------------------------------------------------------
# AGENT STATES
# ---------------------------------------------------------------------------
class PRReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    pr_diff: str
    pr_number: int
    repo: str
    style_issues: list[str]
    security_issues: list[str]
    summary: str
    task_id: str


class SlackMessageState(TypedDict):
    messages: Annotated[list, add_messages]
    channel: str
    user_mention: str
    original_message: str
    response: str
    task_id: str


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOllama(model=get_ollama_model(), temperature=0)


# ---------------------------------------------------------------------------
# PR REVIEW AGENT
# ---------------------------------------------------------------------------
def pr_analyze_node(state: PRReviewState) -> dict:
    """Analyze PR diff for style and security issues."""
    logger.info(f"[pr_review] task={state['task_id']} | PR #{state['pr_number']} | repo={state['repo']}")

    prompt = f"""You are a code reviewer. Analyze this diff:
{state['pr_diff'][:2000]}

List style issues and security issues briefly. Be specific."""

    response = llm.invoke([SystemMessage(content="You are an expert code reviewer."),
                           HumanMessage(content=prompt)])

    content = response.content
    style_issues = []
    security_issues = []

    for line in content.split("\n"):
        line = line.strip()
        if "style" in line.lower() or "format" in line.lower() or "naming" in line.lower():
            style_issues.append(line)
        elif "security" in line.lower() or "injection" in line.lower() or "vulnerab" in line.lower():
            security_issues.append(line)

    return {
        "messages": [response],
        "style_issues": style_issues or ["No major style issues"],
        "security_issues": security_issues or ["No security issues detected"],
    }


def pr_summarize_node(state: PRReviewState) -> dict:
    """Synthesize findings into a PR comment."""
    style = "\n".join(f"- {s}" for s in state["style_issues"])
    security = "\n".join(f"- {s}" for s in state["security_issues"])

    summary = f"""## Automated Code Review — PR #{state["pr_number"]}

**Style Issues:**
{style}

**Security Issues:**
{security}

*Review generated by AI agent — always verify before merging.*"""

    logger.info(f"[pr_review] summary generated | task={state['task_id']}")
    return {"summary": summary}


def build_pr_review_graph(checkpointer):
    builder = StateGraph(PRReviewState)
    builder.add_node("analyze", pr_analyze_node)
    builder.add_node("summarize", pr_summarize_node)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# TASK PROCESSOR (the consumer)
# ---------------------------------------------------------------------------
def process_agent_task(task: AgentTask) -> dict:
    """
    Core processing function — dispatches tasks to the right agent graph.

    This is what the Celery worker calls (or what we call directly in sync mode).
    Pattern: idempotency check → dispatch → store result.
    """
    # IDEMPOTENCY CHECK — critical for event-driven systems
    if idempotency_store.is_processed(task.idempotency_key):
        cached = idempotency_store.get_result(task.idempotency_key)
        logger.info(f"[processor] DUPLICATE | key={task.idempotency_key} | returning cached result")
        return cached

    logger.info(f"[processor] START | task={task.task_id} | type={task.task_type}")
    start = time.perf_counter()

    checkpointer = MemorySaver()
    result = {}

    try:
        if task.task_type == "pr_review":
            graph = build_pr_review_graph(checkpointer)
            config = {"configurable": {"thread_id": task.task_id}}
            state = graph.invoke(
                {
                    "pr_diff": task.payload.get("diff", "# no diff"),
                    "pr_number": task.payload.get("pr_number", 0),
                    "repo": task.payload.get("repo", "unknown"),
                    "task_id": task.task_id,
                    "messages": [],
                    "style_issues": [],
                    "security_issues": [],
                    "summary": "",
                },
                config=config,
            )
            result = {"status": "success", "summary": state["summary"], "task_id": task.task_id}

        elif task.task_type == "slack_message":
            response = llm.invoke([
                SystemMessage(content="You are a helpful Slack bot. Be concise."),
                HumanMessage(content=task.payload.get("message", "")),
            ])
            result = {
                "status": "success",
                "response": response.content,
                "channel": task.payload.get("channel", ""),
                "task_id": task.task_id,
            }

        else:
            result = {"status": "error", "error": f"Unknown task type: {task.task_type}"}

    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error(f"[processor] FAILED | task={task.task_id} | error={exc} | elapsed={elapsed:.0f}ms")

        if task.retry_count < task.max_retries:
            logger.info(f"[processor] RETRY {task.retry_count + 1}/{task.max_retries}")
            task.retry_count += 1
            time.sleep(2 ** task.retry_count)   # exponential backoff
            return process_agent_task(task)

        # Retries exhausted — move to Dead Letter Queue for manual review
        dead_letter_queue.put(task, str(exc))
        result = {"status": "failed", "error": str(exc), "task_id": task.task_id,
                  "dlq": True, "dlq_depth": dead_letter_queue.depth()}

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[processor] DONE | task={task.task_id} | elapsed={elapsed:.0f}ms | status={result['status']}")

    # Store for idempotency
    idempotency_store.mark_processed(task.idempotency_key, result)
    return result


# ---------------------------------------------------------------------------
# CELERY TASK (the async worker)
# ---------------------------------------------------------------------------
if CELERY_AVAILABLE:
    @celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
    def execute_agent_task(self, task_dict: dict):
        """
        Celery task — runs in a separate worker process.

        Worker start command:
          celery -A lesson_19_event_driven.lesson_19_event_driven worker --loglevel=info

        Enterprise pattern:
          - Web server (FastAPI) → enqueues task (returns immediately → HTTP 202 Accepted)
          - Celery worker       → processes task async
          - Redis               → stores result
          - Client              → polls GET /task/{id}/status
        """
        task = AgentTask.from_dict(task_dict)
        try:
            return process_agent_task(task)
        except Exception as exc:
            logger.error(f"[celery] task {task.task_id} failed: {exc}")
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)


# ---------------------------------------------------------------------------
# TASK PRODUCER
# ---------------------------------------------------------------------------
def enqueue_task(task: AgentTask) -> str:
    """
    Enqueue a task for async processing.
    Returns task_id for status polling.

    With Celery: sends to Redis queue, worker picks up async.
    Without Celery: processes synchronously (dev/demo mode).
    """
    if CELERY_AVAILABLE and USE_CELERY:
        result = execute_agent_task.apply_async(args=[task.to_dict()], task_id=task.task_id)
        logger.info(f"[producer] enqueued | task={task.task_id} | celery_id={result.id}")
        return result.id
    else:
        # Sync fallback
        logger.info(f"[producer] sync mode | processing task={task.task_id} directly")
        result = process_agent_task(task)
        return task.task_id


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def run_demo():
    print("\n" + "=" * 60)
    print("DEMO 0: Webhook HMAC signature verification")
    print("=" * 60)
    payload = json.dumps({"pr_number": 42, "repo": "acme/backend"}).encode()
    valid_sig   = sign_webhook_payload(payload)
    invalid_sig = "sha256=deadbeef00000000000000000000000000000000000000000000000000000000"
    print(f"  Valid signature:   {verify_webhook_signature(payload, valid_sig)}")
    print(f"  Invalid signature: {verify_webhook_signature(payload, invalid_sig)}")
    print(f"  Missing header:    {verify_webhook_signature(payload, '')}")

    print("\n" + "=" * 60)
    print("DEMO 1: PR review webhook event")
    print("=" * 60)

    pr_task = AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="pr_review",
        tenant_id="acme-corp",
        user_id="github-webhook",
        idempotency_key="pr-review-acme-42-abc123",
        payload={
            "pr_number": 42,
            "repo": "acme-corp/backend",
            "diff": """
+def get_user(user_id):
+    query = f"SELECT * FROM users WHERE id={user_id}"  # SQL injection risk!
+    return db.execute(query)
+
+def processData( data,extra ):    # style: bad naming, extra space
+    return data+extra
""",
        },
    )

    task_id = enqueue_task(pr_task)
    print(f"  Task enqueued: {task_id}")
    result = idempotency_store.get_result(pr_task.idempotency_key)
    if result:
        print(f"  Status: {result['status']}")
        print(f"  Summary preview:\n{result.get('summary', '')[:300]}")

    print("\n" + "=" * 60)
    print("DEMO 2: Idempotency — same event sent twice (simulates webhook retry)")
    print("=" * 60)

    # Same idempotency_key → should NOT re-process
    pr_task_duplicate = AgentTask(
        task_id=str(uuid.uuid4()),   # different task_id
        task_type="pr_review",
        tenant_id="acme-corp",
        user_id="github-webhook",
        idempotency_key="pr-review-acme-42-abc123",   # SAME key!
        payload={"pr_number": 42, "repo": "acme-corp/backend", "diff": "different diff"},
    )
    task_id2 = enqueue_task(pr_task_duplicate)
    print(f"  Second task enqueued: {task_id2}")
    print("  → Should return cached result without re-processing (check logs above)")

    print("\n" + "=" * 60)
    print("DEMO 3: Slack message event")
    print("=" * 60)

    slack_task = AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="slack_message",
        tenant_id="globex-inc",
        user_id="U123456",
        idempotency_key=f"slack-{uuid.uuid4()}",
        payload={
            "channel": "#general",
            "message": "@bot What is the revenue forecast for Q3?",
        },
    )
    enqueue_task(slack_task)
    result = idempotency_store.get_result(slack_task.idempotency_key)
    if result:
        print(f"  channel={result.get('channel')}")
        print(f"  response={result.get('response', '')[:100]}...")

    print("\n" + "=" * 60)
    print("Architecture: FastAPI + Celery + Redis")
    print("=" * 60)
    print("""
  [Webhook/Client]
       │  POST /tasks/pr-review   (HTTP 202 Accepted immediately)
       ▼
  [FastAPI Server]
       │  enqueue_task() → Redis queue
       ▼
  [Redis Queue]
       │
       ▼
  [Celery Worker(s)]  ← scale horizontally for more throughput
       │  process_agent_task()
       │  graph.invoke()
       ▼
  [Result stored in Redis]

  [Client polls]
       │  GET /tasks/{task_id}/status
       ▼
  [FastAPI Server]
       │  celery_app.AsyncResult(task_id).status
       ▼
  {"status": "SUCCESS", "result": {...}}
    """)


if __name__ == "__main__":
    run_demo()
    print("\n✓ Lesson 19 complete.")
    print("Next: Lesson 20 — Cost control, governance & enterprise capstone")
