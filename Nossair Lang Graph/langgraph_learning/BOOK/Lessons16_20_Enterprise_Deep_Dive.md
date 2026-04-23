# LangGraph Enterprise Deep Dive — Lessons 16–21

> **Who this is for:** Engineers who completed Lessons 1–15 and want staff/principal-level
> understanding of enterprise LangGraph patterns. Each section: internals, failure modes,
> anti-patterns, and 7 interview Q&A.

---

# Lesson 16 — Oracle 19c Persistence & Async Execution

> **Primary:** `lesson_16_postgres_async/lesson_16_oracle_async.py`

---

## How LangGraph Checkpointing Works Internally

Every time a node completes, LangGraph calls `checkpointer.put()`:

```
graph.invoke(state, config)
  ├─ node_1 runs → checkpointer.put(CheckpointTuple(
  │     config={"configurable": {"thread_id": "abc"}},
  │     checkpoint={"id": "uuid", "channel_values": {...state...}},
  │     metadata={"step": 1},
  │  ))
  ├─ node_2 runs → checkpointer.put(..., step=2)
  └─ next invoke() → checkpointer.get_tuple(config) → resumes from last step
```

### OracleSaver must implement

```python
class OracleSaver(BaseCheckpointSaver):
    def get_tuple(self, config) -> CheckpointTuple | None: ...  # load latest
    def put(self, config, checkpoint, metadata, new_versions) -> dict: ...  # save
    def list(self, config, *, filter, before, limit) -> Iterator[CheckpointTuple]: ...  # time-travel
```

---

## Oracle MERGE Anatomy

```sql
MERGE INTO langgraph_checkpoints dst
USING (SELECT :1 AS thread_id FROM dual) src   -- dual = Oracle's 1-row dummy table
ON (dst.thread_id = src.thread_id)
WHEN MATCHED THEN
    UPDATE SET checkpoint = :2, metadata = :3, ts = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint, metadata) VALUES (:4, :5, :6)
```

Oracle holds a row-level exclusive lock for the entire MERGE — concurrent writes from
10 API pods for the same `thread_id` are serialised safely. No `INSERT OR REPLACE`
exists in Oracle; MERGE is the correct pattern.

---

## Connection Pool — Critical Details

```python
pool = oracledb.create_pool(
    min=2,       # 2 warm connections always open (eliminates first-request latency)
    max=10,      # hard cap: api_workers × 2. 11th request BLOCKS until a conn frees
    increment=1, # grow one connection at a time, not burst
    timeout=30,  # fail fast when pool exhausted (don't hang)
    ping_interval=60,  # detect stale connections
)

# CRITICAL: return connection to pool, never close it
conn = pool.acquire()
# ... use conn ...
pool.release(conn)   # ✅ returns to pool
# conn.close()       # ❌ destroys the connection — pool drains silently
```

---

## CLOB vs VARCHAR2

```
VARCHAR2  → max 32,767 bytes
CLOB      → max 4 GB

LangGraph checkpoint sizes:
  5-message chatbot    → ~2 KB  (VARCHAR2 works but is risky)
  100-message session  → ~80 KB (VARCHAR2 FAILS, CLOB required)
  RAG with doc chunks  → ~500 KB
  Multi-agent session  → ~2 MB

Rule: ALWAYS use CLOB. Never size for "current" message count.
```

---

## Async vs Sync — The Event Loop Contract

```python
# ❌ WRONG — blocks entire FastAPI event loop during LLM call
async def ask_endpoint(...):
    result = graph.invoke(state, config)   # sync, blocks for 2-10s

# ✅ CORRECT — async nodes
async def ask_endpoint(...):
    result = await graph.ainvoke(state, config)

# ✅ CORRECT — sync graph in async endpoint
async def ask_endpoint(...):
    result = await asyncio.to_thread(graph.invoke, state, config)
```

### asyncio.gather() timing

```python
# Sequential: time_A + time_B + time_C = 6s (3 × 2s LLM calls)
for user in [a, b, c]:
    await graph.ainvoke(user.state, user.config)

# Concurrent: max(time_A, time_B, time_C) = 2s
await asyncio.gather(
    graph.ainvoke(a.state, a.config),
    graph.ainvoke(b.state, b.config),
    graph.ainvoke(c.state, c.config),
)
```

---

## Anti-Patterns — Lesson 16

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| `conn.close()` in hot path | Destroys pooled connection | `pool.release(conn)` always |
| `graph.invoke()` in `async def` | Blocks event loop | `await graph.ainvoke()` |
| Pool `max` hardcoded to 100 | Exceeds Oracle session limit → ORA-00018 | `max = workers × 2` |
| `VARCHAR2` for checkpoint | Truncates long conversations | `CLOB` always |
| `MemorySaver` without env check | Silent data loss in production | Check `APP_ENV` at startup |
| Pool created per request | 10 pools × 10 conns = 100 sessions per user | Create pool once in `lifespan()` |
| No startup health check | Oracle unreachable only discovered at first request | `conn.ping()` in `lifespan()` |

---

## Interview Q&A — Lesson 16

**Q1: Walk me through exactly what happens when LangGraph calls `checkpointer.put()`.**
`put()` receives a `CheckpointTuple` with the full serialised state, config (containing `thread_id`), metadata (step number), and new channel versions. The implementation serialises the checkpoint dict to JSON, runs a MERGE statement keyed on `thread_id`, and commits. LangGraph calls `put()` after every node — a 5-node graph triggers 5 `put()` calls per `invoke()`. Oracle row-level locking ensures concurrent `put()` for the same `thread_id` are serialised safely.

**Q2: What happens if `checkpointer.put()` fails mid-graph execution?**
The exception propagates to the caller. State written before the failure is persisted; the current node's output is lost. On next `invoke()` for the same `thread_id`, LangGraph calls `get_tuple()` which returns the last successfully persisted checkpoint — the graph resumes from there. This is why every `put()` must be atomic (MERGE + COMMIT in one statement).

**Q3: Why is `asyncio.to_thread()` safer than calling sync `invoke()` in an async endpoint?**
FastAPI runs a single-threaded async event loop. A sync call blocking for 3 seconds blocks the event loop — no other requests are served. `asyncio.to_thread()` moves the call to a ThreadPoolExecutor so the event loop stays free. Set `max_workers` explicitly via `ThreadPoolExecutor(max_workers=N)` to bound concurrency.

**Q4: How does Oracle RAC change connection pool configuration?**
RAC presents multiple DB nodes behind a single SCAN listener. Set `min` slightly higher (4 instead of 2) to ensure warm connections on both nodes. Use the SCAN listener DNS in the DSN, not individual node IPs — SCAN handles failover transparently. Set `pool.reconnect=True` so the pool reconnects if a RAC node fails over.

**Q5: How do you implement time-travel debugging with OracleSaver?**
Add a `step` column: `(thread_id, step)` as composite PK. `put()` inserts a new row per step (not MERGE/overwrite). `get_tuple()` fetches `WHERE thread_id = :1 ORDER BY step DESC FETCH FIRST 1 ROW ONLY`. `list()` returns all rows for the thread. Replay: call `graph.invoke(None, config={"configurable": {"thread_id": X, "checkpoint_id": Y}})`.

**Q6: How do you test OracleSaver without a real Oracle database?**
Three approaches: (1) Unit test with in-memory mock checkpointer to test graph logic independently. (2) Integration test with Oracle XE — free, Docker-available (`gvenzl/oracle-xe`). (3) Contract tests: `put()` then `get_tuple()` must return same data; `put()` twice with same `thread_id` must not create duplicates. Mark with `@pytest.mark.skipif(not ORACLE_AVAILABLE, ...)`.

**Q7: What is the `alg=none` JWT attack and how do you prevent it?**
An attacker creates a JWT with `{"alg": "none"}` in the header and no signature. Buggy JWT libraries skip verification. Prevention: always pass `algorithms=["HS256"]` explicitly to `jose_jwt.decode()`. python-jose raises `JWTError` if the token algorithm doesn't match the allowlist.

---

# Lesson 17 — JWT Auth, RBAC & Multi-Tenancy

> **File:** `lesson_17_auth_rbac/lesson_17_auth_rbac.py`

---

## JWT Structure Internals

```
eyJhbGciOiJIUzI1NiJ9          ← HEADER (base64url)
.eyJzdWIiOiJhbGljZSIsInJvbGUiOiJhbmFseXN0IiwiZXhwIjoxNzM2MDAxNjAwfQ
                               ← PAYLOAD (base64url, NOT encrypted — readable by anyone)
.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
                               ← SIGNATURE = HMAC_SHA256(header.payload, secret)
```

**What the signature guarantees:** any bit change in header or payload invalidates the signature.
**What it does NOT guarantee:** payload is NOT encrypted — never put passwords or PII in claims.

---

## Token Lifecycle

```
Login → create_access_token() (30 min) + create_refresh_token() (7 days)

Normal request:
  Bearer: access_token → decode_token() → claims → auth_node → graph

Token expired:
  Bearer: refresh_token → decode_token() → verify type=="refresh"
  → re-fetch role from DB (role may have changed!)
  → create_access_token(new claims)

Why refresh re-fetches role:
  If user demoted from admin→viewer yesterday, the 7-day refresh token still works
  but new access tokens encode the current role from DB.
  Role changes take effect within ACCESS_TOKEN_EXPIRE_MINUTES (default 30 min).
```

---

## RBAC Graph Flow

```
START
  │
  ▼
auth_node: has_permission(role, action)?
  ├─ DENIED → denied_node → END
  └─ ALLOWED ↓
  ▼
rate_limit_node: sliding window counter (per user, per hour)
  ├─ EXCEEDED → denied_node → END
  └─ OK ↓
  ▼
tenant_chat_node: inject tenant system_prompt, call LLM
  ↓
END
```

---

## Multi-Tenant Isolation — 4 Layers

```
Layer 1: thread_id = f"{tenant_id}-{user_id}-{session}"
  → OracleSaver WHERE thread_id = :1 (partition key)

Layer 2: System prompt injection
  → TENANT_CONFIG[tenant_id]["system_prompt"] prepended to every LLM call

Layer 3: Tool list scoping
  → TENANT_CONFIG[tenant_id]["allowed_tools"] only bound to LLM

Layer 4 (Oracle VPD — production hardening):
  → Policy function appends AND tenant_id = SYS_CONTEXT('USERENV','CLIENT_INFO')
  → DB engine rejects cross-tenant reads even if app code is buggy
```

---

## Sliding Window Rate Limiter

```python
# In-memory (lesson_17): list of timestamps per user, prune old entries
# Problem: resets on restart, not shared across pods

# Production: Redis atomic operations
def check_rate_limit(user_id, limit, window_sec, redis_client) -> bool:
    now = time.time()
    key = f"rate:{user_id}"
    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, now - window_sec)   # remove old entries
    pipe.zadd(key, {str(uuid.uuid4()): now})           # add current
    pipe.zcard(key)                                     # count in window
    pipe.expire(key, window_sec)                        # cleanup TTL
    _, _, count, _ = pipe.execute()
    return count <= limit

# Fixed window has edge case: 50 req at 10:59 + 50 req at 11:00 = 100 in 2 min
# Sliding window is always correct: looks back exactly window_sec from now
```

---

## Anti-Patterns — Lesson 17

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| Role checked inside business node | Auth scattered across codebase | Central `auth_node` before all work |
| Role cached in graph state across sessions | Role change doesn't take effect | Re-validate JWT every request |
| `==` for token comparison | Timing attack | `hmac.compare_digest()` |
| No `exp` claim | Stolen token valid forever | Short-lived access token (30 min max) |
| In-memory rate counter | Resets on restart, not shared | Redis `INCR` with TTL |
| `thread_id` without tenant prefix | Cross-tenant state accessible | Always `f"{tenant_id}-{user_id}-{session}"` |
| Same secret for access + refresh tokens | Compromised secret = full bypass | Separate secrets or strictly check `type` claim |

---

## Interview Q&A — Lesson 17

**Q1: Why should refresh tokens not carry the user's role?**
Roles can change between issuance (7 days ago) and use. If the refresh token encodes `role=admin` and the user was demoted yesterday, they'd have admin privileges for 7 more days. Refresh tokens carry only `sub` and `tenant_id`. The refresh endpoint re-fetches the current role from DB and creates a new access token with the fresh role.

**Q2: How do you implement JWT token revocation before expiry?**
JWTs are stateless — no token store to delete from. Two patterns: (1) Blocklist: store `hash(token)` in Redis with TTL = remaining lifetime. Check on every request. (2) Short expiry: 5–15 min access tokens. Revoke the refresh token (blocklist it) — user can't get new access tokens. Effective revocation = `ACCESS_TOKEN_EXPIRE_MINUTES`.

**Q3: Why put auth check in a graph node vs FastAPI middleware?**
FastAPI middleware checks authentication (valid token). The RBAC graph node checks authorization per action — which depends on what the graph is doing at that point. A user can `ask_question` but not `delete_history`. By putting RBAC in a graph node, you get fine-grained action-level permission checks that adapt to graph state, not just request-level checks.

**Q4: What is Oracle VPD and when should you use it?**
Virtual Private Database appends a WHERE clause to every SQL statement at the engine level. Use it when: compliance (SOC2, HIPAA) mandates DB-level isolation, tenant data is in a shared table, or you want defence-in-depth — even if the application layer is compromised, DB engine prevents cross-tenant reads.

**Q5: How do you handle a user changing roles mid-session?**
Always re-validate role from the JWT on every request — never cache role in graph state across sessions. If a user is demoted from admin to viewer, their next token (after current access token expires in 30 min) will encode the new role, and the auth node will deny admin actions.

**Q6: How do you design RBAC to support custom tenant-defined roles?**
Extend `ROLE_PERMISSIONS` from hardcoded dict to DB-driven config. Table: `tenant_roles(tenant_id, role_name, permissions CLOB)`. Load with TTL cache. `has_permission(tenant_id, role, action)` checks tenant-specific roles first, falls back to global roles. Lets `acme-corp` define `"data-scientist"` role without code changes.

**Q7: How do you authenticate machine-to-machine (agent-to-agent) calls?**
Service account tokens: JWT with `sub=service:pr-review-agent`, `role=service`, stored in secret manager (AWS Secrets Manager, Vault). Calling agent includes `Authorization: Bearer <service_token>`. RBAC node treats `role=service` with least-privilege permissions. Rotate service tokens on schedule — never hardcode.

---

# Lesson 18 — Observability: Prometheus, Tracing & Health

> **File:** `lesson_18_observability/lesson_18_observability.py`

---

## How Prometheus Scraping Works

```
Your FastAPI app                Prometheus server
  │  GET /metrics               │
  │  → text/plain metrics       │◄── scrape every 15s
  │                             │    store in TSDB
  │                             │
  Grafana ◄──────────────────── PromQL queries
```

Prometheus PULLS from your app — no outbound connections needed from your app.

---

## Metric Type Selection

| Type | Use for | Never use for |
|------|---------|---------------|
| **Counter** | Monotonically increasing (requests, errors, tokens) | Current values |
| **Histogram** | Distribution / latency percentiles | Current state |
| **Gauge** | Current state, up+down (active sessions) | Totals |
| **Summary** | Avoid — can't aggregate across pods | Latency (use Histogram) |

**Why Histogram not Summary for latency?**
Summaries compute quantiles per-process — can't aggregate across 10 pods.
Histogram bucket counts sum correctly: `histogram_quantile(0.99, sum(rate(bucket[5m])))`.

---

## Trace ID — Full Propagation Pattern

```python
# API boundary: generate trace_id ONCE
@app.post("/ask")
async def ask(request: Request):
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    state = {"trace_id": trace_id, "messages": [...]}
    result = await graph.ainvoke(state, config)
    return JSONResponse(content={...}, headers={"X-Trace-Id": trace_id})

# Every node: log trace_id
def some_node(state):
    logger.info(f"[some_node] trace={state['trace_id'][:8]} | ...")
    # ELK/Datadog: search trace_id=abc123 → see EVERY log line for this request
```

`trace_id` = one user session. `request_id` = one HTTP request.
A session has many requests — same `trace_id`, many `request_id`s.

---

## SLA Guard — Production Alerting Flow

```
sla_guard_node detects last_latency > 5.0s
  → logger.warning("[SLA_BREACH] ...")
  → ERROR_TOTAL.labels(error_type="sla_breach").inc()
  → Prometheus scrapes every 15s
  → Alertmanager evaluates: rate(sla_breach[5m]) > 0.05
  → Alert fires → PagerDuty → on-call engineer
  → Engineer: histogram_quantile(0.99, ...) → finds slow node
  → Checks LangSmith trace → finds bottleneck
```

---

## Health Check — Kubernetes Probe Design

```yaml
livenessProbe:           # is process alive? (restart if fails)
  httpGet: {path: /health/live, port: 8000}
  initialDelaySeconds: 10
  failureThreshold: 3
  timeoutSeconds: 5

readinessProbe:          # is service ready for traffic? (remove from LB if fails)
  httpGet: {path: /health/ready, port: 8000}
  failureThreshold: 1    # stop routing on FIRST failure
  periodSeconds: 10
```

**Rule:** liveness only fails if process is truly stuck/deadlocked.
Never make liveness depend on external services — a DB outage should NOT restart pods.
Readiness fails on DB disconnect, LLM timeout, etc. — removes pod from load balancer.

---

## JSON Structured Logging

```python
# Enable: USE_JSON_LOGS=true
# Every log line becomes:
{"ts": "2024-01-15T10:30:00.123Z", "level": "INFO", "logger": "lesson_18",
 "msg": "[chat] DONE | trace=abc12345 | tokens~450"}

# Why required in production:
# ELK/Datadog parse JSON fields for filtering
# Can query: level=ERROR AND tenant_id=acme-corp
# Plain text logs cannot be filtered by field
```

---

## Anti-Patterns — Lesson 18

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| Summary for latency | Can't aggregate across pods | Histogram always |
| `trace_id` generated inside a node | Each node gets different ID | Generate at API boundary |
| No `for:` in alert rule | Single spike = alert fatigue | Always `for: 2m` minimum |
| Liveness checks external services | DB outage = pod restart loop | Liveness = process alive only |
| No label on metrics | Can't filter by tenant | Add `[tenant_id, node, status]` labels |

---

## Interview Q&A — Lesson 18

**Q1: Why can't you use Prometheus Summary for p99 across multiple servers?**
Summary computes quantiles in each process — server A p99=2s, server B p99=8s can't be combined. Histogram records bucket counts (integers) that sum across servers. `histogram_quantile(0.99, sum(rate(bucket[5m])))` correctly computes fleet-wide p99.

**Q2: What is the difference between trace_id, span_id, and request_id?**
`trace_id` — one logical operation end-to-end (one user question). Constant across all nodes and services. `span_id` — one unit of work within a trace (one node execution), forms a tree with parent spans. `request_id` — one HTTP request. A trace may span multiple HTTP requests (polling pattern).

**Q3: How do you correlate LangSmith traces with your Prometheus metrics?**
LangSmith auto-traces when `LANGCHAIN_TRACING_V2=true`. Each trace has a `run_id`. Include `run_id` in your structured log as `langsmith_run_id` alongside your `trace_id`. In ELK: search `trace_id` → find `langsmith_run_id` → open LangSmith dashboard → see full LLM call tree.

**Q4: When should liveness fail but readiness remain passing?**
It shouldn't. Liveness = "is process running and not deadlocked?" (very hard to fail). Readiness = "are all dependencies reachable?" (fails on DB/LLM issues). Liveness failure triggers pod restart. Readiness failure removes pod from load balancer without restart. Never make liveness dependent on external services.

**Q5: How do you prevent Prometheus from becoming a GDPR liability?**
Never use user email, name, or PII as label values — stored in TSDB for retention period. Use `tenant_id` as finest granularity label, never `user_id`. Prometheus has no per-label deletion capability.

**Q6: What PromQL queries should every LangGraph engineer know?**
```promql
rate(langgraph_requests_total[1m])                          # RPS
rate(langgraph_errors_total[5m]) / rate(langgraph_requests_total[5m])  # error rate
histogram_quantile(0.99, rate(langgraph_request_latency_seconds_bucket[5m]))  # p99
sum(rate(langgraph_requests_total[5m])) by (tenant_id)     # per-tenant breakdown
sum(langgraph_active_sessions) by (tenant_id)              # concurrency
```

**Q7: How do you implement distributed tracing without OpenTelemetry?**
The lesson's `Span` class shows the minimal approach: generate `trace_id` at request boundary, pass through state to every node, log it on every line. For production add OpenTelemetry: `tracer.start_as_current_span(node_name)` + configure OTLP exporter to Jaeger/Datadog. OpenTelemetry propagates `trace_id` via `traceparent` HTTP header automatically.

---

# Lesson 19 — Event-Driven Agents

> **File:** `lesson_19_event_driven/lesson_19_event_driven.py`

---

## Celery Task Lifecycle

```
FastAPI ──LPUSH celery:default──▶ Redis ◄──BLPOP── Celery Worker
  │  (returns task_id immediately)                    │ process_agent_task()
  │                                                   │ graph.invoke() (2-30s)
  │                               Redis ◄──result─────┘
  │
  │ GET /tasks/{id}/status
  │ AsyncResult(id).state → SUCCESS
  │ AsyncResult(id).result → {...}
```

**`task_acks_late=True`:** Celery acknowledges AFTER success. If worker crashes mid-task,
task goes back to queue and retried by another worker.
**`task_reject_on_worker_lost=True`:** Worker OOM kill → task rejected back to queue.

---

## Webhook HMAC Verification

```python
def verify_webhook_signature(payload_bytes, signature_header, secret):
    # signature_header: "sha256=<hex>"
    received = signature_header[len("sha256="):]
    expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received, expected)  # NEVER use ==

# Why hmac.compare_digest() not ==?
# String == short-circuits on first mismatch → timing attack
# Attacker can measure response time to brute-force signature byte by byte
# compare_digest() always takes the same time regardless of match position
```

**Security layers:**
1. HMAC signature → prevents spoofing
2. Timestamp check (add `X-Timestamp` to signed headers, reject if > 5 min old) → prevents replay
3. Idempotency key → prevents duplicate processing even if replay slips through

---

## Idempotency Key Design

```python
# GitHub PR: deterministic from content
key = f"pr-review-{repo}-{pr_number}-{head_commit_sha}"

# Stripe: use their provided ID
key = f"payment-{charge_id}"

# General webhook: content hash
key = f"webhook-{hashlib.sha256(payload_bytes).hexdigest()[:16]}"

# Redis TTL = expected retry window + margin
redis.setex(f"idempotency:{key}", time=86400, value=json.dumps(result))
```

---

## Dead Letter Queue Operations

```
Normal:  Queue → Worker → ✅ result stored
Retry:   Queue → Worker → ❌ → wait 2s → retry 1
                           ❌ → wait 4s → retry 2
                           ❌ → wait 8s → retry 3
                           ❌ → max_retries → dead_letter_queue.put(task, error)

Alert: dlq_depth > 0 for 5 minutes → engineer inspects
  dead_letter_queue.drain() → {"task": {...}, "error": "ORA-01017"}
  Fix root cause → re-enqueue: enqueue_task(AgentTask.from_dict(entry["task"]))
```

---

## Anti-Patterns — Lesson 19

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| No webhook signature check | Anyone can trigger your agents | HMAC verify on every webhook |
| `==` for signature comparison | Timing attack | `hmac.compare_digest()` |
| No idempotency | Webhook retry → duplicate PR reviews | Check idempotency key before processing |
| Silent task failure | No visibility into failures | DLQ + alert on `dlq_depth > 0` |
| `task_acks_late=False` | Worker crash = task lost silently | `task_acks_late=True` always |
| Celery worker accessing Oracle directly | Pool per worker process → too many sessions | Use OracleSaver pool sized `workers × max_concurrency × 2` |

---

## Interview Q&A — Lesson 19

**Q1: When use job queue instead of direct HTTP for an agent?**
When: (1) Task > 30s — HTTP clients time out. (2) Task must survive server restarts — queued tasks persist in Redis. (3) Need to scale consumers independently — add Celery workers without touching API. (4) Webhook sources retry — idempotency prevents duplicates.

**Q2: What is idempotency and why is it critical for event-driven agents?**
Processing the same event multiple times produces the same result as processing it once. Critical because webhooks retry failed deliveries. Without idempotency: a PR gets reviewed twice, a user gets two emails. Implementation: check deterministic `idempotency_key` before processing; if already processed, return cached result.

**Q3: How does Celery handle failures and retries?**
`raise self.retry(exc=exc, countdown=2 ** self.request.retries)` — exponential backoff. After `max_retries`, task enters FAILURE state. With `task_acks_late=True`, failed tasks are re-queued if worker crashes mid-execution. Monitor with Flower UI (`celery flower`).

**Q4: What is the difference between Celery broker and backend?**
Broker (Redis/RabbitMQ): where tasks are queued — workers PULL from here. Backend (Redis/DB): where results are stored — clients QUERY here for status and results. Can use Redis for both. RabbitMQ preferred for broker (better DLQ, reliable delivery) with Redis as backend (fast lookups).

**Q5: How do you secure webhook endpoints against replay attacks?**
Three layers: (1) HMAC signature verifies sender. (2) Timestamp in signed payload — reject events older than 5 minutes. (3) Idempotency key — even valid replay returns cached result without re-processing. HMAC alone doesn't prevent replay of a captured valid request within the valid window.

**Q6: How do you scale Celery for high-throughput agent tasks?**
Three dimensions: (1) Concurrency: `celery worker --concurrency=8` (I/O-bound tasks benefit). (2) Workers: add more worker instances across servers. (3) Priority queues: separate queues for fast tasks (Slack reply: 1s) vs slow tasks (PR review: 2 min). Route premium tenants to dedicated high-concurrency workers. Use `--autoscale=10,2`.

**Q7: How do you poll for Celery task results in a FastAPI endpoint?**
```python
@app.post("/tasks/pr-review", status_code=202)
async def submit(payload: dict):
    task = AgentTask(...)
    task_id = enqueue_task(task)
    return {"task_id": task_id, "poll_url": f"/tasks/{task_id}/status"}

@app.get("/tasks/{task_id}/status")
async def status(task_id: str):
    if CELERY_AVAILABLE:
        result = celery_app.AsyncResult(task_id)
        return {"status": result.state, "result": result.result if result.ready() else None}
    cached = idempotency_store.get_result(task_id)
    if cached: return {"status": "SUCCESS", "result": cached}
    raise HTTPException(404, "Task not found")
```

---

# Lesson 20 — Cost Control, Governance & Enterprise Capstone

> **File:** `lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py`

---

## Model Routing — Cost Math

```
Without routing: 1000 req/day × $0.015/1k tokens = $15/day
With routing:    800 × $0.0002 + 150 × $0.001 + 50 × $0.015
               = $0.16 + $0.15 + $0.75 = $1.06/day → 93% saving
```

**Routing heuristic (production):**
- Word count < 20, question mark, no complex keywords → `fast`
- Contains "design", "implement", "architect", "complex" → `powerful`
- Everything else → `balanced`

Production upgrade: fine-tune a small classifier model on your own query history.

---

## Token Budget — Redis Production Pattern

```python
# In-memory (lesson_20): resets on restart, not shared across pods
# Production:
def can_proceed(tenant_id, estimated_tokens, redis_client) -> tuple[bool, str]:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key = f"budget:{tenant_id}:{today}"
    budget = BUDGETS.get(tenant_id, 10.0)

    # Atomic check-and-increment (no race conditions)
    cost = estimated_tokens / 1000 * 0.001
    new_total = redis_client.incrbyfloat(key, cost)
    redis_client.expire(key, 90000)  # TTL: 25 hours (survives midnight UTC)

    if new_total > budget:
        redis_client.incrbyfloat(key, -cost)  # rollback
        return False, f"Daily budget ${budget} exhausted"
    return True, "ok"
```

---

## Circuit Breaker State Machine

```
CLOSED → all calls pass → failure count++
  │   failure_count >= threshold (5)
  ▼
OPEN → all calls blocked immediately → return cached/error response
  │   cooldown_seconds elapsed (60s)
  ▼
HALF_OPEN → one test call allowed
  ├─ success → CLOSED (reset counter)
  └─ failure → OPEN (reset timer)
```

**Why it matters:** Without circuit breaker, if the LLM API is slow/down, all requests
pile up waiting, exhausting thread pools and making the entire service unresponsive.
Circuit breaker fails fast and recovers gracefully.

---

## GDPR Implementation

```python
# Article 17 — Right to erasure:
def gdpr_erasure(user_id):
    # 1. Anonymise application records
    for record in get_records(user_id):
        record["content"] = "[ERASED]"
    # 2. Delete Oracle checkpoints
    cursor.execute(
        "DELETE FROM langgraph_checkpoints WHERE thread_id LIKE :1",
        [f"{user_id}-%"]
    )
    # 3. Write immutable audit event
    audit.record(user_id, ..., action="GDPR_ERASURE", result="COMPLETED")
    # Must complete within 30 days (Art. 12)

# Article 20 — Right to portability:
def gdpr_export(user_id):
    return [r for r in interactions if r["user_id"] == user_id]
    # Return as JSON — standard machine-readable format

# Article 5 — Data minimisation:
# Log user_id + action + timestamp only
# Never log raw message content in audit trail
```

---

## Full Enterprise Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  GATEWAY LAYER                                                   │
│    nginx/ALB → FastAPI (4 workers) → JWT auth → rate limiting   │
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER                                             │
│    LangGraph: START → RBAC → Complexity Router → Budget         │
│                    → Circuit → Chat → Compliance → END          │
├─────────────────────────────────────────────────────────────────┤
│  EXECUTION LAYER                                                 │
│    Model routing (fast/balanced/powerful)                        │
│    Celery workers (async tasks > 10s)                           │
├─────────────────────────────────────────────────────────────────┤
│  PERSISTENCE LAYER                                               │
│    OracleSaver/19c (TDE, RAC, Audit Vault, VPD)                 │
│    Redis (session cache, idempotency, rate limits, budgets)     │
├─────────────────────────────────────────────────────────────────┤
│  OBSERVABILITY LAYER                                             │
│    Prometheus → Grafana | OpenTelemetry → Jaeger                │
│    Structured JSON logs → ELK | LangSmith (LLM traces)         │
├─────────────────────────────────────────────────────────────────┤
│  GOVERNANCE LAYER                                                │
│    Token budget/tenant/day | Audit trail (SOC2)                 │
│    GDPR erasure + export | Circuit breaker | DLQ monitoring     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Anti-Patterns — Lesson 20

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| Always using `powerful` model | 15× cost vs `fast` for simple queries | Complexity router |
| Budget check after LLM call | Money already spent | Budget gate node BEFORE LLM call |
| In-memory token budget | Resets on restart, not shared | Redis `INCRBYFLOAT` with daily TTL |
| No circuit breaker | LLM outage → thread pool exhaustion | Circuit breaker with OPEN state |
| Hard-deleting GDPR data | Audit gap, compliance risk | Anonymise records, keep structure |
| GDPR erasure without audit log | Can't prove erasure happened | Write `GDPR_ERASURE` audit event |
| Model routing by word count only | Simple long question → wrong tier | Use a proper complexity classifier |

---

## Interview Q&A — Lesson 20

**Q1: How do you control LLM costs in a multi-tenant production system?**
Three mechanisms: (1) **Model routing** — classify complexity and route to cheapest sufficient model (93% saving in 80/15/5 split). (2) **Token budgets** — per-tenant daily budget enforced in a gate node before LLM call. (3) **Prompt optimisation** — trim history to last N messages, cache repeated queries in Redis. Monitor with `tokens_total` counter and daily cost reports.

**Q2: Explain the circuit breaker pattern for LLM APIs.**
Three states: CLOSED (normal, calls pass through), OPEN (failing, all calls blocked for `cooldown_seconds`), HALF_OPEN (after cooldown, one test call — success → CLOSED, failure → OPEN). Without it: LLM API slowdown causes all requests to pile up waiting, exhausting thread pools and making the entire service unresponsive.

**Q3: How do you implement GDPR right-to-erasure for LangGraph?**
(1) `DELETE /users/{user_id}/data` calls `gdpr_erasure(user_id)`. (2) Anonymise application records — replace content with `"[ERASED]"`. (3) Delete Oracle checkpoints: `DELETE FROM langgraph_checkpoints WHERE thread_id LIKE '{user_id}-%'`. (4) Write immutable audit event `GDPR_ERASURE`. (5) Return `{"erased_records": N}`. Must complete within 30 days (Art. 12).

**Q4: How would you design for 10,000 requests/day from 500 tenants?**
10k/day ≈ 7 req/min average, 100 req/min peak. Architecture: FastAPI 4 workers (async) handles 100+ concurrent. OracleSaver pool max=8/server + RAC for HA. Redis for cache (80% hit rate → 80% cost saving). Celery for tasks > 10s. Per-tenant rate limit: 20 req/min. Model routing: 80% fast, 15% balanced, 5% powerful. Cost: 10k × 500 tokens × $0.001/1k = $5/day.

**Q5: What is horizontal vs vertical scaling for LangGraph systems?**
Vertical: bigger server — hits single-machine limits. Horizontal: more servers — requires: (1) Stateless API pods (state in OracleSaver, not RAM). (2) Shared Redis for rate limiting and cache across pods. (3) Load balancer distributing traffic. Oracle RAC for active-active DB nodes. LangGraph's `thread_id` isolation makes horizontal scaling natural — any pod handles any `thread_id`.

**Q6: How do you design a cost chargeback system for a multi-tenant LangGraph platform?**
Track per-tenant token usage in `budget_manager`. Expose `GET /usage/{tenant_id}` returning `{tokens_today, cost_today_usd, requests_today}`. Store daily snapshots in Oracle: `INSERT INTO usage_daily (tenant_id, date, tokens, cost)`. Send monthly invoice report via scheduled Celery task. For internal chargeback: emit usage events to a data warehouse (Snowflake, BigQuery) via Kafka. Alert when tenant approaches 80% of budget (`remaining_budget < budget * 0.2`).

**Q7: How do you test enterprise agents that depend on Oracle, Redis, and Celery?**
Three-level strategy: (1) **Unit tests** — mock all external services (`MagicMock` for OracleSaver, in-memory dict for Redis). Test graph logic and node behaviour. (2) **Integration tests** — use real services in Docker Compose: `oracle-xe`, `redis:alpine`, `celery` worker. `pytest` with `docker-compose up --wait`. (3) **Contract tests** — test each integration point in isolation: OracleSaver put/get, Redis rate limiter, Celery task enqueue/result. Use `@pytest.mark.integration` to separate from unit tests in CI.

---

# Enterprise Anti-Patterns Master List

| Layer | Anti-pattern | Consequence | Fix |
|-------|-------------|-------------|-----|
| Persistence | `MemorySaver` in multi-pod deploy | State not shared; users lose context on load balancer switch | OracleSaver with shared DB |
| Persistence | `conn.close()` instead of `pool.release()` | Pool drains → ORA-00018 connection errors | Always `pool.release()` |
| Auth | Role checked inside business logic | Auth bypass if someone calls node directly | Central `auth_node` first |
| Auth | No token expiry | Stolen token valid forever | `exp` claim, 30 min max |
| Observability | Summary for latency | Can't aggregate across pods | Histogram |
| Observability | No `trace_id` in state | Can't correlate logs from one request | Generate at API boundary |
| Events | No HMAC webhook verification | Anyone can trigger your agents | `verify_webhook_signature()` |
| Events | No idempotency | Webhook retry = duplicate side effects | Idempotency key check |
| Events | Silent DLQ | Failed tasks disappear with no alert | Alert on `dlq_depth > 0` |
| Cost | Always powerful model | 15× overspend | Complexity router |
| Cost | In-memory budget counter | Resets on restart | Redis `INCRBYFLOAT` with TTL |
| Compliance | Hard-delete for GDPR | Audit gap | Anonymise + audit event |
| Compliance | PII in logs | GDPR violation | Log IDs only, not content |

---

# Enterprise Systems Design Questions

## Design Question: Multi-Tenant Code Review Platform

**Question:** "Design a LangGraph system that processes GitHub PR webhooks from 50 enterprise customers, reviews code for security issues, and posts results as PR comments. Handle 500 PRs/day at peak. SOC2 compliant."

**Model answer:**
```
WEBHOOK LAYER:
  GitHub → POST /webhooks/github
  → verify_webhook_signature() (HMAC-SHA256)
  → check idempotency_key = f"pr-{repo}-{pr_number}-{commit_sha}"
  → enqueue AgentTask to Redis (HTTP 202 immediately)

PROCESSING LAYER:
  Celery worker → process_agent_task()
  → build_pr_review_graph() → analyze → summarize
  → post GitHub comment via API
  → store result in OracleSaver (thread_id = task_id)

GOVERNANCE LAYER:
  Auth: GitHub webhook secret (HMAC, not JWT — service-to-service)
  RBAC: tenant config scopes which repos can be reviewed
  Budget: token budget per tenant per day enforced before LLM call
  Audit: every review logged with tenant_id, repo, pr_number, reviewer_model

SCALE:
  500 PRs/day = 21/hour average, 60/hour peak
  2 Celery workers × 4 concurrency = 8 concurrent reviews
  Average review: 30s → 8 × 2 reviews/min = 16/min capacity
  60/hour peak = 1/min → well within capacity

SOC2:
  Audit trail: every review with who requested, when, what was found
  Secrets: webhook secret in AWS Secrets Manager
  Encryption: OracleSaver with TDE on checkpoint CLOB
```

## Design Question: Enterprise AI Assistant with Cost Control

**Question:** "Design a LangGraph agent for 1000 enterprise employees across 10 companies. Each company has a $500/month budget. Response time SLA: 95% < 3s."

**Model answer:**
```
ARCHITECTURE:
  FastAPI (4 workers, async) + nginx
  → JWT auth (employee token, tenant_id encoded)
  → RBAC node (viewer/analyst/admin per company)
  → Rate limit node (50 req/hr per user, Redis)
  → Complexity router (fast/balanced/powerful)
  → Budget check (Redis INCRBYFLOAT, daily/monthly limits)
  → Circuit breaker (protect against LLM API outage)
  → Chat node (async, ainvoke)
  → Compliance logger (GDPR)
  → OracleSaver (thread_id = {tenant_id}-{user_id}-{date})

COST CONTROL:
  $500/month per company = $500/30 = $16.67/day budget
  1000 employees / 10 companies = 100/company
  100 employees × 10 queries/day = 1000 queries/company/day
  With routing: 80% fast ($0.0002) + 20% balanced ($0.001) = $0.04/query avg
  1000 × $0.04 = $40/day → well within $16.67 budget per company
  Actually $40 > $16.67 → set max 500 queries/company/day with routing

SLA (95% < 3s):
  fast model target: < 1s
  balanced model target: < 3s
  powerful model: async (Celery) for tasks > 10s

OBSERVABILITY:
  Prometheus: p95 latency per company per model tier
  Alert: p95 > 3s for 2 min → PagerDuty
  Alert: company at 80% daily budget → email admin
```

---

# Lesson 21 — AWS Bedrock: Production LLMs Without Ollama

> **Primary:** `lesson_21_aws_bedrock/lesson_21_aws_bedrock.py`

---

## How Bedrock Invocation Works Internally

```
Your code                   AWS Bedrock
──────────                  ───────────
ChatBedrockConverse
  .invoke(messages)
       │
       ▼
boto3 bedrock-runtime client
  .converse(
      modelId="anthropic.claude-3-haiku-...",
      messages=[{"role": "user", "content": [...]}],
      system=[{"text": "..."}],
      inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
      guardrailConfig={...}  ← optional, applied server-side
  )
       │  HTTPS to bedrock-runtime.<region>.amazonaws.com
       │
       ▼
  Bedrock control plane:
    1. Auth:         SigV4 signature verified against IAM
    2. Quota check:  requests-per-minute limit enforced
    3. Guardrail:    content filter applied to prompt (if configured)
    4. Model invoke: request forwarded to model inference cluster
    5. Guardrail:    content filter applied to response (if configured)
    6. Return:       JSON response with content + usage metadata
       │
       ▼
ChatBedrockConverse parses:
  response["output"]["message"]["content"][0]["text"]  → AIMessage.content
  response["usage"]["inputTokens"]                     → response_metadata["usage"]["input_tokens"]
  response["usage"]["outputTokens"]                    → response_metadata["usage"]["output_tokens"]
```

**Key insight:** Every single Bedrock call is authenticated via AWS SigV4. There are no API keys in your code — boto3 signs requests automatically using whatever credentials are in the chain (env vars, IAM role, SSO).

---

## boto3 Credential Chain — Detailed

```
boto3.Session() credential resolution (in order, first match wins):

 1. Environment variables
    AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY [+ AWS_SESSION_TOKEN]
    → Used in: CI/CD pipelines with short-lived credentials from OIDC

 2. AWS config file profile
    AWS_PROFILE=myprofile → reads ~/.aws/credentials [myprofile]
    → Used in: local dev with multiple AWS accounts

 3. AWS SSO
    aws sso login → stores cached token in ~/.aws/sso/cache/
    → Used in: enterprise with AWS IAM Identity Center

 4. EC2 instance metadata service (IMDS v2)
    http://169.254.169.254/latest/meta-data/iam/security-credentials/{role}
    → Used in: EC2 instances with attached IAM role
    → Credentials auto-rotate every ~6 hours

 5. ECS task role credentials
    AWS_CONTAINER_CREDENTIALS_RELATIVE_URI env var (set by ECS)
    → Used in: ECS/Fargate containers

 6. Lambda execution role
    Automatically available inside Lambda functions

Production rule: NEVER use option 1 for long-lived credentials in production.
Use option 4/5/6 (IAM roles). Temporary credentials only.
```

---

## Converse API Request Format — Under the Hood

`ChatBedrockConverse` translates LangChain messages to the Converse API format:

```python
# LangChain input:
[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangGraph?"),
    AIMessage(content="LangGraph is..."),
    HumanMessage(content="Give an example."),
]

# Bedrock Converse API format (boto3):
{
    "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
    "system": [{"text": "You are a helpful assistant."}],  # ← SystemMessage extracted
    "messages": [
        {"role": "user",      "content": [{"text": "What is LangGraph?"}]},
        {"role": "assistant", "content": [{"text": "LangGraph is..."}]},
        {"role": "user",      "content": [{"text": "Give an example."}]},
    ],
    "inferenceConfig": {"maxTokens": 1024, "temperature": 0.1},
}
```

**Why this matters:** Claude requires `system` to be a top-level field, not a message in the `messages` array. `ChatBedrockConverse` handles this automatically. If you used `ChatBedrock` (legacy InvokeModel API), you'd have to handle this yourself per model.

---

## Tool Calling on Bedrock — Internals

```python
# Step 1: bind tools (same as Ollama)
llm_with_tools = llm.bind_tools([get_weather, search_db])

# What ChatBedrockConverse sends to Bedrock:
{
    "toolConfig": {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }
            }
        ]
    }
}

# Bedrock response when model wants to call a tool:
{
    "stopReason": "tool_use",
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "uuid", "name": "get_weather", "input": {"city": "London"}}}
            ]
        }
    }
}

# ChatBedrockConverse converts this to:
AIMessage(
    content="",
    tool_calls=[{"id": "uuid", "name": "get_weather", "args": {"city": "London"}}]
)
# → LangGraph ToolNode processes this identically to Ollama tool calls
```

**Supported for tool calling on Bedrock Converse API:**
- ✅ Claude 3 Haiku, Sonnet, Opus
- ✅ Amazon Titan Text Premier
- ✅ Mistral Large
- ❌ Titan Express/Lite (no tool calling)
- ❌ Llama 3 8B (limited/unreliable)

---

## Streaming Internals

```
Without streaming:   Client waits for full response → receives all at once
With streaming:      Client receives chunks as tokens are generated

Bedrock streaming API:    bedrock-runtime.invoke_model_with_response_stream()
ChatBedrockConverse:      .stream(messages) → Iterator[AIMessageChunk]
                          .astream(messages) → AsyncIterator[AIMessageChunk]

Each chunk:
  AIMessageChunk(content="partial text", response_metadata={})

Final chunk:
  AIMessageChunk(
      content="",
      response_metadata={
          "usage": {"input_tokens": 42, "output_tokens": 187},
          "stopReason": "end_turn"
      }
  )

IMPORTANT: Token usage is ONLY in the FINAL chunk, not intermediate ones.
Accumulate chunks for content, check the last chunk for usage.

FastAPI pattern:
  StreamingResponse(generate(), media_type="text/event-stream")
  Each chunk: f"data: {chunk.content}\n\n"  ← SSE format
  Final:      "data: [DONE]\n\n"
```

---

## Cost Model — Deep Dive

```
Bedrock pricing dimensions:
  1. Input tokens:  all tokens in your request (system + history + current message)
  2. Output tokens: all tokens in the response
  3. No per-request fee — pure token-based

Token counting (approximate — Claude):
  1 token ≈ 4 characters ≈ 0.75 words
  "What is LangGraph?" = ~5 tokens
  A typical 500-word response = ~375 tokens

Real cost example — customer service bot, 100 users/day:
  Average conversation: 3 turns × 200 tokens in + 150 tokens out = 1050 tokens total
  Daily cost (Haiku): 100 users × 1050/1000 × $0.00025 in + $0.00125 out
    = 100 × (0.06¢ in + 0.1875¢ out) = 100 × 0.2475¢ = ~25¢/day = $7.50/month

Context window cost trap:
  Multi-turn chat: input tokens GROW with every turn (full history re-sent)
  Turn 1:  200 tokens in
  Turn 2:  200 + 150 (prev response) + 200 (new msg) = 550 tokens in
  Turn 3:  550 + 150 + 200 = 900 tokens in
  → Trim history to last N messages to control cost (Lesson 20 pattern applies here too)

Savings levers (in order of impact):
  1. Model routing:    Haiku vs Sonnet vs Opus — up to 60x cost difference
  2. History trimming: limit context window to last 5-10 messages
  3. Redis caching:    repeated queries (FAQ, schema lookups) → 0 Bedrock calls
  4. Token budget:     hard daily cap per tenant (Lesson 20 pattern)
  5. Prompt compression: summarize long history instead of full replay
```

---

## Bedrock Guardrails — Internals

```
Guardrail evaluation pipeline (server-side, within Bedrock):

INPUT path (before model):
  1. Word filter:    blocked words in user message → INTERVENED immediately
  2. Topic denial:   classifier detects denied topic → INTERVENED
  3. PII detection:  mask SSN/credit card/email in input (optional)

MODEL INFERENCE (only if input passes all filters)

OUTPUT path (after model, before returning to your code):
  4. Content filter: harmful/sexual/violence score threshold
  5. Topic denial:   check response for denied topics
  6. PII detection:  mask SSN/credit card/email in response
  7. Grounding:      compare response to source docs (RAG use case)

If ANY filter triggers:
  response["amazon-bedrock-guardrailAction"] = "INTERVENED"
  response content replaced with safe message you configured

Cost: Guardrails add ~$0.75 per 1000 text units (on top of model cost)

When to use:
  ✓ Customer-facing apps (block harmful content)
  ✓ Regulated industries (auto-mask PII in responses)
  ✓ Topic restriction (keep agent on-topic, block competitor mentions)
  ✗ Internal dev tools (unnecessary overhead)
  ✗ Air-gapped / ultra-low-latency (adds ~50-100ms)
```

---

## Anti-Patterns — Lesson 21

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Hardcoding `AWS_ACCESS_KEY_ID` in code | Credential leak via git, logs, errors | IAM role (EC2/ECS) or env var injection at deploy time |
| Using `ChatBedrock` (legacy) for new code | Model-specific JSON, no unified tool calling | Use `ChatBedrockConverse` |
| Creating a new boto3 client per request | Connection overhead, hitting session limits | Module-level client singleton, reuse across requests |
| Always using Sonnet/Opus | 3–60x more expensive than needed | Model routing: Haiku for simple queries, escalate only when needed |
| No token usage tracking | Surprise billing, no per-tenant cost attribution | Extract `response_metadata["usage"]` in every `chat_node` |
| Requesting `bedrock:*` in IAM policy | Overly broad — allows model listing, policy changes | Least privilege: only `InvokeModel` + `InvokeModelWithResponseStream` |
| Sending full conversation history every turn | Context window costs grow O(n²) per session | Trim to last N messages or summarize old turns |
| Using Guardrails in dev/test | Adds latency + cost during iteration | Enable only in staging/production via env var flag |
| Ignoring `INTERVENED` guardrail response | User sees confusing blocked message without context | Check `guardrailAction` and return a user-friendly explanation |

---

## Interview Q&A — Lesson 21

**Q1: Explain how AWS SigV4 authentication works for Bedrock calls and why there are no API keys.**
Every boto3 API call is signed with AWS Signature Version 4 (SigV4). The process: (1) Create a canonical request (method + URL + headers + body hash). (2) Create a string-to-sign (algorithm + timestamp + credential scope + canonical request hash). (3) Calculate signature using HMAC-SHA256 with your secret key. (4) Add `Authorization` header to the HTTP request. boto3 does all of this automatically. The credentials come from the credential chain (IAM role IMDS, env vars, etc.) — you never embed them in code. IAM roles on EC2/ECS/Lambda provide temporary credentials that auto-rotate every 6 hours, making them fundamentally more secure than static API keys.

**Q2: What is the Converse API and why should you always use `ChatBedrockConverse` over `ChatBedrock`?**
The Converse API is a unified Bedrock API that accepts the same request format regardless of the underlying model. Before it existed, each model family (Claude, Titan, Llama) required a different JSON request body format. `ChatBedrockConverse` uses Converse, giving you: (1) consistent system message handling across all models, (2) native tool calling support without model-specific hacks, (3) streaming with the same interface everywhere, (4) trivial model switching (change `model_id`, zero other changes). Only fall back to `ChatBedrock` (InvokeModel) if a brand-new model isn't in the Converse API yet.

**Q3: How does Bedrock streaming differ from non-streaming in terms of LangGraph integration?**
Non-streaming: `graph.invoke()` → `chat_node` calls `llm.invoke()` → waits for full response → returns to graph. Streaming: `chat_node` calls `llm.stream()` → iterates chunks → accumulates into a full `AIMessage` → returns to graph. The graph state always receives a complete `AIMessage` — the streaming only affects how `chat_node` internally fetches the response. For the user-facing stream (FastAPI SSE), run `graph.stream()` and yield chunks directly from the `chat` node's output. Token usage is only in the final chunk's `response_metadata` — always read it after the loop ends.

**Q4: How do you design a multi-tenant Bedrock cost attribution system?**
(1) Every request carries `tenant_id` in the LangGraph state. (2) `chat_node` extracts `input_tokens` + `output_tokens` from `response_metadata["usage"]` after every invocation. (3) Write to a `usage` table: `INSERT INTO bedrock_usage (tenant_id, model_id, input_tokens, output_tokens, cost_usd, ts)`. (4) Or: use Redis `INCRBYFLOAT bedrock:cost:{tenant_id}:{date}` for real-time budget enforcement. (5) Daily job aggregates per-tenant cost, compares against budget, sends alert at 80%. (6) Monthly invoice: `SELECT tenant_id, SUM(cost_usd) FROM bedrock_usage WHERE ts > ...`. This integrates directly with the Lesson 20 token budget pattern.

**Q5: How would you implement zero-downtime migration from Ollama to Bedrock in production?**
Feature flag pattern: (1) Add `LLM_PROVIDER` env var (default: `ollama`). (2) `get_llm()` factory returns `ChatOllama` or `ChatBedrockConverse` based on the flag. (3) Deploy the new code with `LLM_PROVIDER=ollama` — zero behaviour change, zero risk. (4) Run parallel shadow traffic: log both Ollama and Bedrock responses for the same queries, compare quality. (5) Canary: set `LLM_PROVIDER=bedrock` for 5% of traffic. Monitor error rate, latency, and cost in Prometheus. (6) Gradually roll to 100%. Rollback = flip env var back to `ollama` — instant, no redeployment. All graph nodes, tools, checkpointers, and state are completely unchanged.

**Q6: What are Bedrock Guardrails and when are they better than prompt engineering for safety?**
Guardrails are server-side content filters applied by AWS inside Bedrock before returning the response to your code. They are better than prompt engineering when: (1) **Bypassing risk** — prompt injection attacks (`"Ignore previous instructions and..."`) cannot bypass Guardrails because they run after the model, independently. Prompt engineering can always be jailbroken. (2) **PII masking** — automatically masks SSN, credit cards, emails in responses without any application code. (3) **Compliance** — Guardrail application is logged in AWS CloudTrail, providing an auditable record that content was filtered. (4) **Consistency** — applies the same policy across all models and all code paths without developer discipline. Use Guardrails for production customer-facing apps; prompt engineering for guidance, not safety.

**Q7: How does Bedrock handle rate limits, and what should your LangGraph agent do when it hits them?**
Bedrock enforces per-account, per-model, per-region quota limits (requests-per-minute and tokens-per-minute). When exceeded, Bedrock returns `ThrottlingException` (HTTP 429). In your LangGraph agent: (1) Wrap `llm.invoke()` in the retry decorator from Lesson 9 (exponential backoff: 1s, 2s, 4s, 8s, max 3 retries). (2) Catch `botocore.exceptions.ClientError` with `error_code == "ThrottlingException"`. (3) If all retries fail, route to circuit breaker (Lesson 20 pattern). (4) For high-volume tenants, request quota increases in the AWS console. (5) Multi-region fallback: if `us-east-1` is throttled, fall back to `us-west-2` with the same model. Bedrock quotas are per region, not global.

---

## Systems Design — Lesson 21

**"Design a multi-tenant AI assistant that uses AWS Bedrock, handles 50,000 requests/day, stays within per-tenant cost budgets, and meets SOC2 compliance."**

```
SCALE MATH:
  50,000 req/day ÷ 86,400s = 0.58 req/s avg
  Peak (10× avg): 6 req/s
  FastAPI 4 workers async: handles 100+ concurrent req → sufficient

ARCHITECTURE:
  Request:
    nginx → FastAPI (4 workers)
      → JWT auth (Lesson 17, python-jose)
      → Rate limit node (Redis sliding window, 20 req/min per user)
      → Budget check (Redis INCRBYFLOAT, daily limit per tenant)
      → Model selector (Haiku 80% / Sonnet 15% / Opus 5%)
      → Bedrock Guardrail (content filter + PII mask for SOC2)
      → ChatBedrockConverse (boto3 IAM role, no hardcoded keys)
      → Token usage → Redis cost accumulator
      → OracleSaver (checkpoint with tenant_id prefix)
      → Prometheus metrics (latency, tokens, cost, errors)

CREDENTIALS:
  EC2 IAM role with bedrock:InvokeModel + bedrock:InvokeModelWithResponseStream
  No AWS keys in code or environment

COST CONTROL:
  50,000 req × avg 600 tokens (in+out) = 30M tokens/day
  80% Haiku: 24M × $0.000375/1k = $9.00/day
  20% Sonnet: 6M × $0.018/1k   = $108.00/day
  Total: ~$117/day with routing
  vs all-Sonnet: 30M × $0.018/1k = $540/day → 78% saving

SOC2 COMPLIANCE:
  ✓ JWT auth + RBAC on every request (Lesson 17)
  ✓ Audit trail: WHO + WHAT + WHEN + RESULT (immutable Oracle log)
  ✓ Bedrock Guardrails: PII masking logged in CloudTrail
  ✓ TDE on Oracle checkpoint CLOB
  ✓ IAM least privilege (no wildcard permissions)
  ✓ CloudTrail: every bedrock:InvokeModel logged with principal + model + request ID
  ✓ Circuit breaker: no cascading failure under LLM outage

MULTI-REGION FAILOVER (bonus):
  Primary: us-east-1 (Bedrock + Oracle primary)
  Failover: us-west-2 (Bedrock only — Oracle Data Guard replica)
  Route53 health check: switch if primary Bedrock >10% error rate
```
