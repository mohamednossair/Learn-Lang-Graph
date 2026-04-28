# LangGraph Enterprise Guide — Part 5: Enterprise Agents

> **Who this is for:** Engineers moving from "it works on my laptop" to production systems
> handling real tenants, real money, real compliance requirements.

---

## The Enterprise Gap

Your Lessons 1–15 gave you a complete senior LangGraph engineer foundation.
Lessons 16–20 fill the gap between **senior** and **staff/principal** level:

| Skill | Lessons 1–15 | Lessons 16–20 |
|-------|-------------|---------------|
| Build agents | ✓ | ✓ |
| Persist state | SQLite | **Oracle 19c (multi-server, enterprise-grade)** |
| Handle concurrency | Sequential | **Async / concurrent** |
| Control access | None | **JWT + RBAC** |
| Multi-tenant | None | **Full isolation** |
| Observe system | LangSmith | **Prometheus + tracing + health** |
| Trigger agents | HTTP only | **Events / webhooks / queues** |
| Control cost | None | **Budget enforcement + model routing** |
| Compliance | None | **GDPR + audit trail + circuit breaker** |

---

# Lesson 16 — Oracle 19c Persistence & Async Execution

> **Primary file:** `lesson_16_postgres_async/lesson_16_oracle_async.py`
> **PostgreSQL reference:** `lesson_16_postgres_async/lesson_16_postgres_async.py` (kept for comparison)

---

## Why SQLite Is Not Enterprise

`SqliteSaver` works perfectly for single-server setups. In enterprise:

| Problem | Effect |
|---------|--------|
| Single file | Only one server can write at a time |
| File locking | 2 API pods = race conditions, data corruption |
| No connection pooling | Each request opens a new file handle |
| No replication | Database host dies = all state lost |

**OracleSaver solves all of these — and adds Oracle 19c enterprise features.**

---

## Oracle 19c vs PostgreSQL vs SQLite

| Feature | SQLite | PostgreSQL | **Oracle 19c** |
|---------|--------|------------|----------------|
| Multi-server writes | ✗ (file lock) | ✓ | ✓ |
| Connection pool | ✗ | ✓ (asyncpg) | ✓ (oracledb pool) |
| ACID transactions | ✓ | ✓ | ✓ |
| High-availability | ✗ | ✓ (Patroni) | **✓ Oracle RAC (built-in)** |
| Transparent encryption | ✗ | pgcrypto ext | **✓ TDE (built-in)** |
| Enterprise audit | ✗ | pgaudit ext | **✓ Audit Vault (built-in)** |
| Fine-grained access | ✗ | Row security | **✓ VPD/FGAC (built-in)** |
| Disaster recovery | ✗ | Streaming rep | **✓ Data Guard (built-in)** |
| Zero-downtime patch | ✗ | ✗ | **✓ RAC rolling patch** |

---

## Checkpointer Decision Tree

```
Which checkpointer should you use?

  APP_ENV == "production" or USE_ORACLE=true
      ↓
  OracleSaver (Oracle 19c, multi-server, enterprise-grade)
      |
      └── Oracle unreachable (e.g. local dev without VPN)
              ↓
          MemorySaver (dev fallback, logged as WARNING)

  APP_ENV == "staging"
      ↓
  SqliteSaver (single file, easy to inspect/reset)

  APP_ENV == "development" (default)
      ↓
  MemorySaver (zero config)
```

**Enterprise rule:** Never hardcode `MemorySaver()` in production code.
Always read from `APP_ENV` / `ORACLE_*` environment variables.

---

## Async Nodes: Why They Matter

```python
# ❌ SYNC — blocks thread while waiting for LLM (200ms–10s)
def chat_node(state):
    response = llm.invoke(state["messages"])   # blocks!
    return {"messages": [response]}

# ✅ ASYNC — releases thread to serve other requests
async def chat_node(state):
    response = await llm.ainvoke(state["messages"])   # non-blocking
    return {"messages": [response]}
```

### Concurrency comparison (4 uvicorn workers, 10 req/s):

| Mode | Throughput | Threads blocked |
|------|-----------|-----------------|
| Sync `invoke()` | ~4 req in flight | 4 |
| Async `ainvoke()` | 100+ req in flight | 0 |

---

## asyncio.gather() — Concurrent Users

```python
# Run 3 users' requests at the same time:
results = await asyncio.gather(
    graph.ainvoke(user_a_state, config_a),
    graph.ainvoke(user_b_state, config_b),
    graph.ainvoke(user_c_state, config_c),
)
# Total time ≈ max(A, B, C) instead of A + B + C
```

---

## Oracle DDL — Run Once as DBA

```sql
-- Create the checkpoints table (run once, DBA privileges required)
CREATE TABLE langgraph_checkpoints (
    thread_id   VARCHAR2(255)  NOT NULL,
    checkpoint  CLOB           NOT NULL,   -- JSON blob (up to 4 GB)
    metadata    CLOB,                      -- JSON blob
    ts          TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT pk_lg_chk PRIMARY KEY (thread_id)
);

-- Grant access to the application user
GRANT SELECT, INSERT, UPDATE, DELETE ON langgraph_checkpoints TO langgraph;

-- Optional: enable column-level TDE encryption
ALTER TABLE langgraph_checkpoints
    MODIFY (checkpoint ENCRYPT USING 'AES256' NO SALT);
```

**Why CLOB?** LangGraph checkpoints are JSON state dicts. Oracle CLOB holds up to 4 GB — supports arbitrarily large conversation histories.

---

## OracleSaver — How the MERGE Works

```python
# Oracle MERGE = atomic upsert (no separate INSERT + UPDATE needed)
cursor.execute("""
    MERGE INTO langgraph_checkpoints dst
    USING (SELECT :1 AS thread_id FROM dual) src
    ON (dst.thread_id = src.thread_id)
    WHEN MATCHED THEN
        UPDATE SET checkpoint = :2, metadata = :3, ts = SYSTIMESTAMP
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint, metadata)
        VALUES (:4, :5, :6)
""", [thread_id, json, meta, thread_id, json, meta])
```

**Why MERGE instead of INSERT OR REPLACE?**
Oracle has no `INSERT OR REPLACE`. MERGE is the Oracle-idiomatic atomic upsert — it acquires a row-level lock, so concurrent writes from multiple API servers are safe.

---

## Oracle Connection Pool in FastAPI

```python
import oracledb
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = oracledb.create_pool(
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        dsn=f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}",
        min=2,        # keep 2 warm connections always open
        max=10,       # tune: workers × 2
        increment=1,  # grow one connection at a time under load
    )
    checkpointer = OracleSaver(pool)
    checkpointer.setup()       # creates table on first run
    app.state.graph = build_async_graph(checkpointer)
    yield
    pool.close()               # graceful shutdown

app = FastAPI(lifespan=lifespan)
```

**Pool sizing rule:** `max = api_workers × 2`.
Oracle 19c Standard Edition: 200 session limit per instance.
Oracle 19c Enterprise + RAC: effectively unlimited across nodes.

---

## astream() — Real-Time UI Updates

```python
async for chunk in graph.astream(state, config):
    for node_name, node_output in chunk.items():
        # Each chunk = one node completing
        # Send to browser via Server-Sent Events (SSE):
        yield f"data: {json.dumps({'node': node_name, 'keys': list(node_output.keys())})}\n\n"
```

FastAPI SSE endpoint:
```python
from fastapi.responses import StreamingResponse

@app.get("/stream/{thread_id}")
async def stream(thread_id: str, question: str):
    async def generate():
        async for chunk in graph.astream(...):
            for node, output in chunk.items():
                yield f"data: node={node}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Interview Q&A — Lesson 16

**Q1: Why choose Oracle 19c over PostgreSQL for LangGraph persistence?**
Oracle 19c ships with enterprise features that PostgreSQL needs third-party extensions for: Transparent Data Encryption (TDE) encrypts data at rest with no application changes; Oracle Audit Vault provides immutable audit trails required for SOC2/ISO27001; Oracle RAC gives active-active multi-node clustering with zero-downtime patching; Virtual Private Database (VPD) enforces row-level security at the engine level. If your organisation already runs Oracle, OracleSaver is the natural choice — no new DB technology to license, operate, or secure.

**Q2: What is the difference between `graph.invoke()` and `graph.ainvoke()`?**
`invoke()` is synchronous — blocks the calling thread until the graph completes. `ainvoke()` is async — returns a coroutine that suspends when awaiting I/O (LLM calls, DB queries), letting other coroutines run. Inside a FastAPI `async def` endpoint, `invoke()` blocks the entire event loop and starves all other requests. Always use `ainvoke()` in async contexts. If you must call sync code, use `await asyncio.to_thread(graph.invoke, state, config)`.

**Q3: How does Oracle MERGE differ from PostgreSQL upsert?**
PostgreSQL: `INSERT ... ON CONFLICT DO UPDATE`. Oracle: `MERGE INTO ... USING ... ON ... WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT`. Both are atomic upserts with row-level locking. Oracle MERGE is more powerful: multiple WHEN clauses, DELETE in the MERGE, merge from subqueries or views. For OracleSaver we MERGE against `DUAL` (Oracle's single-row dummy table) to create the source record.

**Q4: What Oracle pool settings should you use for a LangGraph FastAPI API?**
`min=2, max=workers*2, increment=1`. Example: 4 uvicorn workers → `max=8`. Oracle 19c Standard Edition has a 200-session limit per instance. With 5 API servers × 8 pool = 40 connections — well within limits. For Oracle RAC (2 nodes), each node handles half the connections, effective limit doubles. Enable `ping_interval=60` to detect stale connections.

**Q5: How does thread_id guarantee tenant isolation in Oracle?**
`thread_id` is the primary key of `langgraph_checkpoints`. `OracleSaver.get_tuple()` queries `WHERE thread_id = :1`. Format: `{tenant_id}-{user_id}-{session_date}`. Oracle VPD (Virtual Private Database) adds a DB-level enforcement layer: a policy function appends `AND tenant_id = SYS_CONTEXT('USERENV','CLIENT_INFO')` to every SELECT automatically — even if application code is buggy, the DB engine blocks cross-tenant reads.

---

## Tasks — Lesson 16

**Task 16.1** — Run `lesson_16_oracle_async.py`. Observe all 3 demos (sync, concurrent, streaming). Find the log line showing which checkpointer was selected (`OracleSaver` or fallback).

**Task 16.2** — Add a 4th demo: run 5 users concurrently with `asyncio.gather()`. Print total wall-clock time. Compare against running them sequentially in a loop.

**Task 16.3** — Set `USE_ORACLE=true` (env var). Observe the MemorySaver fallback log when Oracle is unreachable. Add a startup check that raises a `RuntimeError` if `APP_ENV=production` but Oracle connection fails.

**Task 16.4** — Extend `OracleSaver` with a `delete(thread_id)` method (`DELETE FROM langgraph_checkpoints WHERE thread_id = :1`). This is needed for GDPR right-to-erasure (Lesson 20). Test it: create a checkpoint, delete it, verify `get_tuple()` returns `None`.

---

# Lesson 17 — Auth, RBAC & Multi-Tenancy

> **File:** `lesson_17_auth_rbac/lesson_17_auth_rbac.py`

---

## Authentication vs Authorization

| Concept | Question | Implementation |
|---------|----------|---------------|
| **Authentication (AuthN)** | Who are you? | JWT token verification |
| **Authorization (AuthZ)** | What can you do? | RBAC role check |

Both are required. Neither replaces the other.

---

## JWT Authentication Flow

```
Client                     FastAPI                    LangGraph
  │                           │                           │
  │──POST /auth/login─────────▶                           │
  │   {username, password}    │ verify credentials        │
  │◀──{token: "eyJ..."}───────│                           │
  │                           │                           │
  │──POST /ask────────────────▶                           │
  │   Authorization: Bearer   │ decode JWT                │
  │   eyJ...                  │ extract user_id,          │
  │                           │ tenant_id, role           │
  │                           │──graph.invoke(state)─────▶│
  │                           │                    auth_node checks role
  │                           │◀──result──────────────────│
  │◀──{answer: "..."}─────────│                           │
```

---

## RBAC Hierarchy

```
superuser ─── all permissions (*)
    │
   admin ─── delete_history, manage_users, view_audit_log
    │         + all analyst permissions
    │
  analyst ─── run_query, export_data
    │          + all viewer permissions
    │
  viewer ─── ask_question, view_history
```

**Design principle:** permissions are additive (roles inherit from below).
Never hardcode roles in nodes — check `has_permission(role, action)`.

---

## Multi-Tenant Isolation — 3 Layers

### Layer 1: State isolation (thread_id)
```python
thread_id = f"{tenant_id}-{user_id}-{session_id}"
# Company A users can NEVER load Company B state
```

### Layer 2: System prompt isolation
```python
system_msg = SystemMessage(content=TENANT_CONFIG[tenant_id]["system_prompt"])
# Acme's LLM sees Acme's rules; Globex's LLM sees Globex's rules
```

### Layer 3: Tool isolation
```python
allowed_tools = TENANT_CONFIG[tenant_id]["allowed_tools"]
# Acme analyst can search_products but not run_query
# Globex analyst can run_query but not search_products
```

---

## Audit Trail — SOC2 Requirements

SOC2 CC6.1 requires that access to systems is logged and reviewable.

```python
# Every event must record:
{
    "timestamp": "2024-01-15T10:30:00Z",   # immutable
    "user_id":   "alice",                  # WHO
    "tenant_id": "acme-corp",              # WHICH TENANT
    "role":      "analyst",                # WHAT ROLE
    "action":    "run_query",              # WHAT ACTION
    "resource":  "sales_database",         # ON WHAT
    "result":    "ALLOWED",                # OUTCOME
}
```

**Non-negotiable rules:**
1. Audit logs are **append-only** (never delete, never modify)
2. Log **denied** events — security team needs to know who tried what
3. Ship to SIEM in real-time (not batch) — Splunk, Datadog, Elastic

---

## Interview Q&A — Lesson 17

**Q1: How do you enforce tenant isolation in a multi-tenant LangGraph system?**
Three layers: (1) **State**: `thread_id = {tenant_id}-{user_id}-{session}` so OracleSaver only loads that tenant's checkpoints. (2) **System prompt**: each tenant has their own instructions injected before LLM calls — prevents cross-tenant knowledge leakage. (3) **Tools**: only tools in `TENANT_CONFIG[tenant_id]["allowed_tools"]` are bound to the LLM for that tenant. Oracle VPD can enforce a 4th layer at the DB engine level.

**Q2: Why put the auth check in a dedicated graph node vs in a FastAPI middleware?**
FastAPI middleware only checks authentication (valid token). The RBAC node inside the graph checks **authorization per action** — which depends on what the graph is doing at that point (e.g., a user can `ask_question` but not `delete_history`). By putting RBAC in a graph node, you can have fine-grained action-level permission checks that adapt to graph state, not just request-level checks.

**Q3: What is the difference between authentication and authorization, and why do both matter for agents?**
Authentication = "prove you are who you say you are" (JWT token signature). Authorization = "prove you have permission for this action" (RBAC role check). Agents especially need both because: a valid authenticated user (analyst) might try to perform admin actions (delete_history). Without RBAC, any authenticated user can do anything. Real breach scenario: user with viewer role calls an `export_all_data` endpoint that wasn't protected by RBAC.

**Q4: How would you handle a user changing roles mid-session?**
LangGraph state snapshots include the role at the time of each interaction. For future requests: always re-validate role from the JWT on every request — don't cache role in graph state across sessions. If a user is demoted from admin to viewer, their next token will encode the new role, and the auth node will deny admin actions.

**Q5: What audit events are required for GDPR compliance?**
GDPR Article 30 requires: who processed personal data, what processing was done, when, and under what legal basis. For agents: log every user query (which may contain PII), every LLM response, every data export, and every erasure request. Never log raw personal data in audit logs — log user_id + action + timestamp, keep PII in the application DB with encryption at rest.

---

**Q6: How do you implement rate limiting inside a LangGraph graph?**
Add a `rate_limit_node` between `auth` and `chat`. Use a sliding-window counter: keep a list of request timestamps per user; prune entries older than the window (e.g. 1 hour); if `len(timestamps) > max_requests`, return denied. In production use Redis `INCR` with TTL instead of in-memory lists — Redis counters survive server restarts and are shared across all API pods. Set different limits per tenant via `TENANT_CONFIG[tenant_id]["max_requests_per_hour"]`.

---

## Tasks — Lesson 17

**Task 17.1** — Run the demo. Add a 4th role: `"operator"` with permissions `{"ask_question", "view_history", "restart_session"}`. Test that operator can ask but cannot delete.

**Task 17.2** — Add a new tenant `"techstart"` with its own system prompt and tool list. Verify tenant isolation by asking the same question as an acme-corp user and a techstart user.

**Task 17.3** — Implement `gdpr_data_request(user_id)` that returns a JSON export of all audit entries for that user. This is GDPR Article 20 (data portability).

**Task 17.4** — The `rate_limit_node` is already implemented with a sliding window. Lower `max_requests_per_hour` for `acme-corp` to `3`, run 4 requests, and observe the 4th being blocked. Restore the original limit. Then migrate the in-memory counter to Redis: `r.incr(f"rate:{user_id}:{window_key}")` + `r.expire(...)` for multi-server safety.

---

# Lesson 18 — Observability

> **File:** `lesson_18_observability/lesson_18_observability.py`

---

## The Three Pillars of Observability

| Pillar | What it answers | Tool |
|--------|----------------|------|
| **Metrics** | Is the system healthy? What are the trends? | Prometheus + Grafana |
| **Traces** | What happened for THIS request? | OpenTelemetry / LangSmith |
| **Logs** | What did the code say at each step? | ELK / Datadog / CloudWatch |

**All three are required for production.** You cannot diagnose incidents with only one.

---

## Prometheus Metric Types

```python
# Counter — only goes up (requests, errors, tokens)
REQUEST_TOTAL = Counter("langgraph_requests_total", "...", ["tenant_id", "status"])
REQUEST_TOTAL.labels(tenant_id="acme", status="success").inc()

# Histogram — distribution (latency percentiles)
LATENCY = Histogram("langgraph_latency_seconds", "...", buckets=[.1, .5, 1, 5, 10])
LATENCY.observe(elapsed_seconds)

# Gauge — current value, can go up and down (active sessions)
ACTIVE = Gauge("langgraph_active_sessions", "...")
ACTIVE.inc()   # on request start
ACTIVE.dec()   # on request end
```

---

## Trace ID Propagation

```
HTTP Request arrives → generate trace_id (UUID)
  │
  ├─ node: validate    [trace=abc123, span=111]
  │                    log: "trace=abc123 ..."
  │
  ├─ node: chat        [trace=abc123, span=222]
  │                    log: "trace=abc123 ..."
  │
  └─ node: sla_guard   [trace=abc123, span=333]
                       log: "trace=abc123 ..."

→ Search Elasticsearch for trace_id=abc123
  → See EVERY log line for this exact request
  → See EVERY span with duration
```

**Enterprise rule:** `trace_id` must be generated at the API boundary and passed in state to every node. Never generate it inside a node.

---

## Health Check Design

```python
GET /health/live   → 200 always (unless process crashed)
GET /health/ready  → 200 if DB + LLM reachable, else 503

# Kubernetes config:
livenessProbe:
  httpGet: {path: /health/live, port: 8000}
  failureThreshold: 3     # restart after 3 consecutive failures

readinessProbe:
  httpGet: {path: /health/ready, port: 8000}
  failureThreshold: 1     # stop routing traffic immediately on first failure
```

**Key distinction:** liveness failure → Kubernetes **restarts** the pod. Readiness failure → Kubernetes **stops routing traffic** to the pod (but doesn't restart). A pod can be alive but not ready (e.g., warming up, DB temporarily disconnected).

---

## Essential Grafana Panels

```
Dashboard: LangGraph Agent System

Row 1 — Traffic
  Panel: Request Rate (RPS) by tenant
  PromQL: sum(rate(langgraph_requests_total[1m])) by (tenant_id)

  Panel: Error Rate (%)
  PromQL: sum(rate(langgraph_requests_total{status="error"}[1m]))
          / sum(rate(langgraph_requests_total[1m]))

Row 2 — Performance
  Panel: p50 / p90 / p99 Latency
  PromQL: histogram_quantile(0.99, rate(langgraph_request_latency_seconds_bucket[5m]))

  Panel: Active Sessions
  PromQL: sum(langgraph_active_sessions) by (tenant_id)

Row 3 — Cost
  Panel: Token Usage Rate
  PromQL: rate(langgraph_tokens_total[1h])

  Panel: Estimated Cost Today (USD)
  PromQL: increase(langgraph_tokens_total[24h]) * 0.001 / 1000
```

---

## SLA Alerting Rules

```yaml
# alerting rules (Prometheus Alertmanager)
groups:
  - name: langgraph_sla
    rules:
      - alert: HighErrorRate
        expr: error_rate > 0.05
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "Error rate > 5% for 5 minutes"

      - alert: HighLatency
        expr: p99_latency > 5.0
        for: 2m
        labels: {severity: warning}
        annotations:
          summary: "p99 latency above 5s SLA"

      - alert: AgentDown
        expr: up{job="langgraph"} == 0
        for: 1m
        labels: {severity: critical}
```

---

## Interview Q&A — Lesson 18

**Q1: What is the difference between liveness and readiness health checks?**
Liveness: "Is the process alive?" — checked by Kubernetes to decide whether to restart the pod. Readiness: "Is the service ready to receive traffic?" — checked by the load balancer. A pod can be alive (process running) but not ready (database disconnected, cache warming). During startup, set readiness = not ready until all dependencies are connected. During a dependency outage, set readiness = not ready to prevent the load balancer from routing traffic to a pod that will fail.

**Q2: What Prometheus metrics are essential for a LangGraph production system?**
Four required: (1) `requests_total` counter with labels `[tenant, node, status]` — for error rate and request volume. (2) `request_latency_seconds` histogram — for p50/p90/p99 latency SLAs. (3) `active_sessions` gauge — for capacity planning and alerts. (4) `tokens_total` counter — for cost tracking and budget enforcement. Optional: `errors_total` by error type for debugging.

**Q3: How does trace_id help with production debugging?**
A trace_id is generated once per request and propagated through every node's state. Every log line emitted by every node includes the same trace_id. When a user reports an issue, you search your log system (ELK/Datadog) for their trace_id and see the complete request journey: which nodes ran, what state they saw, what errors occurred, and the latency breakdown. Without trace_ids, you cannot correlate log lines from different nodes belonging to the same request.

**Q4: What should you do when p99 latency spikes in production?**
Step-by-step: (1) Check `active_sessions` gauge — if too high, you're overloaded → scale out. (2) Check error rate — if errors spiked simultaneously, the LLM API may be degraded. (3) Check LangSmith traces for slow requests — identify which node is slow (LLM call vs DB vs validation). (4) Check `tokens_total` rate — if token usage spiked, prompts may have become unexpectedly long. (5) Set circuit breaker threshold — if LLM p99 > 10s, open the circuit and return cached/fallback responses.

**Q5: How do you implement distributed tracing for a LangGraph agent deployed on multiple servers?**
Use OpenTelemetry SDK: `from opentelemetry import trace; tracer = trace.get_tracer("langgraph")`. Wrap each node: `with tracer.start_as_current_span("node_name") as span: span.set_attribute("tenant", ...)`. Configure OTLP exporter to ship spans to Jaeger or Datadog. The trace_id is automatically propagated via HTTP headers (`traceparent`) when agents call each other. LangSmith also captures LangChain-level traces automatically when `LANGCHAIN_TRACING_V2=true`.

---

**Q6: How do you implement structured JSON logging and why is it required in production?**
Set `USE_JSON_LOGS=true` to activate `JsonFormatter` which outputs every log line as a parseable JSON object with `ts`, `level`, `logger`, and `msg` fields. Required because: ELK (Elasticsearch/Logstash/Kibana) and Datadog ingestion pipelines expect structured fields for filtering and alerting. Unstructured plain-text logs cannot be queried by field — you cannot filter `level=ERROR AND tenant_id=acme-corp` in a text log. JSON logs enable: per-tenant error dashboards, SLA breach reports, and security audit queries by `user_id`.

---

## Tasks — Lesson 18

**Task 18.1** — Install `prometheus-client` and run the demo with Prometheus enabled. Open `http://localhost:9090/metrics`. Find the `langgraph_requests_total` metric and verify the counter increments.

**Task 18.2** — Add a `latency_sla_node` after the chat node. If `latency_ms > 3000`, log a WARNING with `[SLA_BREACH]` prefix. Trigger it by adding a `time.sleep(3.5)` to the chat node.

**Task 18.3** — Add a `/health/ready` endpoint to a FastAPI app that returns 503 if any health check is False. Simulate a failure by calling `health.set_check("llm", False)`.

**Task 18.4** — Instrument the `auth_node` from Lesson 17 with the `trace_node` context manager from Lesson 18. Verify the trace_id appears in both auth and chat log lines for the same request.

**Task 18.5** — Enable JSON logging: set `USE_JSON_LOGS=true` and run the demo. Pipe the output to `python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"` to pretty-print. Verify each line is valid JSON with `ts`, `level`, `logger`, `msg` fields.

---

# Lesson 19 — Event-Driven Agents

> **File:** `lesson_19_event_driven/lesson_19_event_driven.py`

---

## Why Event-Driven?

HTTP request/response is synchronous: client waits for the full response.
For agents that take 30–300 seconds, this is impractical:

```
❌ Synchronous:
  Client         Server
    │──POST /review──▶│
    │  (waiting...)   │ graph running (2 minutes)
    │◀──{result}──────│  ← HTTP timeout often hits before this

✅ Async event-driven:
  Client         Server         Queue         Worker
    │──POST /review──▶│           │              │
    │◀──{task_id}─────│──enqueue──▶              │
    │                 │           │◀─pick up task─│
    │                 │           │      (2 mins) │
    │──GET /status/id─▶│           │              │
    │◀──{status: done}─│           │              │
    │──GET /result/id──▶│           │              │
    │◀──{result: ...}──│           │              │
```

---

## Celery Architecture

```
┌─────────────┐    enqueue    ┌───────────┐   dequeue    ┌──────────────┐
│  FastAPI    │──────────────▶│   Redis   │─────────────▶│ Celery Worker│
│  (producer) │               │  (broker) │              │  (consumer)  │
└─────────────┘               └───────────┘              └──────┬───────┘
       │                           │                             │
       │                     ┌─────▼──────┐                     │
       │                     │   Redis    │◀────store result─────┘
       │                     │ (backend)  │
       │                     └─────┬──────┘
       │                           │
       └──────GET /status/id────────▶ task.status → SUCCESS/PENDING/FAILURE
```

---

## Idempotency — The Most Important Pattern

```python
# Without idempotency:
# GitHub sends webhook → agent starts PR review
# GitHub resends webhook (network retry) → agent starts ANOTHER PR review
# Result: duplicate PR comments

# With idempotency:
def process_task(task):
    if idempotency_store.is_processed(task.idempotency_key):
        return idempotency_store.get_result(task.idempotency_key)  # cached
    result = run_agent(task)
    idempotency_store.mark_processed(task.idempotency_key, result)
    return result

# idempotency_key = deterministic ID for this logical event
# PR review: "pr-review-{repo}-{pr_number}-{commit_sha}"
# Payment: "payment-{transaction_id}"
# Email: "email-{campaign_id}-{user_id}"
```

---

## Dead-Letter Queue Pattern

```python
# Celery config for DLQ:
task_routes = {
    "langgraph_tasks.execute_agent_task": {"queue": "agent_tasks"},
}
task_queues = (
    Queue("agent_tasks"),
    Queue("agent_tasks_dlq"),   # dead letter queue
)

# After max_retries exhausted → move to DLQ:
@celery_app.task(max_retries=3)
def execute_agent_task(self, task_dict):
    try:
        return process_agent_task(...)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            # Move to DLQ for manual inspection
            dead_letter_queue.put(task_dict)
            logger.error(f"DEAD LETTER: task {task_dict['task_id']}")
            return  # don't retry further
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

---

## Starting Celery Workers

```bash
# Terminal 1: Start Redis
docker run -p 6379:6379 redis

# Terminal 2: Start worker
celery -A lesson_19_event_driven.lesson_19_event_driven worker \
       --loglevel=info \
       --concurrency=4    # 4 concurrent tasks per worker

# Terminal 3: Start Flower (Celery monitoring UI)
pip install flower
celery -A lesson_19_event_driven.lesson_19_event_driven flower
# Open: http://localhost:5555
```

---

## Interview Q&A — Lesson 19

**Q1: When would you use a job queue (Celery) instead of direct HTTP for an agent?**
When: (1) Task takes > 30 seconds — HTTP clients time out. (2) Task must survive server restarts — queued tasks persist in Redis. (3) Need to scale consumers independently — add more Celery workers without touching the API. (4) Webhook sources retry on failure — idempotency in queue prevents duplicates. (5) Tasks must be prioritized — high-priority queue for premium tenants, low-priority for free tier.

**Q2: What is idempotency and why is it critical for event-driven agents?**
Idempotency: processing the same event multiple times produces the same result as processing it once. Critical because: webhooks (GitHub, Stripe, Slack) retry failed deliveries. Network timeouts cause duplicate sends. Message queues have "at-least-once" delivery guarantees. Without idempotency: a PR gets reviewed twice, a user gets two emails, a payment gets charged twice. Implementation: check a deterministic `idempotency_key` before processing; if already processed, return the cached result.

**Q3: How does Celery handle task failures and retries?**
Celery retries are configured per task: `max_retries=3, default_retry_delay=60`. Inside the task: `raise self.retry(exc=exc, countdown=2 ** self.request.retries)` — exponential backoff. After max_retries, the task enters FAILURE state in the result backend. Monitor with Flower UI or `celery_app.AsyncResult(task_id).state`. Route permanently failed tasks to a dead-letter queue for manual inspection.

**Q4: What is the difference between Celery broker and backend?**
Broker: where tasks are queued (Redis or RabbitMQ). Workers pull tasks from the broker. Backend: where task results are stored (Redis or database). Clients query the backend for task status and results. You can use Redis for both. RabbitMQ is preferred for broker (more reliable delivery, better DLQ support) with Redis as backend (fast result retrieval).

**Q5: How do you scale a Celery-based agent system?**
Three dimensions: (1) **Workers**: add more worker instances (`celery worker --concurrency=8`). (2) **Broker**: Redis Cluster for horizontal scaling; RabbitMQ with clustering. (3) **Priority queues**: separate queues for fast tasks (Slack reply → 1 second) vs slow tasks (PR review → 2 minutes). Route premium tenants to dedicated high-concurrency workers. Use `--autoscale=10,2` (max 10, min 2 workers) to scale dynamically.

---

**Q6: How do you secure webhooks against replay and spoofing attacks?**
Three protections: (1) **HMAC signature**: sender computes `sha256=HMAC(secret, payload)` and sends it in a header. Receiver recomputes and compares with `hmac.compare_digest()` — never `==` (timing attack). (2) **Timestamp check**: include `X-Timestamp` in the signed payload; reject events older than 5 minutes (replay protection). (3) **Idempotency key**: even if a valid replay gets through, the idempotency store returns the cached result without re-processing. All three layers are required — HMAC alone doesn't prevent replay of a captured valid request.

**Q7: What is the Dead Letter Queue pattern and when do you need it?**
A DLQ is a separate queue where tasks are placed after exhausting all retries. Without DLQ: failed tasks silently disappear — you don't know what failed or why. With DLQ: every failure is preserved with full context (task payload, error, retry count). Operations team can inspect, fix the root cause, and re-enqueue for processing. Alert on `dlq_depth > 0` for 5 minutes in Grafana — every DLQ entry represents lost work. Production: use Redis `LPUSH dlq:<queue>` or AWS SQS dead letter queue.

---

## Tasks — Lesson 19

**Task 19.1** — Run the demo without Celery. Add a new task type: `"document_summary"` that accepts `{"url": str, "content": str}` and returns a 3-sentence summary. Add the dispatch branch in `process_agent_task`.

**Task 19.2** — Test idempotency: create two `AgentTask` objects with the same `idempotency_key` but different `task_id`. Verify in logs that the second one returns cached result without calling the LLM.

**Task 19.3** — Add a simulated retry: make `process_agent_task` raise an exception on the first call (use a flag). Verify the retry logic kicks in with exponential backoff.

**Task 19.4** — *(Advanced)* If you have Redis running: set `USE_CELERY=true`, install Celery, start a worker, and enqueue a real task. Use `celery_app.AsyncResult(task_id).status` to poll the result.

**Task 19.5** — Webhook security: call `sign_webhook_payload(payload)` to get a valid signature. Then call `verify_webhook_signature(payload, sig)` → True. Tamper with the payload bytes by changing one character → False. Try with an empty signature header → False. Understand why `hmac.compare_digest()` is required instead of `==`.

**Task 19.6** — DLQ exercise: make `process_agent_task` always raise an exception (add `raise RuntimeError("forced failure")`). Run a task and let it exhaust all 3 retries. Verify `dead_letter_queue.depth() == 1`. Call `dead_letter_queue.drain()` and inspect the entry. Remove the forced failure.

---

# Lesson 20 — Cost Control, Governance & Enterprise Capstone

> **File:** `lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py`

---

## Model Routing — Use the Cheapest Sufficient Model

```
Question → complexity classifier
              ↓
  simple (score 1-3) → fast model ($0.0002/1k tokens)
  medium (score 4-6) → balanced model ($0.001/1k tokens)
  complex (score 7-10) → powerful model ($0.015/1k tokens)

Example savings:
  1000 requests/day, 80% simple, 15% medium, 5% complex
  All powerful: 1000 × $0.015 = $15/day
  With routing: 800×$0.0002 + 150×$0.001 + 50×$0.015 = $0.16 + $0.15 + $0.75 = $1.06/day
  Saving: 93% cost reduction
```

---

## Token Budget Enforcement

```python
# Before every LLM call:
ok, reason = budget_manager.can_proceed(tenant_id, estimated_tokens=1000)
if not ok:
    return {"messages": [AIMessage(content=f"Budget exhausted: {reason}")]}

# After LLM call:
budget_manager.record_usage(tenant_id, actual_tokens, model_tier)

# Daily budget resets at midnight UTC:
# Reset logic: compare stored reset_date with today's date
```

**Production implementation:** store in Redis with key `budget:{tenant_id}:{date}` and TTL = 25 hours. Atomic `INCRBYFLOAT` for cost accumulation (no race conditions).

---

## Circuit Breaker States

```
CLOSED (normal) ─────────────────────────────────────────────────────
│  All calls pass through.                                           │
│  Failure counter increments on each error.                         │
│  If failures ≥ threshold → OPEN                                   │
▼                                                                    │
OPEN (failing) ──────────────────────────────────────────────────── │
│  All calls blocked immediately (return error).                     │ recover
│  After cooldown_seconds → HALF_OPEN                               │
▼                                                                    │
HALF_OPEN (testing) ─────────────────────────────────────────────── │
│  One test call allowed through.                                    │
│  If success → CLOSED (reset counter)                               │
│  If failure → OPEN (reset timer)                                  ─┘
```

---

## GDPR Implementation Checklist

```
□ Right to erasure (Art. 17):
    endpoint: DELETE /users/{user_id}/data
    effect: anonymize all records (replace content with "[ERASED]")
    log: audit entry "GDPR_ERASURE"

□ Right to portability (Art. 20):
    endpoint: GET /users/{user_id}/data-export
    format: JSON with all interactions
    response time: < 30 days (legal requirement)

□ Data minimization (Art. 5):
    don't log raw message content in audit trail
    store user_id + action + timestamp only
    PII stays in encrypted application DB

□ Breach notification (Art. 33):
    alert within 72 hours of discovering a breach
    implement: anomaly detection on access patterns
    tool: Datadog SIEM or similar
```

---

## Complete Enterprise Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE AGENT SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│  GATEWAY LAYER                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  nginx/ALB → FastAPI (4 workers) → JWT auth → rate limit │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  ORCHESTRATION LAYER      ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LangGraph StateGraph                                    │    │
│  │  START → RBAC → Complexity Router → Budget → Circuit     │    │
│  │        → Chat (async) → Compliance Logger → END          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  EXECUTION LAYER          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Model routing: fast / balanced / powerful               │    │
│  │  Celery workers: PR review, Slack bot, scheduled reports │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  PERSISTENCE LAYER        ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  OracleSaver/19c (checkpoints, TDE, Audit Vault, VPD)  │    │
│  │  + Redis (cache + idempotency + budgets + rate limits)  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  OBSERVABILITY LAYER      ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Prometheus → Grafana | OpenTelemetry → Jaeger           │    │
│  │  Structured logs → ELK | LangSmith (LLM traces)         │    │
│  │  PagerDuty alerts | Slack notifications                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  GOVERNANCE LAYER         ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Token budgets | Audit trail (SOC2) | GDPR endpoints     │    │
│  │  Circuit breaker | Model fallbacks | Cost reporting      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Interview Q&A — Lesson 20

**Q1: How do you control LLM costs in a multi-tenant production system?**
Three mechanisms: (1) **Model routing**: classify question complexity and route to the cheapest model that can handle it (saves 70-90%). (2) **Token budgets**: per-tenant daily/monthly budget enforced in a gate node before LLM calls — block requests when budget is exhausted. (3) **Prompt optimization**: trim conversation history to the last N messages, use structured outputs (shorter than prose), cache repeated queries in Redis. Monitor with `tokens_total` Prometheus counter and daily cost reports per tenant.

**Q2: Explain the circuit breaker pattern for LLM APIs.**
Three states: CLOSED (normal operation, all calls pass through), OPEN (LLM API is failing, block all calls immediately for `cooldown_seconds`), HALF_OPEN (after cooldown, allow one test call — if it succeeds, go CLOSED; if it fails, go OPEN again). Implementation: count consecutive failures, open after `failure_threshold` is reached. Without circuit breaker: if the LLM API is slow/down, all your requests pile up waiting, exhausting thread pools and making your entire service unresponsive.

**Q3: How do you implement GDPR right-to-erasure for a LangGraph agent?**
(1) Every interaction is recorded with `user_id` as the key. (2) `DELETE /users/{user_id}/data` calls `gdpr_erasure(user_id)` which: replaces all record content with `"[ERASED]"`, removes the user_id index, and writes an audit event `GDPR_ERASURE`. (3) LangGraph checkpoints: `OracleSaver.delete()` runs `DELETE FROM langgraph_checkpoints WHERE thread_id LIKE '{user_id}-%'`. (4) Return `{"erased_records": N}` to confirm. Legal requirement: complete within 30 days.

**Q4: How would you design the system to handle 10,000 requests per day from 500 tenants?**
Math: 10,000/day ≈ 7 req/minute average. Peak: 100 req/minute. Architecture: (1) FastAPI with 4 uvicorn workers (async) → handles 100+ concurrent requests. (2) OracleSaver with pool max=8 per server, Oracle RAC for HA. (3) Redis cache for repeated queries (80% cache hit rate → 80% cost reduction). (4) Celery for tasks > 10 seconds (PR reviews, reports). (5) Per-tenant rate limiting: 20 req/minute max. (6) Model routing: 80% fast model, 15% balanced, 5% powerful. Budget: 10,000 req × average 500 tokens × $0.001/1k = $5/day.

**Q5: What is the difference between horizontal and vertical scaling for a LangGraph agent system?**
Vertical: make one server bigger (more RAM, more CPU cores). Limited — you hit single-machine limits. Horizontal: add more servers. Requires: (1) Stateless API servers (state in OracleSaver, not RAM) — any new server can pick up any request. (2) Shared Redis for rate limiting and caching (each server sees the same limits). (3) Load balancer (nginx, AWS ALB) distributing traffic. Oracle RAC provides active-active DB nodes — no single point of failure. LangGraph's thread_id-based isolation makes horizontal scaling natural — any server handles any thread_id because state is in the shared Oracle DB.

---

## Tasks — Lesson 20

**Task 20.1** — Run the demo. Read the cost report. Observe model routing: which tier was selected for each question? Add a 4th question that forces `powerful` tier.

**Task 20.2** — Lower the `acme-corp` daily budget to $0.0001 (forces budget exhaustion). Run 5 requests and observe the budget node blocking after budget is exhausted.

**Task 20.3** — Trigger the circuit breaker: make `enterprise_chat_node` raise an exception on every call. Watch the breaker open after 5 failures. Add a `time.sleep(cooldown_seconds)` and verify it transitions to HALF_OPEN.

**Task 20.4** — Add a cost reporting endpoint to the enterprise graph: a `cost_report_node` that runs only if `action_requested == "cost_report"` and returns the `get_usage_report()` dict as an AIMessage.

**Task 20.5 (Capstone)** — Wire Lesson 17 (RBAC) + Lesson 18 (observability) + Lesson 20 (budget) together in a single FastAPI app. Endpoints: `POST /ask`, `GET /health/ready`, `GET /usage/{tenant_id}`. Test with 3 tenants, 3 roles, and verify all three systems work together.

---

# Lesson 21 — AWS Bedrock: Production LLMs Without Ollama

> **File:** `lesson_21_aws_bedrock/lesson_21_aws_bedrock.py`

---

## Why Bedrock Instead of Ollama?

Ollama is perfect for learning (free, local, no credentials). But enterprise systems need:

| Feature | Ollama (local) | AWS Bedrock |
|---------|---------------|-------------|
| Model quality | Varies by GPU | Best-in-class (Claude 3) |
| Context window | 8k–128k | 200k (Claude 3) |
| Scalability | Single machine | Fully managed, multi-region |
| Tool calling | Model-dependent | Native (Claude, Titan) |
| Compliance | Your responsibility | SOC2, HIPAA, FedRAMP |
| Cost | Free (GPU power) | Pay per token |
| Production ops | You manage | AWS manages |

**When to use Bedrock:** production systems, team collaboration (no GPU required), compliance requirements, context > 32k, best accuracy.
**When to keep Ollama:** learning/experimentation, air-gapped environments, data that must never leave your machine.

---

## Bedrock Model IDs — Cheatsheet

```python
# ANTHROPIC CLAUDE — best for agents, instruction following, tool calling
"anthropic.claude-3-haiku-20240307-v1:0"    # fast + cheapest  $0.00025/1k in
"anthropic.claude-3-sonnet-20240229-v1:0"   # balanced         $0.003/1k in
"anthropic.claude-3-opus-20240229-v1:0"     # most capable     $0.015/1k in

# AMAZON TITAN — no separate vendor agreement, AWS-native
"amazon.titan-text-express-v1"              # general purpose
"amazon.titan-text-lite-v1"                 # fast + cheap
"amazon.titan-text-premier-v1:0"            # most capable Titan

# META LLAMA — open-weight, closest to Ollama models
"meta.llama3-8b-instruct-v1:0"              # similar to ollama llama3
"meta.llama3-70b-instruct-v1:0"             # large + powerful

# MISTRAL
"mistral.mistral-7b-instruct-v0:2"
"mistral.mixtral-8x7b-instruct-v0:1"
```

**Enable model access:** AWS Console → Bedrock → Model access → Request access.
Haiku/Sonnet are usually available instantly. Opus may need approval.

---

## Migrating from Ollama to Bedrock — One Line Change

```python
# BEFORE (Ollama — lessons 1-20)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2", temperature=0.1)

# AFTER (Bedrock)
import boto3
from langchain_aws import ChatBedrockConverse
llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=boto3.Session().client("bedrock-runtime"),
)

# EVERYTHING ELSE IS IDENTICAL:
llm.invoke(messages)          # same
llm.stream(messages)          # same
llm.bind_tools(tools)         # same
graph.invoke() / ainvoke()    # same
with_structured_output()      # same
```

---

## boto3 Credential Chain

```
AWS credential resolution (automatic, in order):
  1. Env vars:       AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
  2. Named profile:  AWS_PROFILE=myprofile  (~/.aws/credentials)
  3. IAM role:       EC2 / ECS task / Lambda (instance metadata IMDS)
  4. AWS SSO:        aws sso login

Enterprise pattern:
  Local dev:    aws configure  (one-time setup)
  Production:   IAM role on EC2/ECS — NO hardcoded keys in code or env
```

**Minimum IAM policy for Bedrock:**
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": "arn:aws:bedrock:*::foundation-model/*"
}
```

---

## Bedrock Guardrails

```
Without Guardrails:   LLM response → your application → user
With Guardrails:      LLM response → Bedrock Guardrail engine → your application → user

Guardrail capabilities:
  □ Content filters: block harmful/hate/sexual/violence content
  □ Topic denial: block topics you define (competitors, off-topic)
  □ PII detection: auto-mask SSN, credit card, email in responses
  □ Word filters: block specific words/phrases
  □ Grounding check: verify response is grounded in provided context

Usage in code:
  llm = ChatBedrockConverse(
      model_id="...",
      guardrail_config={"guardrailIdentifier": "abc123", "guardrailVersion": "DRAFT"},
  )
  # Guardrail applied by AWS before response returns — zero code change

Detection: if guardrail triggered:
  response.response_metadata["amazon-bedrock-guardrailAction"] == "INTERVENED"
```

---

## ChatBedrockConverse vs ChatBedrock

```
ChatBedrockConverse (recommended):
  - Uses the new Converse API (unified across all models)
  - Supports: streaming, tool calling, system messages natively
  - Same request/response format regardless of model
  - Use for: all new code

ChatBedrock (legacy):
  - Uses InvokeModel API (model-specific JSON format)
  - Requires different request body for Claude vs Titan vs Llama
  - Still works but more complex
  - Use only if: model not yet supported by Converse API
```

---

## Cost Tracking with Token Usage

```python
response = llm.invoke(messages)

# Token usage in response metadata:
usage = response.response_metadata.get("usage", {})
input_tokens  = usage.get("input_tokens", 0)
output_tokens = usage.get("output_tokens", 0)

# Estimate cost:
HAIKU_RATES = (0.00025, 0.00125)  # (input_per_1k, output_per_1k)
cost = (input_tokens / 1000 * HAIKU_RATES[0]) + (output_tokens / 1000 * HAIKU_RATES[1])
```

Always check [https://aws.amazon.com/bedrock/pricing/](https://aws.amazon.com/bedrock/pricing/) — rates change.

---

## Interview Q&A — Lesson 21

**Q1: What is the difference between `ChatBedrockConverse` and `ChatBedrock`?**
`ChatBedrockConverse` uses the newer Converse API which provides a unified request/response format across all Bedrock models (Claude, Titan, Llama, Mistral). It handles system messages, tool calling, and streaming natively without model-specific JSON formatting. `ChatBedrock` uses the older `InvokeModel` API which requires model-specific request bodies — different JSON structure for Claude vs Titan vs Llama. Always use `ChatBedrockConverse` for new code. Only fall back to `ChatBedrock` if a newly-released model hasn't been added to the Converse API yet.

**Q2: How does boto3 resolve AWS credentials, and what's the production best practice?**
boto3 checks in order: (1) explicit constructor params (never do this), (2) env vars `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, (3) named profile from `~/.aws/credentials`, (4) EC2/ECS instance metadata service (IAM role), (5) AWS SSO. Production best practice: attach an IAM role to your EC2 instance or ECS task with minimum required permissions. Zero credentials in code or environment variables — the IMDS provides temporary credentials automatically and rotates them.

**Q3: How do you prevent a Bedrock LLM from responding to off-topic or harmful queries?**
Use Bedrock Guardrails: (1) Create a guardrail in AWS Console with topic denial policies, content filters, and PII masking rules. (2) Pass `guardrail_config={"guardrailIdentifier": "...", "guardrailVersion": "DRAFT"}` to `ChatBedrockConverse`. (3) Bedrock applies the guardrail server-side before returning the response — no application code change required. Check `response.response_metadata["amazon-bedrock-guardrailAction"] == "INTERVENED"` to detect triggered guardrails. This is superior to prompt engineering for safety — it cannot be bypassed by prompt injection.

**Q4: How do you migrate a production LangGraph system from Ollama to Bedrock with zero downtime?**
Feature flag approach: (1) Add env var `LLM_PROVIDER=ollama|bedrock`. (2) `get_llm()` factory returns `ChatOllama` or `ChatBedrockConverse` based on the flag. (3) Deploy new code with `LLM_PROVIDER=ollama` — no behaviour change. (4) Switch `LLM_PROVIDER=bedrock` for 5% of traffic (canary). (5) Monitor error rate, latency, cost. (6) Gradually ramp to 100%. Rollback = set `LLM_PROVIDER=ollama`. All graph nodes, tools, and state are unchanged — only the LLM instantiation differs.

**Q5: How do you stream Bedrock responses in a FastAPI endpoint?**
```python
from fastapi.responses import StreamingResponse

@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    llm = ChatBedrockConverse(model_id=..., streaming=True, client=bedrock_client)
    messages = [HumanMessage(content=request.question)]

    async def generate():
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"  # SSE format
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Q6: How do you select the right Bedrock model for different use cases in an enterprise system?**
Three-tier routing: (1) `haiku` — simple Q&A, short queries (< 20 words, no complexity keywords), fastest response (< 1s), cheapest ($0.00025/1k in). (2) `sonnet` — complex analysis, code generation, multi-step reasoning, 2–5s response. (3) `opus` — hardest tasks only: legal document review, complex architecture decisions. Production: start all traffic on Haiku, measure quality with LangSmith, identify which queries need upgrading. Target: 80% Haiku, 15% Sonnet, 5% Opus → ~93% cost saving vs all-Sonnet.

**Q7: What IAM permissions does a LangGraph agent need to call Bedrock, and why use least privilege?**
Minimum required: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on `arn:aws:bedrock:*::foundation-model/*`. Optionally scope to specific models: `arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0`. Least privilege matters because: if the agent role is compromised (prompt injection → SSRF), the attacker can only call Bedrock, not access S3 buckets, RDS databases, or other services. Never grant `bedrock:*` or `*`. Audit via CloudTrail: every `InvokeModel` call is logged with the IAM principal, model ID, and request ID.

---

## Tasks — Lesson 21

**Task 21.1** — Run the demo. Check which mode it started in (Bedrock or Ollama fallback). If using Ollama, run `aws configure` with test credentials and re-run to see the Bedrock path.

**Task 21.2** — Migrate Lesson 4 (ReAct tools agent) to Bedrock: change the one `ChatOllama` line to `ChatBedrockConverse`. Verify `bind_tools()` and `ToolNode` work identically. Run the same questions and compare responses.

**Task 21.3** — Model selection: lower the word-count threshold in `model_selector_node` to 5. Send a 6-word question and observe it routing to Sonnet. Restore the threshold.

**Task 21.4** — Cost tracking: send 10 questions, track `estimated_cost_usd` per turn, print a summary at the end. Compare Claude Haiku cost vs Sonnet cost for the same questions.

**Task 21.5 (Advanced)** — Set up a Bedrock Guardrail in the AWS Console that blocks questions about competitors. Set `BEDROCK_GUARDRAIL_ID` and test that a competitor question returns the guardrail message while a normal question passes through.

---

```
Week 1 — Foundation to Advanced (Lessons 1-10)
  □ Complete all notebooks
  □ Read all Deep Dive files

Week 2 — Senior Level (Lessons 11-15)
  □ Run all lesson scripts
  □ Pass 18/18 unit tests

Week 3 — Enterprise (Lessons 16-20)
  □ Lesson 16: Run sync + async demos, understand checkpointer selection
  □ Lesson 17: RBAC demo, add new tenant and role
  □ Lesson 18: Prometheus metrics, trace propagation
  □ Lesson 19: Event-driven patterns, idempotency test
  □ Lesson 20: Cost control, circuit breaker, GDPR

Week 4 — Build
  □ Task 20.5: Full enterprise FastAPI app
  □ Deploy to Docker with all layers working
  □ Implement 3 system design questions from Part 4 with enterprise patterns
```

---

## Enterprise Checklist (print and verify before production)

```
AUTHENTICATION & AUTHORIZATION
  □ JWT validation on every endpoint (no unauthenticated routes)
  □ RBAC check in graph auth node before any action
  □ Tenant isolation: thread_id includes tenant_id prefix
  □ Tool list scoped per tenant (no cross-tenant tool access)

PERSISTENCE (Oracle 19c)
  □ OracleSaver in production (not MemorySaver or SqliteSaver)
  □ oracledb.create_pool() with min=2, max=workers×2 (not per-request connections)
  □ langgraph_checkpoints table created with CLOB columns + primary key
  □ GRANT SELECT/INSERT/UPDATE/DELETE to application user (least privilege)
  □ TDE encryption enabled on checkpoint CLOB column (AES256)
  □ Oracle Data Guard or RAC configured for HA/DR
  □ RMAN backup schedule configured (point-in-time recovery)

ASYNC & CONCURRENCY
  □ All nodes are async (async def + await)
  □ graph.ainvoke() in FastAPI (not graph.invoke() in async context)
  □ asyncio.to_thread() if any remaining sync calls
  □ Celery workers for tasks > 10 seconds

OBSERVABILITY
  □ Prometheus metrics: requests, errors, latency, active sessions, tokens
  □ trace_id generated at API boundary, propagated through all nodes
  □ Health: /health/live and /health/ready endpoints
  □ Grafana dashboards with p99 latency and error rate panels
  □ Alerts: error > 5%, p99 > 5s, active_sessions > limit

COST CONTROL
  □ Model routing by complexity (not always powerful model)
  □ Token budget per tenant per day enforced in gate node
  □ Redis cache for repeated queries (schema descriptions, FAQ)
  □ Daily cost report emailed to tenant admin

GOVERNANCE & COMPLIANCE
  □ Immutable audit trail: WHO + WHAT + WHEN + RESULT
  □ GDPR erasure endpoint tested and working
  □ GDPR data export endpoint tested and working
  □ Circuit breaker on all external API calls (LLM, DB)
  □ Exponential backoff on retries
  □ Dead-letter queue for failed tasks

DEPLOYMENT
  □ Secrets in environment variables (never in code)
  □ Docker with slim base image
  □ /health endpoints configured in Kubernetes probes
  □ Rolling deployment (zero downtime)
  □ LangSmith tracing enabled in production
```

---

# Lesson 22 — AWS S3 Integration

> **File:** `lesson_22_aws_s3/lesson_22_aws_s3.py`

---

## Why S3 in the Architecture

The architecture diagram shows an **AWS S3 Bucket** connected to both the EC2 server and the agents. S3 serves two distinct roles:

| Role | What Is Stored | Key Format |
|------|---------------|-----------|
| **Conversation storage** | JSON snapshots of message history | `conversations/{tenant_id}/{thread_id}/{timestamp}.json` |
| **Document storage** | CSV, PDF, JSON files from users | `documents/{tenant_id}/{doc_id}/{filename}` |

---

## S3 vs Database for Conversation History

| Feature | Oracle/SQLite (L16/L8) | S3 |
|---------|------------------------|-----|
| Structured queries | ✓ Fast | ✗ No |
| Long-term cheap storage | ✗ Expensive | ✓ Cheap |
| Cross-server access | ✓ | ✓ |
| Audit snapshots | Partial | ✓ Immutable history |
| GDPR erasure | Row delete | Object delete |

**Use both:** Oracle/SQLite for active session state, S3 for durable conversation archives.

---

## Core Patterns

### 1. Presigned URLs (Frontend → S3 Direct Upload/Download)

```python
url = s3_generate_presigned_url("documents/tenant_acme/doc_001/file.csv", expiry_seconds=3600)
```

The EC2 server generates the URL but is NOT in the download path. The browser communicates directly with S3. This eliminates EC2 as a bottleneck for large files.

### 2. Conversation Snapshots

```python
key = save_conversation_to_s3(tenant_id="tenant_acme", thread_id="thread_001", messages=messages)
snapshot = load_latest_conversation_from_s3(tenant_id, thread_id)
```

Key format: `conversations/tenant_acme/thread_001/20240101_120000.json`
Each snapshot is immutable — you get a full audit trail of all conversation states.

### 3. GDPR Erasure

S3 data must be deleted separately from database data. Many implementations forget this.

```python
result = erase_user_data(tenant_id="tenant_acme", thread_id="thread_001")
# Deletes ALL conversation snapshots AND document files for that tenant/thread
```

### 4. Streaming Uploads (Large Files)

```python
s3_upload_file_stream(key, file_bytes, content_type="text/csv")
```

Uses `BytesIO` to stream bytes directly to S3 without loading the full file into memory as a string.

---

## IAM Policy (Least Privilege)

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::your-bucket-name",
    "arn:aws:s3:::your-bucket-name/*"
  ]
}
```

---

## Q&A

**Q1: Why not store conversation history only in S3?**
S3 has no SQL queries. You cannot ask "give me all threads for tenant X in the last 24 hours" without listing every key. Use Oracle/SQLite for active lookups, S3 for archives.

**Q2: What is a presigned URL and when does it expire?**
A presigned URL is a time-limited signed URL that grants access to a specific S3 object without needing AWS credentials. Typical TTL: 15 minutes (900s) to 1 hour (3600s). After expiry, the URL returns 403.

**Q3: How do you prevent one tenant from accessing another tenant's files?**
The S3 key always includes `{tenant_id}` as a prefix. The API verifies `user["tenant_id"] == request_tenant_id` before generating any presigned URL (see Lesson 23 `/upload` endpoint).

**Q4: What happens to S3 operations when AWS credentials are unavailable?**
`create_s3_client()` catches `NoCredentialsError` and returns `None`. All helper functions check `if s3_client is None` and log a simulation message instead of failing. This allows local dev without S3.

**Q5: Why use S3 key prefixes (conversations/, documents/) instead of separate buckets?**
One bucket with IAM path-based policies is simpler to manage than multiple buckets. S3 buckets are global — naming conflicts are common. Prefixes + bucket policies give equivalent isolation.

---

## Tasks

- **Task 22.1:** Add a `compress=True` option to `s3_upload_text` that gzip-compresses the content before uploading and adds `ContentEncoding: gzip` to the S3 object metadata.
- **Task 22.2:** Write a `list_conversation_threads(tenant_id)` function that returns all unique thread IDs for a tenant by parsing S3 key prefixes.
- **Task 22.3:** Add an expiry mechanism to `save_conversation_to_s3` — pass the number of days after which S3 should auto-delete the object using S3 Object Lifecycle Rules.
- **Task 22.4:** Implement `copy_conversation_to_archive(tenant_id, thread_id)` that copies snapshots to a `archive/` prefix (cheaper storage class: S3 Glacier Instant Retrieval).
- **Task 22.5:** Write a unit test for `save_conversation_to_s3` using `moto` (AWS mocking library) to mock S3 without real credentials.

---

# Lesson 23 — Conversation Management API Layer

> **File:** `lesson_23_conversation_api/lesson_23_conversation_api.py`
> **Run:** `uvicorn lesson_23_conversation_api.lesson_23_conversation_api:app --reload --port 8023`

---

## Architecture Role

This lesson builds the **Conversation Management Layer / Chatbot API** — the single entry point between the Front End and all backend agents.

```
Front End
    └── Conversation Management Layer (THIS LESSON)
           └── EC2 FastAPI Server
                  └── Agent Orchestration
                         ├── Data Analysis Agent
                         ├── DB Retrieval Agent
                         └── General Agent
```

---

## The Session Model

A **session** represents one ongoing conversation between a user and an agent. Key fields:

| Field | Purpose |
|-------|---------|
| `session_id` | Stable handle for the frontend (returned at session creation) |
| `thread_id` | LangGraph `thread_id` — links to MemorySaver/OracleSaver state |
| `agent_type` | Which specialist handles this session: `data_analysis`, `db_retrieval`, `general` |
| `message_count` | Tracks usage for enforcement (MAX_MESSAGES_PER_SESSION) |
| `tenant_id` | Isolation key — sessions cannot cross tenant boundaries |

---

## Agent Routing

The `route_to_agent()` function decides which specialist node handles each message:

1. **Session-level routing:** `agent_type` set at session creation determines the agent for the entire session (e.g. a data analysis session always goes to the Data Analysis Agent).
2. **Keyword fallback:** If `agent_type = "general"`, keywords in the message trigger specialist routing ("analyze" → Data Analysis, "query" → DB Retrieval).

This is **orchestrator-level routing** — different from the ReAct tool-selection in Lesson 4. The Conversation API decides routing before the LLM is invoked.

---

## Usage Limitations (Architecture Box)

Three layers enforced before the LLM is ever called:

```
Request → verify JWT → load session → check_usage_limits() → route_to_agent() → LLM
```

| Limit | Default | Config Env Var |
|-------|---------|---------------|
| Messages per session | 100 | `MAX_MESSAGES_PER_SESSION` |
| Sessions per tenant | 50 | `MAX_SESSIONS_PER_TENANT` |
| Requests per minute per user | 30 | `RATE_LIMIT_RPM` |

All limits return HTTP 429 with a descriptive message. The frontend should show the user a clear explanation.

---

## SSE Streaming

The `/chat/stream` endpoint uses Server-Sent Events (SSE):

```python
async def token_generator():
    for chunk in llm.stream(messages):
        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
    yield "data: [DONE]\n\n"
return StreamingResponse(token_generator(), media_type="text/event-stream")
```

Frontend receives tokens incrementally:
```javascript
const es = new EventSource('/chat/stream?...');
es.onmessage = (e) => { if (e.data !== '[DONE]') appendToken(JSON.parse(e.data).token); }
```

---

## API Endpoints Summary

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/sessions` | Create session → returns `session_id` + JWT |
| POST | `/chat` | Send message → returns agent response |
| POST | `/chat/stream` | Streaming SSE chat |
| GET | `/sessions/{id}/history` | Get conversation history |
| GET | `/sessions` | List tenant sessions |
| POST | `/upload/{session_id}` | Upload document → S3 |
| GET | `/health` | Health check for ALB |

---

## Q&A

**Q1: Why is a session layer needed when LangGraph already has thread_id?**
`thread_id` is a LangGraph internal. The session layer adds: tenant isolation, usage limits, JWT auth, agent routing, and API contracts for the frontend. You cannot expose `thread_id` directly to the frontend.

**Q2: How do you persist sessions across EC2 restarts?**
Replace the in-memory `_session_store` dict with Redis (Lesson 19 pattern): `session_store.setex(session_id, TTL, json.dumps(session))`. Redis survives EC2 restarts and is shared across multiple EC2 instances.

**Q3: What is the difference between `/chat` and `/chat/stream`?**
`/chat` waits for the full LLM response then returns it as JSON. `/chat/stream` yields tokens one by one as SSE events. Use streaming for long responses to improve UX.

**Q4: How does the session layer prevent cross-tenant data leakage?**
JWT contains `tenant_id`. Every session lookup verifies `session["tenant_id"] == jwt["tenant_id"]`. The `thread_id` passed to LangGraph always includes `tenant_id` as a prefix, so no two tenants can share a memory thread.

**Q5: Why is Nginx rate limiting AND Python rate limiting both used?**
Nginx rate limiting drops requests before they reach Python — protects against DDoS and abusive clients cheaply. Python rate limiting enforces per-user business rules (e.g. 30 RPM per paying user). Defense in depth.

---

## Tasks

- **Task 23.1:** Add a `DELETE /sessions/{session_id}` endpoint that marks the session inactive and calls `erase_user_data()` from Lesson 22 if `gdpr=true` query param is set.
- **Task 23.2:** Replace the in-memory `_session_store` with a Redis-backed implementation (use `redis.Redis` from Lesson 19, store sessions with 24h TTL).
- **Task 23.3:** Add a `confidence_score` field to `ChatResponse` by asking the LLM to rate its own answer confidence (0-1) using structured output (`with_structured_output`).
- **Task 23.4:** Add a `/sessions/{id}/summarize` endpoint that uses the LLM to generate a one-paragraph summary of the conversation so far.
- **Task 23.5:** Write an integration test that creates a session, sends 3 messages, verifies the message_count increments, and verifies the 4th message hits the rate limit.

---

# Lesson 24 — EC2 Production Deployment

> **File:** `lesson_24_ec2_deployment/lesson_24_ec2_deployment.py`

---

## The Full Architecture Stack on EC2

This lesson connects all previous lessons into a single running production system:

```
Internet → ALB → Nginx (L24) → uvicorn → FastAPI (L23)
                                              ├── JWT auth (L17)
                                              ├── Agent orchestration (L5/L11)
                                              │      ├── Data Analysis Agent (L6)
                                              │      ├── DB Retrieval Agent (L6/L16)
                                              │      └── General Agent (L3/L4)
                                              ├── Bedrock LLMs (L21)
                                              ├── S3 storage (L22)
                                              ├── Oracle persistence (L16)
                                              ├── Prometheus metrics (L18)
                                              └── CloudWatch logs (L24)
```

---

## IAM Instance Profile (Never Use Access Keys on EC2)

| Approach | Security | Dev Experience |
|---------|----------|---------------|
| Hardcoded keys in code | ❌ Terrible | Easy (don't do this) |
| `.env` file with keys | ❌ In AMI snapshots | Easy (don't do this) |
| Environment variables | ⚠️ OK for dev | OK |
| IAM instance profile | ✅ Best | Transparent — boto3 auto-discovers |

IAM role attached to EC2 → boto3 queries instance metadata at `169.254.169.254` → credentials rotated automatically every ~6 hours → **zero secrets in code, config, or environment files**.

---

## SSM Parameter Store vs Environment Variables

| Feature | `.env` file | SSM Parameter Store |
|---------|------------|-------------------|
| Encryption at rest | ❌ | ✅ KMS-encrypted SecureString |
| Audit trail | ❌ | ✅ CloudTrail logs every read |
| Centralized | ❌ Per-instance | ✅ All EC2s share same params |
| Rotation | Manual | ✅ Can integrate with Secrets Manager |
| Access control | File permissions | ✅ IAM policy per parameter path |

Use pattern: `/p5/{env}/{key}` → `/p5/prod/JWT_SECRET_KEY`, `/p5/dev/JWT_SECRET_KEY`

---

## Health Checks: Live vs Ready

| Endpoint | Checks | Fails When |
|---------|--------|-----------|
| `/health/live` | Process is alive | Process hangs or exits |
| `/health/ready` | All deps reachable | Bedrock, S3, Oracle unreachable |

ALB uses `/health` (maps to `/health/live`). Configure separately:
- **Liveness probe**: 30s interval, 3 failures → restart container/process
- **Readiness probe**: 10s interval, 1 failure → remove from ALB rotation

---

## Nginx: Key Settings for LangGraph APIs

| Setting | Value | Reason |
|---------|-------|--------|
| `proxy_read_timeout` | `120s` | LLM responses can take 30-60s |
| `proxy_buffering off` | On `/chat/stream` | SSE must flow immediately |
| `proxy_http_version 1.1` | All | Required for keepalive and SSE |
| Rate limit zone | `30r/m` | Nginx-level DDoS protection |

---

## Zero-Downtime Deployment Decision Tree

```
Is this a single EC2?
  YES → systemctl reload p5-chatbot   (workers restart one at a time)
  NO  → Using ALB?
          YES → Launch new EC2 → wait /health/ready → register in ALB
                → deregister old EC2 → wait connection drain → terminate
          NO  → Deploy during maintenance window
```

---

## CloudWatch Structured Logging

```python
logger = setup_production_logging(service_name="p5-chatbot-api", log_group="/p5/chatbot")
```

Every log line is a JSON object with: `timestamp`, `level`, `service`, `host`, `logger`, `message`, `trace_id`, `tenant_id`.

CloudWatch Logs Insights query examples:
```sql
fields @timestamp, level, tenant_id, message
| filter level = "ERROR"
| sort @timestamp desc
| limit 50
```

---

## Q&A

**Q1: Why use Nginx in front of uvicorn instead of exposing uvicorn directly?**
uvicorn is an ASGI server, not a hardened proxy. Nginx handles: SSL termination, request buffering (protects against slow-loris attacks), static files, rate limiting, and zero-downtime reload (`nginx -s reload`). Never expose uvicorn directly to the internet.

**Q2: What happens if SSM is unreachable on startup?**
`load_config_from_ssm()` catches all exceptions and falls back to `os.getenv()`. The service starts in degraded mode with a warning log. This prevents the server from refusing to start if SSM is briefly unavailable (e.g., VPC endpoint misconfiguration).

**Q3: How do you update code on EC2 without downtime?**
Pull new code → `pip install` → `sudo systemctl reload p5-chatbot`. The reload sends SIGHUP to uvicorn's master process, which gracefully restarts each worker one at a time. In-flight requests on old workers finish before those workers are replaced.

**Q4: What IAM permissions does the EC2 role need?**
Minimum: `bedrock:InvokeModel`, `s3:GetObject + PutObject + DeleteObject + ListBucket`, `ssm:GetParameters`, `logs:CreateLogGroup + CreateLogStream + PutLogEvents`. Keep the policy scoped to specific resource ARNs (not `*`).

**Q5: How do you debug a production issue on EC2?**
1. `journalctl -u p5-chatbot -f` — live systemd logs
2. CloudWatch Logs Insights — query JSON logs by tenant_id or trace_id
3. `curl http://localhost:8000/health/ready` — check dependency status
4. `sudo systemctl status p5-chatbot` — process state + last 10 log lines

---

## Tasks

- **Task 24.1:** Add a `/health/ready` FastAPI endpoint to Lesson 23's app that calls `get_readiness()` from this lesson and returns HTTP 200 (ready) or HTTP 503 (not ready).
- **Task 24.2:** Write a `rotate_ssm_parameter(key, new_value)` function that updates an SSM SecureString parameter and logs the rotation event to CloudWatch.
- **Task 24.3:** Add a startup lifespan handler to Lesson 23's FastAPI app that calls `production_startup()` and refuses to start if critical checks fail.
- **Task 24.4:** Write a shell script `deploy.sh` that: pulls the latest code from git, installs dependencies, runs `production_startup()` in a check-only mode, and then performs a systemd reload.
- **Task 24.5:** Configure an S3 bucket policy that allows only your EC2 IAM role ARN to read/write objects, and denies all access if requests come without HTTPS (`"Condition": {"Bool": {"aws:SecureTransport": "false"}}`).

---

## Architecture Completion Checklist

After completing Lessons 22–24, the full architecture is covered:

```
✅ Front End                 ← external (not in scope)
✅ Conversation Mgmt Layer   ← Lesson 23 (FastAPI, sessions, routing)
✅ EC2 FastAPI Server        ← Lesson 24 (Nginx, systemd, IAM, SSM)
✅ Agent Orchestration       ← Lessons 5, 11 (multi-agent, subgraphs)
✅ Data Analysis Agent       ← Lesson 6 (DB/data tools)
✅ DB Retrieval Agent        ← Lesson 6, 16 (Oracle async)
✅ Session & Memory Mgmt     ← Lessons 8, 16, 22 (memory + S3 snapshots)
✅ Usage Limitations         ← Lessons 17, 20, 23 (rate limit, token budget)
✅ Amazon Bedrock            ← Lesson 21 (ChatBedrockConverse)
✅ LLM Catalogs              ← Lesson 21 (model routing, Haiku/Sonnet/Opus)
✅ AWS Guardrails            ← Lesson 21 (Bedrock Guardrails)
✅ AWS S3 Bucket             ← Lesson 22 (documents, conversation snapshots)
✅ User Authority            ← Lesson 17 (JWT, RBAC)
✅ Conversation History      ← Lessons 8, 16, 22 (Oracle + S3)
✅ Document Storage          ← Lesson 22 (S3 upload, download, presigned URLs)
✅ Long-Term Memory (Mem0)   ← Lesson 25 (auto-extraction, dedup, GDPR erasure)
✅ Search + RAG (Solr)       ← Lesson 26 (BM25, kNN vector, hybrid search)
```

---

# Lesson 25 — Long-Term Memory with Mem0

> **File:** `lesson_25_mem0/lesson_25_mem0.py`

---

## Why Mem0?

Lesson 13 (Chroma) taught you to build vector memory manually: write an extraction prompt, parse JSON, store documents, handle duplicates yourself. Mem0 automates all of this.

| Capability | Chroma (Lesson 13) | Mem0 (Lesson 25) |
|---|---|---|
| Fact extraction | You write LLM prompt | Automatic |
| Deduplication | None | Built-in (LLM-driven) |
| Contradiction resolution | None | Auto-update/replace |
| Memory categories | Manual metadata | Auto-classified |
| Search | `similarity_search()` | `m.search(query, user_id)` |
| GDPR delete | Manual filter loop | `m.delete_all(user_id)` |
| Deployment | Local only | Local or Mem0 Cloud |

The key insight: when a user says "I now prefer TypeScript" after previously saying "I prefer Python", Mem0 **updates** the stored memory rather than storing a contradiction alongside the original.

---

## Mem0 Architecture

```
User turn
   ↓
[load_memories_node]  → m.search(query, user_id) → Qdrant (vector) + LLM (extraction)
   ↓
[chat_node]           → LLM with memory context injected into system prompt
   ↓
[save_memories_node]  → m.add(messages, user_id)
                          ├── LLM extracts facts
                          ├── Deduplication check
                          ├── Contradiction resolution (update vs add)
                          └── Store in Qdrant
```

---

## Deployment Options

### Option A — Mem0 Cloud (Managed)
```python
from mem0 import MemoryClient
m = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
```
- No infrastructure to manage
- Uses OpenAI for extraction (not local Ollama)
- Best for production

### Option B — Self-hosted (Local Ollama + Qdrant)
```python
config = {
    "llm":          {"provider": "ollama", "config": {"model": "llama3.2", ...}},
    "embedder":     {"provider": "ollama", "config": {"model": "llama3.2", ...}},
    "vector_store": {"provider": "qdrant", "config": {"url": "http://localhost:6333", ...}},
    "version":      "v1.1",
}
m = Memory.from_config(config)
```
- Full local control, no API key needed
- Requires: `docker run -p 6333:6333 qdrant/qdrant`
- Best for enterprise private-cloud deployments

### Option C — In-process Qdrant (no Docker)
Replace `"url"` with `"path"` in the vector store config. Ephemeral — data lost on restart. Good for local testing only.

---

## Core API

```python
# Add a conversation turn — Mem0 auto-extracts facts
m.add(messages=[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
      user_id="user-123")

# Search relevant memories for the current query
results = m.search(query="what languages does the user prefer?", user_id="user-123", limit=5)
# results: {"results": [{"memory": "User prefers Python", "score": 0.91, "id": "..."}]}

# List all memories for a user
all_mems = m.get_all(user_id="user-123")

# Update a specific memory
m.update(memory_id="...", data="User now prefers TypeScript")

# GDPR erasure — delete all memories for a user
m.delete_all(user_id="user-123")

# Memory history (audit trail)
m.history(memory_id="...")
```

---

## Graph Pattern (same topology as Lesson 13)

```
START → load_memories → chat → save_memories → END
```

The only change from Lesson 13: replace every `memory_store.similarity_search()` and manual `add_documents()` call with `mem0_search()` and `mem0_add()`.

---

## Mem0 vs Chroma — Code Comparison

**Lesson 13 (Chroma) — manual:**
```python
extract_prompt = f"Extract facts... User said: {last_human} Return JSON list..."
raw = llm.invoke([HumanMessage(content=extract_prompt)]).content
facts = json.loads(raw[start:end+1])
for fact in facts:
    memory_store.add_documents([Document(page_content=fact, metadata={"user_id": user_id})])
```

**Lesson 25 (Mem0) — automatic:**
```python
mem0_add(messages=[{"role": "user", "content": last_human}], user_id=user_id)
# Mem0 handles extraction, dedup, contradiction resolution internally
```

---

## Interview Q&A

**Q1: What problem does Mem0 solve that a plain vector store does not?**
A: Mem0 handles the full memory lifecycle: extraction (what facts are worth storing), deduplication (don't store the same fact twice), and contradiction resolution (update "User prefers Python" when user later says "I switched to Go"). A plain Chroma store requires you to write and maintain all of this logic yourself.

**Q2: How does Mem0 resolve contradictions?**
A: On each `m.add()` call Mem0 runs an internal LLM pass that compares new extracted facts against existing memories. If a new fact contradicts an existing one (same subject, different value), Mem0 issues an `UPDATE` operation on the stored memory instead of an `ADD`. The previous value is retained in `m.history()` for audit purposes.

**Q3: How is Mem0 memory scoped — what prevents user A from seeing user B's memories?**
A: Every operation (add, search, get_all, delete_all) requires a `user_id` parameter. Mem0 stores `user_id` as metadata on every memory vector in Qdrant and filters all queries by it. There is no API to retrieve memories without specifying a user_id.

**Q4: How would you implement GDPR right-to-erasure with Mem0?**
A: Call `m.delete_all(user_id=user_id)`. This deletes all Qdrant vectors tagged with that user_id. For the memory history/audit log, also call `m.delete_all()` on the history collection. Combine with S3 erasure (Lesson 22) and Oracle checkpoint deletion (Lesson 16) for a full cross-system GDPR purge.

**Q5: When would you NOT use Mem0 and stick with Chroma?**
A: When you need embedding-only retrieval with no LLM extraction overhead (e.g., document RAG where the "memories" are fixed static documents, not dynamic user facts). Mem0 runs an extra LLM inference call on every `add()` to extract facts — that cost is only worth it for dynamic user-preference memory, not for indexing a static knowledge base.

---

## Tasks

- **Task 25.1:** Run the demo with two different `user_id` values. Verify that memories from user A never appear in searches for user B.
- **Task 25.2:** Implement contradiction resolution test: Turn 1 "I love Python", Turn 5 "I switched to Rust". Call `mem0_list_all()` — confirm only one memory about language preference exists.
- **Task 25.3:** Add `agent_id` scoping. Mem0 supports `agent_id` alongside `user_id`. Modify `save_memories_node` to pass `agent_id="langgraph-agent"` and verify memories are namespaced per agent.
- **Task 25.4:** Combine with Lesson 5 (multi-agent): build a two-agent system where each specialist (data agent, DB agent) has its own `agent_id` memory namespace but shares the same `user_id`.
- **Task 25.5:** Implement a `GET /memories/{user_id}` FastAPI endpoint (extend Lesson 23's app) that returns all Mem0 memories for a user. Protect with JWT auth from Lesson 17.

---

# Lesson 26 — RAG with Apache Solr

> **File:** `lesson_26_solr_rag/lesson_26_solr_rag.py`

---

## Why Solr?

Lesson 12 used Chroma — a simple local vector store. Enterprises often already run Apache Solr clusters for product search, document management, or compliance archives. Lesson 26 teaches you to use Solr as the retrieval backend for your LangGraph RAG agent.

| Feature | Chroma (Lesson 12) | Solr (Lesson 26) |
|---|---|---|
| Scale | Single process | SolrCloud cluster |
| Keyword search (BM25) | No | Yes (full Lucene engine) |
| Vector search (kNN) | Yes | Yes (Dense Vector Field, Solr 9+) |
| Hybrid search | No | Yes (BM25 + kNN weighted merge) |
| Faceting & filtering | Basic metadata | Full Solr facets + pivots |
| Enterprise auth | None | Kerberos, PKI, JWT |
| Existing data | Re-ingest required | Leverage existing Solr index |

**Rule of thumb:** Use Chroma for quick local prototypes. Use Solr when the enterprise already has a Solr cluster, needs BM25 keyword precision, or requires hybrid retrieval at scale.

---

## Solr Architecture in a RAG System

```
User question
    ↓
[agent node] → solr_hybrid_search_tool(query)
                    ├── BM25 search  → content field (Lucene inverted index)
                    ├── kNN search   → vector field (HNSW graph, Solr 9+)
                    └── Weighted merge → re-ranked top-K results
    ↑                      ↓
    └────── Retrieved docs ─┘
    ↓
[grounded answer — cite Solr source number]
```

---

## Solr Setup

```bash
# Start Solr 9 with a pre-created collection
docker run -d -p 8983:8983 --name solr solr:9 solr-precreate langgraph_docs

# Access the Solr Admin UI
open http://localhost:8983/solr/#/langgraph_docs
```

**Schema fields needed** (added via Schema API or `setup_solr_schema()`):

| Field | Type | Purpose |
|---|---|---|
| `id` | string (unique key) | Document identifier |
| `content` | text_general | BM25 full-text indexed |
| `source` | string | Metadata / source tag |
| `vector` | knnDenseVector | Embedding for kNN search |

---

## Three Search Modes

### BM25 (Keyword)
```python
results = client.search("content:(MemorySaver SqliteSaver)", rows=3, fl="content,score")
```
Best for: exact terms, function names, structured queries.

### kNN (Vector / Semantic)
```python
vec_str = ",".join(str(v) for v in embeddings.embed_query(query))
results = client.search(f"{{!knn f=vector topK=3}}{vec_str}", rows=3)
```
Best for: meaning-based queries, paraphrase matching.

### Hybrid (BM25 + kNN — production default)
```python
bm25_results = solr_bm25_search(query, top_k=10)
knn_results  = solr_knn_search(query,  top_k=10)
# Normalize each to [0,1], then: score = 0.4*bm25 + 0.6*knn
# Sort by combined score → take top 5
```
Best for: most production queries — combines keyword precision with semantic recall.

---

## LangGraph Agent Topology

```
START → agent ──tool_calls?──► tools (ToolNode) ──► agent
                      │
                      └── no tool_calls ──► END
```

Three tools available to the agent:
- `solr_hybrid_search_tool(query)` — **primary** (BM25 + kNN)
- `solr_keyword_search(query)` — for exact-term queries
- `solr_semantic_search(query)` — for concept-only queries

---

## Self-Correcting RAG Pattern

Same as Lesson 12, applied to Solr:

```
retrieve → check_relevance → [score ≥ 0.5?] → generate → END
                    ↓ no
               refine_query → retrieve (retry, max 2)
```

If the first Solr query returns low-relevance results, the LLM rephrases the query to use more specific Solr-friendly keywords before retrying.

---

## LangChain SolrVectorStore

For teams migrating from Chroma, LangChain provides `SolrVectorStore` with the same interface:

```python
# Lesson 12 (Chroma):
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="...")

# Lesson 26 (Solr) — minimal change:
vectorstore = SolrVectorStore.from_documents(docs, embeddings,
                  solr_url="http://localhost:8983/solr/langgraph_docs")
```

After that, `vectorstore.similarity_search(query, k=3)` works identically.

---

## Interview Q&A

**Q1: What is the difference between BM25 and kNN search, and why does hybrid outperform either alone?**
A: BM25 is term-frequency/inverse-document-frequency scoring — it excels at exact keyword matches but misses semantic synonyms. kNN vector search finds documents whose embedding is closest to the query embedding — it excels at semantic similarity but can miss exact-match documents if they embed differently. Hybrid combines both: BM25 handles "what exact terms are present", kNN handles "what concepts are present", giving better precision and recall than either alone.

**Q2: How do you implement hybrid search when Solr doesn't natively combine BM25 + kNN scores?**
A: Run both queries independently, normalize each result set's scores to [0, 1] (min-max normalization), then merge by document ID computing `final_score = w_bm25 * bm25_norm + w_knn * knn_norm`. Sort by `final_score` descending and take the top-K. This is the Reciprocal Rank Fusion (RRF) alternative approach; RRF is also a valid pattern.

**Q3: What Solr version is required for vector (kNN) search and what field type is used?**
A: Solr 9.0+ with the `DenseVectorField` (also called `knnDenseVector`). The field must be declared with `dims` matching the embedding model's output dimension (e.g., 4096 for llama3). The KNN query syntax is `{!knn f=vector topK=N}<comma-separated-float-vector>`.

**Q4: How would you handle Solr authentication in an enterprise environment?**
A: Solr supports three auth mechanisms: (1) Basic auth — pass `auth=(user, password)` to pysolr; (2) Kerberos/SPNEGO — set up `requests-kerberos` and a Kerberos ticket cache; (3) JWT auth — configure Solr's JWTAuthPlugin with the same JWT issuer as your application auth (Lesson 17). Always use HTTPS (`solrs://`) in production.

**Q5: When would you choose Solr over a purpose-built vector database like Qdrant or Pinecone?**
A: When: (a) the enterprise already runs SolrCloud and IT won't approve a new infrastructure component; (b) you need BM25 keyword search alongside vector search (Qdrant is vector-only); (c) you need Solr's rich faceting, boosting, and query DSL features; (d) compliance requires data to remain in the existing enterprise search cluster. For pure vector-only workloads with no existing Solr, Qdrant or Chroma are simpler to set up.

---

## Tasks

- **Task 26.1:** Start Solr with Docker, call `setup_solr_schema()`, index `KNOWLEDGE_BASE`, and run all three search modes on 5 different queries. Compare BM25 vs kNN vs Hybrid top-1 result quality.
- **Task 26.2:** Tune hybrid weights. Run `solr_hybrid_search()` with `(bm25=0.8, knn=0.2)` vs `(bm25=0.2, knn=0.8)` on 10 queries. Which setting gives better top-1 relevance for your queries?
- **Task 26.3:** Implement a `delete_document(doc_id: str)` function and a `reindex_document(doc_id: str, new_text: str)` function for incremental knowledge base updates.
- **Task 26.4:** Extend the agent to support faceted results: after each retrieval, run a second Solr query to get source facets and include them in the agent's response as "Related topics: ...".
- **Task 26.5:** Compare Lesson 12 (Chroma) vs Lesson 26 (Solr) on the same 10 questions. Which returns better top-1 results? Record your findings in a markdown table.
