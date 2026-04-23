# Enterprise Lessons (16-20) — Setup Guide

## Quick Start

```bash
# 1. Copy environment template
cp .env.example .env
# Edit .env with your values

# 2. Install enterprise dependencies
pip install -r requirements.txt

# 3. Run lessons in order
python lesson_16_postgres_async/lesson_16_oracle_async.py
python lesson_17_auth_rbac/lesson_17_auth_rbac.py
python lesson_18_observability/lesson_18_observability.py
python lesson_19_event_driven/lesson_19_event_driven.py
python lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py
```

---

## Lesson 16 — Oracle 19c Persistence & Async

**File:** `lesson_16_postgres_async/lesson_16_oracle_async.py`
**Reference (PostgreSQL):** `lesson_16_postgres_async/lesson_16_postgres_async.py`

### Oracle Setup (run once as DBA)
```sql
CREATE TABLE langgraph_checkpoints (
    thread_id   VARCHAR2(255)  NOT NULL,
    checkpoint  CLOB           NOT NULL,
    metadata    CLOB,
    ts          TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT pk_lg_chk PRIMARY KEY (thread_id)
);
GRANT SELECT, INSERT, UPDATE, DELETE ON langgraph_checkpoints TO langgraph;
```

### Run with Oracle
```bash
export USE_ORACLE=true
export ORACLE_USER=langgraph
export ORACLE_PASSWORD=...
export ORACLE_HOST=your-host
export ORACLE_PORT=1521
export ORACLE_SERVICE=ORCLPDB1
python lesson_16_postgres_async/lesson_16_oracle_async.py
```

**Without Oracle:** runs automatically with `MemorySaver` fallback (logged as WARNING).

---

## Lesson 17 — JWT Auth, RBAC & Rate Limiting

**File:** `lesson_17_auth_rbac/lesson_17_auth_rbac.py`

### Dependencies
```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

### Graph flow
```
START → auth (RBAC check) → rate_limit (sliding window) → chat → END
              ↓ denied              ↓ denied
             END                   END
```

### Key env vars
| Variable | Default | Purpose |
|----------|---------|---------|
| `JWT_SECRET` | `change-me-...` | HMAC signing key — **must change in production** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_MINUTES` | `10080` | Refresh token lifetime (7 days) |

---

## Lesson 18 — Observability

**File:** `lesson_18_observability/lesson_18_observability.py`

### Dependencies
```bash
pip install prometheus-client opentelemetry-sdk opentelemetry-exporter-otlp
```

### JSON structured logging
```bash
export USE_JSON_LOGS=true
python lesson_18_observability/lesson_18_observability.py
# Each log line is now a JSON object parseable by ELK/Datadog
```

### Prometheus metrics
Set `USE_JSON_LOGS=false` and run the demo — Prometheus serves metrics at `http://localhost:9090/metrics`.

### SLA configuration
```bash
export SLA_P99_LATENCY_SECONDS=3.0   # fail SLA if any request > 3s
```

---

## Lesson 19 — Event-Driven Agents (Celery + Redis)

**File:** `lesson_19_event_driven/lesson_19_event_driven.py`

### Without Redis (default — sync fallback)
```bash
python lesson_19_event_driven/lesson_19_event_driven.py
# Runs tasks synchronously — shows all patterns without Redis
```

### With Redis + Celery
```bash
# Terminal 1: Redis
docker run -p 6379:6379 redis

# Terminal 2: Celery worker
celery -A lesson_19_event_driven.lesson_19_event_driven worker --loglevel=info --concurrency=4

# Terminal 3: Run demo (tasks sent to Redis)
export USE_CELERY=true
python lesson_19_event_driven/lesson_19_event_driven.py
```

### Webhook HMAC secret
```bash
export WEBHOOK_SECRET=your-strong-secret-here
# GitHub: set matching secret in repo Settings → Webhooks
# Slack: set matching secret in app manifest
```

### FastAPI polling endpoints
```python
# HTTP 202 on submit:
POST /tasks/pr-review  →  {"task_id": "...", "poll_url": "/tasks/{id}/status"}

# Poll for result:
GET /tasks/{task_id}/status  →  {"status": "SUCCESS|PENDING|FAILURE", "result": ...}

# Monitor DLQ:
GET  /tasks/dlq/depth   →  {"depth": 0}
POST /tasks/dlq/drain   →  {"drained": N, "entries": [...]}  (admin only)
```

---

## Lesson 20 — Enterprise Capstone

**File:** `lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py`

### What it combines
All enterprise patterns in one graph:
```
START
  → complexity_router  (fast / balanced / powerful model selection)
  → budget_check       (per-tenant daily token budget)
  → [budget exhausted? END]
  → circuit_check      (fault tolerance — CLOSED / OPEN / HALF_OPEN)
  → [circuit OPEN? END]
  → chat               (LLM call + cost tracking + GDPR compliance record)
  → END
```

### 6-Layer Architecture
| Layer | Technology |
|-------|-----------|
| Gateway | nginx/ALB + FastAPI + JWT + rate limiting |
| Orchestration | LangGraph StateGraph |
| Execution | Model routing + Celery async tasks |
| Persistence | **Oracle 19c** (OracleSaver) + Redis |
| Observability | Prometheus + Grafana + OpenTelemetry + LangSmith |
| Governance | Token budget + Circuit breaker + GDPR + Audit trail |

---

## Dependency Summary

| Package | Lesson | Purpose |
|---------|--------|---------|
| `oracledb>=2.0.0` | 16 | Oracle 19c connection pool |
| `python-jose[cryptography]` | 17 | Real JWT encode/decode |
| `passlib[bcrypt]` | 17 | Password hashing |
| `prometheus-client` | 18 | Metrics (Counter, Histogram, Gauge) |
| `opentelemetry-sdk` | 18 | Distributed tracing |
| `celery` | 19 | Async task queue |
| `redis` | 19, 20 | Broker + result backend + cache |

Install all at once:
```bash
pip install oracledb "python-jose[cryptography]" "passlib[bcrypt]" \
            prometheus-client opentelemetry-sdk celery redis
```

---

## Study Materials

| File | Content |
|------|---------|
| [`BOOK/BOOK_Part5_Enterprise.md`](BOOK/BOOK_Part5_Enterprise.md) | Lesson overviews, tasks, Q&A for L16–20 |
| [`BOOK/Lessons16_20_Enterprise_Deep_Dive.md`](BOOK/Lessons16_20_Enterprise_Deep_Dive.md) | Deep internals, anti-patterns, 35 interview Q&A, 2 systems design questions |

---

## Enterprise Checklist Before Production

- [ ] `JWT_SECRET` is ≥ 32 random characters (not the default)
- [ ] `WEBHOOK_SECRET` is ≥ 32 random characters
- [ ] `APP_ENV=production` and `USE_ORACLE=true`
- [ ] Oracle `langgraph_checkpoints` table created + grants applied
- [ ] TDE encryption enabled on CLOB column
- [ ] Oracle Data Guard or RAC configured
- [ ] Prometheus + Grafana dashboards deployed
- [ ] PagerDuty/Slack alerts configured for SLA breaches + DLQ depth > 0
- [ ] GDPR erasure + export endpoints tested
- [ ] Rate limiting per tenant verified
- [ ] Circuit breaker thresholds tuned for your LLM provider SLA
