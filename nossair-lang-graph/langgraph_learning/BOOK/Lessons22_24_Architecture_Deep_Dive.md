# LangGraph Architecture Deep Dive — Lessons 22–24

> **Who this is for:** Engineers who completed Lessons 1–21 and are building the
> full production system from the High Level Architecture diagram.
> Each section covers: internals, failure modes, anti-patterns, and 7 interview Q&A.

---

# Lesson 22 — AWS S3 Deep Dive

> **File:** `lesson_22_aws_s3/lesson_22_aws_s3.py`

---

## How S3 Fits Into LangGraph State

S3 is not a LangGraph concept — it is an external side-effect called from inside nodes.
The pattern is always the same:

```
node_function(state) → calls s3 helper → returns updated state dict
```

LangGraph does NOT know about S3. The state carries S3 keys (strings), not S3 objects.
This is critical: never put boto3 objects or file handles in the state — they cannot be serialized by the checkpointer.

```python
class S3AgentState(TypedDict):
    uploaded_doc_key: Optional[str]     # ✅ S3 key (serializable string)
    presigned_url: Optional[str]        # ✅ URL string
    # uploaded_doc: bytes               # ❌ NEVER — not serializable
```

---

## Credential Chain — How boto3 Finds Credentials

boto3 tries credentials in this exact order, stopping at the first success:

```
1. Explicit constructor args (never use in production)
2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
3. AWS config file (~/.aws/credentials, profile name)
4. AWS SSO / identity center
5. EC2 instance metadata (169.254.169.254) ← PRODUCTION: this is what fires
6. ECS container metadata
7. Lambda function environment
```

On EC2 with an IAM role: step 5 fires automatically. boto3 fetches temporary
credentials from instance metadata, caches them, and rotates them every ~6 hours.
**Zero code change needed between local dev (step 3) and EC2 prod (step 5).**

---

## S3 Consistency Model

Since 2020, S3 provides **strong read-after-write consistency** for all operations:

| Operation | Guarantee |
|-----------|-----------|
| `PUT` new object | Immediately readable after |
| `PUT` overwrite | Immediately consistent |
| `DELETE` | Immediately consistent |
| `LIST` | Immediately consistent |

This means: after `s3_upload_text(key, content)` returns, a subsequent `s3_download_text(key)` is guaranteed to return the new content. No eventual consistency delays to worry about.

---

## Presigned URL Internals

A presigned URL encodes the following in its query parameters:
- **X-Amz-Algorithm**: signing algorithm (AWS4-HMAC-SHA256)
- **X-Amz-Credential**: IAM role ARN + date + region
- **X-Amz-Date**: timestamp of URL creation
- **X-Amz-Expires**: TTL in seconds
- **X-Amz-Signature**: HMAC-SHA256 of the canonical request

When the browser makes the request, S3 recomputes the expected signature and compares it.
If the TTL has expired or the signature doesn't match → 403.

Key insight: **the IAM role that generated the URL is verified at access time, not at generation time.** If you revoke the IAM role after generating a URL but before it expires, the URL still works until expiry.

---

## S3 Key Design: Tenant Isolation

```
conversations/
  {tenant_id}/
    {thread_id}/
      20240101_120000.json   ← snapshot 1
      20240101_130000.json   ← snapshot 2

documents/
  {tenant_id}/
    {doc_id}/
      report.csv
      analysis.pdf
```

**Why not one file per thread?**
Appending to S3 objects is not atomic. Instead, each snapshot is a separate immutable object.
The latest snapshot is `sorted(keys)[-1]` — lexicographic sort on timestamps works because the format is `YYYYMMDD_HHMMSS`.

**Isolation enforcement:**
```python
# In the API layer — ALWAYS verify tenant before generating presigned URL
if user["tenant_id"] != requested_tenant_id:
    raise HTTPException(403, "Tenant mismatch")
url = get_document_url(requested_tenant_id, doc_id, filename)
```

---

## Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|-----------|
| Credentials expired | `NoCredentialsError` at startup | IAM role auto-rotates — only happens if metadata endpoint unreachable |
| Bucket does not exist | `NoSuchBucket` on first write | Check `S3_BUCKET_NAME` env var; create bucket in Terraform/CDK |
| Large file OOM | Process killed on upload | Use `s3_upload_file_stream()` (BytesIO) not `s3_upload_text()` for binary files |
| Partial upload | Object exists but truncated | Use multipart upload for files > 100MB (boto3 `TransferConfig`) |
| Rate limiting | `SlowDown` error | boto3 auto-retries with exponential backoff (default 3 retries) |
| Wrong region | `PermanentRedirect` error | Set `AWS_REGION` or hardcode region in `boto3.Session(region_name=...)` |
| Presigned URL expired | HTTP 403 from browser | Use shorter TTL (900s), regenerate URL on demand |

---

## Anti-Patterns

❌ **Storing large files as base64 in LangGraph state**
Bloats checkpointer storage, OracleSaver CLOB has limits, JSON serialization is slow.
✅ Store the S3 key in state, load content inside the node that needs it.

❌ **One S3 bucket per tenant**
Bucket names are globally unique — naming conflicts, billing complexity, IAM explosion.
✅ One bucket, prefix-based isolation, bucket policy scoped to IAM role.

❌ **Using `s3.upload_file()` with a local file path in a containerized app**
Assumes a persistent filesystem. Containers and Lambda have ephemeral storage.
✅ Use `s3_upload_file_stream(key, bytes_in_memory)` — no temp file needed.

❌ **Not setting `ContentType` on uploaded objects**
S3 serves everything as `application/octet-stream` — browsers won't render inline.
✅ Set `ContentType` based on file extension (see `content_type_map` in `upload_document()`).

❌ **GDPR erasure that only deletes from the database**
S3 objects survive and contain conversation PII.
✅ `erase_user_data()` deletes both DB records (Lesson 20 GDPR) AND S3 objects.

---

## 7 Interview Q&A

**Q1: How does boto3 authenticate to S3 on EC2 without any credentials in the code?**
boto3 checks the instance metadata service (IMDS) at `http://169.254.169.254/latest/meta-data/iam/security-credentials/{role-name}`. This returns temporary `AccessKeyId`, `SecretAccessKey`, and `Token` that expire in ~6 hours. The EC2 hypervisor injects these credentials — no code change needed. In local dev, boto3 falls back to `~/.aws/credentials`.

**Q2: What is the difference between `s3.put_object()` and `s3.upload_fileobj()`?**
`put_object()` takes a `Body` parameter (bytes or string) and uploads the entire content in a single HTTP PUT. `upload_fileobj()` accepts a file-like object and uses multipart upload automatically for files > 8MB (configurable via `TransferConfig`). For large files, `upload_fileobj()` is more memory-efficient and retries individual parts on failure.

**Q3: How do you make S3 automatically delete old conversation snapshots after 90 days?**
Apply an S3 Lifecycle Rule to the bucket:
```json
{
  "Rules": [{
    "Filter": {"Prefix": "conversations/"},
    "Status": "Enabled",
    "Expiration": {"Days": 90}
  }]
}
```
S3 deletes matching objects automatically. This is cheaper and more reliable than a cron job. For GDPR compliance, use 30 days instead of 90.

**Q4: Why is S3 eventual consistency no longer a concern?**
Since December 2020, AWS upgraded S3 to strong read-after-write consistency for all operations in all regions. Before that, `LIST` operations and overwrite `GET` could return stale data for several seconds. This change means the pattern `upload → list → download latest` is now safe without any delays or retries.

**Q5: How do you prevent a tenant from guessing another tenant's presigned URL?**
The URL contains an HMAC-SHA256 signature over the full object key, which includes the tenant prefix. To access `documents/tenant_b/...`, an attacker would need tenant_b's IAM credentials to generate a valid signature. The signature is not derivable from tenant_a's URL. Additionally, bucket policies can enforce that presigned URLs only work for objects under the tenant's own prefix.

**Q6: What happens if `s3_client` is `None` (simulation mode) in production?**
All helper functions check `if s3_client is None` and return simulated responses — uploads succeed (logged as `[SIMULATE]`), downloads return a placeholder JSON string, lists return fake keys. This is intentional for local dev but **must never reach production**. Add a startup assertion: `assert s3_client is not None, "S3 client required in production"` in `production_startup()` (Lesson 24).

**Q7: How do you implement S3 server-side encryption for conversation data (GDPR requirement)?**
Enable SSE-S3 (AES-256, AWS-managed keys) or SSE-KMS (customer-managed KMS key):
```python
s3_client.put_object(
    Bucket=bucket, Key=key, Body=content,
    ServerSideEncryption="aws:kms",
    SSEKMSKeyId="arn:aws:kms:us-east-1:123456789:key/your-key-id",
)
```
For GDPR: KMS allows key rotation and deletion. Deleting the KMS key renders all encrypted data permanently unreadable — an alternative to object-by-object deletion for full tenant erasure.

---

# Lesson 23 — Conversation Management API Deep Dive

> **File:** `lesson_23_conversation_api/lesson_23_conversation_api.py`

---

## The Session Layer: Why It Exists

LangGraph's `thread_id` is an internal persistence key. It has no concept of:
- Who the user is
- Which tenant they belong to
- How many messages they have sent
- Which specialist agent should handle them
- Whether they have exceeded their usage quota

The **Session Layer** wraps `thread_id` with all of this business logic:

```
Frontend → session_id (opaque UUID) → API session store → thread_id → LangGraph
```

The frontend never sees or handles `thread_id` directly. It only uses `session_id`.

---

## Session Lifecycle

```
POST /sessions
  → creates session dict (session_id, thread_id, tenant_id, agent_type)
  → stores in session store (dict / Redis)
  → returns session_id + JWT to frontend

POST /chat (n times)
  → verify JWT
  → load session by session_id
  → check_usage_limits()
  → invoke LangGraph graph with thread_id
  → update message_count, last_active
  → return response

(optional) DELETE /sessions/{id}
  → mark session inactive
  → trigger GDPR erasure if requested
```

---

## Routing Architecture: Orchestrator vs ReAct

Two fundamentally different routing approaches:

| | ReAct (Lesson 4) | Orchestrator (Lesson 23) |
|-|-----------------|------------------------|
| Who decides? | LLM inside the agent | API layer (Python code) |
| When decided? | At each LLM turn | At session creation + keyword check |
| Cost | LLM call per routing decision | Zero LLM cost for routing |
| Flexibility | High (LLM can reason) | Lower (rule-based) |
| Predictability | Low (LLM may choose wrong tool) | High (deterministic) |
| Best for | Single agent with many tools | Multi-specialist routing |

The Conversation API uses **orchestrator routing** because:
1. Routing decisions are cheap (no LLM call needed)
2. Specialists are completely separate agents, not tools
3. Session context (agent_type) gives strong routing signal upfront

---

## Rate Limiting: Sliding Window Algorithm

```python
now = time.time()
window_start = now - 60.0                           # 1-minute window
user_requests = [t for t in history if t > window_start]  # keep only recent
if len(user_requests) >= RATE_LIMIT_RPM:
    raise HTTPException(429, "Rate limit exceeded")
user_requests.append(now)
_rate_limit_store[user_id] = user_requests
```

**Sliding window** vs **fixed window**:
- Fixed window: 30 req/min resets at :00, :01, :02... A user can send 30 at :59 and 30 at :00 = 60 in 2 seconds.
- Sliding window: always looks at the last 60 seconds. Max 30 per any 60s period.

**Production upgrade**: replace the in-memory dict with Redis sorted sets:
```python
redis.zadd(f"ratelimit:{user_id}", {str(now): now})
redis.zremrangebyscore(f"ratelimit:{user_id}", 0, now - 60)
count = redis.zcard(f"ratelimit:{user_id}")
```
Redis sorted sets give O(log N) sliding window that works across multiple EC2 instances.

---

## SSE Streaming Internals

HTTP SSE (Server-Sent Events) is a standard protocol for server→client streaming:

```
Client: GET /chat/stream
        Accept: text/event-stream

Server: HTTP/1.1 200 OK
        Content-Type: text/event-stream
        Cache-Control: no-cache
        Connection: keep-alive

        data: {"token": "The "}\n\n
        data: {"token": "answer "}\n\n
        data: {"token": "is..."}\n\n
        data: [DONE]\n\n
```

Key rules:
- Each event ends with **two newlines** (`\n\n`)
- `Content-Type` must be `text/event-stream`
- Nginx must have `proxy_buffering off` for `/chat/stream` (see Lesson 24)
- The `[DONE]` sentinel tells the client the stream is finished

**Why SSE over WebSockets?**
SSE is unidirectional (server→client) over plain HTTP/1.1. WebSockets are bidirectional over HTTP/1.1 upgrade. For chat streaming (client sends one message, server streams response), SSE is simpler: no WebSocket library needed, works through all HTTP proxies, and reconnects automatically if the connection drops.

---

## Thread Safety: In-Memory Session Store Limitation

```python
_session_store: dict[str, dict] = {}  # ← NOT thread-safe with multiple workers
```

With uvicorn `--workers 4`, each worker process has its own memory. A session created in worker 1 is invisible to worker 3. This causes:
- `Session not found` errors (session in worker 1, request routed to worker 2)
- Duplicate sessions if two requests hit different workers simultaneously

**Fix: Redis**
```python
import redis
r = redis.Redis(host=REDIS_HOST, decode_responses=True)

def get_session(session_id: str):
    raw = r.get(f"session:{session_id}")
    return json.loads(raw) if raw else None

def create_session(...):
    session = {...}
    r.setex(f"session:{session_id}", 86400, json.dumps(session))  # 24h TTL
    return session
```

This is Task 23.2. Redis sessions work across all workers and EC2 instances.

---

## Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|-----------|
| Session not found (multi-worker) | HTTP 404 on valid session_id | Migrate to Redis session store (Task 23.2) |
| JWT expired mid-session | HTTP 401 after 1 hour | Client should refresh JWT before expiry (Lesson 17 refresh flow) |
| LLM timeout | Request hangs | Set `proxy_read_timeout 120s` in Nginx; add LLM timeout in `get_llm()` |
| Rate limit store grows unbounded | Memory leak | Prune `_rate_limit_store` entries older than 60s; or use Redis with TTL |
| SSE connection dropped mid-stream | Client gets partial response | Client-side EventSource reconnects automatically; add `Last-Event-ID` for resumability |
| Agent routing wrong specialist | User frustrated | Log `agent_used` per session; adjust keyword list; allow user to switch agent_type |

---

## Anti-Patterns

❌ **Exposing `thread_id` to the frontend**
If the frontend sends `thread_id` directly, any user can read another user's conversation by guessing or brute-forcing thread IDs.
✅ Always use `session_id` (opaque UUID) as the frontend handle. Map to `thread_id` server-side.

❌ **Running usage limit checks after invoking the LLM**
The LLM call costs money — checking limits after means you pay even for rejected requests.
✅ `check_usage_limits()` is called before `ORCHESTRATOR.invoke()`.

❌ **Synchronous SSE generator blocking the event loop**
```python
# BAD — blocks uvicorn's async event loop
for chunk in llm.stream(messages):   # synchronous generator
    yield f"data: {chunk.content}\n\n"
```
✅ For true async streaming, use `llm.astream()`:
```python
async def token_generator():
    async for chunk in llm.astream(messages):
        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
```

❌ **Returning the full conversation history in every `/chat` response**
Sends large payloads on every message — bandwidth waste, slow frontend.
✅ Return only the latest response. Frontend calls `/history` separately when needed.

---

## 7 Interview Q&A

**Q1: Why does the session layer exist when LangGraph already manages state with thread_id?**
`thread_id` is a persistence key with no business logic. The session layer adds: (1) opaque external ID decoupled from internal thread_id, (2) tenant isolation enforced at API level, (3) usage limit enforcement before LLM invocation, (4) agent routing based on session context, (5) JWT auth binding user identity to session. Without the session layer, any authenticated user could read any thread by guessing its ID.

**Q2: How does the orchestrator route a message to the correct agent without an LLM call?**
`route_to_agent()` uses two signals: (1) `session.agent_type` set at session creation (explicit routing — the frontend chose "data analysis" when creating the session), and (2) keyword matching on the last message content for `agent_type="general"` sessions. Both are O(1) Python operations — no LLM invocation, no latency, no cost.

**Q3: What is the difference between a session and a thread?**
A `thread_id` is a LangGraph internal concept: it identifies a checkpoint sequence in the checkpointer. A `session` is a business object: it wraps `thread_id` with user identity, tenant, agent type, message count, timestamps, and status. One session maps to exactly one thread. Threads outlive sessions — you can have an "inactive" session whose thread still has history in the checkpointer.

**Q4: How do you scale the Conversation API to handle 10,000 concurrent sessions?**
(1) Replace in-memory session store with Redis Cluster — O(1) lookups, shared across all instances. (2) Replace in-memory rate limiter with Redis sorted sets. (3) Run multiple EC2 instances behind an ALB — sessions are now stateless from the EC2 perspective. (4) Use async LangGraph (`ainvoke`) to avoid blocking uvicorn workers on LLM calls. (5) Set uvicorn `--workers` to `2 * CPU_cores + 1`.

**Q5: How does SSE differ from WebSockets for LLM token streaming?**
SSE is unidirectional (server→client), uses plain HTTP/1.1 GET, and reconnects automatically. WebSockets are bidirectional, require an HTTP upgrade handshake, and need explicit reconnect logic. For LLM streaming where the client sends one message and the server streams the response, SSE is sufficient and simpler. WebSockets are preferred when the client needs to interrupt mid-stream or send multiple messages while a response is streaming.

**Q6: How do you resume an SSE stream if the connection drops mid-response?**
Use the `id:` field in SSE events and the `Last-Event-ID` request header. The server assigns a monotonically increasing ID to each token chunk. On reconnect, the client sends `Last-Event-ID: 42`. The server looks up the stored response and replays from chunk 43 onward. In practice, most implementations restart the LLM call from the beginning on reconnect — resumability requires caching the full streaming output (e.g., in Redis).

**Q7: What happens to in-flight SSE streams when the EC2 instance is replaced during a deployment?**
The ALB connection draining period (default 300s) keeps the old instance in the pool until all existing connections close. SSE connections are long-lived (the LLM may stream for 30-60s). Clients connected to the old instance during deployment experience the full stream, then reconnect to the new instance for their next message. To reduce disruption: (1) Set ALB deregistration delay to 120s. (2) Ensure new instances are fully healthy before deregistering old ones. (3) The client-side `EventSource` reconnects transparently.

---

# Lesson 24 — EC2 Production Deployment Deep Dive

> **File:** `lesson_24_ec2_deployment/lesson_24_ec2_deployment.py`

---

## EC2 vs Container (ECS/EKS): When to Choose EC2

| Aspect | EC2 (bare VM) | ECS/EKS (containers) |
|--------|--------------|---------------------|
| Control | Full OS access | Container boundary |
| Startup time | 1-3 min (AMI boot) | 5-30s (image pull) |
| State | Persistent disk | Ephemeral (stateless preferred) |
| IAM | Instance profile | Task role |
| Best for | Long-running, stateful, Oracle clients | Microservices, burst scaling |
| This architecture | ✅ Good fit (Oracle drivers, persistent connections) | Also viable |

The High Level Architecture shows a single EC2 instance per environment. This is appropriate for:
- Oracle 19c client connections (persistent, pool-based)
- LangGraph MemorySaver (in-process, single instance)
- Simpler ops (no container registry, no Kubernetes YAML)

---

## IAM Instance Profile: How It Works Under the Hood

```
EC2 Instance
  └── Hypervisor injects IAM role credentials at:
        http://169.254.169.254/latest/meta-data/iam/security-credentials/{role-name}
              ↑ Link-local address — only accessible from inside the EC2 instance

Response:
{
  "Code": "Success",
  "Type": "AWS-HMAC",
  "AccessKeyId": "ASIA...",
  "SecretAccessKey": "...",
  "Token": "...",
  "Expiration": "2024-01-01T13:00:00Z"   ← rotated every ~6 hours
}
```

boto3's `InstanceMetadataProvider` fetches these credentials, caches them, and refreshes before expiry. **This is transparent to application code** — `boto3.Session()` works identically in local dev (reads `~/.aws/credentials`) and on EC2 (reads IMDS).

IMDSv2 (recommended): requires a token-based handshake to prevent SSRF attacks:
```python
# boto3 uses IMDSv2 automatically since botocore 1.25.0
```

---

## SSM Parameter Store: Encryption and Access Control

```
/p5/prod/JWT_SECRET_KEY  (SecureString — encrypted with KMS key)
/p5/prod/S3_BUCKET_NAME  (String — not sensitive)
/p5/dev/JWT_SECRET_KEY   (SecureString — different value for dev)
```

**Access control with IAM:**
```json
{
  "Effect": "Allow",
  "Action": ["ssm:GetParameters", "ssm:GetParameter"],
  "Resource": "arn:aws:ssm:us-east-1:123456789:parameter/p5/prod/*"
}
```

The EC2 role can only read `/p5/prod/*` — not `/p5/dev/*` and not other apps.

**Audit trail:**
Every `GetParameters` call appears in CloudTrail:
```json
{"eventName": "GetParameters", "userIdentity": {"arn": "arn:aws:iam::123::role/ChatbotEC2Role"},
 "requestParameters": {"names": ["/p5/prod/JWT_SECRET_KEY"]}}
```

**Vs Secrets Manager:**
- SSM Parameter Store SecureString: free, pull-based (code fetches on startup)
- Secrets Manager: $0.40/secret/month, supports automatic rotation, push-based webhook

Use SSM for config, Secrets Manager for database passwords that need automatic rotation.

---

## Nginx: Request Flow and Buffering

```
Client → Nginx (port 80) → uvicorn (port 8000 on 127.0.0.1)
```

**Why proxy_read_timeout 120s?**
Default Nginx timeout is 60s. Claude Sonnet can take 45-90s for long responses. Without increasing the timeout, Nginx drops the connection at 60s and returns 504 Gateway Timeout to the client, even though uvicorn is still waiting for the LLM.

**Why proxy_buffering off for /chat/stream?**
With buffering on (default), Nginx collects the full upstream response before sending it to the client. This defeats SSE — the client would receive all tokens at once after the LLM finishes. `proxy_buffering off` sends each `data: {...}\n\n` chunk immediately as it arrives from uvicorn.

**Nginx reload vs restart:**
```bash
sudo nginx -s reload    # graceful: finishes in-flight requests, then loads new config
sudo nginx -s restart   # abrupt: kills all connections immediately
```
Always use `reload` for zero-downtime config changes.

---

## Systemd: Process Supervision Internals

```
systemd (PID 1)
  └── p5-chatbot.service
        └── uvicorn (master process)
              ├── worker 1 (handles requests)
              ├── worker 2
              ├── worker 3
              └── worker 4
```

**What `Restart=always` does:**
systemd watches the main process PID. If it exits (for any reason — crash, OOM, unhandled exception), systemd waits `RestartSec=5s` then starts a new process. This means:
- Unhandled Python exception → uvicorn exits → systemd restarts in 5s
- OOM killer terminates uvicorn → systemd restarts in 5s
- EC2 reboot → systemd starts on boot (`WantedBy=multi-user.target`)

**Graceful shutdown (`KillSignal=SIGTERM`):**
When `systemctl stop` or `systemctl reload` is called:
1. systemd sends `SIGTERM` to uvicorn master
2. uvicorn stops accepting new requests
3. Workers finish in-flight requests (up to `TimeoutStopSec=30s`)
4. Workers exit cleanly
5. If any worker is still alive after 30s → `SIGKILL`

---

## CloudWatch Logs: Log Insights Queries

Every log line is a JSON object. CloudWatch Logs Insights can query fields directly:

```sql
-- Find all errors in the last hour
fields @timestamp, level, tenant_id, message
| filter level = "ERROR"
| sort @timestamp desc
| limit 100

-- Count requests per tenant
fields tenant_id
| filter ispresent(tenant_id)
| stats count() as request_count by tenant_id
| sort request_count desc

-- Find slow LLM responses (> 10 seconds)
fields @timestamp, tenant_id, message, duration_ms
| filter duration_ms > 10000
| sort duration_ms desc

-- Error rate over time
fields @timestamp, level
| stats count(level="ERROR") as errors, count() as total by bin(5m)
| sort @timestamp asc
```

**Log group naming:** `/p5/{env}` — one group per environment. Use log streams per instance per day: `{hostname}-{YYYYMMDD}`.

---

## Zero-Downtime Deployment: Deep Dive

### Single EC2 — `systemctl reload`

```
1. git pull / aws s3 cp new code
2. pip install -r requirements.txt
3. sudo systemctl reload p5-chatbot
   ↓
   systemd → SIGHUP → uvicorn master
                          ↓
   uvicorn master starts new worker (with new code)
   uvicorn master waits for old workers to finish in-flight requests
   uvicorn master kills old workers
   (no downtime — at least one worker always accepting requests)
```

### Multi-EC2 + ALB — Blue/Green

```
Existing: EC2-A (old code) registered in ALB target group
Deploy:
  1. Launch EC2-B (new AMI)
  2. EC2-B passes /health/ready → register in ALB
  3. ALB routes new requests to BOTH EC2-A and EC2-B
  4. Deregister EC2-A from ALB
  5. ALB connection drain: 300s for in-flight requests on EC2-A to complete
  6. Terminate EC2-A
```

**Connection draining** is critical for SSE streams — a client streaming a 60s LLM response must not be cut off mid-stream during deployment.

---

## Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|-----------|
| systemd not starting on boot | Server silent after reboot | `systemctl enable p5-chatbot` sets `WantedBy=multi-user.target` |
| OOM kill | uvicorn silently restarts | Set `--workers` based on available RAM; add CloudWatch memory alarm |
| SSM unreachable at startup | Config loads from env vars (degraded) | `production_startup()` warns but does not fail — acceptable for non-secret config |
| ALB health check fails | Instance removed from rotation | `/health` must return 200 within 5s; do not add slow DB queries to liveness check |
| Nginx config syntax error | 502 Bad Gateway after reload | Always run `sudo nginx -t` before reload; automate in deploy script |
| Workers exhaust file descriptors | `Too many open files` error | Set `LimitNOFILE=65536` in systemd unit; check `ulimit -n` |
| CloudWatch PUT throttled | Logs delayed or dropped | watchtower queues logs locally and retries; data is not lost |

---

## Anti-Patterns

❌ **Putting AWS credentials in `/etc/p5/.env`**
The env file is readable by all users with `cat`. In an AMI, it is baked into disk snapshots.
✅ Secrets in SSM Parameter Store (KMS-encrypted), fetched at runtime.

❌ **Running uvicorn directly on port 80 without Nginx**
Requires root (port < 1024), no SSL, no rate limiting, no buffering.
✅ Nginx on port 80/443 → uvicorn on localhost:8000.

❌ **Liveness endpoint that checks database connectivity**
If Oracle is briefly unavailable (failover, maintenance), the liveness check fails → systemd kills and restarts uvicorn repeatedly → thrashing with no benefit.
✅ Liveness = is the process alive (always fast). Readiness = are deps reachable. Use separate endpoints.

❌ **`sudo systemctl restart` for code deployments**
Restart sends SIGKILL immediately → all in-flight requests dropped.
✅ `sudo systemctl reload` sends SIGTERM → graceful drain → zero-downtime.

❌ **Hardcoding `--workers 4` without considering RAM**
Each uvicorn worker loads the full Python app into memory (~200-500MB for LangGraph apps).
4 workers × 400MB = 1.6GB. On a `t3.medium` (4GB RAM), this leaves 2.4GB for Oracle connection pool, S3 buffers, and OS.
✅ Start with `2 * nproc + 1` workers, monitor memory in CloudWatch, tune based on actual usage.

---

## 7 Interview Q&A

**Q1: Why should you never use hardcoded AWS access keys in production EC2 code?**
Three reasons: (1) Keys in code end up in git history permanently — even after deletion. (2) If the EC2 instance is compromised, the attacker gets permanent keys that must be manually rotated. (3) IAM instance profiles use temporary credentials that rotate every 6 hours automatically — if the credentials leak, they expire quickly. Use IAM roles; the credential chain makes the code identical for both.

**Q2: What is the difference between SSM Parameter Store SecureString and AWS Secrets Manager?**
Both store encrypted secrets using KMS. Parameter Store SecureString costs nothing (free tier), is pull-based (code fetches on startup), and requires manual rotation. Secrets Manager costs $0.40/secret/month but supports automatic rotation (Lambda-triggered), cross-account access, and a dedicated SDK. For static secrets (JWT key, bucket name), use Parameter Store. For database passwords that need rotation, use Secrets Manager.

**Q3: How does Nginx `proxy_buffering off` work for SSE and why is it required?**
Normally, Nginx buffers the entire upstream response in memory before forwarding to the client (for performance — it can then serve from its buffer even if the upstream closes). With SSE, each token chunk must reach the client immediately. `proxy_buffering off` disables this buffer: every byte written by uvicorn is forwarded to the client immediately. Without this setting, the client receives all tokens at once after the LLM finishes — defeating the purpose of streaming.

**Q4: How does systemd's `Restart=always` interact with uvicorn's graceful shutdown?**
When uvicorn receives SIGTERM (from `systemctl stop` or `reload`): (1) The master process stops the listening socket. (2) In-flight requests continue to completion. (3) Workers exit when their current request finishes. (4) Master exits with code 0. systemd sees the process exit and, because `Restart=always`, schedules a restart after `RestartSec`. For `reload` (not `stop`), systemd does not restart — it only sends SIGHUP which uvicorn handles as a graceful worker rotation.

**Q5: How do you ensure the ALB health check does not trigger false positives during a cold start?**
EC2 instances take 30-90 seconds to start the service after boot (Python import, connection pool warmup). During this time, `/health` returns errors, and the ALB may immediately mark the instance unhealthy. Mitigation: (1) Set ALB health check `healthy threshold = 2` (requires 2 consecutive successes). (2) Set `unhealthy threshold = 3` (requires 3 consecutive failures). (3) Use EC2 Auto Scaling `WarmupSeconds` to delay health check evaluation. (4) Add a `startup_time` check in `/health/live` — return 200 even if deps are not ready within the first 60s.

**Q6: How do you structure CloudWatch alarms for the Chatbot API?**
Three alarm tiers: (1) **Liveness** — ALB HealthyHostCount < 1 → immediate page (service is completely down). (2) **Latency** — ALB TargetResponseTime p99 > 30s for 5 minutes → warning (LLM is slow). (3) **Error rate** — ALB HTTPCode_Target_5XX_Count > 5% of requests for 5 minutes → alert (application errors). Also: EC2 CPUUtilization > 80% for 10 minutes → scale-out trigger, and custom metric for active sessions > 80% of MAX_SESSIONS_PER_TENANT.

**Q7: A user reports their chat session disappeared after you deployed new code. What happened and how do you fix it?**
Root cause: the in-memory `_session_store` dict was lost when the uvicorn process restarted during deployment. Fix: (1) Immediate — check if conversation history is still in the LangGraph MemorySaver/OracleSaver (thread_id persisted to DB). The session dict is lost but the conversation is not. (2) Long-term — implement Redis-backed sessions (Task 23.2). Redis is external to the EC2 process; deployments, restarts, and even instance replacements do not clear the session store. (3) Enhancement — expose a `POST /sessions/restore` endpoint that recreates the session dict from OracleSaver thread history.

---

## Systems Design: Full Architecture on EC2

**Scenario:** Design the complete Chatbot backend that handles 500 concurrent users across 10 tenants, with per-tenant usage limits, conversation history, and document analysis.

**Answer:**

```
Layer 1 — Entry: ALB → Nginx (2 EC2 instances, active-active)
Layer 2 — App:   uvicorn workers (4 per EC2) running Lesson 23 FastAPI
Layer 3 — Cache: Redis (ElastiCache) for sessions + rate limit counters
Layer 4 — Agents: LangGraph orchestrator → Bedrock (Claude Haiku/Sonnet)
Layer 5 — Storage: Oracle 19c (RDS) for active state + S3 for archives
Layer 6 — Infra:  IAM roles, SSM, CloudWatch, SNS alerts
```

Key decisions:
- **Why 2 EC2 + ALB?** Single point of failure eliminated; blue/green deploys.
- **Why Redis for sessions?** Shared across both EC2 instances; survives restarts.
- **Why Bedrock Haiku for most requests?** 85% of queries are simple — Haiku handles them at 1/10 the cost of Sonnet. Model selector node routes to Sonnet only for complex analysis.
- **Why S3 for conversation archives?** Oracle stores the last N messages (active window). S3 stores full history at $0.023/GB vs Oracle at ~$0.10/GB. 10:1 cost reduction.
- **Usage limits enforced where?** At Redis (rate limit), at API layer (session count), at LangGraph gate node (token budget). Three independent enforcement points — no single bypass.
