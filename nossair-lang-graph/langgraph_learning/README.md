# LangGraph Learning — Complete Practical Curriculum (Beginner → Enterprise)

Hands-on, progressive curriculum to learn LangGraph from zero to enterprise-grade
production agents. Uses **Ollama** (local, free, no API key needed).

21 lessons across 4 levels: Foundation → Advanced → Senior → Enterprise.

---

## Quick Start

```bash
# 1. Install Ollama + model
#    Download from https://ollama.com, then:
ollama pull llama3.2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template (enterprise lessons)
cp .env.example .env
# Edit .env with your Oracle/Redis/JWT settings (optional — all lessons have fallbacks)

# 4. Run any lesson
python lesson_01_basics/lesson_01_basics.py
python lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py
```

---

## Full Curriculum (20 Lessons)

### Part 1 — Foundation (Lessons 1–5)
| # | Folder | Key Concepts |
|---|--------|-------------|
| 1 | `lesson_01_basics/` | `StateGraph`, `TypedDict`, nodes, edges, `compile`, `invoke` |
| 2 | `lesson_02_conditional/` | `add_conditional_edges`, routing functions, branching |
| 3 | `lesson_03_chatbot/` | `add_messages` reducer, `ChatOllama`, multi-turn memory |
| 4 | `lesson_04_tools_agent/` | `@tool`, `bind_tools`, `ToolNode`, ReAct loop |
| 5 | `lesson_05_multi_agent/` | Supervisor pattern, specialist agents, task handoff |

### Part 2 — Advanced (Lessons 6–10)
| # | Folder | Key Concepts |
|---|--------|-------------|
| 6 | `lesson_06_database_agent/` | SQLite tools, NL→SQL, schema inspection, read-only guard |
| 7 | `lesson_07_human_in_loop/` | `interrupt()`, `Command(resume=)`, 3 HITL patterns |
| 8 | `lesson_08_memory_persistence/` | `MemorySaver` vs `SqliteSaver`, `thread_id`, state history |
| 9 | `lesson_09_best_practices/` | Logging, Pydantic, structured output, retry, streaming, `Send()` |
| 10 | `lesson_10_capstone/` | Full agent: DB QA + HITL + memory + streaming + logging |

### Part 3 — Senior Level (Lessons 11–15)
| # | Folder | Key Concepts |
|---|--------|-------------|
| 11 | `lesson_11_subgraphs/` | Subgraphs, state sharing, reusable modules |
| 12 | `lesson_12_rag_agent/` | Embeddings, Chroma, retrieval, agentic RAG, self-correcting |
| 13 | `lesson_13_vector_memory/` | Long-term semantic memory, fact extraction |
| 14 | `lesson_14_testing/` | Unit tests, mock LLM, evaluation datasets, pytest |
| 15 | `lesson_15_deployment/` | FastAPI wrapper, Docker, LangSmith, production ops |

### Part 4 — Enterprise (Lessons 16–20)
| # | Folder | Key Concepts |
|---|--------|-------------|
| 16 | `lesson_16_postgres_async/` | **Oracle 19c** `OracleSaver`, MERGE upsert, oracledb pool, `ainvoke`, `asyncio.gather` |
| 17 | `lesson_17_auth_rbac/` | `python-jose` JWT, RBAC, `rate_limit_node`, multi-tenant isolation, SOC2 audit |
| 18 | `lesson_18_observability/` | Prometheus, `JsonFormatter`, `sla_guard_node`, trace_id, health checks |
| 19 | `lesson_19_event_driven/` | Celery + Redis, HMAC webhook verify, `DeadLetterQueue`, idempotency |
| 20 | `lesson_20_enterprise_capstone/` | Model routing, token budgets, circuit breaker, GDPR, 6-layer architecture |
| 21 | `lesson_21_aws_bedrock/` | `boto3`, `ChatBedrockConverse`, Claude/Titan/Llama model IDs, streaming, Guardrails, Ollama → Bedrock migration |

> **Enterprise lessons run without Oracle/Redis** — automatic fallback to `MemorySaver` / sync mode.
> See [ENTERPRISE_SETUP.md](ENTERPRISE_SETUP.md) for full production setup.

---

## Book & Study Materials

All study materials are in the `BOOK/` folder:

| File | Content |
|------|---------|
| `BOOK_Index.md` | Master index — curriculum map, topic index, learning paths |
| `BOOK_Part1_Foundation.md` | Lessons 1–5 overview + tasks + Q&A |
| `BOOK_Part2_Advanced.md` | Lessons 6–10 overview + tasks + Q&A |
| `BOOK_Part3_Senior.md` | Lessons 11–15 overview + tasks + Q&A |
| `BOOK_Part4_Interview_Master.md` | 50 senior interview Q&A, system design, cheatsheet |
| `BOOK_Part5_Enterprise.md` | Lessons 16–20 overview + tasks + Q&A |
| `Lesson01_Deep_Dive.md` | L1 internals, anti-patterns, 7 Q&A |
| `Lesson02_Deep_Dive.md` | L2 routing internals, 6 patterns, 7 Q&A |
| `Lessons03_04_05_Deep_Dive.md` | Reducers, ReAct loop, multi-agent — 6 Q&A each |
| `Lessons06_07_08_Deep_Dive.md` | DB agent, interrupt(), checkpointers — 5 Q&A each |
| `Lessons09_10_Best_Practices.md` | All 9 production pillars, streaming, parallel |
| `Lessons11_15_Senior_Deep_Dive.md` | Subgraphs, RAG, vector memory, testing, Docker |
| `Lessons16_20_Enterprise_Deep_Dive.md` | OracleSaver, JWT, Prometheus, Celery, GDPR — 7 Q&A each + 2 systems design |

**Start here:** [`BOOK/BOOK_Index.md`](BOOK/BOOK_Index.md)

---

## Key Architecture Patterns

```
ReAct Agent:         START → agent ←→ tools → END
HITL:                agent → interrupt() → [human] → Command(resume) → continue
Supervisor:          START → supervisor → specialist → supervisor → FINISH
Database Agent:      list_tables → describe_table → run_sql → answer
Persistent Memory:   OracleSaver/SqliteSaver + thread_id = per-user cross-session state
Enterprise:          Gateway → RBAC → Rate Limit → Budget → Circuit → Chat → Audit → END
```

---

## Running

```bash
# Foundation/Advanced lessons (notebooks recommended)
jupyter notebook

# Run any lesson script directly
python lesson_06_database_agent/lesson_06_database_agent.py
python lesson_10_capstone/lesson_10_capstone.py

# Enterprise lessons (all run without external services via fallback)
python lesson_16_postgres_async/lesson_16_oracle_async.py
python lesson_17_auth_rbac/lesson_17_auth_rbac.py
python lesson_18_observability/lesson_18_observability.py
python lesson_19_event_driven/lesson_19_event_driven.py
python lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py
python lesson_21_aws_bedrock/lesson_21_aws_bedrock.py

# Run all tests
pytest lesson_14_testing/test_agents.py -v
```

---

## Enterprise Setup

See **[ENTERPRISE_SETUP.md](ENTERPRISE_SETUP.md)** for:
- Oracle 19c DDL + grants
- Environment variable reference
- Redis + Celery worker startup
- Pre-production checklist
