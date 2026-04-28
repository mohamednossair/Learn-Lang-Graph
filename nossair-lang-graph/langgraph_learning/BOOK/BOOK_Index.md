# LangGraph Senior Engineer Guide — Complete Index

## How to Read This Book

| Your Level | Start Here | Goal |
|------------|-----------|------|
| Beginner | [Part 1 → Lesson 1](#lesson-1--stategraph-nodes-edges) | Understand graphs, state, nodes |
| Intermediate | [Part 2 → Lesson 6](#lesson-6--database-agent) | Build real agents |
| Advanced | [Part 3 → Lesson 11](#lesson-11--subgraphs) | Senior-level patterns |
| Interview Prep | [Part 4](#part-4--interview-master) | Pass senior engineer interviews |

---

## Book Structure

### Original Parts (overview + tasks + basic Q&A)

| File                                                           | Content | Lessons |
|----------------------------------------------------------------|---------|---------|
| [BOOK_Part1_Foundation.md](BOOK_Part1_Foundation.md)      | StateGraph, routing, chatbots, tools, multi-agent | 1–5 |
| [BOOK_Part2_Advanced.md](BOOK_Part2_Advanced.md)               | Database agent, HITL, memory, best practices, capstone | 6–10 |
| [BOOK_Part3_Senior.md](BOOK_Part3_Senior.md)                   | Subgraphs, RAG, vector memory, testing, deployment | 11–15 |
| [BOOK_Part4_Interview_Master.md](BOOK_Part4_Interview_Master.md) | 50 Q&A, system design, cheatsheet, self-assessment | — |
| [BOOK_Part5_Enterprise.md](BOOK_Part5_Enterprise.md)           | Oracle 19c async, RBAC, observability, event-driven, cost governance, AWS Bedrock, S3, Conversation API, EC2 deployment | 16–24 |

### Deep Dive Files (rich theory + under-the-hood + anti-patterns + full Q&A)

| File | Content | Lessons |
|------|---------|---------|
| [Lesson01_Deep_Dive.md](Lesson01_Deep_Dive.md) | StateGraph internals, state merging, node anatomy, anti-patterns, 7 Q&A | 1 |
| [Lesson02_Deep_Dive.md](Lesson02_Deep_Dive.md) | Routing under-the-hood, 6 patterns, before/after code, anti-patterns, 7 Q&A | 2 |
| [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | Reducers deep dive, ReAct loop mechanics, multi-agent — 6 Q&A each | 3–5 |
| [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | DB agent safety, interrupt() internals, checkpointer hierarchy, 5 Q&A each | 6–8 |
| [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | All 9 production pillars with full code, streaming, parallel, deploy checklist | 9–10 |
| [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | Subgraph state sharing, RAG 4 stages, vector memory lifecycle, testing levels, Docker | 11–15 |
| [Lessons16_20_Enterprise_Deep_Dive.md](Lessons16_20_Enterprise_Deep_Dive.md) | OracleSaver internals, JWT lifecycle, Prometheus internals, Celery lifecycle, GDPR, Bedrock SigV4/Converse/Guardrails — 7 Q&A each + 3 systems design | 16–21 |
| [Lessons22_24_Architecture_Deep_Dive.md](Lessons22_24_Architecture_Deep_Dive.md) | S3 internals, credential chain, presigned URL signing, session lifecycle, SSE internals, rate limiting, IAM/SSM/Nginx/systemd/CloudWatch — 7 Q&A each + 1 system design | 22–24 |
| [Lessons25_26_Memory_Search_Deep_Dive.md](Lessons25_26_Memory_Search_Deep_Dive.md) | Mem0 internals (5-step pipeline, Qdrant scoping, contradiction resolution, async saves), Solr internals (Lucene BM25, HNSW kNN, hybrid late-fusion, SolrCloud), failure modes, anti-patterns — 7 Q&A each + 2 system designs | 25–26 |

### How to Read
1. **First pass:** Read original Part files (1–4) for each lesson overview
2. **Deep study:** After running each notebook, read the matching Deep Dive file
3. **Interview prep:** [Part 4](BOOK_Part4_Interview_Master.md) + all Q&A sections in the Deep Dive files

---

## Full Curriculum Map

### Part 1 — Foundation

| Lesson | Overview | Deep Dive | Notebook | Script | Key Concepts |
|--------|----------|-----------|----------|--------|-------------|
| **1** | [Part 1](BOOK_Part1_Foundation.md) | [Deep Dive](Lesson01_Deep_Dive.md) | [.ipynb](../lesson_01_basics/lesson_01_basics.ipynb) | [.py](../lesson_01_basics/lesson_01_basics.py) | StateGraph, TypedDict, nodes, edges, compile, invoke |
| **2** | [Part 1](BOOK_Part1_Foundation.md) | [Deep Dive](Lesson02_Deep_Dive.md) | [.ipynb](../lesson_02_conditional/lesson_02_conditional.ipynb) | [.py](../lesson_02_conditional/lesson_02_conditional.py) | add_conditional_edges, routing function, Literal |
| **3** | [Part 1](BOOK_Part1_Foundation.md) | [Deep Dive](Lessons03_04_05_Deep_Dive.md) | [.ipynb](../lesson_03_chatbot/lesson_03_chatbot.ipynb) | [.py](../lesson_03_chatbot/lesson_03_chatbot.py) | add_messages reducer, ChatOllama, multi-turn memory |
| **4** | [Part 1](BOOK_Part1_Foundation.md) | [Deep Dive](Lessons03_04_05_Deep_Dive.md) | [.ipynb](../lesson_04_tools_agent/lesson_04_tools_agent.ipynb) | [.py](../lesson_04_tools_agent/lesson_04_tools_agent.py) | @tool, bind_tools, ToolNode, ReAct loop |
| **5** | [Part 1](BOOK_Part1_Foundation.md) | [Deep Dive](Lessons03_04_05_Deep_Dive.md) | [.ipynb](../lesson_05_multi_agent/lesson_05_multi_agent.ipynb) | [.py](../lesson_05_multi_agent/lesson_05_multi_agent.py) | Supervisor pattern, specialists, task handoff |

### Part 2 — Advanced

| Lesson | Overview | Deep Dive | Notebook | Script | Key Concepts |
|--------|----------|-----------|----------|--------|-------------|
| **6** | [Part 2](BOOK_Part2_Advanced.md) | [Deep Dive](Lessons06_07_08_Deep_Dive.md) | [.ipynb](../lesson_06_database_agent/lesson_06_database_agent.ipynb) | [.py](../lesson_06_database_agent/lesson_06_database_agent.py) | SQLite tools, NL→SQL, schema inspection, read-only guard |
| **7** | [Part 2](BOOK_Part2_Advanced.md) | [Deep Dive](Lessons06_07_08_Deep_Dive.md) | [.ipynb](../lesson_07_human_in_loop/lesson_07_human_in_loop.ipynb) | [.py](../lesson_07_human_in_loop/lesson_07_human_in_loop.py) | interrupt(), Command(resume), 3 HITL patterns |
| **8** | [Part 2](BOOK_Part2_Advanced.md) | [Deep Dive](Lessons06_07_08_Deep_Dive.md) | [.ipynb](../lesson_08_memory_persistence/lesson_08_memory_persistence.ipynb) | [.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) | MemorySaver, SqliteSaver, thread_id, history |
| **9** | [Part 2](BOOK_Part2_Advanced.md) | [Deep Dive](Lessons09_10_Best_Practices.md) | [.ipynb](../lesson_09_best_practices/lesson_09_best_practices.ipynb) | [.py](../lesson_09_best_practices/lesson_09_best_practices.py) | Logging, validation, retry, streaming, Send() |
| **10** | [Part 2](BOOK_Part2_Advanced.md) | [Deep Dive](Lessons09_10_Best_Practices.md) | [.ipynb](../lesson_10_capstone/lesson_10_capstone.ipynb) | [.py](../lesson_10_capstone/lesson_10_capstone.py) | All lessons combined — full production agent |

### Part 3 — Senior Level

| Lesson | Overview | Deep Dive | Script | Key Concepts |
|--------|----------|-----------|--------|-------------|
| **11** | [Part 3](BOOK_Part3_Senior.md) | [Deep Dive](Lessons11_15_Senior_Deep_Dive.md) | [.py](../lesson_11_subgraphs/lesson_11_subgraphs.py) | Subgraphs, state sharing, reusable modules, Send() |
| **12** | [Part 3](BOOK_Part3_Senior.md) | [Deep Dive](Lessons11_15_Senior_Deep_Dive.md) | [.py](../lesson_12_rag_agent/lesson_12_rag_agent.py) | Embeddings, Chroma, retrieval, agentic RAG, self-correcting |
| **13** | [Part 3](BOOK_Part3_Senior.md) | [Deep Dive](Lessons11_15_Senior_Deep_Dive.md) | [.py](../lesson_13_vector_memory/lesson_13_vector_memory.py) | Long-term semantic memory, Chroma, fact extraction |
| **14** | [Part 3](BOOK_Part3_Senior.md) | [Deep Dive](Lessons11_15_Senior_Deep_Dive.md) | [tests](../lesson_14_testing/test_agents.py) | Unit tests, mock LLM, evaluation datasets, pytest |
| **15** | [Part 3](BOOK_Part3_Senior.md) | [Deep Dive](Lessons11_15_Senior_Deep_Dive.md) | [api.py](../lesson_15_deployment/api.py) | FastAPI, Docker, LangSmith, production ops |

### Part 4 — Interview Master

| File | Sections |
|------|---------|
| [BOOK_Part4_Interview_Master.md](BOOK_Part4_Interview_Master.md) | 50 Q&A · System Design · Cheatsheet · Self-Assessment · Mistake Patterns |

### Part 5 — Enterprise Level

| Lesson | Overview | Deep Dive | Script | Key Concepts |
|--------|----------|-----------|--------|--------------|
| **16** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [Oracle .py](../lesson_16_postgres_async/lesson_16_oracle_async.py) · [PG ref](../lesson_16_postgres_async/lesson_16_postgres_async.py) | OracleSaver (19c), MERGE upsert, oracledb pool, ainvoke, astream, asyncio.gather |
| **17** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) | python-jose JWT, RBAC, rate_limit_node, multi-tenant, SOC2 audit |
| **18** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [.py](../lesson_18_observability/lesson_18_observability.py) | Prometheus, JsonFormatter, sla_guard_node, trace_id, health checks, Grafana |
| **19** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [.py](../lesson_19_event_driven/lesson_19_event_driven.py) | Celery, Redis, HMAC webhook verify, idempotency, DeadLetterQueue, task polling |
| **20** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [.py](../lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py) | Model routing, token budgets, circuit breaker, GDPR, 6-layer architecture |
| **21** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md#lesson-21) | [.py](../lesson_21_aws_bedrock/lesson_21_aws_bedrock.py) | `ChatBedrockConverse`, boto3 credential chain, Claude/Titan/Llama, streaming, Guardrails, Ollama→Bedrock migration |
| **22** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons22_24_Architecture_Deep_Dive.md) | [.py](../lesson_22_aws_s3/lesson_22_aws_s3.py) | S3 upload/download, conversation snapshots, presigned URLs, document storage, GDPR S3 erasure |
| **23** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons22_24_Architecture_Deep_Dive.md) | [.py](../lesson_23_conversation_api/lesson_23_conversation_api.py) | Chatbot API, session management, agent routing, usage limits, SSE streaming, `/chat` + `/upload` endpoints |
| **24** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons22_24_Architecture_Deep_Dive.md) | [.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) | EC2 IAM role, SSM Parameter Store, Nginx config, systemd service, CloudWatch logging, zero-downtime deploy |
| **25** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons25_26_Memory_Search_Deep_Dive.md) | [.py](../lesson_25_mem0/lesson_25_mem0.py) | Mem0, automatic memory extraction, deduplication, contradiction resolution, GDPR erasure, Qdrant backend |
| **26** | [Part 5](BOOK_Part5_Enterprise.md) | [Deep Dive](Lessons25_26_Memory_Search_Deep_Dive.md) | [.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) | Apache Solr, BM25 keyword search, kNN vector search, hybrid BM25+kNN, self-correcting RAG, SolrVectorStore |

---

## Topic Index (Find Anything Fast)

| Topic | Book File | Lesson Code |
|-------|-----------|-------------|
| `add_conditional_edges` | [Lesson02_Deep_Dive.md](Lesson02_Deep_Dive.md) | [lesson_02.py](../lesson_02_conditional/lesson_02_conditional.py) |
| `add_messages` reducer | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_03.py](../lesson_03_chatbot/lesson_03_chatbot.py) |
| Agent evaluation | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [test_agents.py](../lesson_14_testing/test_agents.py) |
| `@tool` decorator | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_04.py](../lesson_04_tools_agent/lesson_04_tools_agent.py) |
| `bind_tools()` | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_04.py](../lesson_04_tools_agent/lesson_04_tools_agent.py) |
| Checkpointers | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_08.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) |
| Circuit breaker | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| `Command(resume)` | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_07.py](../lesson_07_human_in_loop/lesson_07_human_in_loop.py) |
| `compile()` | [Lesson01_Deep_Dive.md](Lesson01_Deep_Dive.md) | [lesson_01.py](../lesson_01_basics/lesson_01_basics.py) |
| Conditional edges | [Lesson02_Deep_Dive.md](Lesson02_Deep_Dive.md) | [lesson_02.py](../lesson_02_conditional/lesson_02_conditional.py) |
| Docker deployment | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [api.py](../lesson_15_deployment/api.py) |
| Error handling | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Exponential backoff | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| FastAPI wrapper | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [api.py](../lesson_15_deployment/api.py) |
| GDPR compliance | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_08.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) |
| Human-in-the-loop | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_07.py](../lesson_07_human_in_loop/lesson_07_human_in_loop.py) |
| `interrupt()` | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_07.py](../lesson_07_human_in_loop/lesson_07_human_in_loop.py) |
| `invoke()` | [Lesson01_Deep_Dive.md](Lesson01_Deep_Dive.md) | [lesson_01.py](../lesson_01_basics/lesson_01_basics.py) |
| LangSmith tracing | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [api.py](../lesson_15_deployment/api.py) |
| Logging | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Long-term memory | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [lesson_13.py](../lesson_13_vector_memory/lesson_13_vector_memory.py) |
| `MemorySaver` | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_08.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) |
| Message trimming | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_03.py](../lesson_03_chatbot/lesson_03_chatbot.py) |
| `MessagesState` | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_03.py](../lesson_03_chatbot/lesson_03_chatbot.py) |
| Mock LLM testing | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [test_agents.py](../lesson_14_testing/test_agents.py) |
| Multi-agent supervisor | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_05.py](../lesson_05_multi_agent/lesson_05_multi_agent.py) |
| NL to SQL | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_06.py](../lesson_06_database_agent/lesson_06_database_agent.py) |
| Parallel execution (`Send()`) | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Pydantic validation | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| RAG | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [lesson_12.py](../lesson_12_rag_agent/lesson_12_rag_agent.py) |
| ReAct loop | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_04.py](../lesson_04_tools_agent/lesson_04_tools_agent.py) |
| Recursion limit | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Reducers | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_03.py](../lesson_03_chatbot/lesson_03_chatbot.py) |
| Retry logic | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| SQL injection prevention | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_06.py](../lesson_06_database_agent/lesson_06_database_agent.py) |
| `SqliteSaver` | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_08.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) |
| `START` / `END` | [Lesson01_Deep_Dive.md](Lesson01_Deep_Dive.md) | [lesson_01.py](../lesson_01_basics/lesson_01_basics.py) |
| `StateGraph` | [Lesson01_Deep_Dive.md](Lesson01_Deep_Dive.md) | [lesson_01.py](../lesson_01_basics/lesson_01_basics.py) |
| `stream()` | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Structured output | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Subgraphs | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [lesson_11.py](../lesson_11_subgraphs/lesson_11_subgraphs.py) |
| Supervisor pattern | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_05.py](../lesson_05_multi_agent/lesson_05_multi_agent.py) |
| System design questions | [BOOK_Part4_Interview_Master.md](BOOK_Part4_Interview_Master.md) | — |
| Testing agents | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [test_agents.py](../lesson_14_testing/test_agents.py) |
| `thread_id` | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_08.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) |
| Time-travel debugging | [Lessons06_07_08_Deep_Dive.md](Lessons06_07_08_Deep_Dive.md) | [lesson_08.py](../lesson_08_memory_persistence/lesson_08_memory_persistence.py) |
| `ToolNode` | [Lessons03_04_05_Deep_Dive.md](Lessons03_04_05_Deep_Dive.md) | [lesson_04.py](../lesson_04_tools_agent/lesson_04_tools_agent.py) |
| `TypedDict` | [Lesson01_Deep_Dive.md](Lesson01_Deep_Dive.md) | [lesson_01.py](../lesson_01_basics/lesson_01_basics.py) |
| Vector memory | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [lesson_13.py](../lesson_13_vector_memory/lesson_13_vector_memory.py) |
| `with_structured_output` | [Lessons09_10_Best_Practices.md](Lessons09_10_Best_Practices.md) | [lesson_09.py](../lesson_09_best_practices/lesson_09_best_practices.py) |
| Zero-downtime deploy | [Lessons11_15_Senior_Deep_Dive.md](Lessons11_15_Senior_Deep_Dive.md) | [api.py](../lesson_15_deployment/api.py) |
| `ainvoke` / `astream` | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_16_oracle.py](../lesson_16_postgres_async/lesson_16_oracle_async.py) |
| Oracle connection pool | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_16_oracle.py](../lesson_16_postgres_async/lesson_16_oracle_async.py) |
| `asyncio.gather()` | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_16_oracle.py](../lesson_16_postgres_async/lesson_16_oracle_async.py) |
| Oracle MERGE upsert | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_16_oracle.py](../lesson_16_postgres_async/lesson_16_oracle_async.py) |
| CLOB vs VARCHAR2 | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_16_oracle.py](../lesson_16_postgres_async/lesson_16_oracle_async.py) |
| OracleSaver (19c) | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_16_oracle.py](../lesson_16_postgres_async/lesson_16_oracle_async.py) |
| PostgresSaver (reference) | [BOOK_Part5_Enterprise.md](BOOK_Part5_Enterprise.md) | [lesson_16_pg.py](../lesson_16_postgres_async/lesson_16_postgres_async.py) |
| Audit trail (SOC2) | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| JWT authentication | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| JWT refresh token | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| RBAC permissions | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| Multi-tenancy | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| Rate limiting (sliding window) | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| Oracle VPD | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_17.py](../lesson_17_auth_rbac/lesson_17_auth_rbac.py) |
| Prometheus metrics | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_18.py](../lesson_18_observability/lesson_18_observability.py) |
| Trace ID propagation | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_18.py](../lesson_18_observability/lesson_18_observability.py) |
| Health checks (liveness/readiness) | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_18.py](../lesson_18_observability/lesson_18_observability.py) |
| Grafana dashboards | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_18.py](../lesson_18_observability/lesson_18_observability.py) |
| JSON structured logging | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_18.py](../lesson_18_observability/lesson_18_observability.py) |
| SLA guard / alerting | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_18.py](../lesson_18_observability/lesson_18_observability.py) |
| Celery job queue | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_19.py](../lesson_19_event_driven/lesson_19_event_driven.py) |
| Event-driven agents | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_19.py](../lesson_19_event_driven/lesson_19_event_driven.py) |
| Idempotency | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_19.py](../lesson_19_event_driven/lesson_19_event_driven.py) |
| Dead-letter queue | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_19.py](../lesson_19_event_driven/lesson_19_event_driven.py) |
| Webhook HMAC verification | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_19.py](../lesson_19_event_driven/lesson_19_event_driven.py) |
| FastAPI task polling | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_19.py](../lesson_19_event_driven/lesson_19_event_driven.py) |
| Circuit breaker | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_20.py](../lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py) |
| Token budget | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_20.py](../lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py) |
| Model routing (cost) | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_20.py](../lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py) |
| GDPR erasure | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | [lesson_20.py](../lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.py) |
| Enterprise systems design | [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) | — |
| AWS Bedrock models | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_21.py](../lesson_21_aws_bedrock/lesson_21_aws_bedrock.py) |
| `ChatBedrockConverse` | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_21.py](../lesson_21_aws_bedrock/lesson_21_aws_bedrock.py) |
| boto3 credential chain | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_21.py](../lesson_21_aws_bedrock/lesson_21_aws_bedrock.py) |
| Bedrock Guardrails | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_21.py](../lesson_21_aws_bedrock/lesson_21_aws_bedrock.py) |
| Ollama → Bedrock migration | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_21.py](../lesson_21_aws_bedrock/lesson_21_aws_bedrock.py) |
| S3 upload / download | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_22.py](../lesson_22_aws_s3/lesson_22_aws_s3.py) |
| S3 presigned URLs | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_22.py](../lesson_22_aws_s3/lesson_22_aws_s3.py) |
| S3 conversation snapshots | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_22.py](../lesson_22_aws_s3/lesson_22_aws_s3.py) |
| GDPR S3 erasure | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_22.py](../lesson_22_aws_s3/lesson_22_aws_s3.py) |
| Session management (API) | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_23.py](../lesson_23_conversation_api/lesson_23_conversation_api.py) |
| Agent routing (orchestrator) | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_23.py](../lesson_23_conversation_api/lesson_23_conversation_api.py) |
| SSE streaming (FastAPI) | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_23.py](../lesson_23_conversation_api/lesson_23_conversation_api.py) |
| Usage limits enforcement | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_23.py](../lesson_23_conversation_api/lesson_23_conversation_api.py) |
| EC2 IAM instance profile | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_24.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) |
| SSM Parameter Store | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_24.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) |
| Nginx reverse proxy | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_24.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) |
| Systemd service (auto-restart) | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_24.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) |
| CloudWatch logging (watchtower) | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_24.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) |
| Zero-downtime EC2 deploy | [Part 5](BOOK_Part5_Enterprise.md) | [lesson_24.py](../lesson_24_ec2_deployment/lesson_24_ec2_deployment.py) |
| Mem0 long-term memory | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_25.py](../lesson_25_mem0/lesson_25_mem0.py) |
| Mem0 contradiction resolution | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_25.py](../lesson_25_mem0/lesson_25_mem0.py) |
| Mem0 GDPR erasure | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_25.py](../lesson_25_mem0/lesson_25_mem0.py) |
| Mem0 Cloud vs self-hosted | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_25.py](../lesson_25_mem0/lesson_25_mem0.py) |
| Qdrant vector store | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_25.py](../lesson_25_mem0/lesson_25_mem0.py) |
| Apache Solr RAG | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| BM25 keyword search | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| kNN vector search (Solr) | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| Hybrid BM25+kNN search | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| SolrVectorStore (LangChain) | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| Lucene BM25 internals | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| HNSW graph (Solr kNN) | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| Late-fusion hybrid search | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_26.py](../lesson_26_solr_rag/lesson_26_solr_rag.py) |
| Mem0 5-step add() pipeline | [Deep Dive L25-26](Lessons25_26_Memory_Search_Deep_Dive.md) | [lesson_25.py](../lesson_25_mem0/lesson_25_mem0.py) |

---

## Learning Path Recommendations

### Path A — "I want to build a database agent fast" (2-3 days)
1. [Lesson 1](Lesson01_Deep_Dive.md) — run notebook, do Task 1.1
2. [Lesson 2](Lesson02_Deep_Dive.md) — run notebook, do Task 2.1
3. [Lesson 4](Lessons03_04_05_Deep_Dive.md) — full lesson (tools are essential)
4. [Lesson 6](Lessons06_07_08_Deep_Dive.md) — full lesson (the target)

### Path B — "I'm preparing for a senior interview" (1 week)
1. All 15 lessons — run every notebook linked in the [Curriculum Map](#full-curriculum-map)
2. Do at least 2 tasks per lesson from the Deep Dive files
3. Read all Interview Q&A sections
4. Study [Part 4](BOOK_Part4_Interview_Master.md) — system design section
5. Memorize the cheatsheet in [Part 4](BOOK_Part4_Interview_Master.md)

### Path D — "I'm building an enterprise system" (2 weeks)
1. Complete Paths A + B first (Lessons 1–15)
2. [Lesson 16](BOOK_Part5_Enterprise.md) — Oracle 19c async + OracleSaver (foundation for all enterprise work)
3. [Lesson 17](BOOK_Part5_Enterprise.md) — JWT auth, RBAC, rate limiting, multi-tenancy
4. [Lesson 18](BOOK_Part5_Enterprise.md) — observability (Prometheus, JSON logs, SLA guard)
5. [Lesson 19](BOOK_Part5_Enterprise.md) — event-driven (HMAC webhooks, DLQ, Celery)
6. [Lesson 20](BOOK_Part5_Enterprise.md) — cost control + governance capstone
7. [Lesson 21](BOOK_Part5_Enterprise.md) — AWS Bedrock (migrate from Ollama to production LLMs)
8. [Lesson 22](BOOK_Part5_Enterprise.md) — AWS S3 (document storage, conversation snapshots, presigned URLs)
9. [Lesson 23](BOOK_Part5_Enterprise.md) — Conversation Management API (Chatbot API, sessions, SSE)
10. [Lesson 24](BOOK_Part5_Enterprise.md) — EC2 deployment (IAM role, SSM, Nginx, systemd, CloudWatch)
11. [Lesson 25](BOOK_Part5_Enterprise.md) — Mem0 long-term memory (auto-extraction, dedup, contradiction resolution, GDPR)
12. [Lesson 26](BOOK_Part5_Enterprise.md) — Solr RAG (BM25, kNN vector, hybrid search, SolrVectorStore)
13. [Enterprise Deep Dive](Lessons16_20_Enterprise_Deep_Dive.md) — internals, anti-patterns, 35 Q&A
14. Practice 2 systems design questions from the Deep Dive

### Path C — "I'm deploying to production tomorrow" (1 day)
1. [Lesson 9](Lessons09_10_Best_Practices.md) — all 9 production pillars
2. [Lesson 8](Lessons06_07_08_Deep_Dive.md) — SqliteSaver setup
3. [Lesson 7](Lessons06_07_08_Deep_Dive.md) — HITL patterns
4. [Lesson 15](Lessons11_15_Senior_Deep_Dive.md) — FastAPI + Docker deployment
5. [Part 4 cheatsheet](BOOK_Part4_Interview_Master.md)

---

## Progress Tracker

Mark each lesson complete as you go:

```
Part 1 — Foundation:
  [ ] Lesson 1  — StateGraph basics
  [ ] Lesson 2  — Conditional edges
  [ ] Lesson 3  — Chatbot with memory
  [ ] Lesson 4  — ReAct agent with tools
  [ ] Lesson 5  — Multi-agent supervisor

Part 2 — Advanced:
  [ ] Lesson 6  — Database agent
  [ ] Lesson 7  — Human-in-the-loop
  [ ] Lesson 8  — Persistent memory
  [ ] Lesson 9  — Production best practices
  [ ] Lesson 10 — Capstone project

Part 3 — Senior Level:
  [ ] Lesson 11 — Subgraphs
  [ ] Lesson 12 — RAG agent
  [ ] Lesson 13 — Vector memory
  [ ] Lesson 14 — Testing agents
  [ ] Lesson 15 — Deployment

Part 4 — Interview:
  [ ] Read all 50 Q&A
  [ ] Practice 3 system design questions
  [ ] Memorize cheatsheet
  [ ] Complete self-assessment (target: all 4-5)

Part 5 — Enterprise:
  [ ] Lesson 16 — Oracle 19c async (OracleSaver, MERGE, oracledb pool)
  [ ] Lesson 17 — JWT auth, RBAC, rate limiting, multi-tenancy
  [ ] Lesson 18 — Observability (Prometheus, JSON logs, SLA guard)
  [ ] Lesson 19 — Event-driven (Celery, HMAC webhooks, DLQ, idempotency)
  [ ] Lesson 20 — Cost control, circuit breaker, GDPR, capstone
  [ ] Lesson 21 — AWS Bedrock (ChatBedrockConverse, credential chain, Guardrails)
  [ ] Lesson 22 — AWS S3 (upload/download, snapshots, presigned URLs, GDPR erasure)
  [ ] Lesson 23 — Conversation Management API (sessions, agent routing, SSE, usage limits)
  [ ] Lesson 24 — EC2 deployment (IAM role, SSM Parameter Store, Nginx, systemd, CloudWatch)
  [ ] Lesson 25 — Mem0 long-term memory (auto-extraction, dedup, contradiction resolution, GDPR)
  [ ] Lesson 26 — Solr RAG (BM25, kNN vector, hybrid BM25+kNN, SolrVectorStore)
  [ ] Deep Dive L25-26 — Mem0+Solr internals, 14 Q&A + 2 system designs
  [ ] Enterprise Deep Dive L16-21 — all 5 sections + 35 Q&A + 2 systems design
  [ ] Architecture Deep Dive L22-24 — S3/API/EC2 internals + 21 Q&A + 1 system design
  [ ] Task 20.5 — Full enterprise FastAPI app (6 layers combined)
  [ ] Architecture Completion Checklist (BOOK_Part5_Enterprise.md) — all 15 components covered
```
