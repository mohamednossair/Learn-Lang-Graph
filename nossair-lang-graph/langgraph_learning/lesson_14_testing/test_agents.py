# =============================================================
# LESSON 14 — Testing and Evaluating LangGraph Agents
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. Unit tests — test each tool in isolation
#   2. Node tests — test each node function
#   3. Mock LLM tests — test routing logic without real LLM calls
#   4. Integration tests — test full graph paths end-to-end
#   5. Evaluation tests — assess answer quality with ground truth
#
# RUN:
#   pip install pytest
#   pytest lesson_14_testing/test_agents.py -v
#   pytest lesson_14_testing/test_agents.py -v -m "not integration"  (fast only)
#   pytest lesson_14_testing/test_agents.py -v -m integration         (slow only)
#
# WHY TESTING AGENTS IS HARD:
#   - LLMs are non-deterministic (same input → different output)
#   - Long execution paths with many conditional branches
#   - External dependencies (database, LLM, tools)
#   - State-dependent behavior
#
# SOLUTIONS:
#   - Mock the LLM for logic tests
#   - Use temperature=0 for reproducibility in integration tests
#   - Test tools independently (no LLM needed)
#   - Use ground truth keywords, not exact string matching
# =============================================================

import os
import sys
import sqlite3
import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# =============================================================
# FIXTURES — shared setup for tests
# =============================================================

@pytest.fixture(scope="session")
def capstone_db():
    """Set up capstone database once for all tests."""
    from lesson_10_capstone.lesson_10_capstone import setup_database
    setup_database()
    yield
    # Cleanup not needed for read-only tests


@pytest.fixture
def capstone_graph(capstone_db):
    """Build a fresh capstone graph for each test."""
    from lesson_10_capstone.lesson_10_capstone import build_capstone_graph
    from langgraph.checkpoint.memory import MemorySaver
    return build_capstone_graph(checkpointer=MemorySaver())


@pytest.fixture
def test_config():
    """Standard test config."""
    import time
    return {"configurable": {"thread_id": f"test-{int(time.time())}"}, "recursion_limit": 25}


# =============================================================
# LEVEL 1 — TOOL UNIT TESTS
# Test each tool function independently. No LLM needed.
# These should run in milliseconds.
# =============================================================

class TestDatabaseTools:
    """Test all database tools in isolation."""

    def test_list_tables_returns_string(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import list_tables
        result = list_tables.invoke({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_tables_contains_expected_tables(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import list_tables
        result = list_tables.invoke({})
        result_lower = result.lower()
        assert "employees" in result_lower or "departments" in result_lower

    def test_run_sql_accepts_select(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import run_sql
        result = run_sql.invoke({"query": "SELECT COUNT(*) FROM employees"})
        assert "ERROR" not in result.upper() or "Only SELECT" not in result

    def test_run_sql_rejects_delete(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import run_sql
        result = run_sql.invoke({"query": "DELETE FROM employees WHERE id=1"})
        assert "ERROR" in result.upper() or "Only SELECT" in result

    def test_run_sql_rejects_drop(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import run_sql
        result = run_sql.invoke({"query": "DROP TABLE employees"})
        assert "ERROR" in result.upper() or "Only SELECT" in result

    def test_run_sql_rejects_update(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import run_sql
        result = run_sql.invoke({"query": "UPDATE employees SET salary=0"})
        assert "ERROR" in result.upper() or "Only SELECT" in result

    def test_run_sql_handles_invalid_query(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import run_sql
        result = run_sql.invoke({"query": "SELECT * FROM nonexistent_table"})
        assert isinstance(result, str)  # should return error string, not raise

    def test_describe_table_returns_schema(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import describe_table
        result = describe_table.invoke({"table_name": "employees"})
        assert isinstance(result, str)
        assert len(result) > 10

    def test_describe_table_invalid_table(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import describe_table
        result = describe_table.invoke({"table_name": "nonexistent"})
        assert isinstance(result, str)  # should not raise


# =============================================================
# LEVEL 2 — NODE TESTS (with mock LLM)
# Test routing and logic without calling the real LLM.
# These run fast because no actual LLM calls happen.
# =============================================================

class TestNodeLogic:
    """Test node functions with mocked LLM responses."""

    def test_supervisor_routes_to_db_agent(self):
        """Supervisor should route DB questions to db_agent."""
        mock_response = AIMessage(content='{"next": "db_agent"}')
        with patch("lesson_10_capstone.lesson_10_capstone.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from lesson_10_capstone.lesson_10_capstone import supervisor_node
            state = {
                "messages": [HumanMessage(content="How many employees are there?")],
                "user_id": "test-user",
                "next_agent": "",
                "needs_approval": False
            }
            result = supervisor_node(state)
            assert result.get("next_agent") == "db_agent"

    def test_supervisor_routes_to_finish(self):
        """Supervisor should route to FINISH when task is done."""
        mock_response = AIMessage(content='{"next": "FINISH"}')
        with patch("lesson_10_capstone.lesson_10_capstone.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from lesson_10_capstone.lesson_10_capstone import supervisor_node
            state = {
                "messages": [HumanMessage(content="Thanks!")],
                "user_id": "test-user",
                "next_agent": "",
                "needs_approval": False
            }
            result = supervisor_node(state)
            assert result.get("next_agent") == "FINISH"

    def test_supervisor_handles_malformed_json(self):
        """Supervisor should default to FINISH if JSON is malformed."""
        mock_response = AIMessage(content="I don't know what to do")
        with patch("lesson_10_capstone.lesson_10_capstone.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from lesson_10_capstone.lesson_10_capstone import supervisor_node
            state = {
                "messages": [HumanMessage(content="Something unclear")],
                "user_id": "test-user",
                "next_agent": "",
                "needs_approval": False
            }
            result = supervisor_node(state)
            # Should not crash — defaults gracefully
            assert "next_agent" in result

    def test_route_supervisor_mapping(self):
        """route_supervisor should map next_agent to correct graph destinations."""
        from lesson_10_capstone.lesson_10_capstone import route_supervisor

        assert route_supervisor({"next_agent": "db_agent"})    == "db_agent"
        assert route_supervisor({"next_agent": "analyst"})     == "analyst"
        assert route_supervisor({"next_agent": "human_review"}) == "human_review"
        assert route_supervisor({"next_agent": "FINISH"})      == "__end__"
        assert route_supervisor({"next_agent": "unknown"})     == "__end__"
        assert route_supervisor({"next_agent": ""})            == "__end__"


# =============================================================
# LEVEL 3 — GRAPH COMPILATION TESTS
# Verify graphs build without errors.
# =============================================================

class TestGraphCompilation:
    """Test that all lesson graphs compile correctly."""

    def test_lesson_01_compiles(self):
        from lesson_01_basics.lesson_01_basics import graph
        assert graph is not None

    def test_lesson_02_compiles(self):
        from lesson_02_conditional.lesson_02_conditional import graph
        assert graph is not None

    def test_lesson_06_compiles(self, capstone_db):
        from lesson_06_database_agent.lesson_06_database_agent import graph
        assert graph is not None

    def test_capstone_compiles(self, capstone_db):
        from lesson_10_capstone.lesson_10_capstone import build_capstone_graph
        g = build_capstone_graph()
        assert g is not None

    def test_lesson_11_compiles(self):
        from lesson_11_subgraphs.lesson_11_subgraphs import graph
        assert graph is not None


# =============================================================
# LEVEL 4 — INTEGRATION TESTS (require real LLM)
# Mark with @pytest.mark.integration to run separately.
# Run: pytest -v -m integration
# =============================================================

@pytest.mark.integration
class TestIntegration:
    """End-to-end tests using the real Ollama LLM."""

    def test_lesson_01_basic_flow(self):
        from lesson_01_basics.lesson_01_basics import graph
        result = graph.invoke({"message": "hello", "processed": "", "final": ""})
        assert result["processed"] == "HELLO"
        assert "HELLO" in result["final"]

    def test_lesson_02_positive_route(self):
        from lesson_02_conditional.lesson_02_conditional import graph
        result = graph.invoke({"review": "This is amazing and great!", "sentiment": "", "response": ""})
        assert result["sentiment"] == "positive"
        assert len(result["response"]) > 0

    def test_lesson_02_negative_route(self):
        from lesson_02_conditional.lesson_02_conditional import graph
        result = graph.invoke({"review": "Terrible experience, I hate it.", "sentiment": "", "response": ""})
        assert result["sentiment"] == "negative"

    def test_validation_subgraph_accepts_valid(self):
        from lesson_11_subgraphs.lesson_11_subgraphs import validation_subgraph
        result = validation_subgraph.invoke({
            "content": "Great Python IDE for developers.", "is_valid": True, "validation_errors": []
        })
        assert result["is_valid"] is True
        assert len(result["validation_errors"]) == 0

    def test_validation_subgraph_rejects_short(self):
        from lesson_11_subgraphs.lesson_11_subgraphs import validation_subgraph
        result = validation_subgraph.invoke({
            "content": "Hi", "is_valid": True, "validation_errors": []
        })
        assert result["is_valid"] is False

    def test_validation_subgraph_rejects_forbidden(self):
        from lesson_11_subgraphs.lesson_11_subgraphs import validation_subgraph
        result = validation_subgraph.invoke({
            "content": "This product is totally spam.", "is_valid": True, "validation_errors": []
        })
        assert result["is_valid"] is False


# =============================================================
# LEVEL 5 — EVALUATION TESTS (LLM answer quality)
# =============================================================

EVAL_DATASET = [
    {
        "id": "E01",
        "question": "How many employees are in the Engineering department?",
        "expected_keywords": ["2", "two", "alice", "bob"],
        "description": "Count Engineering employees"
    },
    {
        "id": "E02",
        "question": "What is the highest salary in the company?",
        "expected_keywords": ["95000", "95,000", "alice"],
        "description": "Find max salary"
    },
    {
        "id": "E03",
        "question": "What departments exist in the company?",
        "expected_keywords": ["engineering", "sales", "marketing"],
        "description": "List all departments"
    },
    {
        "id": "E04",
        "question": "What is the average salary across all employees?",
        "expected_keywords": ["77", "average", "salary"],
        "description": "Calculate average salary (should be ~77k)"
    },
]


@pytest.mark.integration
class TestEvaluation:
    """Evaluate agent answer quality against ground truth."""

    @pytest.mark.parametrize("case", EVAL_DATASET)
    def test_agent_answer_quality(self, capstone_graph, case):
        """Agent should produce answers containing expected keywords."""
        import time
        config = {
            "configurable": {"thread_id": f"eval-{case['id']}-{int(time.time())}"},
            "recursion_limit": 25
        }
        result = capstone_graph.invoke(
            {"messages": [HumanMessage(content=case["question"])],
             "user_id": "eval-bot", "next_agent": ""},
            config=config
        )
        last_msg = next(
            (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        last_msg_lower = last_msg.lower()
        matched = any(kw.lower() in last_msg_lower for kw in case["expected_keywords"])
        assert matched, (
            f"\nEval case [{case['id']}]: {case['description']}\n"
            f"Question: {case['question']}\n"
            f"Answer: {last_msg[:300]}\n"
            f"Expected one of: {case['expected_keywords']}"
        )


# =============================================================
# UTILITY: Run evaluation and print a report
# =============================================================

def run_evaluation_report():
    """Run all evaluation cases and print a pass/fail report."""
    import time
    from lesson_10_capstone.lesson_10_capstone import build_capstone_graph, setup_database
    from langgraph.checkpoint.memory import MemorySaver

    setup_database()
    graph = build_capstone_graph(checkpointer=MemorySaver())

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    total = len(EVAL_DATASET)
    passed = 0

    for case in EVAL_DATASET:
        config = {"configurable": {"thread_id": f"eval-{case['id']}"}, "recursion_limit": 25}
        start = time.time()
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=case["question"])],
                 "user_id": "eval-bot", "next_agent": ""},
                config=config
            )
            elapsed = time.time() - start
            last = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
            match = any(kw.lower() in last.lower() for kw in case["expected_keywords"])
            status = "✅ PASS" if match else "❌ FAIL"
            if match:
                passed += 1
        except Exception as e:
            elapsed = time.time() - start
            status = f"💥 ERROR: {e}"
            last = ""

        print(f"  [{case['id']}] {status} ({elapsed:.1f}s) — {case['description']}")
        if "FAIL" in status or "ERROR" in status:
            print(f"         Answer: {last[:100]}")
            print(f"         Expected one of: {case['expected_keywords']}")

    print(f"\n  Score: {passed}/{total} ({passed/total*100:.0f}%)")
    return passed, total


if __name__ == "__main__":
    # Run evaluation report when executed directly
    passed, total = run_evaluation_report()
    print(f"\nRun all tests: pytest lesson_14_testing/test_agents.py -v")
    print(f"Run fast only: pytest lesson_14_testing/test_agents.py -v -m 'not integration'")
