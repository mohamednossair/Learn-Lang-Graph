"""Task 14.3 — Evaluation Suite."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_CASES = [
    {"input": "What is 2 + 3?", "expected_contains": ["5", "five"], "should_use_tools": True},
    {"input": "What is the weather in Paris?", "expected_contains": ["sunny", "Sunny"], "should_use_tools": True},
    {"input": "What is 4 * 5?", "expected_contains": ["20", "twenty"], "should_use_tools": True},
    {"input": "Hello, how are you?", "expected_contains": [], "should_use_tools": False},
]

def run_suite(graph, verbose=True):
    passed = failed = 0
    if verbose: print("=" * 50 + "\nTASK 14.3 — EVALUATION SUITE\n" + "=" * 50)
    for test in TEST_CASES:
        result = graph.invoke({"messages": [{"role": "user", "content": test["input"]}]})
        response = result["messages"][-1].content.lower()
        ok = all(kw.lower() in response for kw in test["expected_contains"]) if test["expected_contains"] else len(response) > 0
        if ok: passed += 1; print(f"✅ {test['input'][:40]}...")
        else: failed += 1; print(f"❌ {test['input'][:40]}... Got: {response[:50]}")
    print(f"\nResults: {passed}/{len(TEST_CASES)} passed")
    return {"passed": passed, "failed": failed}

if __name__ == "__main__":
    from lesson_04_tools_agent.lesson_04_tools_agent import graph
    run_suite(graph)
