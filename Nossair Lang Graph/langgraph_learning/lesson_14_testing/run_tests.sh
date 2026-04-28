#!/bin/bash
# Task 14.4 — CI Integration Script

echo "============================================"
echo "Running LangGraph Test Suite"
echo "============================================"

FAILED=0

# 1. Unit tests
echo ""
echo "1. Running unit tests..."
python lesson_14_testing/task_14_1_test.py || FAILED=1

# 2. Evaluation suite
echo ""
echo "2. Running evaluation suite..."
python lesson_14_testing/task_14_3_suite.py || FAILED=1

# 3. Syntax check all Python files
echo ""
echo "3. Syntax check..."
python -m py_compile lesson_04_tools_agent/lesson_04_tools_agent.py || FAILED=1
echo "   All files compile OK"

# Final result
echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    exit 1
fi
