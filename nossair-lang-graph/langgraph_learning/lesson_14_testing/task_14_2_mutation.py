"""Task 14.2 — Mutation Testing."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Original functions (mirroring Lesson 4 tools)
def original_add(a, b): return a + b
def original_multiply(a, b): return a * b
def original_weather(city):
    db = {"london": "Cloudy, 15°C", "paris": "Sunny, 22°C"}
    return db.get(city.lower(), "No data")

# Mutant generators
def mutate_add_sub(a, b): return a - b          # + → -
def mutate_add_zero(a, b): return 0             # → constant 0
def mutate_mul_div(a, b): return a / b if b else 0  # * → /
def mutate_weather_empty(city): return "Unknown" # → always unknown

def run_test(func, args, expected, label, expect_kill=False):
    try:
        result = func(*args)
        if result == expected:
            if expect_kill:
                print(f"  ❌ Mutant SURVIVED: {label}")
                return "survived"
            else:
                print(f"  ✅ Original passed: {label}")
                return "passed"
        else:
            if expect_kill:
                print(f"  ✅ Mutant KILLED: {label} (got {result}, expected {expected})")
                return "killed"
            else:
                print(f"  ❌ Original FAILED: {label}")
                return "failed"
    except Exception as e:
        if expect_kill:
            print(f"  ✅ Mutant KILLED (exception): {label} — {e}")
            return "killed"
        else:
            print(f"  ❌ Original FAILED (exception): {label}")
            return "failed"

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 14.2 — MUTATION TESTING")
    print("=" * 50)

    killed = survived = 0

    # Test originals
    print("\n--- Original functions ---")
    run_test(original_add, (2, 3), 5, "add(2,3)=5")
    run_test(original_multiply, (4, 5), 20, "mul(4,5)=20")
    run_test(original_weather, ("London",), "Cloudy, 15°C", "weather(London)")

    # Test mutants
    print("\n--- Mutant 1: add → subtract ---")
    r = run_test(mutate_add_sub, (2, 3), 5, "add(2,3)=5", expect_kill=True)
    if r == "killed": killed += 1
    else: survived += 1

    print("\n--- Mutant 2: add → constant 0 ---")
    r = run_test(mutate_add_zero, (2, 3), 5, "add(2,3)=5", expect_kill=True)
    if r == "killed": killed += 1
    else: survived += 1

    print("\n--- Mutant 3: multiply → divide ---")
    r = run_test(mutate_mul_div, (4, 5), 20, "mul(4,5)=20", expect_kill=True)
    if r == "killed": killed += 1
    else: survived += 1

    print("\n--- Mutant 4: weather → always unknown ---")
    r = run_test(mutate_weather_empty, ("London",), "Cloudy, 15°C", "weather(London)", expect_kill=True)
    if r == "killed": killed += 1
    else: survived += 1

    total = killed + survived
    score = (killed / total * 100) if total else 0
    print(f"\n{'=' * 50}")
    print(f"Mutation score: {score:.0f}% ({killed}/{total} killed)")
    if score >= 80:
        print("🎯 Good mutation coverage!")
    else:
        print("⚠️ Add more tests to kill surviving mutants")
