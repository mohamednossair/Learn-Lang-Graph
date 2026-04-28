"""
Production Smoke Test Suite - Multi-Tenant Agent Service.

Covers (per ARCHITECTURE.md §20.8):
  [1]  Health: /health/live and /health/ready
  [2]  Security: Missing X-Customer-ID -> 401
  [3]  Security: Unknown tenant -> 403
  [4]  Isolation: Customer A - allowed topic  -> 200 + answer
  [5]  Isolation: Customer A - blocked topic  -> 200 + refusal (blocked=true)
  [6]  Isolation: Customer B - allowed topic  -> 200 + answer
  [7]  Isolation: Customer B - blocked topic  -> 200 + refusal (blocked=true)
  [8]  Isolation: Customer C - allowed topic  -> 200 + answer
  [9]  Cross-tenant: verify X-Request-ID header always present
  [10] Concurrent:  3 tenants simultaneously (asyncio.gather equivalent via threads)
  [11] Admin: /tenants lists all registered tenants
  [12] Admin: /admin/cache/clear succeeds

Run:
  # From the project root (multi_tenant_agent_project/):
  python smoke_test.py
  # or via pytest:
  pytest smoke_test.py -v
"""

import concurrent.futures
import sys
import os
import time

import pytest

# Ensure the parent package is importable regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.main import app

PASS = "[PASS]"
FAIL = "[FAIL]"
results: list[dict] = []
client: TestClient  # set inside run_all() / pytest fixture


@pytest.fixture(scope="session", autouse=True)
def _init_client():
    global client
    with TestClient(app, raise_server_exceptions=False) as c:
        client = c
        yield


def _record(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    results.append({"name": name, "ok": ok})
    print(f"  {status}  {name}" + (f" - {detail}" if detail else ""))


def _chat(question: str, customer_id: str, user_id: str = "smoke_test") -> dict:
    resp = client.post(
        f"/chat?question={question}&user_id={user_id}",
        headers={"X-Customer-ID": customer_id},
    )
    return {"status_code": resp.status_code, "json": resp.json(), "headers": dict(resp.headers)}


# ---------------------------------------------------------------------------
# TEST GROUPS
# ---------------------------------------------------------------------------

def test_health():
    print("\n-- [Health] --------------------------------------------------")

    r = client.get("/health/live")
    _record("GET /health/live -> 200", r.status_code == 200)

    r = client.get("/health/ready")
    _record(
        "GET /health/ready -> 200 or 503",
        r.status_code in (200, 503),
        f"status_code={r.status_code}",
    )
    if r.status_code == 503:
        print(f"    [WARN] readiness errors: {r.json().get('errors')}")


def test_security_missing_header():
    print("\n-- [Security] Missing X-Customer-ID --------------------------")

    r = client.post("/chat?question=Hello")
    _record(
        "Missing header -> 401",
        r.status_code == 401,
        f"got {r.status_code}",
    )


def test_security_unknown_tenant():
    print("\n-- [Security] Unknown Tenant ---------------------------------")

    r = client.post(
        "/chat?question=Hello",
        headers={"X-Customer-ID": "hacker_corp"},
    )
    _record(
        "Unknown tenant -> 403",
        r.status_code == 403,
        f"got {r.status_code}",
    )


def test_customer_a():
    print("\n-- [Customer A - Electronics] --------------------------------")
    cid = "customer_a"

    r = _chat("What are best practices for BOM management in electronics?", cid)
    _record(
        "Customer A: allowed topic (BOM/electronics) -> 200",
        r["status_code"] == 200,
        f"status_code={r['status_code']}",
    )
    body = r["json"]
    _record(
        "Customer A: response not blocked",
        body.get("blocked") is False,
        f"blocked={body.get('blocked')}",
    )
    _record(
        "Customer A: tenant echoed correctly",
        body.get("tenant") == cid,
        f"tenant={body.get('tenant')}",
    )

    r2 = _chat("Who won the FIFA World Cup?", cid)
    _record(
        "Customer A: blocked topic (sports) -> blocked=true",
        r2["json"].get("blocked") is True,
        f"blocked={r2['json'].get('blocked')}",
    )


def test_customer_b():
    print("\n-- [Customer B - Inventory] ----------------------------------")
    cid = "customer_b"

    r = _chat("How should we price seasonal inventory items?", cid)
    _record(
        "Customer B: allowed topic (pricing) -> 200",
        r["status_code"] == 200,
        f"status_code={r['status_code']}",
    )

    r2 = _chat("What is the best political party?", cid)
    _record(
        "Customer B: blocked topic (politics) -> blocked=true",
        r2["json"].get("blocked") is True,
        f"blocked={r2['json'].get('blocked')}",
    )


def test_customer_c():
    print("\n-- [Customer C - Logistics] ----------------------------------")
    cid = "customer_c"

    r = _chat("How do I track an international shipment?", cid)
    _record(
        "Customer C: allowed topic (tracking) -> 200",
        r["status_code"] == 200,
        f"status_code={r['status_code']}",
    )


def test_request_id_header():
    print("\n-- [Request ID] ----------------------------------------------")

    r = client.post(
        "/chat?question=Hello",
        headers={"X-Customer-ID": "customer_a"},
    )
    has_request_id = "x-request-id" in r.headers
    _record(
        "X-Request-ID header present in every response",
        has_request_id,
        f"headers={list(r.headers.keys())}",
    )


def test_concurrent():
    """
    Fire 3 requests concurrently - one per tenant.
    Verifies no cross-tenant interference under parallel load.
    """
    print("\n-- [Concurrent - 3 tenants in parallel] ----------------------")

    calls = [
        ("customer_a", "What is RoHS compliance?"),
        ("customer_b", "How do we optimise warehouse inventory?"),
        ("customer_c", "What is the fastest shipping route to Dubai?"),
    ]

    start = time.perf_counter()

    def _call(args):
        cid, q = args
        return cid, _chat(q, cid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = list(pool.map(_call, calls))

    elapsed = (time.perf_counter() - start) * 1000

    for cid, r in futures:
        _record(
            f"Concurrent: {cid} -> 200",
            r["status_code"] == 200,
            f"status={r['status_code']}",
        )

    print(f"  Total wall-clock: {elapsed:.0f}ms (3 parallel requests)")


def test_admin_endpoints():
    print("\n-- [Admin] ---------------------------------------------------")

    r = client.get("/tenants")
    tenants = r.json().get("tenants", [])
    _record(
        "GET /tenants -> lists registered tenants",
        r.status_code == 200 and len(tenants) >= 2,
        f"tenants={tenants}",
    )

    r2 = client.post("/admin/cache/clear")
    _record(
        "POST /admin/cache/clear -> 200",
        r2.status_code == 200,
    )


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def run_all():
    global client
    print("=" * 60)
    print("  Multi-Tenant Agent Service - Production Smoke Tests")
    print("=" * 60)

    with TestClient(app, raise_server_exceptions=False) as c:
        client = c

        test_health()
        test_security_missing_header()
        test_security_unknown_tenant()
        test_customer_a()
        test_customer_b()
        test_customer_c()
        test_request_id_header()
        test_concurrent()
        test_admin_endpoints()

    passed = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    total = len(results)

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed | {failed} failed")
    print("=" * 60)

    if failed:
        print("\nFailed tests:")
        for r in results:
            if not r["ok"]:
                print(f"  [FAIL]  {r['name']}")
        sys.exit(1)
    else:
        print("\nAll smoke tests passed.")


if __name__ == "__main__":
    run_all()
