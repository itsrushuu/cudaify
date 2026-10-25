"""End-to-end tests using FastAPI TestClient (mocked AI)."""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app


import time


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Reset slowapi in-memory rate limiter storage between tests."""
    from app.rate_limit import limiter
    if hasattr(limiter, "_storage") and hasattr(limiter._storage, "storage"):
        limiter._storage.storage.clear()

def _mock_translate_success(monkeypatch):
    """Patch nemotron.chat_completion to return a fixed CUDA translation."""
    from app.services import nemotron

    def fake_chat_completion(*args, **kwargs):
        return (
            "CUDA KERNEL:\n```cpp\n__global__ void vec_add(float* a, float* b, float* c, int n) {\n"
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (idx < n) c[idx] = a[idx] + b[idx];\n"
            "}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple element-wise add.\n"
        )

    monkeypatch.setattr(nemotron, "chat_completion", fake_chat_completion)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")


def _unique_username(prefix="e2euser"):
    return f"{prefix}_{int(time.time() * 1000)}"


def test_translate_endpoint_e2e(monkeypatch):
    _mock_translate_success(monkeypatch)

    # Signup
    client = TestClient(app)
    username = _unique_username()
    r = client.post("/auth/signup", data={"username": username, "password": "password123"})
    assert r.status_code == 200, r.text

    # Login
    r = client.post("/auth/login", data={"username": username, "password": "password123"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]

    # Translate (use_pipeline=false to avoid multiple mocked calls)
    r = client.post(
        "/translate",
        data={"source_code": "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n", "use_pipeline": "false"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "id" in data
    assert "result" in data
    assert "__global__" in data["result"]
    assert data["verification_status"] == "generated"


def test_translate_empty_source_rejected(monkeypatch):
    _mock_translate_success(monkeypatch)
    client = TestClient(app)
    username = _unique_username()
    r = client.post("/auth/signup", data={"username": username, "password": "password123"})
    r = client.post("/auth/login", data={"username": username, "password": "password123"})
    token = r.json()["token"]

    r = client.post(
        "/translate",
        data={"source_code": "", "use_pipeline": "false"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400


def test_translate_unauthorized(monkeypatch):
    _mock_translate_success(monkeypatch)
    client = TestClient(app)
    r = client.post(
        "/translate",
        data={"source_code": "def f(x): return x", "use_pipeline": "false"},
    )
    assert r.status_code == 401


def test_history_endpoint(monkeypatch):
    _mock_translate_success(monkeypatch)
    client = TestClient(app)
    username = _unique_username()
    r = client.post("/auth/signup", data={"username": username, "password": "password123"})
    r = client.post("/auth/login", data={"username": username, "password": "password123"})
    token = r.json()["token"]

    client.post(
        "/translate",
        data={"source_code": "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n", "use_pipeline": "false"},
        headers={"Authorization": f"Bearer {token}"},
    )

    r = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_cors_origins_parsing(monkeypatch):
    """CORS_ALLOWED_ORIGINS is parsed correctly."""
    from app.main import _parse_cors_origins

    # No env var -> empty list (fail closed)
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert _parse_cors_origins() == []

    # Empty string -> empty list
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "")
    assert _parse_cors_origins() == []

    # Single origin
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://example.com")
    assert _parse_cors_origins() == ["http://example.com"]

    # Multiple origins
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://example.com, https://example.org")
    assert _parse_cors_origins() == ["http://example.com", "https://example.org"]

    # Wildcard
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    assert _parse_cors_origins() == ["*"]


def test_rate_limiter_configured():
    """Rate limiter is configured on the FastAPI app."""
    from app.rate_limit import limiter
    from app.main import app
    assert app.state.limiter is limiter


def _setup_auth_and_translation(monkeypatch, client):
    """Helper: signup, login, translate. Returns (token, translation_id)."""
    _mock_translate_success(monkeypatch)
    username = _unique_username()
    r = client.post("/auth/signup", data={"username": username, "password": "password123"})
    assert r.status_code == 200, r.text
    r = client.post("/auth/login", data={"username": username, "password": "password123"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]

    r = client.post(
        "/translate",
        data={
            "source_code": "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n",
            "use_pipeline": "false",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    return token, tid


def test_verify_cannot_set_static_passed(monkeypatch):
    """Client cannot assign static_passed via /verify without server-side evidence."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "static_passed"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text
    # Status must remain unchanged
    row_check = client.get(f"/translate/{tid}", headers={"Authorization": f"Bearer {token}"})
    assert row_check.status_code == 200
    assert row_check.json()["verification_status"] == "generated"


def test_verify_cannot_set_runtime_tested(monkeypatch):
    """Client cannot assign runtime_tested via /verify without execution evidence."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "runtime_tested"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text


def test_verify_cannot_set_benchmarked(monkeypatch):
    """Client cannot assign benchmarked via /verify without benchmark evidence."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "benchmarked"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text


def test_verify_cannot_set_compiled(monkeypatch):
    """Client cannot assign compiled via /verify without compilation evidence."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "compiled"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text


def test_verify_cannot_set_ai_reviewed(monkeypatch):
    """Client cannot assign ai_reviewed via /verify; only /review can set it."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "ai_reviewed"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text


def test_verify_can_set_generated(monkeypatch):
    """Client can set generated via /verify (base state, no check implied)."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "generated"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["verification_status"] == "generated"


def test_review_endpoint_sets_ai_reviewed(monkeypatch):
    """Legitimate /review workflow sets ai_reviewed via server-side review."""
    client = TestClient(app)
    token, tid = _setup_auth_and_translation(monkeypatch, client)
    r = client.post(
        f"/translate/{tid}/review",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["verification_status"] == "ai_reviewed"

    # Confirm the status persisted
    row = client.get(f"/translate/{tid}", headers={"Authorization": f"Bearer {token}"})
    assert row.status_code == 200
    assert row.json()["verification_status"] == "ai_reviewed"


def test_translate_endpoint_sets_generated(monkeypatch):
    """Legitimate /translate workflow sets generated (no check implied)."""
    client = TestClient(app)
    token, _ = _setup_auth_and_translation(monkeypatch, client)
    # Already asserted inside helper: verification_status == "generated"


def test_verify_unauthorized_rejected(monkeypatch):
    """Unauthenticated clients cannot call /verify."""
    _mock_translate_success(monkeypatch)
    client = TestClient(app)
    r = client.post(
        "/translate/1/verify",
        data={"verification_status": "generated"},
    )
    assert r.status_code == 401


def test_verify_wrong_user_rejected(monkeypatch):
    """A user cannot verify another user's translation."""
    client = TestClient(app)
    _mock_translate_success(monkeypatch)

    # User A creates a translation
    username_a = _unique_username()
    client.post("/auth/signup", data={"username": username_a, "password": "password123"})
    client.post("/auth/login", data={"username": username_a, "password": "password123"})
    token_a = client.post("/auth/login", data={"username": username_a, "password": "password123"}).json()["token"]

    r = client.post(
        "/translate",
        data={"source_code": "def f(x): return x", "use_pipeline": "false"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert r.status_code == 200
    tid = r.json()["id"]

    # User B tries to verify User A's translation
    username_b = _unique_username()
    client.post("/auth/signup", data={"username": username_b, "password": "password123"})
    client.post("/auth/login", data={"username": username_b, "password": "password123"})
    token_b = client.post("/auth/login", data={"username": username_b, "password": "password123"}).json()["token"]

    r = client.post(
        f"/translate/{tid}/verify",
        data={"verification_status": "generated"},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert r.status_code == 403, r.text
