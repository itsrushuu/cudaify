"""End-to-end tests using FastAPI TestClient (mocked AI)."""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app


import time

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