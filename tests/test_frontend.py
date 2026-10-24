"""Frontend smoke tests using Streamlit's AppTest (no real backend, no network)."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

SRC = "import numpy as np\n\ndef f(x, y):\n    return 2.0 * x + np.sin(y)\n"
K0 = "__global__ void k(int n){ int i = 0; }"
K1 = "__global__ void k(int n){ int i = 0; if (i < n) {} }"
PIPELINE = {
    "final_status": "ai_reviewed", "iterations": 2, "converged": True, "final_cuda_code": K1,
    "review_findings": [{"severity": "high", "category": "bounds", "message": "missing guard"}],
    "references": [{"title": "CUDA Best Practices Guide", "url": "https://docs.nvidia.com/cuda/x", "snippet": "s"}],
    "agent_results": [
        {"agent": "translator", "model": "m", "success": True, "content": "x",
         "sections": {"CUDA KERNEL": K0}, "iteration": 0},
        {"agent": "verifier", "model": "m", "success": True, "content": "x",
         "sections": {"VERDICT": "passed"}, "iteration": 2},
    ],
}


def _result(with_pipeline=True):
    res = {"id": 1, "result": K1, "model": "m", "sections": {"CUDA KERNEL": K1},
           "verification_status": "ai_reviewed" if with_pipeline else "generated",
           "ir_summary": "", "tuning": {}, "ir": {}, "ir_comparison": {}}
    if with_pipeline:
        res["pipeline"] = PIPELINE
    return res


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.ok = status, payload, status < 400

    def json(self):
        return self._p

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


@pytest.fixture
def app(monkeypatch):
    import requests

    monkeypatch.setenv("BACKEND_URL", "http://backend.test")     # skip the embedded backend
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp(200, []))

    def make(result):
        at = AppTest.from_file("../streamlit_app.py", default_timeout=60)
        at.session_state["token"] = "t"
        at.session_state["user"] = {"id": 1, "username": "tester"}
        at.session_state["last_result"] = result
        at.session_state["last_source"] = SRC
        return at

    return make


def test_result_panels_render(app):
    at = app(_result()).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "What was verified" in md and "Compiled with nvcc" in md and "nvcc -O2" in md
    assert "CUDA Best Practices Guide" in md                       # Tavily source shown
    labels = [e.label for e in at.expander]
    assert any("review/fix loop changed" in lbl for lbl in labels)
    assert any("actually need custom CUDA" in lbl for lbl in labels)
    assert "diff" in [c.language for c in at.code]


def test_renders_without_pipeline(app):
    at = app(_result(with_pipeline=False)).run()
    assert not at.exception
    assert not any("review/fix loop changed" in e.label for e in at.expander)
    assert any("actually need custom CUDA" in e.label for e in at.expander)


def test_expired_session_returns_to_login(app, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp(401, {"detail": "expired"}))
    at = app(_result()).run()
    assert not at.exception                                         # used to crash with KeyError
    assert any("Login" in t.label for t in at.tabs)
