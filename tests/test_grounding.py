"""Tests for optional Tavily grounding (no network)."""
import requests

from app.services import agent_pipeline as ap
from app.services import grounding


class Resp:
    def __init__(self, payload, ok=True):
        self._p, self._ok = payload, ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.exceptions.HTTPError("boom")

    def json(self):
        return self._p


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append(1))
    assert grounding.fetch_references("x = 1") == []
    assert not called                                  # no network call at all


def test_fetch_references_parses_and_limits(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return Resp({"results": [
            {"title": "A", "url": "https://docs.nvidia.com/a", "content": "x" * 900},
            {"title": "", "url": "https://developer.nvidia.com/b", "content": "y"},
            {"title": "no url", "url": "", "content": "z"},
        ]})

    monkeypatch.setattr(requests, "post", fake_post)
    refs = grounding.fetch_references("for i in range(3):\n    out[i] = x[i-1] + x[i+1]\n")
    assert seen["headers"]["Authorization"] == "Bearer tvly-test"
    assert seen["body"]["include_domains"] == grounding.DOC_DOMAINS
    assert "stencil" in seen["body"]["query"]
    assert [r["url"] for r in refs] == ["https://docs.nvidia.com/a", "https://developer.nvidia.com/b"]
    assert len(refs[0]["snippet"]) == grounding.MAX_SNIPPET_CHARS
    assert refs[1]["title"] == "https://developer.nvidia.com/b"      # falls back to the URL


def test_fetch_references_swallows_errors(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp({}, ok=False))
    assert grounding.fetch_references("x") == []

    def timeout(*a, **k):
        raise requests.exceptions.Timeout()
    monkeypatch.setattr(requests, "post", timeout)
    assert grounding.fetch_references("x") == []


def test_format_for_prompt():
    assert grounding.format_for_prompt([]) == ""
    txt = grounding.format_for_prompt([{"title": "T", "url": "https://u", "snippet": "S"}])
    assert "[1] T (https://u): S" in txt and "never invent citations" in txt


def test_pipeline_passes_references_to_reviewer(monkeypatch):
    prompts = {}
    refs = [{"title": "Guide", "url": "https://docs.nvidia.com/g", "snippet": "use shared memory"}]
    monkeypatch.setattr(ap, "is_configured", lambda: True)
    monkeypatch.setattr(ap, "fetch_references", lambda src: refs)

    def fake_call(role, system_prompt, user_content, iteration=0, model_override=None, **kw):
        prompts[role] = user_content
        secs = {
            "translator": {"CUDA KERNEL": "__global__ void k(){}", "HOST CODE": "//"},
            "reviewer": {"FINDINGS": "none", "BLOCKING": "no", "SUMMARY": "ok", "FIX_REQUIRED": "no"},
            "verifier": {"VERDICT": "passed"},
        }[role]
        return ap.AgentResult(agent=role, model="m", success=True, content="x", sections=secs, iteration=iteration)

    monkeypatch.setattr(ap, "_call_agent", fake_call)
    result = ap.run_pipeline("def f(): pass", 1)
    assert "use shared memory" in prompts["reviewer"]
    assert result.references == refs
    assert ap.pipeline_result_to_dict(result)["references"] == refs
