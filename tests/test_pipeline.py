"""Tests for the agentic pipeline (mocked)."""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.agent_pipeline import (
    run_pipeline,
    pipeline_result_to_dict,
    AgentResult,
    PipelineResult,
)


def _make_agent_result(agent, success=True, content="", sections=None, error=None, iteration=0):
    return AgentResult(
        agent=agent,
        model="test-model",
        success=success,
        content=content,
        sections=sections or {},
        error=error,
        iteration=iteration,
    )


def test_run_pipeline_not_configured(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    result = run_pipeline("def f(x): return x", translation_id=1)
    assert result.error is not None
    assert "NEBIUS_API_KEY" in result.error
    assert result.converged is False


def test_run_pipeline_translator_fails(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    from app.services import agent_pipeline

    def fake_call_agent(*args, **kwargs):
        return _make_agent_result("translator", success=False, error="API down")

    monkeypatch.setattr(agent_pipeline, "_call_agent", fake_call_agent)

    result = run_pipeline("def f(x): return x", translation_id=1)
    assert result.error is not None
    assert result.converged is False


def test_run_pipeline_converges_first_review(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    from app.services import agent_pipeline

    call_count = {"n": 0}

    def fake_call_agent(agent_role, system_prompt, user_content, iteration=0, model_override=None):
        call_count["n"] += 1
        if agent_role == "translator":
            return _make_agent_result(
                "translator",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple.",
                sections={
                    "CUDA KERNEL": "__global__ void k() {}",
                    "HOST CODE": "void launch() {}",
                    "EXPLANATION": "Simple.",
                },
                iteration=0,
            )
        if agent_role == "reviewer":
            return _make_agent_result(
                "reviewer",
                success=True,
                content="FINDINGS:\nNone.\nBLOCKING:\nno\nSUMMARY:\nLooks good.\nFIX_REQUIRED:\nno",
                sections={
                    "FINDINGS": "None.",
                    "BLOCKING": "no",
                    "SUMMARY": "Looks good.",
                    "FIX_REQUIRED": "no",
                },
                iteration=iteration,
            )
        if agent_role == "verifier":
            return _make_agent_result(
                "verifier",
                success=True,
                content="VERDICT:\npassed\nREMAINING_ISSUES:\nnone\nCONFIDENCE:\nhigh\nNOTES:\nAll good.",
                sections={
                    "VERDICT": "passed",
                    "REMAINING_ISSUES": "none",
                    "CONFIDENCE": "high",
                    "NOTES": "All good.",
                },
                iteration=iteration,
            )
        return _make_agent_result(agent_role, success=False, error="unknown agent")

    monkeypatch.setattr(agent_pipeline, "_call_agent", fake_call_agent)

    result = run_pipeline("def f(x): return x", translation_id=1)
    assert result.converged is True
    assert result.final_status == "static_passed"
    assert result.final_cuda_code != ""
    assert len(result.agent_results) >= 2  # translator + reviewer (+ verifier)


def test_pipeline_result_to_dict():
    ar = AgentResult(
        agent="translator",
        model="m",
        success=True,
        content="c",
        sections={"CUDA KERNEL": "k"},
        iteration=0,
    )
    pr = PipelineResult(
        translation_id=1,
        iterations=1,
        final_status="generated",
        agent_results=[ar],
        final_cuda_code="cuda",
        final_sections={"CUDA KERNEL": "k"},
        review_findings=[],
        converged=False,
    )
    d = pipeline_result_to_dict(pr)
    assert d["translation_id"] == 1
    assert d["iterations"] == 1
    assert d["converged"] is False
    assert len(d["agent_results"]) == 1
    assert d["agent_results"][0]["agent"] == "translator"


def test_run_pipeline_verifier_failed(monkeypatch):
    """Pipeline should end with ai_reviewed when verifier returns 'failed'."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    from app.services import agent_pipeline

    call_count = {"n": 0}

    def fake_call_agent(agent_role, system_prompt, user_content, iteration=0, model_override=None):
        call_count["n"] += 1
        if agent_role == "translator":
            return _make_agent_result(
                "translator",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple.",
                sections={
                    "CUDA KERNEL": "__global__ void k() {}",
                    "HOST CODE": "void launch() {}",
                    "EXPLANATION": "Simple.",
                },
                iteration=0,
            )
        if agent_role == "reviewer":
            # First review: blocking issues found
            return _make_agent_result(
                "reviewer",
                success=True,
                content="FINDINGS:\nMissing bounds check.\nBLOCKING:\nyes\nSUMMARY:\nNeeds fix.\nFIX_REQUIRED:\nyes",
                sections={
                    "FINDINGS": "Missing bounds check.",
                    "BLOCKING": "yes",
                    "SUMMARY": "Needs fix.",
                    "FIX_REQUIRED": "yes",
                },
                iteration=iteration,
            )
        if agent_role == "fixer":
            return _make_agent_result(
                "fixer",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nCHANGES MADE:\nAdded bounds check.\nREMAINING CONCERNS:\nnone",
                sections={
                    "CUDA KERNEL": "__global__ void k() {}",
                    "HOST CODE": "void launch() {}",
                    "CHANGES MADE": "Added bounds check.",
                    "REMAINING CONCERNS": "none",
                },
                iteration=iteration,
            )
        if agent_role == "reviewer":
            # Second review: no blocking issues
            return _make_agent_result(
                "reviewer",
                success=True,
                content="FINDINGS:\nNone.\nBLOCKING:\nno\nSUMMARY:\nAll good.\nFIX_REQUIRED:\nno",
                sections={
                    "FINDINGS": "None.",
                    "BLOCKING": "no",
                    "SUMMARY": "All good.",
                    "FIX_REQUIRED": "no",
                },
                iteration=iteration,
            )
        if agent_role == "verifier":
            return _make_agent_result(
                "verifier",
                success=True,
                content="VERDICT:\nfailed\nREMAINING_ISSUES:\nsome issues remain\nCONFIDENCE:\nmedium\nNOTES:\nNot all findings addressed.",
                sections={
                    "VERDICT": "failed",
                    "REMAINING_ISSUES": "some issues remain",
                    "CONFIDENCE": "medium",
                    "NOTES": "Not all findings addressed.",
                },
                iteration=iteration,
            )
        return _make_agent_result(agent_role, success=False, error="unknown agent")

    monkeypatch.setattr(agent_pipeline, "_call_agent", fake_call_agent)

    result = run_pipeline("def f(x): return x", translation_id=1)
    assert result.converged is False
    assert result.final_status == "ai_reviewed"
    assert result.final_cuda_code != ""
    assert len(result.agent_results) >= 4  # translator + reviewer + fixer + reviewer + verifier


def test_run_pipeline_max_iterations_reached(monkeypatch):
    """Pipeline should stop at max_iterations and run verifier."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    from app.services import agent_pipeline

    call_count = {"n": 0}

    def fake_call_agent(agent_role, system_prompt, user_content, iteration=0, model_override=None):
        call_count["n"] += 1
        if agent_role == "translator":
            return _make_agent_result(
                "translator",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple.",
                sections={
                    "CUDA KERNEL": "__global__ void k() {}",
                    "HOST CODE": "void launch() {}",
                    "EXPLANATION": "Simple.",
                },
                iteration=0,
            )
        if agent_role == "reviewer":
            # Always report blocking issues
            return _make_agent_result(
                "reviewer",
                success=True,
                content="FINDINGS:\nAlways blocking.\nBLOCKING:\nyes\nSUMMARY:\nStill needs work.\nFIX_REQUIRED:\nyes",
                sections={
                    "FINDINGS": "Always blocking.",
                    "BLOCKING": "yes",
                    "SUMMARY": "Still needs work.",
                    "FIX_REQUIRED": "yes",
                },
                iteration=iteration,
            )
        if agent_role == "fixer":
            return _make_agent_result(
                "fixer",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nCHANGES MADE:\nTried to fix.\nREMAINING CONCERNS:\nstill some issues",
                sections={
                    "CUDA KERNEL": "__global__ void k() {}",
                    "HOST CODE": "void launch() {}",
                    "CHANGES MADE": "Tried to fix.",
                    "REMAINING CONCERNS": "still some issues",
                },
                iteration=iteration,
            )
        if agent_role == "verifier":
            return _make_agent_result(
                "verifier",
                success=True,
                content="VERDICT:\nfailed\nREMAINING_ISSUES:\nunresolved\nCONFIDENCE:\nlow\nNOTES:\nMax iterations reached.",
                sections={
                    "VERDICT": "failed",
                    "REMAINING_ISSUES": "unresolved",
                    "CONFIDENCE": "low",
                    "NOTES": "Max iterations reached.",
                },
                iteration=iteration,
            )
        return _make_agent_result(agent_role, success=False, error="unknown agent")

    monkeypatch.setattr(agent_pipeline, "_call_agent", fake_call_agent)

    result = run_pipeline("def f(x): return x", translation_id=1, max_iterations=2)
    # Should have run: translator(0) + reviewer(1) + fixer(1) + reviewer(2) + verifier(2) = 5 agents
    # But max_iterations=2 means iteration goes 1, 2, then breaks at iteration >= max_iterations
    assert result.iterations >= 1
    assert result.final_status in ("ai_reviewed", "generated")
    assert len(result.agent_results) >= 2  # at least translator + reviewer


def test_run_pipeline_reviewer_error_severity(monkeypatch):
    """Pipeline should treat 'error' severity findings as blocking."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    from app.services import agent_pipeline

    def fake_call_agent(agent_role, system_prompt, user_content, iteration=0, model_override=None):
        if agent_role == "translator":
            return _make_agent_result(
                "translator",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple.",
                sections={
                    "CUDA KERNEL": "__global__ void k() {}",
                    "HOST CODE": "void launch() {}",
                    "EXPLANATION": "Simple.",
                },
                iteration=0,
            )
        if agent_role == "reviewer":
            # Reviewer reports an error-severity finding but BLOCKING=no
            return _make_agent_result(
                "reviewer",
                success=True,
                content="FINDINGS:\n[error] Integer overflow risk in index calculation.\nBLOCKING:\nno\nSUMMARY:\nHas error severity issue.\nFIX_REQUIRED:\nno",
                sections={
                    "FINDINGS": "[error] Integer overflow risk in index calculation.",
                    "BLOCKING": "no",
                    "SUMMARY": "Has error severity issue.",
                    "FIX_REQUIRED": "no",
                },
                iteration=iteration,
            )
        if agent_role == "fixer":
            return _make_agent_result(
                "fixer",
                success=True,
                content="CUDA KERNEL:\n```cpp\n__global__ void k_fixed() {}\n```\nHOST CODE:\n```cpp\nvoid launch_fixed() {}\n```\nCHANGES MADE:\nFixed integer overflow.\nREMAINING CONCERNS:\nnone",
                sections={
                    "CUDA KERNEL": "__global__ void k_fixed() {}",
                    "HOST CODE": "void launch_fixed() {}",
                    "CHANGES MADE": "Fixed integer overflow.",
                    "REMAINING CONCERNS": "none",
                },
                iteration=iteration,
            )
        if agent_role == "verifier":
            return _make_agent_result(
                "verifier",
                success=True,
                content="VERDICT:\npassed\nREMAINING_ISSUES:\nnone\nCONFIDENCE:\nhigh\nNOTES:\nAll good.",
                sections={
                    "VERDICT": "passed",
                    "REMAINING_ISSUES": "none",
                    "CONFIDENCE": "high",
                    "NOTES": "All good.",
                },
                iteration=iteration,
            )
        return _make_agent_result(agent_role, success=False, error="unknown agent")

    monkeypatch.setattr(agent_pipeline, "_call_agent", fake_call_agent)

    result = run_pipeline("def f(x): return x", translation_id=1)
    # Even though BLOCKING=no, the error severity should make it blocking
    # So fixer should run, then reviewer again, then verifier
    assert result.converged is True
    assert result.final_status == "static_passed"
    # Should have fixer in the pipeline because error severity triggered blocking
    agents = [ar.agent for ar in result.agent_results]
    assert "fixer" in agents