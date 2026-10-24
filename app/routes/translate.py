"""Code translation routes with agentic self-healing pipeline."""

import json
from typing import Any, Dict, List, Optional
from dataclasses import asdict
from fastapi import APIRouter, Form, HTTPException, Depends, Path
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models import (
    save_translation,
    translation_to_dict,
    get_translation,
    update_verification,
    update_pipeline,
    VERIFICATION_STATUSES,
)
from app.services.agent_pipeline import run_pipeline, pipeline_result_to_dict
from app.services.code_review import review_cuda_code
from app.services.nemotron import model_for, translate_to_cuda, is_configured
from app.services.cuda_ir import build_ir, compare_ir_with_cuda, ir_summary
from app.services.auto_tune import generate_tuning_recommendations
from app.services.semantic_equiv import (
    run_equivalence_tests,
    equivalence_result_to_dict,
    TestInputGenerator,
    ReferenceExecutor,
    arrays_close,
    save_equivalence_result,
)
from app.routes.auth import get_current_user

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Maximum source code size in characters
MAX_SOURCE_LENGTH = 50_000


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    """Authenticate via Bearer token and return the user id."""
    user = get_current_user(credentials)
    return user["id"]


def _build_translation_response(
    *,
    tid: int,
    result: str,
    model: str,
    sections: Dict[str, Any],
    verification_status: str,
    ir: Any,
    ir_comparison: Dict[str, Any],
    tuning: Dict[str, Any],
    pipeline: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Build a consistent JSON response for translation endpoints."""
    payload = {
        "id": tid,
        "result": result,
        "model": model,
        "sections": sections or {},
        "verification_status": verification_status,
        "ir": ir.to_dict() if hasattr(ir, "to_dict") else {},
        "ir_comparison": ir_comparison,
        "ir_summary": ir_summary(ir),
        "tuning": tuning,
    }
    if pipeline is not None:
        payload["pipeline"] = pipeline
    return JSONResponse(payload)


@router.post("")
def translate(
    source_code: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    user_id: int = Depends(get_current_user_id),
    use_pipeline: str = Form("true"),
):
    """Translate Python/NumPy code to CUDA via the agentic self-healing pipeline.

    If use_pipeline="true" (default), runs the full Translator→Reviewer→Fixer→
    Verifier loop. Otherwise performs a single-shot translation.
    """
    # --- Input validation ---
    if not source_code or not source_code.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty.")

    if len(source_code) > MAX_SOURCE_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Source code too large (max {MAX_SOURCE_LENGTH:,} characters).",
        )

    # --- AI provider availability ---
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "AI provider not configured. Set NEBIUS_API_KEY in .env "
                "and restart the server."
            ),
        )

    source = source_code.strip()

    # Build CUDA Intermediate Representation from source
    ir = build_ir(source)
    ir_json = json.dumps(ir.to_dict(), ensure_ascii=False)

    if use_pipeline.lower() == "true":
        # --- Agentic self-healing pipeline ---
        result = run_pipeline(source, translation_id=0, model=model)

        if result.error and not result.agent_results:
            raise HTTPException(status_code=502, detail=result.error)

        # Compare IR with generated CUDA code
        ir_comparison = compare_ir_with_cuda(ir, result.final_cuda_code)
        ir_score = ir_comparison["score"]

        # Save the initial translation record
        model_used = (
            result.agent_results[0].model
            if result.agent_results
            else (model or model_for("translator"))
        )

        tid = save_translation(
            user_id=user_id,
            source_code=source,
            cuda_code=result.final_cuda_code,
            explanation=result.final_sections.get("EXPLANATION"),
            model_used=model_used,
            verification_status=result.final_status,
            review_findings=json.dumps(result.review_findings, ensure_ascii=False) if result.review_findings else None,
            sections_json=json.dumps(result.final_sections, ensure_ascii=False) if result.final_sections else None,
            ir_json=ir_json,
            ir_cuda_match_score=ir_score,
            pipeline_iterations=result.iterations,
            agent_results_json=json.dumps(
                [asdict(ar) for ar in result.agent_results],
                ensure_ascii=False,
            ) if result.agent_results else None,
            pipeline_converged=1 if result.converged else 0,
        )

        # Update with the actual translation id
        update_pipeline(
            translation_id=tid,
            pipeline_iterations=result.iterations,
            agent_results_json=json.dumps(
                [asdict(ar) for ar in result.agent_results],
                ensure_ascii=False,
            ) if result.agent_results else None,
            pipeline_converged=1 if result.converged else 0,
            verification_status=result.final_status,
            review_findings=json.dumps(result.review_findings, ensure_ascii=False) if result.review_findings else None,
            ir_json=ir_json,
            ir_cuda_match_score=ir_score,
        )

        return _build_translation_response(
            tid=tid,
            result=result.final_cuda_code,
            model=model_used,
            sections=result.final_sections,
            verification_status=result.final_status,
            ir=ir,
            ir_comparison=ir_comparison,
            tuning=generate_tuning_recommendations(ir, result.final_cuda_code),
            pipeline=pipeline_result_to_dict(result),
        )
    else:
        # --- Single-shot translation (legacy mode) ---
        trans_result = translate_to_cuda(source, model=model)

        if "error" in trans_result:
            raise HTTPException(status_code=502, detail=trans_result["error"])

        sections = trans_result.get("sections", {})
        sections_json = json.dumps(sections, ensure_ascii=False) if sections else None
        explanation = sections.get("EXPLANATION")

        # Compare IR with generated CUDA code
        ir_comparison = compare_ir_with_cuda(ir, trans_result["content"])
        ir_score = ir_comparison["score"]

        tid = save_translation(
            user_id=user_id,
            source_code=source,
            cuda_code=trans_result["content"],
            explanation=explanation,
            model_used=trans_result["model"],
            verification_status="generated",
            sections_json=sections_json,
            ir_json=ir_json,
            ir_cuda_match_score=ir_score,
        )

        return _build_translation_response(
            tid=tid,
            result=trans_result["content"],
            model=trans_result["model"],
            sections=sections,
            verification_status="generated",
            ir=ir,
            ir_comparison=ir_comparison,
            tuning=generate_tuning_recommendations(ir, trans_result["content"]),
        )


@router.get("/{translation_id}")
def get_translation_endpoint(
    translation_id: int = Path(..., ge=1),
    user_id: int = Depends(get_current_user_id),
):
    """Fetch a single translation by id (scoped to the authenticated user)."""
    row = get_translation(translation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Translation not found.")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this translation.")

    result = translation_to_dict(row)
    if result.get("sections_json"):
        try:
            result["sections"] = json.loads(result["sections_json"])
        except (json.JSONDecodeError, TypeError):
            result["sections"] = {}
    if result.get("agent_results_json"):
        try:
            result["agent_results"] = json.loads(result["agent_results_json"])
        except (json.JSONDecodeError, TypeError):
            result["agent_results"] = []
    if result.get("ir_json"):
        try:
            result["ir"] = json.loads(result["ir_json"])
        except (json.JSONDecodeError, TypeError):
            result["ir"] = {}
    return result


@router.post("/{translation_id}/verify")
def verify_translation(
    translation_id: int = Path(..., ge=1),
    verification_status: str = Form(...),
    review_findings: Optional[str] = Form(None),
    user_id: int = Depends(get_current_user_id),
):
    """Update the verification status of a translation.

    Valid statuses: generated, ai_reviewed, static_passed, compiled,
    runtime_tested, benchmarked.
    """
    if verification_status not in VERIFICATION_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid verification status. Valid values: {', '.join(VERIFICATION_STATUSES)}",
        )

    row = get_translation(translation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Translation not found.")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to verify this translation.")

    updated = update_verification(translation_id, verification_status, review_findings)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update verification status.")

    return JSONResponse(
        {
            "id": translation_id,
            "verification_status": verification_status,
            "review_findings": review_findings,
        }
    )


@router.post("/{translation_id}/review")
def review_translation(
    translation_id: int = Path(..., ge=1),
    user_id: int = Depends(get_current_user_id),
):
    """Run independent static code review on a translation's CUDA code.

    Findings are AI-generated static analysis — not compiler diagnostics
    and not runtime verification.
    """
    row = get_translation(translation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Translation not found.")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to review this translation.")

    cuda_code = row["cuda_code"] or ""
    review = review_cuda_code(cuda_code)

    # Store findings as JSON in review_findings and bump status to ai_reviewed.
    findings_json = json.dumps(review["findings"], ensure_ascii=False)
    update_verification(translation_id, "ai_reviewed", findings_json)

    return JSONResponse(
        {
            "id": translation_id,
            "verification_status": "ai_reviewed",
            "review": review,
        }
    )


@router.post("/{translation_id}/equiv")
def run_equivalence(
    translation_id: int = Path(..., ge=1),
    user_id: int = Depends(get_current_user_id),
):
    """Run semantic equivalence tests against the Python reference.

    Generates test inputs for corpus workloads, runs the Python reference
    implementation, and compares against expected outputs. Results are
    stored and the verification status is updated to 'runtime_tested'
    when all cases pass and CUDA is available, or 'static_passed' when
    only reference tests pass.

    Since CUDA cannot be compiled/executed locally (no nvcc), this
    validates the test framework itself. When CUDA becomes available,
    the same test cases can compare CUDA output against the reference.
    """
    row = get_translation(translation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Translation not found.")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to run equivalence tests on this translation.")

    # Check if nvcc/CUDA is available
    cuda_available = False
    try:
        import subprocess
        result = subprocess.run(["nvcc", "--version"], capture_output=True, check=True, timeout=5)
        cuda_available = result.returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        cuda_available = False

    result = run_equivalence_tests(cuda_available=cuda_available)

    # Save results to the translation record
    save_equivalence_result(translation_id, result)

    return JSONResponse(
        {
            "id": translation_id,
            "equivalence": equivalence_result_to_dict(result),
            "cuda_available": cuda_available,
        }
    )
