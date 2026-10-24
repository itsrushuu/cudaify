"""Agentic self-healing CUDA translation pipeline.

Orchestrates multiple Nemotron agent roles in a feedback loop:

    Translator → Reviewer → Fixer → Reviewer → ... → Verifier

Each agent is a specialized Nemotron prompt. The pipeline iterates until
the reviewer reports no blocking issues or the maximum iteration count is
reached. Every stage is recorded in the database for transparency and
auditability.

This is the technical differentiator: most code-to-CUDA tools generate
once and ship. CUDAify iteratively self-heals its own output.
"""

import json
import re
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field

from app.services.grounding import fetch_references, format_for_prompt
from app.services.nemotron import (
    NemotronError,
    _SECTION_RE,
    _parse_sections,
    chat_completion,
    is_configured,
    model_for,
)
import requests  # noqa: F401  (kept so tests can patch pipeline.requests.post)

AGENT_CONFIGS = {
    "translator": {
        "temperature": 0.2,
        "max_tokens": None,
    },
    "reviewer": {
        "temperature": 0.1,
        "max_tokens": None,
    },
    "fixer": {
        "temperature": 0.3,
        "max_tokens": None,
    },
    "verifier": {
        "temperature": 0.1,
        "max_tokens": None,
    },
}

MAX_ITERATIONS = 5
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2


@dataclass
class AgentResult:
    agent: str
    model: str
    success: bool
    content: str
    sections: Dict[str, str]
    error: Optional[str] = None
    iteration: int = 0
    retry_count: int = 0


@dataclass
class PipelineResult:
    translation_id: int
    iterations: int
    final_status: str
    agent_results: List[AgentResult]
    final_cuda_code: str
    final_sections: Dict[str, str]
    review_findings: List[Dict[str, Any]]
    converged: bool
    error: Optional[str] = None
    references: List[Dict[str, str]] = field(default_factory=list)


TRANSLATOR_SYSTEM_PROMPT = """You are an expert CUDA programmer. Translate Python/NumPy code to optimized CUDA C++.

Return exactly these sections, each headed by the section name in ALL CAPS:

1. CUDA KERNEL: device kernel code in a ```cpp fenced block
2. HOST CODE: host launcher/wrapper code in a ```cpp fenced block
3. EXPLANATION: how the Python maps to parallel GPU execution
4. INPUT ASSUMPTIONS: shapes, ranks, contiguity, in-place modification
5. DATA TYPES: CUDA types for each input/output
6. BOUNDARY CONDITIONS: out-of-bounds guards, edge cases
7. MEMORY CONSIDERATIONS: coalescing, shared memory, bank conflicts
8. CORRECTNESS RISKS: overflow, race conditions, uninitialized memory, etc.
9. OPTIMIZATION RECOMMENDATIONS: specific optimizations, no measured speedups
10. LIMITATIONS: what this translation does NOT handle
11. VERIFICATION STATUS: Generated (not compiled, not runtime-tested)

Rules:
- Return ONLY these sections. No introduction, no conclusion.
- Use fenced code blocks (```cpp and ```) for all code.
- Do not invent performance numbers.
- Be honest about limitations and risks.
- If the code cannot be safely translated, say so in LIMITATIONS and return a minimal stub.
"""

REVIEWER_SYSTEM_PROMPT = """You are an expert CUDA correctness reviewer. Review the following generated CUDA C++ code for correctness issues.

Check for:
1. Thread-index bounds guarding (every global memory access guarded by if (idx < n))
2. Out-of-bounds memory access risks
3. Missing cudaDeviceSynchronize() after kernel launch
4. Missing CUDA API error checking (cudaGetLastError after cudaMalloc, cudaMemcpy, etc.)
5. Shared memory (__shared__) without __syncthreads() barriers
6. Potential race conditions (multiple threads writing same location without atomic)
7. Integer overflow in index calculations (blockIdx * blockDim without size_t cast)
8. Incomplete host-side code (missing kernel launch config, missing cudaMalloc/cudaFree)
9. Differences in numerical behavior from the original Python
10. Missing cudaFree for allocated device memory

Return your review in these sections:
1. FINDINGS: list of issues found, each with severity (error/warning/info), category, and description
2. BLOCKING: whether any issue is blocking - "yes" or "no"
3. SUMMARY: one-paragraph summary
4. FIX_REQUIRED: whether the code needs fixing before it can be considered correct - "yes" or "no"

Be thorough but concise. Label each finding clearly.
"""

FIXER_SYSTEM_PROMPT = """You are an expert CUDA programmer fixing correctness issues in generated CUDA C++ code.

You will receive:
1. The original Python/NumPy source code
2. The current CUDA code (kernel + host)
3. A list of review findings with categories and descriptions

Your job: produce corrected CUDA C++ that addresses ALL the review findings while
preserving the original computation's semantics.

Return exactly these sections:
1. CUDA KERNEL: corrected device kernel code in ```cpp block
2. HOST CODE: corrected host launcher/wrapper code in ```cpp block
3. CHANGES MADE: list of what you changed and why (reference review finding categories)
4. REMAINING CONCERNS: any issues you could not fix, or "none"
5. VERIFICATION STATUS: ai_reviewed (pending static checks)

Rules:
- Address every review finding. Do not ignore any.
- Preserve the original computation semantics.
- Use fenced code blocks for all code.
- Be honest about remaining concerns.
"""

VERIFIER_SYSTEM_PROMPT = """You are an expert CUDA verification engineer. Review the FIXED CUDA code against
the original review findings to confirm all issues were addressed.

You will receive:
1. The original Python/NumPy source code
2. The original review findings
3. The fixed CUDA code (kernel + host)
4. The changes made by the fixer

Determine:
1. VERDICT: "passed" or "failed" - did the fixer address all blocking findings?
2. REMAINING_ISSUES: any issues still present, or "none"
3. CONFIDENCE: "high", "medium", or "low"
4. NOTES: explanation of your verdict

Return these sections:
1. VERDICT
2. REMAINING_ISSUES
3. CONFIDENCE
4. NOTES

Be strict. If any blocking finding is unresolved, verdict is "failed".
"""


def _call_agent_with_retry(
    agent_role: str,
    system_prompt: str,
    user_content: str,
    iteration: int = 0,
    model_override: Optional[str] = None,
) -> AgentResult:
    """Call a single Nemotron agent with retry logic for transient failures."""
    if not is_configured():
        return AgentResult(
            agent=agent_role,
            model="",
            success=False,
            content="",
            sections={},
            error="NEBIUS_API_KEY not configured",
            iteration=iteration,
            retry_count=0,
        )

    config = AGENT_CONFIGS.get(agent_role, AGENT_CONFIGS["translator"])
    model = model_override or model_for(agent_role)

    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            content = chat_completion(
                model,
                system_prompt,
                user_content,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            return AgentResult(
                agent=agent_role,
                model=model,
                success=True,
                content=content,
                sections=_parse_sections(content),
                iteration=iteration,
                retry_count=attempt,
            )
        except NemotronError as exc:
            last_error = str(exc)
            # Don't retry on authentication errors (401) — they won't succeed
            if "401" in last_error or "unauthenticated" in last_error.lower():
                break
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            break

    return AgentResult(
        agent=agent_role,
        model=model,
        success=False,
        content="",
        sections={},
        error=last_error or f"{agent_role} agent failed after {MAX_RETRIES + 1} attempts",
        iteration=iteration,
        retry_count=MAX_RETRIES + 1,
    )


def _call_agent(
    agent_role: str,
    system_prompt: str,
    user_content: str,
    iteration: int = 0,
    model_override: Optional[str] = None,
) -> AgentResult:
    """Backward-compatible wrapper that delegates to the retry-enabled version."""
    return _call_agent_with_retry(
        agent_role, system_prompt, user_content, iteration, model_override
    )


def _extract_cuda_code(sections: Dict[str, str], raw_content: str = "") -> str:
    """Extract the full CUDA code (kernel + host) from parsed sections.

    Falls back to the raw content if no structured sections were found.
    """
    parts = []
    if sections.get("CUDA KERNEL"):
        parts.append("// CUDA KERNEL\n" + sections["CUDA KERNEL"])
    if sections.get("HOST CODE"):
        parts.append("// HOST CODE\n" + sections["HOST CODE"])

    if parts:
        return "\n\n".join(parts)

    # Fallback: return raw content if sections weren't parsed
    if raw_content.strip():
        return raw_content.strip()

    return ""


def _parse_verdict(vr: AgentResult) -> Optional[str]:
    """Parse the verifier's verdict with multiple fallback patterns."""
    texts_to_check = []
    if vr.sections.get("VERDICT"):
        texts_to_check.append(vr.sections["VERDICT"])
    if vr.content:
        texts_to_check.append(vr.content)

    for text in texts_to_check:
        # Pattern 1: "VERDICT: passed" or "VERDICT: failed"
        m = re.search(
            r"verdict\s*[:#\-]?\s*\"?(pass(?:ed)?|fail(?:ed)?)\b",
            text,
            re.I,
        )
        if m:
            return "passed" if m.group(1).lower().startswith("pass") else "failed"

        # Pattern 2: Just the word "passed" or "failed" on its own line
        m = re.search(
            r"^\s*(pass(?:ed)?|fail(?:ed)?)\s*$",
            text,
            re.I | re.M,
        )
        if m:
            return "passed" if m.group(1).lower().startswith("pass") else "failed"

    # Pattern 3: Check the stripped section value directly
    first = vr.sections.get("VERDICT", "").strip().lower().strip("*_\"' ")
    if first.startswith("pass"):
        return "passed"
    if first.startswith("fail"):
        return "failed"

    return None


def _detect_severity(line: str) -> str:
    """Detect severity from a finding line with multiple keyword patterns."""
    lower = line.lower()
    if any(kw in lower for kw in ["error", "critical", "severe", "fatal", "blocking"]):
        return "error"
    if any(kw in lower for kw in ["warning", "warn", "caution", "issue"]):
        return "warning"
    return "info"


def _detect_category(line: str) -> str:
    """Detect category from a finding line with multiple keyword patterns."""
    lower = line.lower()
    category_keywords = [
        ("bounds", ["bounds", "out.of.bounds", "index", "guard", "idx", "threadidx"]),
        ("synchronization", ["synchron", "syncthreads", "barrier", "cuda_device_sync"]),
        ("error_checking", ["error.check", "cuda_getlasterror", "cuda_success", "cudaapi"]),
        ("shared_memory", ["shared.memory", "__shared__", "shared mem"]),
        ("race_condition", ["race", "atomic", "concurrent", "write conflict"]),
        ("index_overflow", ["overflow", "32.bit", "size_t", "int64", "blockidx.*blockdim"]),
        ("memory", ["memory", "malloc", "memcpy", "cuda_free", "allocation"]),
    ]
    for cat, keywords in category_keywords:
        if any(re.search(kw, lower) for kw in keywords):
            return cat
    return "general"


def _parse_blocking_findings(
    review_sections: Dict[str, str],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Parse reviewer output to determine if there are blocking findings.

    Returns (is_blocking, findings_list).
    """
    findings_text = review_sections.get("FINDINGS", "")
    blocking_text = review_sections.get("BLOCKING", "").strip().lower()
    fix_required = review_sections.get("FIX_REQUIRED", "").strip().lower()
    summary_text = review_sections.get("SUMMARY", "").lower()

    findings: List[Dict[str, Any]] = []
    for line in findings_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip bullet markers and numbering
        clean_line = re.sub(r"^[-•\*\d\.\s]+", "", line).strip()
        if not clean_line:
            continue

        sev = _detect_severity(clean_line)
        cat = _detect_category(clean_line)

        findings.append({
            "severity": sev,
            "category": cat,
            "message": clean_line,
            "ai_generated": True,
        })

    # Determine blocking status from multiple signals
    explicit_blocking = blocking_text in ("yes", "true", "1")
    explicit_fix_required = fix_required in ("yes", "true", "1")
    has_error_severity = any(f["severity"] == "error" for f in findings)
    summary_indicates_issues = any(
        kw in summary_text for kw in ["error", "critical", "blocking", "failed"]
    )

    is_blocking = (
        explicit_blocking
        or explicit_fix_required
        or has_error_severity
        or summary_indicates_issues
    )

    return is_blocking, findings


def _has_valid_cuda_code(cuda_code: str) -> bool:
    """Check if the CUDA code looks valid (has at least a kernel or host function)."""
    if not cuda_code or not cuda_code.strip():
        return False
    lower = cuda_code.lower()
    return (
        "__global__" in lower
        or "__device__" in lower
        or "__host__" in lower
        or "void " in lower
        or "kernel" in lower
    )


def run_pipeline(
    source_code: str,
    translation_id: int,
    model: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> PipelineResult:
    """Run the full agentic self-healing pipeline.

    Stages:
      1. Translator generates CUDA from Python
      2. Reviewer reviews for correctness
      3. If blocking issues: Fixer fixes, Reviewer re-reviews (loop)
      4. Verifier gives final verdict

    Args:
        source_code: Python/NumPy source to translate.
        translation_id: Database id of the translation record.
        model: Optional model override.
        max_iterations: Maximum fix-review iterations.

    Returns:
        PipelineResult with all agent results and final state.
    """
    agent_results: List[AgentResult] = []
    iteration = 0
    converged = False
    final_cuda_code = ""
    final_sections: Dict[str, str] = {}
    review_findings: List[Dict[str, Any]] = []
    final_status = "generated"
    error: Optional[str] = None

    if not is_configured():
        return PipelineResult(
            translation_id=translation_id,
            iterations=0,
            final_status="generated",
            agent_results=[],
            final_cuda_code="",
            final_sections={},
            review_findings=[],
            converged=False,
            error="NEBIUS_API_KEY not configured",
        )

    # --- Stage 1: Translate ---
    translator_prompt = (
        f"Original Python/NumPy code:\n\n{source_code}\n\n"
        "Translate to CUDA C++ now."
    )
    tr = _call_agent("translator", TRANSLATOR_SYSTEM_PROMPT, translator_prompt, iteration=0)
    agent_results.append(tr)

    if not tr.success:
        error = tr.error or "Translation failed"
        return PipelineResult(
            translation_id=translation_id,
            iterations=iteration,
            final_status="generated",
            agent_results=agent_results,
            final_cuda_code="",
            final_sections={},
            review_findings=[],
            converged=False,
            error=error,
        )

    final_cuda_code = _extract_cuda_code(tr.sections, tr.content)
    final_sections = tr.sections
    final_status = "generated"
    iteration = 1

    # Optional Tavily grounding: NVIDIA doc excerpts for the Reviewer (empty if disabled).
    references = fetch_references(source_code)
    ref_block = format_for_prompt(references)

    # Validate that we got some CUDA code
    if not _has_valid_cuda_code(final_cuda_code):
        # Still proceed to review — the reviewer may catch this
        pass

    # --- Stage 2+: Review → Fix → Review loop ---
    while iteration <= max_iterations:
        review_prompt = (
            f"Original Python/NumPy code:\n\n{source_code}\n\n"
            f"CUDA code to review:\n\n{final_cuda_code}\n\n"
            f"{ref_block}"
            "Review this CUDA code for correctness issues now."
        )
        rv = _call_agent("reviewer", REVIEWER_SYSTEM_PROMPT, review_prompt, iteration=iteration)
        agent_results.append(rv)

        if not rv.success:
            error = rv.error or "Review failed"
            break

        is_blocking, findings = _parse_blocking_findings(rv.sections)
        review_findings = findings

        if not is_blocking:
            converged = True
            final_status = "ai_reviewed"
            break

        if iteration >= max_iterations:
            final_status = "ai_reviewed"
            break

        # --- Fix ---
        fix_prompt = (
            f"Original Python/NumPy code:\n\n{source_code}\n\n"
            f"Current CUDA code:\n\n{final_cuda_code}\n\n"
            f"Review findings to fix:\n\n{json.dumps(findings, indent=2)}\n\n"
            "Produce corrected CUDA code now."
        )
        fx = _call_agent("fixer", FIXER_SYSTEM_PROMPT, fix_prompt, iteration=iteration)
        agent_results.append(fx)

        if not fx.success:
            error = fx.error or "Fix failed"
            break

        new_cuda_code = _extract_cuda_code(fx.sections, fx.content)
        if _has_valid_cuda_code(new_cuda_code) or not _has_valid_cuda_code(final_cuda_code):
            final_cuda_code = new_cuda_code
        final_sections = fx.sections
        final_status = "ai_reviewed"
        iteration += 1

    # --- Stage Final: Verify ---
    should_verify = converged or iteration > 1
    if should_verify and final_cuda_code:
        verify_prompt = (
            f"Original Python/NumPy code:\n\n{source_code}\n\n"
            f"Original review findings:\n\n{json.dumps(review_findings, indent=2)}\n\n"
            f"Fixed CUDA code:\n\n{final_cuda_code}\n\n"
            f"Changes summary: Iterative self-healing over {iteration} iteration(s).\n\n"
            "Verify that all blocking findings were addressed."
        )
        vr = _call_agent("verifier", VERIFIER_SYSTEM_PROMPT, verify_prompt, iteration=iteration)
        agent_results.append(vr)

        if vr.success:
            verdict = _parse_verdict(vr)
            if verdict == "passed":
                # AI verifier approval only establishes AI review, not
                # compilation or runtime verification. static_passed requires
                # actual static compilation (Phase 2).
                final_status = "ai_reviewed"
                converged = True
            elif verdict == "failed":
                final_status = "ai_reviewed"
                converged = False
            # If verdict unreadable, keep the reviewer's result
    elif should_verify and not final_cuda_code:
        # No CUDA code to verify
        final_status = "generated"
        converged = False

    return PipelineResult(
        translation_id=translation_id,
        iterations=iteration,
        final_status=final_status,
        agent_results=agent_results,
        final_cuda_code=final_cuda_code,
        final_sections=final_sections,
        review_findings=review_findings,
        converged=converged,
        error=error,
        references=references,
    )


def pipeline_result_to_dict(result: PipelineResult) -> Dict[str, Any]:
    """Convert PipelineResult to a JSON-serializable dict."""
    return {
        "translation_id": result.translation_id,
        "iterations": result.iterations,
        "final_status": result.final_status,
        "converged": result.converged,
        "error": result.error,
        "final_cuda_code": result.final_cuda_code,
        "final_sections": result.final_sections,
        "review_findings": result.review_findings,
        "references": result.references,
        "agent_results": [
            {
                "agent": r.agent,
                "model": r.model,
                "success": r.success,
                "content": r.content,
                "sections": r.sections,
                "error": r.error,
                "iteration": r.iteration,
                "retry_count": r.retry_count,
            }
            for r in result.agent_results
        ],
    }
