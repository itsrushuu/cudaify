"""Independent static code review for generated CUDA code.

This module examines CUDA C++ output for common correctness issues without
compiling or executing it. Findings are clearly labeled as AI-generated
static analysis — they are NOT compiler diagnostics and NOT runtime
verification results.
"""

import re
from typing import Any, Dict, List, Optional


def review_cuda_code(cuda_code: str) -> Dict[str, Any]:
    """Perform static review of CUDA code."""
    findings: List[Dict[str, Any]] = []

    lines = cuda_code.split("\n")
    code_blocks = _extract_code_blocks(cuda_code)

    if not code_blocks:
        code_blocks = [("full_text", cuda_code)]

    for block_name, block_code in code_blocks:
        block_lines = block_code.split("\n")
        _check_bounds_guarding(block_lines, block_name, findings)
        _check_synchronize(block_lines, block_name, findings)
        _check_api_error_checking(block_lines, block_name, findings)
        _check_shared_memory_sync(block_lines, block_name, findings)
        _check_index_overflow(block_lines, block_name, findings)
        _check_race_conditions(block_lines, block_name, findings)

    seen = set()
    unique_findings = []
    for f in findings:
        key = (f["category"], f["message"])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    severity_order = {"error": 0, "warning": 1, "info": 2}
    unique_findings.sort(key=lambda f: (severity_order.get(f["severity"], 9), f.get("line", 0)))

    summary = _build_summary(unique_findings)

    return {
        "findings": unique_findings,
        "summary": summary,
        "ai_generated": True,
        "note": (
            "These are AI-generated static-analysis findings. They are not "
            "compiler diagnostics and have not been verified by nvcc or "
            "runtime execution."
        ),
    }


def _extract_code_blocks(text: str) -> List[tuple]:
    blocks: List[tuple] = []
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(text):
        lang = m.group(1).strip() or "unknown"
        code = m.group(2)
        blocks.append((lang, code))
    return blocks


def _add(
    findings: List[Dict[str, Any]],
    severity: str,
    category: str,
    message: str,
    block_name: str,
    line: Optional[int] = None,
) -> None:
    findings.append(
        {
            "severity": severity,
            "category": category,
            "message": message,
            "block": block_name,
            "line": line,
            "ai_generated": True,
        }
    )


def _check_bounds_guarding(lines, block_name, findings):
    has_bounds_guard = False
    has_global_access = False
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if re.search(r"\b[a-zA-Z_]\w*\s*\[\s*(?:blockIdx|threadIdx|idx|i)\s*\(", stripped) or \
           re.search(r"\b[a-zA-Z_]\w*\s*\[\s*(?:blockIdx|threadIdx|idx|i)\s*\]", stripped):
            has_global_access = True
        if re.search(r"if\s*\(\s*(?:idx|i|tid)\s*<\s*\w+\s*\)", stripped):
            has_bounds_guard = True
        if re.search(r"if\s*\(\s*!\s*\(idx\s*<\s*\w+\)", stripped):
            has_bounds_guard = True
    if has_global_access and not has_bounds_guard:
        _add(
            findings, "warning", "bounds_guarding",
            "Global memory access detected without an obvious bounds guard "
            "(e.g. 'if (idx < n)'). Out-of-bounds threads may read/write "
            "invalid memory.", block_name,
        )


def _check_synchronize(lines, block_name, findings):
    has_kernel_launch = False
    has_sync = False
    for line in lines:
        if re.search(r"<<<.*>>>", line):
            has_kernel_launch = True
        if "cudaDeviceSynchronize" in line:
            has_sync = True
    if has_kernel_launch and not has_sync:
        _add(
            findings, "warning", "synchronization",
            "Kernel launch detected without cudaDeviceSynchronize(). "
            "Host may read results before the kernel completes.", block_name,
        )


def _check_api_error_checking(lines, block_name, findings):
    cuda_api_calls = []
    error_checks = []
    for i, line in enumerate(lines, start=1):
        if re.search(r"\bcuda(Malloc|Memcpy|Memset|Free|EventCreate|StreamCreate)\b", line):
            cuda_api_calls.append(i)
        if "cudaGetLastError" in line or "cudaSuccess" in line:
            error_checks.append(i)
    if cuda_api_calls and not error_checks:
        _add(
            findings, "warning", "error_checking",
            f"CUDA API call(s) at line(s) {cuda_api_calls} without visible "
            "error checking (cudaGetLastError / cudaSuccess).", block_name,
        )


def _check_shared_memory_sync(lines, block_name, findings):
    has_shared = any("__shared__" in line for line in lines)
    has_sync = any("__syncthreads()" in line for line in lines)
    if has_shared and not has_sync:
        _add(
            findings, "warning", "shared_memory_sync",
            "Shared memory (__shared__) declared but no __syncthreads() "
            "barrier found. Threads may read uninitialized shared data.", block_name,
        )


def _check_index_overflow(lines, block_name, findings):
    for i, line in enumerate(lines, start=1):
        if re.search(r"blockIdx\.\w+\s*\*\s*blockDim\.\w+", line):
            _add(
                findings, "info", "index_overflow",
                "Index calculation multiplies blockIdx by blockDim without "
                "an explicit cast to size_t or int64_t. Large grids may "
                "overflow 32-bit int.", block_name, i,
            )
        break


def _check_race_conditions(lines, block_name, findings):
    has_atomic = any("atomic" in line.lower() for line in lines)
    has_reduction_keyword = any(
        kw in line.lower() for line in lines
        for kw in ("sum", "reduce", "accumulate", "max", "min")
    )
    has_shared_write = any(
        "__shared__" in line and ("=" in line or "+=" in line) for line in lines
    )
    if has_shared_write and not has_atomic:
        _add(
            findings, "info", "race_condition",
            "Shared memory writes detected without atomic operations. "
            "Multiple threads may write to the same location causing "
            "race conditions unless properly synchronized.", block_name,
        )


def _build_summary(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "No obvious static-analysis issues detected."
    errors = [f for f in findings if f["severity"] == "error"]
    warnings = [f for f in findings if f["severity"] == "warning"]
    infos = [f for f in findings if f["severity"] == "info"]
    parts = []
    if errors:
        parts.append(f"{len(errors)} error(s)")
    if warnings:
        parts.append(f"{len(warnings)} warning(s)")
    if infos:
        parts.append(f"{len(infos)} info(s)")
    return "; ".join(parts) + " found during static review."