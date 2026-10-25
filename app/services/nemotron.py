"""AI provider integration for code-to-CUDA translation.

Supports NVIDIA Nemotron models served through Nebius Token Factory.
Reads configuration from environment variables (never hardcodes secrets).

The translation workflow is correctness-aware: Nemotron is prompted to
produce structured output with explicit sections for kernel code, host
launch code, input assumptions, memory considerations, correctness risks,
optimization recommendations, and limitations. A response parser splits
the raw text into named sections so the frontend can display them
individually and the verification pipeline can track each stage.
"""

import os
import re
import time
from typing import Optional, Dict, Any, List
import requests
from dotenv import load_dotenv

# Load .env if present (safe: never commits real secrets)
load_dotenv()

NEBIUS_BASE_URL = os.getenv(
    "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
).rstrip("/")

# Default model: the NVIDIA Nemotron ID shown in Nebius' own Token Factory docs.
# Override with NEBIUS_MODEL, or per agent role with NEMOTRON_<ROLE>_MODEL.
DEFAULT_MODEL = "nvidia/Nemotron-3.5-Super-VL"
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL") or DEFAULT_MODEL

_ROLE_ENV = {
    "translator": "NEMOTRON_TRANSLATOR_MODEL",
    "reviewer": "NEMOTRON_REVIEWER_MODEL",
    "fixer": "NEMOTRON_FIXER_MODEL",
    "verifier": "NEMOTRON_VERIFIER_MODEL",
}


def _get_ai_provider() -> str:
    """Read AI_PROVIDER at call time so tests can monkeypatch env vars."""
    return os.getenv("AI_PROVIDER", "nebius").strip().lower()


def _get_local_base_url() -> str:
    """Read LOCAL_BASE_URL at call time."""
    return os.getenv("LOCAL_BASE_URL", "http://localhost:1235/v1").rstrip("/")


def _get_local_model() -> str:
    """Read LOCAL_MODEL at call time."""
    return os.getenv("LOCAL_MODEL", "").strip()


def _get_local_api_key() -> str:
    """Read LOCAL_API_KEY at call time (placeholder; never requires a real key)."""
    return os.getenv("LOCAL_API_KEY", "lm-studio")


def model_for(role: str) -> str:
    """Resolve the model ID for an agent role from the environment (read at call time)."""
    if _get_ai_provider() == "local":
        return _get_local_model() or DEFAULT_MODEL
    return (
        os.getenv(_ROLE_ENV.get(role, ""), "").strip()
        or os.getenv("NEBIUS_MODEL", "").strip()
        or DEFAULT_MODEL
    )


def _max_tokens() -> int:
    try:
        return max(1024, int(os.getenv("NEMOTRON_MAX_TOKENS", "8192")))
    except ValueError:
        return 8192

# Recognized section headers in Nemotron output (case-insensitive).
# Includes translator output sections AND agent-specific sections
# (reviewer, fixer, verifier) used by the agentic pipeline.
_SECTION_HEADERS = [
    "CUDA KERNEL",
    "HOST CODE",
    "EXPLANATION",
    "INPUT ASSUMPTIONS",
    "DATA TYPES",
    "BOUNDARY CONDITIONS",
    "MEMORY CONSIDERATIONS",
    "CORRECTNESS RISKS",
    "OPTIMIZATION RECOMMENDATIONS",
    "LIMITATIONS",
    "VERIFICATION STATUS",
    # Reviewer agent sections
    "FINDINGS",
    "BLOCKING",
    "SUMMARY",
    "FIX_REQUIRED",
    # Fixer agent sections
    "CHANGES MADE",
    "REMAINING CONCERNS",
    # Verifier agent sections
    "VERDICT",
    "REMAINING ISSUES",
    "REMAINING_ISSUES",
    "CONFIDENCE",
    "NOTES",
]

# Build a regex that matches any of the known section headers at the start
# of a line, optionally followed by a colon and whitespace. Inline content
# after the header on the same line is treated as the start of the body.
_SECTION_RE = re.compile(
    r"^[#>\s]*(?:\d+[.)]\s*)?[*_]*(?P<header>"
    + "|".join(re.escape(h) for h in _SECTION_HEADERS)
    + r")[*_]*\s*:?\s*[*_]*\s*",
    re.IGNORECASE | re.MULTILINE,
)

_CODE_SECTIONS = {"CUDA KERNEL", "HOST CODE"}
_FENCE_LINE_RE = re.compile(r"^\s*```[\w+-]*\s*$")

# Python constructs that are explicitly unsupported for CUDA translation.
# Patterns are kept simple to avoid regex backreference issues.
_UNSUPPORTED_PYTHON = [
    ("generator", r"\byield\b"),
    ("class_definition", r"\bclass\s+\w+"),
    ("decorator", r"@\w+"),
    ("global_statement", r"\bglobal\b"),
    ("nonlocal_statement", r"\bnonlocal\b"),
    ("try_except", r"\btry\b"),
    ("with_statement", r"\bwith\b"),
    ("lambda", r"\blambda\b"),
    ("list_comprehension", r"\[.*\bfor\b.*\bin\b.*\]"),
    ("dict_comprehension", r"\{.*\bfor\b.*\bin\b.*\}"),
    ("set_comprehension", r"\{.*\bfor\b.*\bin\b.*\}"),
    ("eval_exec", r"\b(eval|exec)\s*\("),
    ("dynamic_import", r"\b(import|from)\s+\w+\s+.*\bimport\b"),
    ("getattr_setattr", r"\b(getattr|setattr|delattr)\s*\("),
]


def validate_python_input(source_code: str) -> Dict[str, Any]:
    """Check Python source for constructs that cannot be reliably translated to CUDA.

    Returns a dict with keys:
      - "supported": bool - True if no unsupported constructs found
      - "unsupported": list[str] - human-readable descriptions of issues
      - "warnings": list[str] - non-blocking concerns
    """
    unsupported: List[str] = []
    warnings: List[str] = []

    # Check for unsupported constructs
    for name, pattern in _UNSUPPORTED_PYTHON:
        if re.search(pattern, source_code, re.IGNORECASE | re.MULTILINE):
            unsupported.append(name)

    # Check for missing function definition
    if not re.search(r"\bdef\s+\w+", source_code):
        unsupported.append("no_function_definition")

    # Check for missing array/list usage (nothing to parallelize)
    if not re.search(r"\b\w+\s*\[\s*", source_code):
        warnings.append("no_array_access_detected")

    return {
        "supported": len(unsupported) == 0,
        "unsupported": unsupported,
        "warnings": warnings,
    }


def validate_cuda_output(cuda_code: str) -> Dict[str, Any]:
    """Validate that generated CUDA C++ contains the minimal required structure.

    Returns a dict with keys:
      - "valid": bool - True if output looks like real CUDA code
      - "issues": list[str] - problems found
      - "has_kernel": bool - contains __global__ kernel
      - "has_launch": bool - contains kernel launch (<<< >>>)
      - "has_host": bool - contains host wrapper function
    """
    issues: List[str] = []
    lower = cuda_code.lower()

    has_kernel = "__global__" in lower
    has_launch = "<<<" in cuda_code
    has_host = "void " in lower or "cuda" in lower
    has_sync = "devicesynchronize" in lower
    has_error_check = "cudagetterror" in lower or "cudasuccess" in lower
    has_bounds = "if (" in lower and ("<" in lower or "<=" in lower)

    if not has_kernel:
        issues.append("No __global__ kernel function found")
    if not has_launch:
        issues.append("No kernel launch configuration (<<< >>>) found")
    if not has_host:
        issues.append("No host wrapper function found")
    if not has_sync:
        issues.append("No cudaDeviceSynchronize() after kernel launch")
    if not has_error_check:
        issues.append("No CUDA error checking (cudaGetLastError / cudaSuccess)")
    if not has_bounds:
        issues.append("No bounds guarding (if (idx < n)) detected")

    return {
        "valid": has_kernel and has_launch and has_host,
        "issues": issues,
        "has_kernel": has_kernel,
        "has_launch": has_launch,
        "has_host": has_host,
        "has_sync": has_sync,
        "has_error_check": has_error_check,
        "has_bounds": has_bounds,
    }


def strip_code_fences(text: str) -> str:
    """Remove markdown ``` fence lines so the result is compilable source."""
    lines = [ln for ln in text.splitlines() if not _FENCE_LINE_RE.match(ln)]
    return "\n".join(lines).strip()


class NemotronError(Exception):
    """Raised when the Nemotron API call fails."""

    def __init__(self, message: str, retryable: bool = False, status_code: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def clean_model_output(text: Optional[str]) -> str:
    """Strip chain-of-thought tags that Nemotron reasoning models may emit."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _is_retryable_error(exc: requests.exceptions.RequestException) -> bool:
    """Determine if an error is worth retrying."""
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response else None
        # Retry on rate limiting (429) and server errors (5xx)
        if status in (429, 500, 502, 503, 504):
            return True
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    return False


def chat_completion(
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: int = 180,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> str:
    """Call Nebius Token Factory (OpenAI-compatible) and return the cleaned text.

    Nemotron models on Token Factory are reasoning models: `content` can be
    null/empty when the whole token budget is spent on reasoning. That case
    is reported as a clear NemotronError instead of crashing.

    Retries on transient failures (429, 5xx, network errors) with exponential backoff.
    Does NOT retry on authentication errors (401) or client errors (4xx other than 429).
    """
    provider = _get_ai_provider()

    if provider == "nebius":
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise NemotronError(
                "AI provider not configured. "
                "Set NEBIUS_API_KEY in .env for the nebius provider, "
                "or switch AI_PROVIDER to local.",
                retryable=False,
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{NEBIUS_BASE_URL}/chat/completions"
        provider_name = "Nebius Token Factory"
    elif provider == "local":
        api_key = _get_local_api_key()
        local_model = _get_local_model()
        if not local_model:
            raise NemotronError(
                "LOCAL_MODEL is not configured. "
                "Set LOCAL_MODEL in .env for the local AI provider.",
                retryable=False,
            )
        model = local_model
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{_get_local_base_url()}/chat/completions"
        provider_name = "local AI provider"
    else:
        raise NemotronError(
            f"Unknown AI provider '{provider}'. Valid values: nebius, local",
            retryable=False,
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens or _max_tokens(),
    }

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            break  # Success
        except requests.exceptions.HTTPError as exc:
            body = ""
            try:
                body = exc.response.text[:300]
            except Exception:
                pass
            status = exc.response.status_code if exc.response else None
            last_error = (
                f"{provider_name} request failed ({model}): {exc} {body}".strip()
            )
            if status == 401:
                raise NemotronError(last_error, retryable=False, status_code=401) from exc
            if status and 400 <= status < 500 and status != 429:
                raise NemotronError(last_error, retryable=False, status_code=status) from exc
            if attempt < max_retries and _is_retryable_error(exc):
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise NemotronError(last_error, retryable=True, status_code=status) from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = f"{provider_name} request failed: {exc}"
            if attempt < max_retries:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise NemotronError(last_error, retryable=True) from exc
        except ValueError as exc:
            raise NemotronError(f"{provider_name} returned invalid JSON: {exc}") from exc

    try:
        choice = data["choices"][0]
        content = clean_model_output(choice["message"].get("content"))
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise NemotronError(f"Unexpected Token Factory response shape: {exc}") from exc

    if not content:
        finish_reason = choice.get("finish_reason") if "choice" in dir() else None
        if finish_reason == "length":
            raise NemotronError(
                f"{model} used its entire token budget on reasoning and returned no answer. "
                "Increase NEMOTRON_MAX_TOKENS in .env.",
                retryable=False,
            )
        raise NemotronError(f"{model} returned an empty response.", retryable=False)
    return content


def is_configured() -> bool:
    """Return True if the configured AI provider is ready (checked at call time).

    Nebius mode requires NEBIUS_API_KEY. Local mode requires LOCAL_MODEL.
    An unknown provider is treated as not configured.
    """
    if _get_ai_provider() == "nebius":
        return bool(os.getenv("NEBIUS_API_KEY"))
    if _get_ai_provider() == "local":
        return bool(_get_local_model())
    return False


SYSTEM_PROMPT = """You are an expert CUDA programmer and correctness reviewer.
Translate Python/NumPy code into optimized, correct CUDA C++.

You must return output in the following exact sections, each headed by the
section name in ALL CAPS followed by a colon or on its own line:

1. CUDA KERNEL:
   The device kernel code as a fenced CUDA C++ code block. Use proper
   thread indexing (blockIdx, blockDim, threadIdx), grid-stride loops
   for arbitrary sizes, and __syncthreads() where shared memory is used.

2. HOST CODE:
   The host-side launcher/wrapper code as a fenced C++ code block. Include
   the kernel launch configuration (grid size, block size), cudaMalloc /
   cudaFree calls, cudaMemcpy host<->device transfers, and error checking
   after every CUDA API call.

3. EXPLANATION:
   A concise paragraph describing how the Python computation maps to
   parallel execution on the GPU. Mention the parallelism pattern
   (element-wise, reduction, stencil, matrix, etc.) and the work per thread.

4. INPUT ASSUMPTIONS:
   List the assumptions about input arrays: shapes, ranks, contiguity,
   whether inputs are modified in place, and any required alignment.

5. DATA TYPES:
   State the CUDA data types used for each input/output (float, double,
   int, size_t, etc.) and any type promotions from Python/NumPy.

6. BOUNDARY CONDITIONS:
   Describe how out-of-bounds threads are guarded (e.g. if (idx < n)).
   Mention edge cases for empty inputs, single-element arrays, and
   non-multiple-of-block-size problem sizes.

7. MEMORY CONSIDERATIONS:
   Describe global vs shared memory usage, coalesced access patterns,
   bank conflicts, and whether reads/writes are coalesced.

8. CORRECTNESS RISKS:
   List potential correctness issues: integer overflow in index
   calculations, race conditions, uninitialized shared memory,
   floating-point non-associativity in reductions, missing cudaDeviceSynchronize,
   and missing error checking.

9. OPTIMIZATION RECOMMENDATIONS:
   Suggest specific optimizations: vectorized loads/stores, shared memory
   tiling, loop unrolling, constant memory, or tensor core usage where
   applicable. Do not claim measured speedups.

10. LIMITATIONS:
     Explicitly state what this translation does NOT handle: dynamic shapes,
     strided arrays, complex dtypes, multi-GPU, etc.

11. VERIFICATION STATUS:
     State: Generated (not compiled, not runtime-tested).

Rules:
- Return ONLY the sections above. No introduction, no conclusion, no chatter.
- Use fenced code blocks (```cpp and ```) for all code.
- Do not invent performance numbers or measured speedups.
- Be honest about limitations and risks.
- If the Python code cannot be safely translated to CUDA, say so in
  LIMITATIONS and return a minimal stub kernel.
"""


def _parse_sections(content: str) -> Dict[str, str]:
    """Split Nemotron output into named sections.

    Returns a dict mapping lowercase section names to their text content.
    Unknown or trailing content before the first recognized header is
    stored under the key 'preamble'.
    """
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(content))

    if not matches:
        sections["preamble"] = content.strip()
        return sections

    # Content before the first header
    first = matches[0]
    if first.start() > 0:
        sections["preamble"] = content[: first.start()].strip()

    for i, m in enumerate(matches):
        header = m.group("header").strip().upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if header in _CODE_SECTIONS:
            body = strip_code_fences(body)
        sections[header] = body

    return sections


def translate_to_cuda(
    source_code: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate Python/NumPy code to CUDA using Nemotron via Nebius Token Factory.

    Args:
        source_code: The Python/NumPy source code to translate.
        model: Optional model override (defaults to NEBIUS_MODEL env var).

    Returns:
        Dict with keys: success, content, model, sections, validation, confidence,
        or error.
        When successful, 'sections' contains the parsed named sections.
        'validation' contains CUDA output validation results.
        'confidence' is a float 0.0-1.0 estimating translation reliability.
    """
    if not source_code or not source_code.strip():
        return {"error": "Source code cannot be empty."}

    # --- Validate Python input before sending to AI ---
    input_validation = validate_python_input(source_code)
    if not input_validation["supported"]:
        return {
            "error": (
                f"Source code contains unsupported constructs: "
                f"{', '.join(input_validation['unsupported'])}. "
                "CUDAify cannot reliably translate these patterns."
            ),
            "input_validation": input_validation,
        }

    model = model or model_for("translator")

    try:
        content = chat_completion(
            model,
            SYSTEM_PROMPT,
            f"Python code to translate:\n\n{source_code}",
            temperature=0.2,
        )
    except NemotronError as exc:
        return {"error": str(exc)}

    sections = _parse_sections(content)

    # --- Validate CUDA output ---
    cuda_validation = validate_cuda_output(content)

    # Compute confidence: starts at 1.0, penalizes for missing sections and issues
    confidence = 1.0
    if not sections.get("CUDA KERNEL"):
        confidence -= 0.4
    if not sections.get("HOST CODE"):
        confidence -= 0.3
    if not sections.get("EXPLANATION"):
        confidence -= 0.1
    confidence -= len(cuda_validation["issues"]) * 0.05
    confidence = max(0.0, min(1.0, round(confidence, 2)))

    # Check for preamble content (AI added chatter before first section)
    if sections.get("preamble"):
        confidence -= 0.1
        confidence = max(0.0, confidence)

    return {
        "success": True,
        "content": content,
        "model": model,
        "sections": sections,
        "validation": cuda_validation,
        "confidence": confidence,
        "input_validation": input_validation,
    }
