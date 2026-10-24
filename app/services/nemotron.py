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
from typing import Optional, Dict, Any
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


def model_for(role: str) -> str:
    """Resolve the model ID for an agent role from the environment (read at call time)."""
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
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise NemotronError(
            "NEBIUS_API_KEY is not configured. "
            "Copy .env.example to .env and set your Nebius Token Factory API key.",
            retryable=False,
        )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{NEBIUS_BASE_URL}/chat/completions"
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
                f"Nebius Token Factory request failed ({model}): {exc} {body}".strip()
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
            last_error = f"Nebius Token Factory request failed: {exc}"
            if attempt < max_retries:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise NemotronError(last_error, retryable=True) from exc
        except ValueError as exc:
            raise NemotronError(f"Token Factory returned invalid JSON: {exc}") from exc

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
    """Return True if the API key is present (checked at call time)."""
    return bool(os.getenv("NEBIUS_API_KEY"))


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
        Dict with keys: success, content, model, sections, or error.
        When successful, 'sections' contains the parsed named sections.
    """
    if not source_code or not source_code.strip():
        return {"error": "Source code cannot be empty."}

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
    return {
        "success": True,
        "content": content,
        "model": model,
        "sections": _parse_sections(content),
    }
