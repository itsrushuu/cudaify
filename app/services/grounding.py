"""Optional Tavily grounding for the Reviewer agent.

When TAVILY_API_KEY is set, the pipeline searches NVIDIA's official CUDA docs for guidance
relevant to the user's code pattern and gives the excerpts to the Reviewer. The sources
are returned to the UI so users can see exactly what the Reviewer consulted.

Never raises: any failure (no key, timeout, HTTP error, bad JSON) returns an empty list,
so the pipeline behaves exactly as before.
"""
from __future__ import annotations

import os
from typing import Dict, List

import requests

from app.services.advisor import detect_patterns

TAVILY_URL = "https://api.tavily.com/search"
DOC_DOMAINS = ["docs.nvidia.com", "developer.nvidia.com"]
MAX_SNIPPET_CHARS = 500

_QUERIES = {
    "stencil": "CUDA stencil kernel shared memory halo cells best practices",
    "reduction": "CUDA parallel reduction shared memory warp shuffle best practices",
    "blas": "CUDA tiled matrix multiplication shared memory best practices",
    "nested-loops": "CUDA 2D grid block thread indexing bounds check best practices",
}
_DEFAULT_QUERY = "CUDA kernel thread indexing bounds check memory coalescing best practices"


def is_enabled() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def build_query(source_code: str) -> str:
    patterns = detect_patterns(source_code)
    for key in ("stencil", "reduction", "blas", "nested-loops"):
        if key in patterns:
            return _QUERIES[key]
    return _DEFAULT_QUERY


def fetch_references(source_code: str, max_results: int = 3, timeout: float = 12.0) -> List[Dict[str, str]]:
    """Return [{title, url, snippet}] from NVIDIA docs, or [] on any problem."""
    if not is_enabled():
        return []
    try:
        resp = requests.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY', '').strip()}",
                     "Content-Type": "application/json"},
            json={
                "query": build_query(source_code),
                "search_depth": "basic",
                "max_results": max_results,
                "include_domains": DOC_DOMAINS,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception:
        return []

    refs: List[Dict[str, str]] = []
    for r in results[:max_results]:
        url = str(r.get("url", "")).strip()
        if not url:
            continue
        refs.append({
            "title": str(r.get("title", "")).strip() or url,
            "url": url,
            "snippet": str(r.get("content", "")).strip()[:MAX_SNIPPET_CHARS],
        })
    return refs


def format_for_prompt(refs: List[Dict[str, str]]) -> str:
    """Text block appended to the Reviewer prompt ('' when there are no references)."""
    if not refs:
        return ""
    body = "\n".join(f"[{i}] {r['title']} ({r['url']}): {r['snippet']}" for i, r in enumerate(refs, 1))
    return (
        "Reference excerpts from NVIDIA documentation. Use them only where relevant, "
        "and never invent citations:\n" + body + "\n\n"
    )
