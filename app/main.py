"""CUDAify FastAPI application entrypoint."""

import os
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.rate_limit import limiter
from app.models import init_db
from app.routes import auth, translate, history

# Initialize database on import
init_db()


def _parse_cors_origins() -> List[str]:
    """Parse CORS_ALLOWED_ORIGINS env var into a list of origins.

    Returns an empty list (fail closed) when the env var is missing or empty,
    so production deployments are not accidentally open to all origins.
    Returns ["*"] only when the env var is explicitly set to "*".
    """
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="CUDAify",
    description="Code-to-CUDA Translator using NVIDIA Nemotron on Nebius Token Factory",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth")
app.include_router(translate.router, prefix="/translate")
app.include_router(history.router, prefix="/history")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "cudaify",
        "ai_configured": bool(os.getenv("NEBIUS_API_KEY")),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}