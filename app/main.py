"""CUDAify FastAPI application entrypoint."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import init_db
from app.routes import auth, translate, history

# Initialize database on import
init_db()

app = FastAPI(
    title="CUDAify",
    description="Code-to-CUDA Translator using NVIDIA Nemotron on Nebius Token Factory",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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