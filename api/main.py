"""Minimal FastAPI application for ClaimRank AI."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .datasets import router as datasets_router
from .evaluation import router as evaluation_router
from .fact_checks import jobs_router as fact_check_jobs_router
from .fact_checks import run_router as fact_checks_router
from .prioritization_runs import router as prioritization_runs_router

DEFAULT_FRONTEND_ORIGIN = "http://localhost:5173"


def _frontend_origins() -> list[str]:
    """Return configured frontend origins, defaulting to the Vite dev server."""
    raw_origins = os.getenv("FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN)
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or [DEFAULT_FRONTEND_ORIGIN]


app = FastAPI(title="ClaimRank AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(datasets_router)
app.include_router(prioritization_runs_router)
app.include_router(fact_checks_router)
app.include_router(fact_check_jobs_router)
app.include_router(evaluation_router)


@app.get("/api/v1/health", tags=["health"])
def health() -> dict[str, str]:
    """Confirm that the API is running."""
    return {"status": "ok", "service": "claimrank-api"}
