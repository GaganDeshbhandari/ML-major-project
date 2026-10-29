"""User-triggered FactLLaMA fact checks backed by in-memory jobs.

FactLLaMA is invoked only when a fact-check job is created for a completed
prioritization run. The job reuses the claim's stored evidence; it never
retrieves new evidence. Job state and the reusable model service live in the
current process, so they are lost on restart and are not shared between
Uvicorn workers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.pipeline import (
    ClaimRankEntry,
    FactLLaMASelectionError,
    FactLLaMASelectionService,
    FactLLaMAVerificationResult,
    PrioritizationRetrievalResult,
)

from .prioritization_runs import PrioritizationRun, get_completed_run

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR_MESSAGE = "Fact check failed unexpectedly."

#: One lock for the process-wide lazy service. It serializes inference so
#: concurrent jobs cannot load or run duplicate model instances.
_EXECUTION_LOCK = Lock()

FactCheckStatus = Literal["queued", "processing", "completed", "failed"]

LIMITATIONS = (
    "The verdict is based only on evidence stored by the completed "
    "prioritization run; no new evidence was retrieved.",
    "FactLLaMA returns an evidence-based classification, not a proven fact, "
    "confidence score, or probability.",
    "The existing FactLLaMA prompt requests a single verdict and does not "
    "produce a structured explanation.",
    "Fact-check jobs and model state are held in process memory and are lost "
    "on restart.",
)

run_router = APIRouter(
    prefix="/api/v1/prioritization-runs",
    tags=["fact-checks"],
)
jobs_router = APIRouter(
    prefix="/api/v1/fact-check-jobs",
    tags=["fact-checks"],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class FactCheckJob:
    """Mutable in-memory state for one user-triggered fact check."""

    job_id: str
    run_id: str
    claim_id: str | int
    claim: str
    status: FactCheckStatus = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    verdict: str | None = None
    explanation: str | None = None
    raw_output: str | None = None
    evidence_references: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    error: dict[str, str] | None = None

    def payload(self) -> dict[str, Any]:
        """Return the JSON-safe job view."""
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "claim_id": self.claim_id,
            "claim": self.claim,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "raw_output": self.raw_output,
            "evidence_references": [
                dict(reference) for reference in self.evidence_references
            ],
            "limitations": list(self.limitations),
            "error": None if self.error is None else dict(self.error),
        }


class FactCheckJobStore:
    """Thread-safe storage for fact-check jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, FactCheckJob] = {}
        self._lock = Lock()

    def clear(self) -> None:
        """Remove all jobs; used by tests."""
        with self._lock:
            self._jobs.clear()

    def create(self, *, run_id: str, entry: ClaimRankEntry) -> FactCheckJob:
        job = FactCheckJob(
            job_id=f"job_{uuid4().hex}",
            run_id=run_id,
            claim_id=entry.claim_id,
            claim=entry.claim,
            limitations=list(LIMITATIONS),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return replace(job)

    def snapshot(self, job_id: str) -> FactCheckJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return replace(
                job,
                evidence_references=[
                    dict(reference) for reference in job.evidence_references
                ],
                limitations=list(job.limitations),
                error=None if job.error is None else dict(job.error),
            )

    def mark_processing(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return False
            job.status = "processing"
            job.started_at = _utc_now()
            return True

    def mark_completed(
        self,
        job_id: str,
        verification: FactLLaMAVerificationResult,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.completed_at = _utc_now()
            job.verdict = verification.verdict
            job.explanation = None
            job.raw_output = verification.raw_output
            job.evidence_references = _evidence_references(verification.evidence)
            job.error = None

    def mark_failed(
        self,
        job_id: str,
        *,
        error_type: str,
        message: str,
        raw_output: str | None = None,
        evidence: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.completed_at = _utc_now()
            job.verdict = None
            job.explanation = None
            job.raw_output = raw_output
            job.evidence_references = _evidence_references(evidence)
            job.error = {
                "type": error_type,
                "message": message,
            }


fact_check_store = FactCheckJobStore()

_SERVICE_LOCK = Lock()
_factllama_service: FactLLaMASelectionService | None = None


def get_factllama_service() -> FactLLaMASelectionService:
    """Return the process-wide service, constructing it at most once."""
    global _factllama_service
    if _factllama_service is None:
        with _SERVICE_LOCK:
            if _factllama_service is None:
                _factllama_service = FactLLaMASelectionService()
    return _factllama_service


def reset_factllama_service() -> None:
    """Clear the cached service; intended for tests."""
    global _factllama_service
    with _SERVICE_LOCK:
        _factllama_service = None


def _evidence_references(
    evidence: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return JSON-safe evidence metadata, preserving order."""
    references: list[dict[str, Any]] = []
    for index, record in enumerate(evidence, start=1):
        if not isinstance(record, dict):
            references.append({"index": index, "content": str(record)})
            continue
        reference: dict[str, Any] = {"index": index}
        for key in (
            "title",
            "url",
            "source",
            "published_at",
            "content",
            "text",
            "evidence",
        ):
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                reference[key] = value
            else:
                reference[key] = str(value)
        references.append(reference)
    return references


def _has_usable_evidence(evidence: tuple[dict[str, Any], ...]) -> bool:
    """Mirror the prompt builder's supported evidence text fields."""
    for record in evidence:
        if isinstance(record, str) and record.strip():
            return True
        if not isinstance(record, dict):
            continue
        for key in ("content", "text", "evidence"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _claim_or_error(run: PrioritizationRun, claim_id: str) -> ClaimRankEntry:
    assert run.results is not None
    entry = next(
        (
            item
            for item in run.results.results
            if str(item.claim_id) == claim_id
        ),
        None,
    )
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Claim not found in this prioritization run.",
        )
    if not entry.claim.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "message": "FactLLaMA cannot verify an empty claim.",
                "claim_id": entry.claim_id,
            },
        )
    if entry.retrieval_error is not None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "FactLLaMA cannot verify this claim because evidence "
                    "retrieval failed."
                ),
                "claim_id": entry.claim_id,
                "retrieval_error": {
                    "error_type": entry.retrieval_error.error_type,
                    "message": entry.retrieval_error.message,
                },
            },
        )
    if not entry.evidence or not _has_usable_evidence(entry.evidence):
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "FactLLaMA cannot verify this claim because no usable "
                    "stored evidence is available."
                ),
                "claim_id": entry.claim_id,
            },
        )
    return entry


def _adapted_result(
    run: PrioritizationRun,
    entry: ClaimRankEntry,
) -> PrioritizationRetrievalResult:
    """Reuse stored evidence through the existing selection-service contract."""
    return PrioritizationRetrievalResult(
        claim_id=entry.claim_id,
        claim=entry.claim,
        objective=run.objective.pipeline_objective,
        initial_rank=entry.original.rank,
        evidence=list(entry.evidence),
        predicted_truth=entry.original.predicted_truth,
        predicted_tpb=entry.original.predicted_tpb,
        disputability=entry.original.disputability,
        retrieval_error=entry.retrieval_error,
        ecs_result=entry.conflict,
        ecs_error=entry.ecs_error,
    )


def _execute_fact_check(
    job_id: str,
    service: FactLLaMASelectionService,
    run: PrioritizationRun,
    entry: ClaimRankEntry,
) -> None:
    """Run FactLLaMA for one queued job without leaking failures."""
    if not fact_check_store.mark_processing(job_id):
        return

    try:
        adapted = _adapted_result(run, entry)
        with _EXECUTION_LOCK:
            verification = service.verify_selected_claim(
                [adapted],
                entry.claim_id,
            )
    except FactLLaMASelectionError as exc:
        fact_check_store.mark_failed(
            job_id,
            error_type=type(exc).__name__,
            message=str(exc),
            evidence=entry.evidence,
        )
        return
    except Exception:
        logger.exception("Unexpected failure in fact-check job %s", job_id)
        fact_check_store.mark_failed(
            job_id,
            error_type="UnexpectedError",
            message=UNEXPECTED_ERROR_MESSAGE,
            evidence=entry.evidence,
        )
        return

    if verification.error is not None:
        fact_check_store.mark_failed(
            job_id,
            error_type=verification.error.error_type,
            message=verification.error.message,
            raw_output=verification.raw_output,
            evidence=verification.evidence,
        )
        return

    fact_check_store.mark_completed(job_id, verification)


@run_router.post(
    "/{run_id}/claims/{claim_id}/fact-check",
    status_code=202,
)
def create_fact_check_job(
    run_id: str,
    claim_id: str,
    background_tasks: BackgroundTasks,
    service: FactLLaMASelectionService = Depends(get_factllama_service),
) -> dict[str, Any]:
    """Queue one user-triggered FactLLaMA verification for a stored claim."""
    run = get_completed_run(run_id)
    entry = _claim_or_error(run, claim_id)
    job = fact_check_store.create(run_id=run.run_id, entry=entry)
    background_tasks.add_task(
        _execute_fact_check,
        job.job_id,
        service,
        run,
        entry,
    )
    return job.payload()


@jobs_router.get("/{job_id}")
def get_fact_check_job(job_id: str) -> dict[str, Any]:
    """Return the current state of one fact-check job."""
    job = fact_check_store.snapshot(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Fact-check job not found.",
        )
    return job.payload()
