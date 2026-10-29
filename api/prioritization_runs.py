"""In-memory prioritization runs with background execution.

Run state exists only in the current process. It is lost on restart and is not
shared between multiple Uvicorn workers; deployments with multiple workers need
sticky routing so status polling reaches the worker that created the run.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from math import fsum, isfinite
from threading import Lock
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)

from src.evidence_conflict import ECSStatus
from src.pipeline import (
    ClaimRankEntry,
    ClaimRankObjective,
    ClaimRankRun,
    ClaimRankService,
    ClaimRankServiceError,
    parse_objective,
)
from src.prioritization import Claim, InputValidationError, UserRating, prepare_ratings

from .datasets import ParsedCsv, parse_dataset_csv, read_upload

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR_MESSAGE = "Unexpected prioritization pipeline failure."
_EXECUTION_LOCK = Lock()

router = APIRouter(
    prefix="/api/v1/prioritization-runs",
    tags=["prioritization-runs"],
)

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100
HIGH_CONFLICT_ECS_THRESHOLD = 0.7
MEDIUM_CONFLICT_ECS_THRESHOLD = 0.4

SortField: TypeAlias = Literal[
    "final_rank",
    "initial_rank",
    "claim_id",
    "ecs_score",
    "final_score",
    "disputability",
    "rank_change",
]

EXPORT_HEADERS = (
    "Claim ID",
    "Claim",
    "Objective",
    "Original Rank",
    "Original Predicted Truth",
    "Original Predicted TPB",
    "Original Disputability",
    "Final Rank",
    "Rank Change",
    "Final Priority Score (RRF)",
    "ECS Score",
    "ECS Rank",
    "Conflict Status",
    "Supporting Evidence Count",
    "Refuting Evidence Count",
    "Not Enough Info Count",
    "Stance Coverage",
    "Refutation Share",
    "Support Share",
    "Mean Stance Confidence",
    "Evidence Count",
    "Evidence URLs",
    "Evidence (JSON)",
    "Conflict Predictions (JSON)",
    "Retrieval Error",
    "ECS Error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PrioritizationRun:
    """Mutable in-memory state for one prioritization run."""

    run_id: str
    objective: ClaimRankObjective
    claim_count: int
    rating_count: int
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    results: ClaimRankRun | None = None
    error: dict[str, str] | None = None

    def status_payload(self) -> dict[str, Any]:
        """Return the JSON-safe status view without full results."""
        return {
            "run_id": self.run_id,
            "objective": self.objective.value,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "claim_count": self.claim_count,
            "rating_count": self.rating_count,
            "result_count": (
                len(self.results.results) if self.results is not None else None
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class PrioritizationRunStore:
    """Thread-safe temporary storage for prioritization runs."""

    def __init__(self) -> None:
        self._runs: dict[str, PrioritizationRun] = {}
        self._lock = Lock()

    def clear(self) -> None:
        """Remove all runs; used by tests."""
        with self._lock:
            self._runs.clear()

    def create(
        self,
        *,
        objective: ClaimRankObjective,
        claim_count: int,
        rating_count: int,
    ) -> PrioritizationRun:
        run = PrioritizationRun(
            run_id=f"run_{uuid4().hex}",
            objective=objective,
            claim_count=claim_count,
            rating_count=rating_count,
        )
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def status(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else run.status_payload()

    def snapshot(self, run_id: str) -> PrioritizationRun | None:
        """Return a shallow snapshot for read-only API responses."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return replace(
                run,
                error=None if run.error is None else dict(run.error),
            )

    def mark_processing(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = "processing"
            run.stage = "running_pipeline"
            run.progress = 0.1
            run.started_at = _utc_now()

    def mark_completed(self, run_id: str, results: ClaimRankRun) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = "completed"
            run.stage = "completed"
            run.progress = 1.0
            run.results = results
            run.completed_at = _utc_now()

    def mark_failed(
        self,
        run_id: str,
        *,
        stage: str,
        error_type: str,
        message: str,
    ) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = "failed"
            run.stage = "failed"
            run.completed_at = _utc_now()
            run.error = {
                "stage": stage,
                "type": error_type,
                "message": message,
            }


run_store = PrioritizationRunStore()


@lru_cache(maxsize=1)
def get_claimrank_service() -> ClaimRankService:
    """Return the shared service so its lazy NLI model is reused."""
    return ClaimRankService()


def _claims_from(parsed: ParsedCsv) -> list[Claim]:
    return [
        Claim(claim_id=row["claim_id"], claim=row["claim"])
        for row in parsed.rows
    ]


def _ratings_from(parsed: ParsedCsv) -> list[UserRating]:
    return [
        UserRating(claim_id=row["claim_id"], rating=row["rating"])
        for row in parsed.rows
    ]


def _validation_error(message: str, errors: list[str]) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"message": message, "errors": errors},
    )


def _execute_run(
    run_id: str,
    claims: list[Claim],
    ratings: list[UserRating],
    objective: ClaimRankObjective,
    service: ClaimRankService,
) -> None:
    run_store.mark_processing(run_id)
    try:
        # Shared service and model state are kept serialized for this MVP.
        with _EXECUTION_LOCK:
            results = service.rank_claims(claims, ratings, objective)
        if not isinstance(results, ClaimRankRun):
            raise TypeError("ClaimRankService returned an invalid result.")
    except ClaimRankServiceError as exc:
        run_store.mark_failed(
            run_id,
            stage=exc.stage,
            error_type=type(exc).__name__,
            message=exc.message,
        )
    except Exception as exc:
        logger.exception("Prioritization run %s failed unexpectedly.", run_id)
        run_store.mark_failed(
            run_id,
            stage="pipeline",
            error_type=type(exc).__name__,
            message=UNEXPECTED_ERROR_MESSAGE,
        )
    else:
        run_store.mark_completed(run_id, results)


def _to_jsonable(value: Any) -> Any:
    """Convert result dataclasses and enums into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _entry_payload(
    entry: ClaimRankEntry,
    *,
    include_rank_change: bool = False,
) -> dict[str, Any]:
    payload = _to_jsonable(entry)
    if include_rank_change:
        # Positive means the claim moved up in the final ranking.
        payload["rank_change"] = entry.original.rank - entry.final.rank
    return payload


def _csv_optional(value: Any) -> Any:
    """Leave unavailable values blank instead of fabricating zeros."""
    return "" if value is None else value


def _error_text(error: Any) -> str:
    if error is None:
        return ""
    return f"{error.error_type}: {error.message}"


def _evidence_urls(evidence: Iterable[dict[str, Any]]) -> str:
    urls = []
    for item in evidence:
        url = item.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())
    return " | ".join(urls)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(
        _to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _export_row(entry: ClaimRankEntry) -> list[Any]:
    conflict = entry.conflict
    counts = conflict.counts if conflict is not None else None
    return [
        entry.claim_id,
        entry.claim,
        entry.objective.value,
        entry.original.rank,
        _csv_optional(entry.original.predicted_truth),
        _csv_optional(entry.original.predicted_tpb),
        _csv_optional(entry.original.disputability),
        entry.final.rank,
        entry.original.rank - entry.final.rank,
        entry.final.score,
        _csv_optional(entry.ecs_score),
        _csv_optional(entry.ecs_rank),
        _csv_optional(conflict.status.value if conflict is not None else None),
        _csv_optional(counts.supports if counts is not None else None),
        _csv_optional(counts.refutes if counts is not None else None),
        _csv_optional(counts.not_enough_info if counts is not None else None),
        _csv_optional(
            conflict.evidence_coverage if conflict is not None else None
        ),
        _csv_optional(
            conflict.refutation_share if conflict is not None else None
        ),
        _csv_optional(conflict.support_share if conflict is not None else None),
        _csv_optional(
            conflict.mean_confidence if conflict is not None else None
        ),
        len(entry.evidence),
        _evidence_urls(entry.evidence),
        _json_cell(entry.evidence),
        _json_cell(conflict.predictions if conflict is not None else None),
        _error_text(entry.retrieval_error),
        _error_text(entry.ecs_error),
    ]


def _build_export_csv(run: PrioritizationRun) -> bytes:
    assert run.results is not None
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_HEADERS)
    for entry in run.results.results:
        writer.writerow(_export_row(entry))
    return buffer.getvalue().encode("utf-8")


def _snapshot_or_404(run_id: str) -> PrioritizationRun:
    run = run_store.snapshot(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Prioritization run not found.",
        )
    return run


def get_completed_run(run_id: str) -> PrioritizationRun:
    """Return a completed run or raise the standard API error."""
    run = _snapshot_or_404(run_id)
    if run.status == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Prioritization run failed.",
                "status": run.status,
                "stage": run.stage,
                "error": run.error,
            },
        )
    if run.status != "completed" or run.results is None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Prioritization run is not complete.",
                "status": run.status,
                "stage": run.stage,
                "progress": run.progress,
            },
        )
    return run


def _numeric_stats(values: Iterable[float | None]) -> dict[str, Any] | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": fsum(numbers) / len(numbers),
    }


def _label_counts(values: Iterable[str | None]) -> dict[str, int] | None:
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items())) if counts else None


def _conflict_band(ecs_score: float | None) -> str | None:
    if ecs_score is None:
        return None
    if ecs_score >= HIGH_CONFLICT_ECS_THRESHOLD:
        return "high"
    if ecs_score >= MEDIUM_CONFLICT_ECS_THRESHOLD:
        return "medium"
    return "low"


def _invalid_filter(field_name: str, messages: list[str]) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "message": f"Invalid {field_name} filter.",
            "errors": messages,
        },
    )


def _filter_by_priority(
    entries: list[ClaimRankEntry],
    objective: ClaimRankObjective,
    raw_priority: str,
) -> list[ClaimRankEntry]:
    value = raw_priority.strip().casefold()
    if objective is ClaimRankObjective.REMOVE_FALSE_CLAIMS:
        if value not in {"true", "false"}:
            raise _invalid_filter(
                "priority",
                ["For remove_false_claims, priority must be 'True' or 'False'."],
            )
        return [
            entry
            for entry in entries
            if (entry.original.predicted_truth or "").casefold() == value
        ]
    if objective is ClaimRankObjective.CORRECT_USER_MISCONCEPTIONS:
        if value not in {"high", "low"}:
            raise _invalid_filter(
                "priority",
                ["For correct_user_misconceptions, priority must be 'High' or 'Low'."],
            )
        return [
            entry
            for entry in entries
            if (entry.original.predicted_tpb or "").casefold() == value
        ]
    raise _invalid_filter(
        "priority",
        [
            "priority filtering is not available for reduce_user_disagreement; "
            "use min_disputability or max_disputability instead."
        ],
    )


def _claim_id_sort_key(claim_id: str | int) -> tuple[int, Any]:
    if isinstance(claim_id, int):
        return (0, claim_id)
    text = str(claim_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text.casefold())


def _sort_optional(
    entries: list[ClaimRankEntry],
    getter: Any,
    *,
    reverse: bool,
) -> list[ClaimRankEntry]:
    present: list[tuple[Any, ClaimRankEntry]] = []
    missing: list[ClaimRankEntry] = []
    for entry in entries:
        value = getter(entry)
        if value is None:
            missing.append(entry)
        else:
            present.append((value, entry))
    present.sort(key=lambda item: item[0], reverse=reverse)
    return [entry for _, entry in present] + missing


def _sort_entries(
    entries: list[ClaimRankEntry],
    sort_field: SortField,
    order: Literal["asc", "desc"],
) -> list[ClaimRankEntry]:
    reverse = order == "desc"
    if sort_field == "final_rank":
        return sorted(entries, key=lambda entry: entry.final.rank, reverse=reverse)
    if sort_field == "initial_rank":
        return sorted(entries, key=lambda entry: entry.original.rank, reverse=reverse)
    if sort_field == "claim_id":
        return sorted(
            entries,
            key=lambda entry: _claim_id_sort_key(entry.claim_id),
            reverse=reverse,
        )
    if sort_field == "ecs_score":
        return _sort_optional(entries, lambda entry: entry.ecs_score, reverse=reverse)
    if sort_field == "final_score":
        return sorted(entries, key=lambda entry: entry.final.score, reverse=reverse)
    if sort_field == "disputability":
        return _sort_optional(
            entries,
            lambda entry: entry.original.disputability,
            reverse=reverse,
        )
    return _sort_optional(
        entries,
        lambda entry: entry.original.rank - entry.final.rank,
        reverse=reverse,
    )


@router.post("", status_code=202)
async def create_prioritization_run(
    background_tasks: BackgroundTasks,
    claims_file: UploadFile = File(...),
    perception_file: UploadFile = File(...),
    objective: str = Form(...),
    service: ClaimRankService = Depends(get_claimrank_service),
) -> dict[str, Any]:
    """Validate both CSVs and queue a background prioritization run."""
    try:
        selected_objective = parse_objective(objective)
    except ClaimRankServiceError as exc:
        raise _validation_error("Invalid objective.", [exc.message]) from exc

    claims_content = await read_upload(claims_file)
    perception_content = await read_upload(perception_file)

    claims_parsed, claims_errors, _ = parse_dataset_csv(claims_content, "claims")
    if claims_errors:
        raise _validation_error("Claims CSV validation failed.", claims_errors)

    perception_parsed, perception_errors, _ = parse_dataset_csv(
        perception_content,
        "perception",
    )
    if perception_errors:
        raise _validation_error(
            "Perception CSV validation failed.",
            perception_errors,
        )

    claims = _claims_from(claims_parsed)
    ratings = _ratings_from(perception_parsed)
    try:
        prepare_ratings(claims, ratings)
    except InputValidationError as exc:
        raise _validation_error(
            "Claims and perception data are inconsistent.",
            [str(exc)],
        ) from exc

    run = run_store.create(
        objective=selected_objective,
        claim_count=len(claims),
        rating_count=len(ratings),
    )
    background_tasks.add_task(
        _execute_run,
        run.run_id,
        claims,
        ratings,
        selected_objective,
        service,
    )
    return run.status_payload()


@router.get("/{run_id}/status")
def get_prioritization_run_status(run_id: str) -> dict[str, Any]:
    """Return status, stage, counts, and concise failure details."""
    payload = run_store.status(run_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="Prioritization run not found.",
        )
    return payload


@router.get("/{run_id}/results")
def get_prioritization_run_results(run_id: str) -> dict[str, Any]:
    """Return the complete ranked results from a completed run."""
    run = get_completed_run(run_id)
    assert run.results is not None
    results = run.results
    return {
        "run_id": run.run_id,
        "objective": run.objective.value,
        "status": run.status,
        "stage": run.stage,
        "progress": run.progress,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "fusion_method": results.fusion_method,
        "rrf_k": results.rrf_k,
        "result_count": len(results.results),
        "results": [
            _entry_payload(entry)
            for entry in results.results
        ],
    }


@router.get("/{run_id}/export")
def export_prioritization_run_results(run_id: str) -> Response:
    """Download completed stored results as UTF-8 CSV."""
    run = get_completed_run(run_id)
    filename = f"claimrank_{run_id}.csv"
    return Response(
        content=_build_export_csv(run),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/{run_id}/summary")
def get_prioritization_run_summary(run_id: str) -> dict[str, Any]:
    """Return descriptive statistics computed only from stored results."""
    run = get_completed_run(run_id)
    assert run.results is not None
    entries = list(run.results.results)
    total_claims = len(entries)

    claims_with_evidence = sum(1 for entry in entries if entry.evidence)
    evidence_coverage = (
        claims_with_evidence / total_claims if total_claims else None
    )

    high_conflict_claims = sum(
        1
        for entry in entries
        if entry.ecs_score is not None
        and entry.ecs_score >= HIGH_CONFLICT_ECS_THRESHOLD
    )
    conflicting_claims = sum(
        1
        for entry in entries
        if entry.ecs_score is not None and entry.ecs_score > 0.0
    )

    conflict_status_counts = {status.value: 0 for status in ECSStatus}
    stance_coverages: list[float] = []
    for entry in entries:
        if entry.conflict is not None:
            conflict_status_counts[entry.conflict.status.value] += 1
            if entry.conflict.evidence_coverage is not None:
                stance_coverages.append(entry.conflict.evidence_coverage)
    stance_coverage_stats = _numeric_stats(stance_coverages)

    return {
        "run_id": run.run_id,
        "objective": run.objective.value,
        "status": run.status,
        "fusion_method": run.results.fusion_method,
        "rrf_k": run.results.rrf_k,
        "total_claims": total_claims,
        "input_claim_count": run.claim_count,
        "high_conflict_claim_count": high_conflict_claims,
        "high_conflict_threshold": HIGH_CONFLICT_ECS_THRESHOLD,
        "conflicting_claim_count": conflicting_claims,
        "claims_with_evidence": claims_with_evidence,
        "claims_without_evidence": total_claims - claims_with_evidence,
        "evidence_coverage": evidence_coverage,
        "mean_stance_coverage": (
            stance_coverage_stats["mean"]
            if stance_coverage_stats is not None
            else None
        ),
        "conflict_status_counts": conflict_status_counts,
        "priority_label_counts": {
            "predicted_truth": _label_counts(
                entry.original.predicted_truth for entry in entries
            ),
            "predicted_tpb": _label_counts(
                entry.original.predicted_tpb for entry in entries
            ),
        },
        "component_error_counts": {
            "retrieval": sum(
                1 for entry in entries if entry.retrieval_error is not None
            ),
            "ecs": sum(1 for entry in entries if entry.ecs_error is not None),
        },
        "score_summaries": {
            "final_score": _numeric_stats(
                entry.final.score for entry in entries
            ),
            "ecs_score": _numeric_stats(
                entry.ecs_score for entry in entries
            ),
            "disputability": _numeric_stats(
                entry.original.disputability for entry in entries
            ),
        },
    }


@router.get("/{run_id}/claims")
def list_prioritization_run_claims(
    run_id: str,
    q: str | None = Query(None, max_length=200),
    search: str | None = Query(None, max_length=200),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    sort: SortField | None = Query(None),
    order: Literal["asc", "desc"] | None = Query(None),
    ranking: Literal["final", "initial"] = Query("final"),
    priority: str | None = Query(None, max_length=32),
    conflict: Literal["high", "medium", "low"] | None = Query(None),
    conflict_status: ECSStatus | None = Query(None),
    has_evidence: bool | None = Query(None),
    min_ecs: float | None = Query(None, ge=0.0, le=1.0),
    max_ecs: float | None = Query(None, ge=0.0, le=1.0),
    min_disputability: float | None = Query(None, ge=0.0),
    max_disputability: float | None = Query(None, ge=0.0),
) -> dict[str, Any]:
    """List ranked claims with search, filters, and bounded pagination."""
    run = get_completed_run(run_id)
    assert run.results is not None

    if q is not None and search is not None:
        raise _invalid_filter("search", ["Use either q or search, not both."])
    if min_ecs is not None and max_ecs is not None and min_ecs > max_ecs:
        raise _invalid_filter("ecs range", ["min_ecs must be <= max_ecs."])
    if (
        min_disputability is not None
        and max_disputability is not None
        and min_disputability > max_disputability
    ):
        raise _invalid_filter(
            "disputability range",
            ["min_disputability must be <= max_disputability."],
        )

    entries = list(run.results.results)
    text_query = q if q is not None else search
    if text_query is not None and text_query.strip():
        needle = text_query.strip().casefold()
        entries = [
            entry
            for entry in entries
            if needle in entry.claim.casefold()
            or needle in str(entry.claim_id).casefold()
        ]

    if priority is not None:
        entries = _filter_by_priority(entries, run.objective, priority)

    if conflict is not None:
        entries = [
            entry
            for entry in entries
            if _conflict_band(entry.ecs_score) == conflict
        ]

    if conflict_status is not None:
        entries = [
            entry
            for entry in entries
            if entry.conflict is not None
            and entry.conflict.status is conflict_status
        ]

    if has_evidence is not None:
        entries = [
            entry
            for entry in entries
            if bool(entry.evidence) is has_evidence
        ]

    if min_ecs is not None:
        entries = [
            entry
            for entry in entries
            if entry.ecs_score is not None and entry.ecs_score >= min_ecs
        ]
    if max_ecs is not None:
        entries = [
            entry
            for entry in entries
            if entry.ecs_score is not None and entry.ecs_score <= max_ecs
        ]
    if min_disputability is not None:
        entries = [
            entry
            for entry in entries
            if entry.original.disputability is not None
            and entry.original.disputability >= min_disputability
        ]
    if max_disputability is not None:
        entries = [
            entry
            for entry in entries
            if entry.original.disputability is not None
            and entry.original.disputability <= max_disputability
        ]

    resolved_sort: SortField = (
        sort
        if sort is not None
        else ("initial_rank" if ranking == "initial" else "final_rank")
    )
    resolved_order: Literal["asc", "desc"] = (
        order
        if order is not None
        else (
            "asc"
            if resolved_sort in {"final_rank", "initial_rank", "claim_id"}
            else "desc"
        )
    )
    entries = _sort_entries(entries, resolved_sort, resolved_order)

    matched_claims = len(entries)
    page = entries[offset : offset + limit]
    return {
        "run_id": run.run_id,
        "objective": run.objective.value,
        "status": run.status,
        "ranking": ranking,
        "sort": resolved_sort,
        "order": resolved_order,
        "total_claims": len(run.results.results),
        "matched_claims": matched_claims,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": offset + len(page) < matched_claims,
        "claims": [
            _entry_payload(entry, include_rank_change=True)
            for entry in page
        ],
    }


@router.get("/{run_id}/claims/{claim_id}")
def get_prioritization_run_claim(
    run_id: str,
    claim_id: str,
) -> dict[str, Any]:
    """Return one claim by its stable claim ID from the stored results."""
    run = get_completed_run(run_id)
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
    return {
        "run_id": run.run_id,
        "objective": run.objective.value,
        "status": run.status,
        "claim": _entry_payload(entry, include_rank_change=True),
    }

