"""Opt-in Reciprocal Rank Fusion for prioritization/retrieval results.

This adapter consumes results that have already been prioritized and, when
enabled, scored with ECS. It does not retrieve evidence, run NLI, rescore
O1/O2/O3, or change the default order returned by
:func:`src.pipeline.prioritize_and_retrieve`.

Callers enable fusion explicitly by passing pipeline results to
:func:`fuse_prioritization_results`. The original result objects are retained
inside :class:`FusedPrioritizationResult`; their ranks, objective scores, and
evidence dictionaries are never mutated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from src.evidence_conflict import ECSStatus
from src.prioritization import Objective
from src.ranking_fusion import (
    DEFAULT_RRF_K,
    RankingFusionError,
    fuse_rankings,
    validate_rrf_k,
)

from .prioritization_retrieval import PrioritizationRetrievalResult


@dataclass(frozen=True, slots=True)
class FusedPrioritizationResult:
    """One pipeline result plus its additive RRF rank information."""

    result: PrioritizationRetrievalResult
    baseline_rank: int
    ecs_rank: int | None
    rrf_score: float
    fused_rank: int

    @property
    def claim_id(self) -> str | int:
        """Return the original claim ID."""

        return self.result.claim_id

    @property
    def objective(self) -> Objective:
        """Return the selected objective of the original result."""

        return self.result.objective

    @property
    def ecs_score(self) -> float | None:
        """Return the computed ECS value, or ``None`` when unavailable."""

        return _computed_ecs(self.result)


def _materialize_results(
    results: Iterable[PrioritizationRetrievalResult],
) -> list[PrioritizationRetrievalResult]:
    if isinstance(results, (str, bytes, bytearray)):
        raise RankingFusionError(
            "results must be an iterable of PrioritizationRetrievalResult "
            "objects, not a string."
        )
    try:
        values = list(results)
    except TypeError as exc:
        raise RankingFusionError(
            "results must be an iterable of PrioritizationRetrievalResult "
            "objects."
        ) from exc

    for index, result in enumerate(values):
        if not isinstance(result, PrioritizationRetrievalResult):
            raise RankingFusionError(
                f"results[{index}] must be a PrioritizationRetrievalResult; "
                f"got {type(result).__name__}."
            )
    return values


def _computed_ecs(result: PrioritizationRetrievalResult) -> float | None:
    """Return ECS only for a COMPUTED result with a finite value."""

    ecs_result = result.ecs_result
    if ecs_result is None or ecs_result.status is not ECSStatus.COMPUTED:
        return None

    value = ecs_result.ecs
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RankingFusionError(
            f"Computed ECS for claim {result.claim_id!r} must be a finite "
            f"number or None; got {type(value).__name__}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RankingFusionError(
            f"Computed ECS for claim {result.claim_id!r} must be a finite "
            "number or None."
        ) from exc
    if not math.isfinite(number):
        return None
    return number


def _validate_result_identity(
    values: list[PrioritizationRetrievalResult],
) -> tuple[list[str | int], dict[str | int, int], dict[str | int, float | None]]:
    if not values:
        return [], {}, {}

    objectives = {result.objective for result in values}
    if len(objectives) != 1:
        raise RankingFusionError(
            "All results must belong to the same selected objective; got "
            f"{sorted(str(objective) for objective in objectives)}."
        )

    claim_ids: list[str | int] = []
    baseline_ranks: dict[str | int, int] = {}
    ecs_scores: dict[str | int, float | None] = {}
    seen: set[str | int] = set()

    for index, result in enumerate(values):
        claim_id = result.claim_id
        if isinstance(claim_id, bool) or not isinstance(claim_id, (str, int)):
            raise RankingFusionError(
                f"results[{index}].claim_id must be a string or integer; "
                f"got {type(claim_id).__name__}."
            )
        if isinstance(claim_id, str) and not claim_id.strip():
            raise RankingFusionError(
                f"results[{index}].claim_id must be a non-empty string."
            )
        if claim_id in seen:
            raise RankingFusionError(
                f"Duplicate claim ID {claim_id!r} in results."
            )
        seen.add(claim_id)

        rank = result.initial_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise RankingFusionError(
                f"results[{index}].initial_rank must be a positive integer; "
                f"got {rank!r}."
            )

        claim_ids.append(claim_id)
        baseline_ranks[claim_id] = rank
        ecs_scores[claim_id] = _computed_ecs(result)

    return claim_ids, baseline_ranks, ecs_scores


def fuse_prioritization_results(
    results: Iterable[PrioritizationRetrievalResult],
    *,
    k: float = DEFAULT_RRF_K,
) -> list[FusedPrioritizationResult]:
    """Fuse one selected objective's results with computed ECS using RRF.

    The caller opts in by invoking this function. Results are not reordered in
    place, and the original ``initial_rank``, objective score fields, evidence
    lists, and ECS result objects remain unchanged.

    Claims without a valid ECS keep their baseline contribution only. Every
    input claim appears exactly once in the returned list.
    """

    values = _materialize_results(results)
    if not values:
        validate_rrf_k(k)
        return []

    claim_ids, baseline_ranks, ecs_scores = _validate_result_identity(values)
    fused = fuse_rankings(
        claim_ids,
        baseline_ranks,
        ecs_scores,
        k=k,
    )
    by_claim_id = {result.claim_id: result for result in values}

    return [
        FusedPrioritizationResult(
            result=by_claim_id[item.claim_id],
            baseline_rank=item.baseline_rank,
            ecs_rank=item.ecs_rank,
            rrf_score=item.rrf_score,
            fused_rank=item.fused_rank,
        )
        for item in fused
    ]


__all__ = [
    "FusedPrioritizationResult",
    "fuse_prioritization_results",
]
