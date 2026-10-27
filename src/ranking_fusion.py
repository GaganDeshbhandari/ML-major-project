"""Pure Reciprocal Rank Fusion (RRF) for claim rankings.

The fusion formula uses 1-based ranks:

``score(claim) = 1 / (k + baseline_rank)``

plus, when the claim has a valid ECS rank:

``1 / (k + ecs_rank)``

ECS ranks are assigned only to claims with a finite, already-computed ECS
value. Higher ECS ranks first; ECS ties preserve baseline order. Claims without
a valid ECS keep only their baseline contribution. Missing ECS is never
converted to ``0.0``.

This module does not compute ECS, run NLI, retrieve evidence, load models, or
change any objective-specific score. It only combines caller-supplied ranks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

#: Claim identifiers used by the existing prioritization and evaluation code.
ClaimId: TypeAlias = str | int

#: Standard RRF smoothing constant used when the caller does not override it.
DEFAULT_RRF_K: float = 60.0


class RankingFusionError(ValueError):
    """Raised when ranking-fusion inputs are malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class FusedRank:
    """One claim's baseline, ECS, and fused rank information."""

    claim_id: ClaimId
    baseline_rank: int
    ecs_rank: int | None
    rrf_score: float
    fused_rank: int


def validate_rrf_k(k: object) -> float:
    """Validate and return the positive finite RRF smoothing constant."""

    if isinstance(k, bool) or not isinstance(k, Real):
        raise RankingFusionError(
            f"k must be a positive finite number; got {type(k).__name__}."
        )
    try:
        number = float(k)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RankingFusionError(
            f"k must be a positive finite number; got {k!r}."
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise RankingFusionError(
            f"k must be a positive finite number; got {k!r}."
        )
    return number


def _validate_claim_id(value: object, field: str) -> ClaimId:
    if isinstance(value, bool):
        raise RankingFusionError(
            f"{field} must be a string or integer claim ID; got bool."
        )
    if isinstance(value, str):
        if not value.strip():
            raise RankingFusionError(
                f"{field} must be a non-empty claim ID string."
            )
        return value
    if isinstance(value, int):
        return value
    raise RankingFusionError(
        f"{field} must be a string or integer claim ID; "
        f"got {type(value).__name__}."
    )


def _validate_claim_ids(claim_ids: Iterable[ClaimId]) -> list[ClaimId]:
    if isinstance(claim_ids, (str, bytes, bytearray)):
        raise RankingFusionError(
            "claim_ids must be an iterable of claim IDs, not a string."
        )
    try:
        raw_ids = list(claim_ids)
    except TypeError as exc:
        raise RankingFusionError(
            "claim_ids must be an iterable of claim IDs."
        ) from exc

    ids: list[ClaimId] = []
    seen: set[ClaimId] = set()
    for index, raw_id in enumerate(raw_ids):
        claim_id = _validate_claim_id(raw_id, f"claim_ids[{index}]")
        if claim_id in seen:
            raise RankingFusionError(
                f"claim_ids contains duplicate claim ID {claim_id!r}."
            )
        seen.add(claim_id)
        ids.append(claim_id)
    return ids


def _mapping_key_details(
    expected: set[ClaimId],
    supplied: Mapping[ClaimId, object],
) -> str:
    missing = sorted(repr(claim_id) for claim_id in expected - set(supplied))
    extra = sorted(repr(claim_id) for claim_id in set(supplied) - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if extra:
        details.append(f"unexpected {extra}")
    return "; ".join(details)


def _validate_baseline_ranks(
    claim_ids: Sequence[ClaimId],
    baseline_ranks: Mapping[ClaimId, int],
) -> dict[ClaimId, int]:
    if not isinstance(baseline_ranks, Mapping):
        raise RankingFusionError(
            "baseline_ranks must be a mapping of claim IDs to 1-based ranks."
        )

    expected = set(claim_ids)
    if set(baseline_ranks) != expected:
        raise RankingFusionError(
            "baseline_ranks must contain exactly one entry for every claim ID "
            f"({_mapping_key_details(expected, baseline_ranks)})."
        )

    ranks: dict[ClaimId, int] = {}
    for claim_id in claim_ids:
        raw_rank = baseline_ranks[claim_id]
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise RankingFusionError(
                f"baseline_ranks[{claim_id!r}] must be a positive integer; "
                f"got {type(raw_rank).__name__}."
            )
        if raw_rank < 1:
            raise RankingFusionError(
                f"baseline_ranks[{claim_id!r}] must be >= 1; got {raw_rank}."
            )
        ranks[claim_id] = raw_rank
    return ranks


def _normalize_ecs(value: object, claim_id: ClaimId) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RankingFusionError(
            f"ecs_scores[{claim_id!r}] must be a finite number or None; "
            f"got {type(value).__name__}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RankingFusionError(
            f"ecs_scores[{claim_id!r}] must be a finite number or None."
        ) from exc
    if not math.isfinite(number):
        return None
    return number


def _validate_ecs_scores(
    claim_ids: Sequence[ClaimId],
    ecs_scores: Mapping[ClaimId, float | None],
) -> dict[ClaimId, float | None]:
    if not isinstance(ecs_scores, Mapping):
        raise RankingFusionError(
            "ecs_scores must be a mapping of claim IDs to finite numbers or "
            "None."
        )

    expected = set(claim_ids)
    if set(ecs_scores) != expected:
        raise RankingFusionError(
            "ecs_scores must contain exactly one entry for every claim ID "
            f"({_mapping_key_details(expected, ecs_scores)})."
        )

    scores: dict[ClaimId, float | None] = {}
    for claim_id in claim_ids:
        scores[claim_id] = _normalize_ecs(
            ecs_scores[claim_id],
            claim_id,
        )
    return scores


def _validated_inputs(
    claim_ids: Iterable[ClaimId],
    baseline_ranks: Mapping[ClaimId, int],
    ecs_scores: Mapping[ClaimId, float | None],
) -> tuple[list[ClaimId], dict[ClaimId, int], dict[ClaimId, float | None]]:
    ids = _validate_claim_ids(claim_ids)
    ranks = _validate_baseline_ranks(ids, baseline_ranks)
    scores = _validate_ecs_scores(ids, ecs_scores)
    return ids, ranks, scores


def _ordered_ecs_claim_ids(
    claim_ids: Sequence[ClaimId],
    baseline_ranks: Mapping[ClaimId, int],
    ecs_scores: Mapping[ClaimId, float | None],
) -> list[ClaimId]:
    eligible: list[tuple[int, ClaimId, float]] = []
    for index, claim_id in enumerate(claim_ids):
        score = ecs_scores[claim_id]
        if score is not None:
            eligible.append((index, claim_id, score))
    eligible.sort(
        key=lambda item: (
            -item[2],
            baseline_ranks[item[1]],
            item[0],
        )
    )
    return [claim_id for _, claim_id, _ in eligible]


def rank_claim_ids_by_ecs(
    claim_ids: Iterable[ClaimId],
    baseline_ranks: Mapping[ClaimId, int],
    ecs_scores: Mapping[ClaimId, float | None],
) -> tuple[ClaimId, ...]:
    """Return claims with valid ECS, highest first, ties in baseline order.

    ``None`` and non-finite ECS values receive no ECS rank and are omitted from
    the returned ranking. A finite ECS of ``0.0`` is valid.
    """

    ids, ranks, scores = _validated_inputs(
        claim_ids,
        baseline_ranks,
        ecs_scores,
    )
    return tuple(_ordered_ecs_claim_ids(ids, ranks, scores))


def fuse_rankings(
    claim_ids: Iterable[ClaimId],
    baseline_ranks: Mapping[ClaimId, int],
    ecs_scores: Mapping[ClaimId, float | None],
    *,
    k: float = DEFAULT_RRF_K,
) -> tuple[FusedRank, ...]:
    """Fuse one baseline ranking with valid ECS ranks using RRF.

    Every supplied claim appears exactly once. Claims without a valid ECS keep
    only their baseline contribution. The final order is descending RRF score,
    then baseline rank, then original input order.
    """

    ids, ranks, scores = _validated_inputs(
        claim_ids,
        baseline_ranks,
        ecs_scores,
    )
    smoothing = validate_rrf_k(k)
    ordered_ecs = _ordered_ecs_claim_ids(ids, ranks, scores)
    ecs_ranks = {
        claim_id: rank
        for rank, claim_id in enumerate(ordered_ecs, start=1)
    }

    indexed: list[tuple[int, ClaimId, int, int | None, float]] = []
    for index, claim_id in enumerate(ids):
        baseline_rank = ranks[claim_id]
        ecs_rank = ecs_ranks.get(claim_id)
        score = 1.0 / (smoothing + baseline_rank)
        if ecs_rank is not None:
            score += 1.0 / (smoothing + ecs_rank)
        indexed.append(
            (index, claim_id, baseline_rank, ecs_rank, score)
        )

    indexed.sort(
        key=lambda item: (
            -item[4],
            item[2],
            item[0],
        )
    )
    return tuple(
        FusedRank(
            claim_id=claim_id,
            baseline_rank=baseline_rank,
            ecs_rank=ecs_rank,
            rrf_score=score,
            fused_rank=position,
        )
        for position, (
            _,
            claim_id,
            baseline_rank,
            ecs_rank,
            score,
        ) in enumerate(indexed, start=1)
    )


__all__ = [
    "DEFAULT_RRF_K",
    "ClaimId",
    "FusedRank",
    "RankingFusionError",
    "fuse_rankings",
    "rank_claim_ids_by_ecs",
    "validate_rrf_k",
]
