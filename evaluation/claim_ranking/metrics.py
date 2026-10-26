"""Pure ranking metrics for claim-ranking evaluation.

This module does not import production ranking, retrieval, or model code. It
defines the metric conventions used by the claim-ranking evaluation harness:

- Precision@K and Recall@K use binary relevance (``grade >= 1`` by default).
- NDCG@K uses the full graded relevance scale with exponential gain.
- A K larger than the number of labeled claims in the evaluation set is
  reported as ``None`` rather than computed on an undersized set.
- Rankings are total orders. The module preserves a supplied order and never
  silently reorders ties.

The helper :func:`stable_rank_by_score` is evaluation-only. It provides a
deterministic order for score-based methods but does not combine signals or
change production ranking behavior.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real

#: Default K values to pre-register for a claim-ranking evaluation.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

#: Minimum grade counted as relevant for Precision@K and Recall@K.
DEFAULT_RELEVANT_MIN_GRADE = 1

#: Highest allowed human relevance grade. Grades are 0, 1, 2, and 3.
MAX_RELEVANCE_GRADE = 3


class EvaluationInputError(ValueError):
    """Raised when metric inputs are malformed or inconsistent."""


def validate_k(k: object) -> int:
    """Validate and return one positive integer K."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise EvaluationInputError(
            f"K must be an integer; got {type(k).__name__}."
        )
    if k <= 0:
        raise EvaluationInputError(f"K must be >= 1; got {k}.")
    return k


def validate_k_values(k_values: Iterable[object]) -> tuple[int, ...]:
    """Validate and return a non-empty tuple of unique positive K values."""

    if isinstance(k_values, (str, bytes, bytearray)):
        raise EvaluationInputError(
            "k_values must be an iterable of positive integers."
        )
    try:
        values = tuple(k_values)
    except TypeError as exc:
        raise EvaluationInputError(
            "k_values must be an iterable of positive integers."
        ) from exc
    if not values:
        raise EvaluationInputError("k_values must contain at least one K value.")
    validated = tuple(validate_k(value) for value in values)
    if len(set(validated)) != len(validated):
        raise EvaluationInputError("k_values must not contain duplicates.")
    return validated


def validate_relevant_min_grade(value: object) -> int:
    """Validate the minimum grade used for binary relevance."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationInputError(
            "relevant_min_grade must be an integer from 1 to "
            f"{MAX_RELEVANCE_GRADE}; got {type(value).__name__}."
        )
    if value < 1 or value > MAX_RELEVANCE_GRADE:
        raise EvaluationInputError(
            "relevant_min_grade must be from 1 to "
            f"{MAX_RELEVANCE_GRADE}; got {value}."
        )
    return value


def _validate_relevance(relevance: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(relevance, Mapping):
        raise EvaluationInputError(
            "relevance must be a mapping of claim IDs to integer grades."
        )

    validated: dict[str, int] = {}
    for claim_id, grade in relevance.items():
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise EvaluationInputError(
                "Every relevance key must be a non-empty claim ID string."
            )
        if isinstance(grade, bool) or not isinstance(grade, int):
            raise EvaluationInputError(
                f"relevance[{claim_id!r}] must be an integer grade; "
                f"got {type(grade).__name__}."
            )
        if grade < 0 or grade > MAX_RELEVANCE_GRADE:
            raise EvaluationInputError(
                f"relevance[{claim_id!r}] must be from 0 to "
                f"{MAX_RELEVANCE_GRADE}; got {grade}."
            )
        validated[claim_id] = grade
    return validated


def _validate_ranking(
    ranked_claim_ids: Sequence[str],
    relevance: Mapping[str, int],
) -> list[str]:
    if isinstance(ranked_claim_ids, (str, bytes, bytearray)):
        raise EvaluationInputError(
            "ranked_claim_ids must be a sequence of claim IDs, not a string."
        )
    try:
        ranking = list(ranked_claim_ids)
    except TypeError as exc:
        raise EvaluationInputError(
            "ranked_claim_ids must be a sequence of claim IDs."
        ) from exc

    seen: set[str] = set()
    for index, claim_id in enumerate(ranking):
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise EvaluationInputError(
                f"Ranking item {index} must be a non-empty claim ID string."
            )
        if claim_id in seen:
            raise EvaluationInputError(
                f"Ranking contains duplicate claim ID {claim_id!r} "
                f"at index {index}."
            )
        if claim_id not in relevance:
            raise EvaluationInputError(
                f"Ranking contains claim ID {claim_id!r} that is not present "
                "in the relevance labels."
            )
        seen.add(claim_id)
    return ranking


def _precision(
    ranking: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
    relevant_min_grade: int,
) -> float | None:
    if k > len(relevance):
        return None
    relevant = sum(
        1
        for claim_id in ranking[:k]
        if relevance[claim_id] >= relevant_min_grade
    )
    return relevant / k


def _recall(
    ranking: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
    relevant_min_grade: int,
) -> float | None:
    if k > len(relevance):
        return None
    total_relevant = sum(
        1
        for grade in relevance.values()
        if grade >= relevant_min_grade
    )
    if total_relevant == 0:
        return None
    relevant = sum(
        1
        for claim_id in ranking[:k]
        if relevance[claim_id] >= relevant_min_grade
    )
    return relevant / total_relevant


def _ndcg(
    ranking: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float | None:
    if k > len(relevance):
        return None

    dcg = 0.0
    for position, claim_id in enumerate(ranking[:k], start=1):
        gain = (2 ** relevance[claim_id]) - 1
        dcg += gain / math.log2(position + 1)

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for position, grade in enumerate(ideal_grades, start=1):
        gain = (2 ** grade) - 1
        idcg += gain / math.log2(position + 1)

    if idcg == 0.0:
        return None
    return dcg / idcg


def precision_at_k(
    ranked_claim_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
    *,
    relevant_min_grade: int = DEFAULT_RELEVANT_MIN_GRADE,
) -> float | None:
    """Return Precision@K, or ``None`` when K exceeds the labeled set."""

    k = validate_k(k)
    threshold = validate_relevant_min_grade(relevant_min_grade)
    validated_relevance = _validate_relevance(relevance)
    ranking = _validate_ranking(ranked_claim_ids, validated_relevance)
    return _precision(ranking, validated_relevance, k, threshold)


def recall_at_k(
    ranked_claim_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
    *,
    relevant_min_grade: int = DEFAULT_RELEVANT_MIN_GRADE,
) -> float | None:
    """Return Recall@K, or ``None`` when undefined."""

    k = validate_k(k)
    threshold = validate_relevant_min_grade(relevant_min_grade)
    validated_relevance = _validate_relevance(relevance)
    ranking = _validate_ranking(ranked_claim_ids, validated_relevance)
    return _recall(ranking, validated_relevance, k, threshold)


def ndcg_at_k(
    ranked_claim_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float | None:
    """Return graded NDCG@K, or ``None`` when undefined.

    Gain is ``2**grade - 1`` and the discount is ``1 / log2(position + 1)``.
    The ideal DCG uses all labels in the evaluation set, sorted by descending
    grade and truncated to K.
    """

    k = validate_k(k)
    validated_relevance = _validate_relevance(relevance)
    ranking = _validate_ranking(ranked_claim_ids, validated_relevance)
    return _ndcg(ranking, validated_relevance, k)


def evaluate_ranking(
    ranked_claim_ids: Sequence[str],
    relevance: Mapping[str, int],
    k_values: Iterable[object] = DEFAULT_K_VALUES,
    *,
    relevant_min_grade: int = DEFAULT_RELEVANT_MIN_GRADE,
) -> dict[int, dict[str, float | None]]:
    """Compute all requested metrics for one ranking.

    The relevance mapping is the complete evaluation set, including relevant
    and non-relevant claims. A ranking may contain only a subset of that set;
    missing positions are treated as non-relevant for Precision@K, Recall@K,
    and NDCG@K. Duplicate ranking IDs and unknown IDs are rejected.

    A metric is ``None`` when K exceeds the number of labeled claims. Recall
    and NDCG are also ``None`` when the evaluation set contains no relevant
    claims.
    """

    validated_relevance = _validate_relevance(relevance)
    ranking = _validate_ranking(ranked_claim_ids, validated_relevance)
    validated_k_values = validate_k_values(k_values)
    threshold = validate_relevant_min_grade(relevant_min_grade)

    return {
        k: {
            "precision": _precision(
                ranking,
                validated_relevance,
                k,
                threshold,
            ),
            "recall": _recall(
                ranking,
                validated_relevance,
                k,
                threshold,
            ),
            "ndcg": _ndcg(ranking, validated_relevance, k),
        }
        for k in validated_k_values
    }


def stable_rank_by_score(
    claim_ids: Iterable[str],
    scores: Mapping[str, float | None],
) -> list[str]:
    """Return a deterministic evaluation-only order for one score signal.

    Defined scores sort descending. ``None`` scores sort after all defined
    scores. Ties preserve the input claim order, matching the production
    prioritizer's stable-sort behavior. This helper does not combine signals.
    """

    if isinstance(claim_ids, (str, bytes, bytearray)):
        raise EvaluationInputError(
            "claim_ids must be an iterable of claim IDs, not a string."
        )
    try:
        ids = list(claim_ids)
    except TypeError as exc:
        raise EvaluationInputError(
            "claim_ids must be an iterable of claim IDs."
        ) from exc

    seen: set[str] = set()
    for index, claim_id in enumerate(ids):
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise EvaluationInputError(
                f"claim_ids item {index} must be a non-empty string."
            )
        if claim_id in seen:
            raise EvaluationInputError(
                f"claim_ids contains duplicate claim ID {claim_id!r}."
            )
        seen.add(claim_id)

    if not isinstance(scores, Mapping):
        raise EvaluationInputError(
            "scores must be a mapping of claim IDs to numbers or None."
        )

    normalized: dict[str, float | None] = {}
    for claim_id in ids:
        if claim_id not in scores:
            raise EvaluationInputError(
                f"Missing score for claim ID {claim_id!r}."
            )
        value = scores[claim_id]
        if value is None:
            normalized[claim_id] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EvaluationInputError(
                f"scores[{claim_id!r}] must be a finite number or None; "
                f"got {type(value).__name__}."
            )
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationInputError(
                f"scores[{claim_id!r}] must be a finite number or None."
            ) from exc
        if not math.isfinite(number):
            raise EvaluationInputError(
                f"scores[{claim_id!r}] must be finite; got {number!r}."
            )
        normalized[claim_id] = number

    indexed = list(enumerate(ids))
    indexed.sort(
        key=lambda item: (
            1 if normalized[item[1]] is None else 0,
            0.0 if normalized[item[1]] is None else -normalized[item[1]],
            item[0],
        )
    )
    return [claim_id for _, claim_id in indexed]


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_RELEVANT_MIN_GRADE",
    "MAX_RELEVANCE_GRADE",
    "EvaluationInputError",
    "evaluate_ranking",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "stable_rank_by_score",
    "validate_k",
    "validate_k_values",
    "validate_relevant_min_grade",
]
