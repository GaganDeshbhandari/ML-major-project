"""Offline tests for claim-ranking metric calculations.

All fixtures are synthetic and exist only in memory. They are not saved as
evaluation results and do not imply real-world performance.
"""

from __future__ import annotations

import math

import pytest

from evaluation.claim_ranking.metrics import (
    EvaluationInputError,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    stable_rank_by_score,
)


def _graded_relevance() -> dict[str, int]:
    return {"A": 3, "B": 2, "C": 1, "D": 0, "E": 0}


def test_precision_at_k_hand_computed() -> None:
    relevance = _graded_relevance()
    ranking = ["A", "D", "B", "C", "E"]

    assert precision_at_k(ranking, relevance, 1) == pytest.approx(1.0)
    assert precision_at_k(ranking, relevance, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranking, relevance, 5) == pytest.approx(3 / 5)


def test_recall_at_k_hand_computed() -> None:
    relevance = _graded_relevance()
    ranking = ["A", "D", "B", "C", "E"]

    assert recall_at_k(ranking, relevance, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, relevance, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, relevance, 5) == pytest.approx(1.0)


def test_ndcg_at_k_graded_hand_computed() -> None:
    relevance = _graded_relevance()
    ranking = ["A", "D", "B", "C", "E"]

    # DCG@3 = 7 + 0 + 3/log2(4) = 8.5.
    # IDCG@3 = 7 + 3/log2(3) + 1/log2(4) = 9.392789...
    assert ndcg_at_k(ranking, relevance, 3) == pytest.approx(
        8.5 / (7.0 + 3.0 / math.log2(3.0) + 0.5),
        abs=1e-12,
    )


def test_ndcg_at_k_binary_hand_computed() -> None:
    relevance = {"A": 1, "B": 0, "C": 1}
    ranking = ["A", "B", "C"]

    expected = 1.5 / (1.0 + 1.0 / math.log2(3.0))
    assert ndcg_at_k(ranking, relevance, 3) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    relevance = {"A": 3, "B": 2, "C": 1, "D": 0}
    ranking = ["A", "B", "C", "D"]

    assert ndcg_at_k(ranking, relevance, 4) == pytest.approx(1.0)


def test_binary_threshold_can_be_changed() -> None:
    relevance = {"A": 3, "B": 2, "C": 1}
    ranking = ["A", "C", "B"]

    assert precision_at_k(
        ranking,
        relevance,
        3,
        relevant_min_grade=2,
    ) == pytest.approx(2 / 3)
    assert recall_at_k(
        ranking,
        relevance,
        3,
        relevant_min_grade=2,
    ) == pytest.approx(1.0)


def test_empty_dataset_returns_none() -> None:
    result = evaluate_ranking([], {}, [1, 3])

    assert result[1] == {
        "precision": None,
        "recall": None,
        "ndcg": None,
    }
    assert result[3] == {
        "precision": None,
        "recall": None,
        "ndcg": None,
    }


def test_undersized_dataset_returns_none_for_k_above_size() -> None:
    relevance = {"A": 1, "B": 0, "C": 0}
    result = evaluate_ranking(["A", "B", "C"], relevance, [1, 3, 5])

    assert result[1]["precision"] == pytest.approx(1.0)
    assert result[5] == {
        "precision": None,
        "recall": None,
        "ndcg": None,
    }


def test_short_ranking_on_large_enough_set_is_penalized() -> None:
    relevance = {"A": 1, "B": 1, "C": 0}
    result = evaluate_ranking(["A"], relevance, [3])[3]

    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["ndcg"] == pytest.approx(
        1.0 / (1.0 + 1.0 / math.log2(3.0)),
        abs=1e-12,
    )


def test_no_relevant_items_handling() -> None:
    relevance = {"A": 0, "B": 0}
    result = evaluate_ranking(["A", "B"], relevance, [1])[1]

    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] is None
    assert result["ndcg"] is None


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "1"])
def test_invalid_k_rejected(k: object) -> None:
    with pytest.raises(EvaluationInputError):
        precision_at_k(["A"], {"A": 1}, k)  # type: ignore[arg-type]


def test_duplicate_k_values_rejected() -> None:
    with pytest.raises(EvaluationInputError):
        evaluate_ranking(["A"], {"A": 1}, [1, 1])


def test_duplicate_ranking_ids_rejected() -> None:
    with pytest.raises(EvaluationInputError):
        precision_at_k(["A", "A"], {"A": 1}, 1)


def test_unknown_ranking_id_rejected() -> None:
    with pytest.raises(EvaluationInputError):
        precision_at_k(["A", "Z"], {"A": 1}, 1)


@pytest.mark.parametrize("grade", [-1, 4, True, 1.5, "3"])
def test_invalid_relevance_grade_rejected(grade: object) -> None:
    with pytest.raises(EvaluationInputError):
        precision_at_k(["A"], {"A": grade}, 1)  # type: ignore[dict-item]


@pytest.mark.parametrize("threshold", [0, 4, True, 1.5])
def test_invalid_relevance_threshold_rejected(threshold: object) -> None:
    with pytest.raises(EvaluationInputError):
        precision_at_k(
            ["A"],
            {"A": 1},
            1,
            relevant_min_grade=threshold,  # type: ignore[arg-type]
        )


def test_metric_calculations_are_reproducible() -> None:
    relevance = _graded_relevance()
    ranking = ["A", "D", "B", "C", "E"]

    first = evaluate_ranking(ranking, relevance, [1, 3, 5])
    second = evaluate_ranking(ranking, relevance, [1, 3, 5])

    assert first == second


def test_stable_rank_by_score_preserves_input_order_for_ties() -> None:
    ordered = stable_rank_by_score(
        ["A", "B", "C"],
        {"A": 0.5, "B": 0.5, "C": 0.9},
    )

    assert ordered == ["C", "A", "B"]


def test_stable_rank_by_score_places_missing_scores_last() -> None:
    ordered = stable_rank_by_score(
        ["A", "B", "C", "D"],
        {"A": None, "B": 0.1, "C": None, "D": 0.9},
    )

    assert ordered == ["D", "B", "A", "C"]


def test_stable_rank_by_score_rejects_missing_score() -> None:
    with pytest.raises(EvaluationInputError):
        stable_rank_by_score(["A", "B"], {"A": 1.0})


def test_stable_rank_by_score_rejects_duplicate_ids() -> None:
    with pytest.raises(EvaluationInputError):
        stable_rank_by_score(["A", "A"], {"A": 1.0})


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, True])
def test_stable_rank_by_score_rejects_invalid_scores(score: object) -> None:
    with pytest.raises(EvaluationInputError):
        stable_rank_by_score(["A"], {"A": score})  # type: ignore[dict-item]
