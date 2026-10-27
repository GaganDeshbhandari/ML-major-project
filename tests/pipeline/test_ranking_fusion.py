"""Offline tests for opt-in ECS ranking fusion with RRF.

All claims, ECS results, and evidence are synthetic fixtures. These tests do not
load an NLI model, retrieve evidence, access the network, or use a GPU.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.evidence_conflict import (
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.pipeline import (
    DEFAULT_RRF_K,
    FusedPrioritizationResult,
    PrioritizationRetrievalResult,
    RankingFusionError,
    fuse_prioritization_results,
    prioritize_and_retrieve,
)
from src.prioritization import (
    FEATURE_COLUMNS,
    Claim,
    Objective,
    UserRating,
)
from src.ranking_fusion import (
    fuse_rankings,
    rank_claim_ids_by_ecs,
    validate_rrf_k,
)

_MISSING = object()


def _ecs_result(
    claim: str,
    ecs: float | None,
    status: ECSStatus,
) -> EvidenceConflictResult:
    return EvidenceConflictResult(
        claim=claim,
        predictions=(),
        counts=StanceCounts(),
        status=status,
        ecs=ecs,
    )


def _result(
    claim_id: str,
    rank: int,
    ecs: object = _MISSING,
    *,
    status: ECSStatus | None = None,
    evidence: list[dict[str, Any]] | None = None,
    predicted_truth: str | None = None,
    predicted_tpb: str | None = None,
    disputability: float | None = None,
) -> PrioritizationRetrievalResult:
    if status is None:
        status = (
            ECSStatus.COMPUTED
            if ecs is not _MISSING
            else ECSStatus.NO_EVIDENCE
        )
    value = None if ecs is _MISSING else ecs
    return PrioritizationRetrievalResult(
        claim_id=claim_id,
        claim=f"Claim {claim_id}",
        objective=Objective.O1,
        initial_rank=rank,
        evidence=[] if evidence is None else evidence,
        predicted_truth=predicted_truth,
        predicted_tpb=predicted_tpb,
        disputability=disputability,
        ecs_result=_ecs_result(
            f"Claim {claim_id}",
            value,  # type: ignore[arg-type]
            status,
        ),
    )


class _IdentityScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


class _ConstantModel:
    def __init__(self, label: str, classes: tuple[str, ...]) -> None:
        self.label = label
        self.classes_ = np.asarray(classes, dtype=object)

    def predict(self, values: object) -> np.ndarray:
        return np.asarray(
            [self.label] * len(np.asarray(values)),
            dtype=object,
        )

    def predict_proba(self, values: object) -> np.ndarray:
        return np.asarray(
            [
                [0.9 if value == self.label else 0.1 for value in self.classes_]
                for _ in np.asarray(values)
            ]
        )


class _FakeStore:
    def __init__(self) -> None:
        self.scaler = _IdentityScaler()
        self.o1_model = _ConstantModel("True", ("False", "True"))
        self.o2_model = _ConstantModel("Low", ("High", "Low"))


class _FakeStanceClassifier:
    def __init__(
        self,
        predictions_by_claim: dict[str, list[StancePrediction]],
    ) -> None:
        self.predictions_by_claim = predictions_by_claim
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def predict(
        self,
        claim: str,
        evidence: list[dict[str, Any]],
    ) -> list[StancePrediction]:
        self.calls.append((claim, evidence))
        return list(self.predictions_by_claim[claim])


def _pipeline_evidence(claim: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "title": f"{claim} source {index}",
            "url": f"https://example.test/{claim[-1]}/{index}",
            "content": f"{claim} evidence {index}",
            "relevance_score": 0.9 - 0.1 * index,
            "rerank_score": 0.8 - 0.1 * index,
        }
        for index in range(count)
    ]


def _pipeline_prediction(
    evidence_index: int,
    label: StanceLabel,
) -> StancePrediction:
    if label is StanceLabel.SUPPORTS:
        probabilities = StanceProbabilities(0.8, 0.1, 0.1)
    elif label is StanceLabel.REFUTES:
        probabilities = StanceProbabilities(0.1, 0.8, 0.1)
    else:
        probabilities = StanceProbabilities(0.1, 0.1, 0.8)
    return StancePrediction(
        evidence_index=evidence_index,
        label=label,
        confidence=0.8,
        probabilities=probabilities,
    )


def test_rrf_hand_calculated_scores_and_ranks() -> None:
    results = [
        _result("A", 1, 0.9),
        _result("B", 2, 0.8),
        _result("C", 3),
    ]

    fused = fuse_prioritization_results(results, k=60)

    assert [item.claim_id for item in fused] == ["A", "B", "C"]
    assert [item.fused_rank for item in fused] == [1, 2, 3]
    assert [item.ecs_rank for item in fused] == [1, 2, None]
    assert fused[0].ecs_score == pytest.approx(0.9)
    assert fused[2].ecs_score is None
    assert fused[0].rrf_score == pytest.approx(2 / 61)
    assert fused[1].rrf_score == pytest.approx(2 / 62)
    assert fused[2].rrf_score == pytest.approx(1 / 63)


def test_rrf_uses_one_based_ranks() -> None:
    fused = fuse_rankings(["A"], {"A": 1}, {"A": 1.0}, k=1)

    assert fused[0].baseline_rank == 1
    assert fused[0].ecs_rank == 1
    assert fused[0].rrf_score == pytest.approx(1.0)


def test_rrf_k_is_configurable() -> None:
    default = fuse_rankings(["A"], {"A": 1}, {"A": 1.0})
    custom = fuse_rankings(["A"], {"A": 1}, {"A": 1.0}, k=10)

    assert default[0].rrf_score == pytest.approx(2 / 61)
    assert custom[0].rrf_score == pytest.approx(2 / 11)
    assert validate_rrf_k(DEFAULT_RRF_K) == pytest.approx(60.0)


@pytest.mark.parametrize(
    "k",
    [0, -1, True, float("nan"), float("inf"), -float("inf"), "60"],
)
def test_invalid_rrf_k_rejected(k: object) -> None:
    with pytest.raises(RankingFusionError):
        fuse_rankings(["A"], {"A": 1}, {"A": None}, k=k)  # type: ignore[arg-type]


def test_empty_and_single_claim_inputs() -> None:
    assert fuse_prioritization_results([]) == []
    with pytest.raises(RankingFusionError):
        fuse_prioritization_results([], k=0)

    fused = fuse_prioritization_results([_result("A", 1)])

    assert len(fused) == 1
    assert fused[0].fused_rank == 1
    assert fused[0].ecs_rank is None
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_fusion_is_reproducible() -> None:
    results = [
        _result("A", 1, 0.9),
        _result("B", 2, 0.8),
        _result("C", 3),
    ]

    assert fuse_prioritization_results(results) == fuse_prioritization_results(
        results
    )


def test_high_ecs_can_outrank_a_better_baseline_rank() -> None:
    results = [
        _result("A", 1, 0.1),
        _result("B", 2, 0.9),
        _result("C", 3, 0.5),
    ]

    fused = fuse_prioritization_results(results, k=60)

    assert [item.claim_id for item in fused] == ["B", "A", "C"]
    assert fused[0].ecs_rank == 1
    assert fused[0].baseline_rank == 2
    assert fused[0].result.initial_rank == 2
    assert fused[0].fused_rank == 1
    assert fused[1].ecs_rank == 3


def test_computed_zero_ecs_is_valid() -> None:
    assert rank_claim_ids_by_ecs(
        ["A", "B"],
        {"A": 1, "B": 2},
        {"A": 0.0, "B": 0.5},
    ) == ("B", "A")

    fused = fuse_prioritization_results(
        [
            _result("A", 1, 0.0),
            _result("B", 2, 0.5),
            _result("C", 3),
        ],
        k=60,
    )
    by_id = {item.claim_id: item for item in fused}

    assert by_id["A"].ecs_rank == 2
    assert by_id["A"].rrf_score == pytest.approx(1 / 61 + 1 / 62)


def test_missing_ecs_is_not_treated_as_zero() -> None:
    fused = fuse_rankings(
        ["A", "B"],
        {"A": 1, "B": 2},
        {"A": None, "B": 0.0},
        k=1,
    )
    by_id = {item.claim_id: item for item in fused}

    assert by_id["A"].ecs_rank is None
    assert by_id["A"].rrf_score == pytest.approx(1 / 2)
    assert by_id["B"].ecs_rank == 1
    assert by_id["B"].rrf_score == pytest.approx(1 / 3 + 1 / 2)
    assert [item.claim_id for item in fused] == ["B", "A"]


def test_result_without_ecs_result_remains_eligible() -> None:
    result = _result("A", 1, 0.5)
    result.ecs_result = None

    fused = fuse_prioritization_results([result])

    assert len(fused) == 1
    assert fused[0].ecs_rank is None
    assert fused[0].ecs_score is None
    assert fused[0].rrf_score == pytest.approx(1 / 61)


@pytest.mark.parametrize(
    "status",
    [
        ECSStatus.DISABLED,
        ECSStatus.NO_EVIDENCE,
        ECSStatus.RETRIEVAL_ERROR,
        ECSStatus.NLI_ERROR,
        ECSStatus.INSUFFICIENT_STANCE,
    ],
)
def test_non_computed_statuses_receive_no_ecs_rank(
    status: ECSStatus,
) -> None:
    fused = fuse_prioritization_results(
        [
            _result("A", 1, status=status),
            _result("B", 2, 0.5),
        ],
        k=60,
    )
    by_id = {item.claim_id: item for item in fused}

    assert by_id["A"].ecs_rank is None
    assert by_id["A"].rrf_score == pytest.approx(1 / 61)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_ecs_is_excluded_from_ecs_ranking(value: float) -> None:
    fused = fuse_rankings(
        ["A", "B"],
        {"A": 1, "B": 2},
        {"A": value, "B": 0.5},
        k=1,
    )
    by_id = {item.claim_id: item for item in fused}

    assert by_id["A"].ecs_rank is None
    assert by_id["A"].rrf_score == pytest.approx(1 / 2)


def test_production_adapter_excludes_non_finite_ecs() -> None:
    result = _result("A", 1, 0.5)
    result.ecs_result = SimpleNamespace(  # type: ignore[assignment]
        status=ECSStatus.COMPUTED,
        ecs=math.inf,
    )

    fused = fuse_prioritization_results(
        [result, _result("B", 2, 0.5)],
        k=1,
    )
    by_id = {item.claim_id: item for item in fused}

    assert by_id["A"].ecs_rank is None
    assert by_id["A"].rrf_score == pytest.approx(1 / 2)


def test_ecs_ties_preserve_baseline_order() -> None:
    assert rank_claim_ids_by_ecs(
        ["A", "B", "C"],
        {"A": 2, "B": 1, "C": 3},
        {"A": 0.5, "B": 0.5, "C": 0.5},
    ) == ("B", "A", "C")


def test_final_rrf_ties_resolve_by_baseline_rank() -> None:
    fused = fuse_rankings(
        ["B", "A"],
        {"B": 2, "A": 1},
        {"B": 2.0, "A": 1.0},
        k=60,
    )

    assert [item.claim_id for item in fused] == ["A", "B"]
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)


def test_final_rrf_ties_resolve_by_input_order_when_ranks_tie() -> None:
    fused = fuse_rankings(
        ["B", "A"],
        {"B": 1, "A": 1},
        {"B": None, "A": None},
        k=60,
    )

    assert [item.claim_id for item in fused] == ["B", "A"]
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)


def test_fusion_does_not_mutate_inputs_or_evidence() -> None:
    evidence = [
        {
            "title": "Source A",
            "url": "https://example.test/a",
            "content": "Synthetic evidence.",
            "relevance_score": 0.9,
        }
    ]
    result = _result(
        "A",
        1,
        0.5,
        evidence=evidence,
        predicted_truth="False",
        predicted_tpb="High",
        disputability=0.4,
    )
    original = copy.deepcopy(result)

    fused = fuse_prioritization_results([result])

    assert result == original
    assert result.evidence is evidence
    assert result.evidence == original.evidence
    assert result.initial_rank == 1
    assert result.predicted_truth == "False"
    assert result.predicted_tpb == "High"
    assert result.disputability == pytest.approx(0.4)
    assert fused[0].result is result
    assert fused[0].baseline_rank == 1


def test_every_input_claim_appears_exactly_once() -> None:
    results = [
        _result("A", 1, 0.9),
        _result("B", 2),
        _result("C", 3, 0.1),
    ]

    fused = fuse_prioritization_results(results)

    assert len(fused) == len(results)
    assert {item.claim_id for item in fused} == {
        result.claim_id for result in results
    }
    assert all(isinstance(item, FusedPrioritizationResult) for item in fused)


def test_fusion_does_not_reorder_the_input_result_list() -> None:
    results = [
        _result("A", 1, 0.1),
        _result("B", 2, 0.9),
    ]
    before = [result.claim_id for result in results]

    fuse_prioritization_results(results)

    assert [result.claim_id for result in results] == before


def test_end_to_end_pipeline_results_can_be_fused() -> None:
    evidence_counts = {"Claim A": 1, "Claim B": 2, "Claim C": 4}

    def retrieve(
        claim: str,
        **kwargs: object,
    ) -> list[dict[str, Any]]:
        return _pipeline_evidence(claim, evidence_counts[claim])

    classifier = _FakeStanceClassifier(
        {
            "Claim A": [
                _pipeline_prediction(0, StanceLabel.SUPPORTS),
            ],
            "Claim B": [
                _pipeline_prediction(0, StanceLabel.SUPPORTS),
                _pipeline_prediction(1, StanceLabel.REFUTES),
            ],
            "Claim C": [
                _pipeline_prediction(0, StanceLabel.SUPPORTS),
                _pipeline_prediction(1, StanceLabel.REFUTES),
                _pipeline_prediction(2, StanceLabel.NOT_ENOUGH_INFO),
                _pipeline_prediction(3, StanceLabel.NOT_ENOUGH_INFO),
            ],
        }
    )
    claims = [
        Claim("a", "Claim A"),
        Claim("b", "Claim B"),
        Claim("c", "Claim C"),
    ]
    ratings = [
        UserRating("a", "True"),
        UserRating("a", "False"),
        UserRating("b", "True"),
        UserRating("b", "Mostly True"),
        UserRating("c", "Mixture"),
        UserRating("c", "Mostly True"),
    ]

    results = prioritize_and_retrieve(
        claims,
        ratings,
        Objective.O1,
        artifact_store=_FakeStore(),
        retrieve=retrieve,
        stance_classifier=classifier,
    )

    assert [result.claim_id for result in results] == ["a", "b", "c"]
    assert [result.initial_rank for result in results] == [1, 2, 3]
    assert [len(call[1]) for call in classifier.calls] == [1, 2, 4]

    ecs_by_id = {
        result.claim_id: result.ecs_result
        for result in results
    }
    assert ecs_by_id["a"] is not None
    assert ecs_by_id["b"] is not None
    assert ecs_by_id["c"] is not None
    assert ecs_by_id["a"].status is ECSStatus.COMPUTED
    assert ecs_by_id["b"].status is ECSStatus.COMPUTED
    assert ecs_by_id["c"].status is ECSStatus.COMPUTED
    assert ecs_by_id["a"].ecs == pytest.approx(0.0)
    assert ecs_by_id["b"].ecs == pytest.approx(1.0)
    assert ecs_by_id["c"].ecs == pytest.approx(0.5)

    fused = fuse_prioritization_results(results, k=60)

    assert [item.claim_id for item in fused] == ["b", "a", "c"]
    assert [item.fused_rank for item in fused] == [1, 2, 3]
    assert [item.baseline_rank for item in fused] == [2, 1, 3]
    assert [item.ecs_rank for item in fused] == [1, 3, 2]
    assert [item.ecs_score for item in fused] == pytest.approx(
        [1.0, 0.0, 0.5]
    )
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2].rrf_score == pytest.approx(1 / 63 + 1 / 62)

    assert [result.claim_id for result in results] == ["a", "b", "c"]
    assert [result.initial_rank for result in results] == [1, 2, 3]
    assert {id(item.result) for item in fused} == {
        id(result) for result in results
    }


def test_mixed_objectives_are_rejected() -> None:
    first = _result("A", 1)
    second = _result("B", 2)
    second.objective = Objective.O2

    with pytest.raises(RankingFusionError, match="same selected objective"):
        fuse_prioritization_results([first, second])


def test_duplicate_claim_ids_are_rejected() -> None:
    with pytest.raises(RankingFusionError, match="Duplicate claim ID"):
        fuse_prioritization_results(
            [
                _result("A", 1),
                _result("A", 2),
            ]
        )


def test_invalid_initial_rank_is_rejected() -> None:
    result = _result("A", 1)
    result.initial_rank = 0

    with pytest.raises(RankingFusionError, match="initial_rank"):
        fuse_prioritization_results([result])


def test_invalid_rank_mapping_inputs_are_rejected() -> None:
    with pytest.raises(RankingFusionError, match="exactly one entry"):
        fuse_rankings(["A", "B"], {"A": 1}, {"A": None, "B": None})
    with pytest.raises(RankingFusionError, match="exactly one entry"):
        fuse_rankings(
            ["A"],
            {"A": 1},
            {"A": None, "B": None},
        )
    with pytest.raises(RankingFusionError, match=">= 1"):
        fuse_rankings(["A"], {"A": 0}, {"A": None})
    with pytest.raises(RankingFusionError, match="positive integer"):
        fuse_rankings(["A"], {"A": True}, {"A": None})
