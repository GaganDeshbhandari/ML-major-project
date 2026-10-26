"""Integration tests for optional ECS scoring in the pipeline.

The tests use fake prioritization artifacts, injected retrieval, and fake
stance classifiers. They do not load an NLI model, access the network, or use
a GPU.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.evidence_conflict import (
    ECSError,
    ECSConfig,
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.evidence_conflict.errors import StanceInferenceError
from src.pipeline import (
    ECSErrorInfo,
    PrioritizationRetrievalResult,
    RetrievalErrorInfo,
    prioritize_and_retrieve,
)
from src.prioritization import FEATURE_COLUMNS, Claim, Objective, UserRating
from src.retrieval import EvidenceSearchError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTS = StanceLabel.SUPPORTS
REFUTES = StanceLabel.REFUTES
NOT_ENOUGH_INFO = StanceLabel.NOT_ENOUGH_INFO
EXPECTED_EVIDENCE_KEYS = {
    "title",
    "url",
    "content",
    "relevance_score",
    "rerank_score",
}


class IdentityScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


class FakeModel:
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


class FakeStore:
    def __init__(
        self,
        o1_model: FakeModel | None = None,
        o2_model: FakeModel | None = None,
    ) -> None:
        self.scaler = IdentityScaler()
        self.o1_model = o1_model
        self.o2_model = o2_model


class FakeStanceClassifier:
    def __init__(
        self,
        predictions_by_claim: dict[str, list[StancePrediction]],
        *,
        errors_by_claim: dict[str, Exception] | None = None,
    ) -> None:
        self.predictions_by_claim = predictions_by_claim
        self.errors_by_claim = errors_by_claim or {}
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict(
        self,
        claim: str,
        evidence: list[dict[str, object]],
    ) -> list[StancePrediction]:
        self.calls.append((claim, evidence))
        error = self.errors_by_claim.get(claim)
        if error is not None:
            raise error
        return list(self.predictions_by_claim[claim])


class ConfigRecordingClassifier:
    def __init__(self, config: ECSConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict(
        self,
        claim: str,
        evidence: list[dict[str, object]],
    ) -> list[StancePrediction]:
        self.calls.append((claim, evidence))
        return [
            _prediction(index, SUPPORTS)
            for index in range(len(evidence))
        ]


def _claims() -> list[Claim]:
    return [
        Claim("a", "Claim A"),
        Claim("b", "Claim B"),
        Claim("c", "Claim C"),
    ]


def _ratings() -> list[UserRating]:
    return [
        UserRating("a", "True"),
        UserRating("a", "False"),
        UserRating("b", "True"),
        UserRating("b", "Mostly True"),
        UserRating("c", "Mixture"),
        UserRating("c", "Mostly True"),
    ]


def _evidence(claim: str, count: int = 2) -> list[dict[str, object]]:
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


def _retrieve_with_evidence(
    claim: str,
    **kwargs: object,
) -> list[dict[str, object]]:
    return _evidence(claim)


def _probabilities(label: StanceLabel) -> StanceProbabilities:
    if label is SUPPORTS:
        return StanceProbabilities(0.8, 0.1, 0.1)
    if label is REFUTES:
        return StanceProbabilities(0.1, 0.8, 0.1)
    return StanceProbabilities(0.1, 0.1, 0.8)


def _prediction(
    evidence_index: int,
    label: StanceLabel,
    confidence: float = 0.8,
) -> StancePrediction:
    return StancePrediction(
        evidence_index=evidence_index,
        label=label,
        confidence=confidence,
        probabilities=_probabilities(label),
    )


def _predictions(*labels: StanceLabel) -> list[StancePrediction]:
    return [
        _prediction(index, label)
        for index, label in enumerate(labels)
    ]


def _ecs_result(result: PrioritizationRetrievalResult) -> EvidenceConflictResult:
    assert result.ecs_result is not None
    return result.ecs_result


def _legacy_fields(
    result: PrioritizationRetrievalResult,
) -> tuple[object, ...]:
    return (
        result.claim_id,
        result.claim,
        result.objective,
        result.initial_rank,
        result.evidence,
        result.predicted_truth,
        result.predicted_tpb,
        result.disputability,
        result.retrieval_error,
    )


def test_ecs_disabled_preserves_legacy_output_and_skips_classifier() -> None:
    retrieval = Mock(side_effect=_retrieve_with_evidence)
    store = FakeStore(FakeModel("True", ("False", "True")))

    with patch(
        "src.evidence_conflict.nli_classifier.NLIStanceClassifier"
    ) as classifier_cls:
        results = prioritize_and_retrieve(
            _claims(),
            _ratings(),
            Objective.O1,
            artifact_store=store,
            retrieve=retrieval,
        )

    classifier_cls.assert_not_called()
    assert [result.claim_id for result in results] == ["a", "b", "c"]
    assert [result.initial_rank for result in results] == [1, 2, 3]
    assert all(result.predicted_truth == "True" for result in results)
    assert all(result.ecs_result is None for result in results)
    assert all(result.ecs_error is None for result in results)
    assert all(result.retrieval_error is None for result in results)
    assert results[0].evidence == _evidence("Claim A")


def test_ecs_disabled_subprocess_does_not_import_heavy_dependencies() -> None:
    code = "\n".join(
        [
            "import sys",
            "from src.pipeline import prioritize_and_retrieve",
            "from src.prioritization import Claim, UserRating",
            "claims = [Claim('a', 'Claim A')]",
            "ratings = [UserRating('a', 'True'), UserRating('a', 'False')]",
            "results = prioritize_and_retrieve(",
            "    claims, ratings, 'O3',",
            "    retrieve=lambda claim, **kwargs: [],",
            ")",
            "assert results[0].ecs_result is None",
            "assert results[0].ecs_error is None",
            "assert 'torch' not in sys.modules",
            "assert 'transformers' not in sys.modules",
            "assert 'sentence_transformers' not in sys.modules",
            "assert 'src.evidence_conflict.nli_classifier' not in sys.modules",
            "assert 'src.evidence_conflict.scoring' not in sys.modules",
            "print('ok')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_mixed_evidence_attaches_a_computed_ecs_result() -> None:
    evidence = _evidence("Claim A", 2)
    classifier = FakeStanceClassifier(
        {"Claim A": _predictions(SUPPORTS, REFUTES)}
    )

    results = prioritize_and_retrieve(
        [Claim("a", "Claim A")],
        [UserRating("a", "True"), UserRating("a", "False")],
        "O3",
        retrieve=Mock(return_value=evidence),
        stance_classifier=classifier,
    )

    ecs = _ecs_result(results[0])
    assert results[0].ecs_error is None
    assert ecs.status is ECSStatus.COMPUTED
    assert ecs.counts == StanceCounts(supports=1, refutes=1)
    assert ecs.ecs == 1.0
    assert ecs.evidence_coverage == 1.0
    assert ecs.support_share == 0.5
    assert ecs.refutation_share == 0.5
    assert ecs.mean_confidence == pytest.approx(0.8)
    assert classifier.calls == [("Claim A", evidence)]


def test_one_sided_evidence_is_computed_with_zero_ecs() -> None:
    classifier = FakeStanceClassifier(
        {"Claim A": _predictions(SUPPORTS, SUPPORTS)}
    )

    results = prioritize_and_retrieve(
        [Claim("a", "Claim A")],
        [UserRating("a", "True"), UserRating("a", "False")],
        "O3",
        retrieve=Mock(return_value=_evidence("Claim A", 2)),
        stance_classifier=classifier,
    )

    ecs = _ecs_result(results[0])
    assert ecs.status is ECSStatus.COMPUTED
    assert ecs.ecs == 0.0
    assert ecs.evidence_coverage == 1.0
    assert ecs.support_share == 1.0
    assert ecs.refutation_share == 0.0
    assert ecs.counts == StanceCounts(supports=2)


def test_all_not_enough_info_preserves_insufficient_stance() -> None:
    classifier = FakeStanceClassifier(
        {
            "Claim A": _predictions(
                NOT_ENOUGH_INFO,
                NOT_ENOUGH_INFO,
            )
        }
    )

    results = prioritize_and_retrieve(
        [Claim("a", "Claim A")],
        [UserRating("a", "True"), UserRating("a", "False")],
        "O3",
        retrieve=Mock(return_value=_evidence("Claim A", 2)),
        stance_classifier=classifier,
    )

    ecs = _ecs_result(results[0])
    assert ecs.status is ECSStatus.INSUFFICIENT_STANCE
    assert ecs.ecs is None
    assert ecs.evidence_coverage == 0.0
    assert ecs.support_share is None
    assert ecs.refutation_share is None
    assert ecs.counts == StanceCounts(not_enough_info=2)


def test_empty_evidence_yields_no_evidence_without_classifier() -> None:
    with patch(
        "src.evidence_conflict.nli_classifier.NLIStanceClassifier"
    ) as classifier_cls:
        results = prioritize_and_retrieve(
            [Claim("a", "Claim A")],
            [UserRating("a", "True"), UserRating("a", "False")],
            "O3",
            retrieve=Mock(return_value=[]),
            ecs_config=ECSConfig(),
        )

    classifier_cls.assert_not_called()
    ecs = _ecs_result(results[0])
    assert results[0].ecs_error is None
    assert ecs.status is ECSStatus.NO_EVIDENCE
    assert ecs.ecs is None
    assert ecs.evidence_coverage is None
    assert ecs.support_share is None
    assert ecs.refutation_share is None
    assert ecs.mean_confidence is None


def test_retrieval_failure_remains_retrieval_error_and_skips_nli() -> None:
    def retrieve(
        claim: str,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        if claim == "Claim B":
            raise EvidenceSearchError("search unavailable")
        return _evidence(claim)

    classifier = FakeStanceClassifier(
        {
            "Claim A": _predictions(SUPPORTS, REFUTES),
            "Claim C": _predictions(SUPPORTS, SUPPORTS),
        }
    )

    results = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        Objective.O1,
        artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
        retrieve=retrieve,
        stance_classifier=classifier,
    )

    by_id = {result.claim_id: result for result in results}
    failed = by_id["b"]
    assert failed.evidence == []
    assert failed.retrieval_error == RetrievalErrorInfo(
        error_type="EvidenceSearchError",
        message="search unavailable",
    )
    assert failed.ecs_result is None
    assert failed.ecs_error is None
    assert [call[0] for call in classifier.calls] == ["Claim A", "Claim C"]
    assert by_id["a"].ecs_result is not None
    assert by_id["c"].ecs_result is not None


def test_nli_failure_is_exposed_without_corrupting_legacy_output() -> None:
    evidence = _evidence("Claim A", 2)
    classifier = FakeStanceClassifier(
        {},
        errors_by_claim={
            "Claim A": StanceInferenceError("inference boom")
        },
    )

    results = prioritize_and_retrieve(
        [Claim("a", "Claim A")],
        [UserRating("a", "True"), UserRating("a", "False")],
        Objective.O1,
        artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
        retrieve=Mock(return_value=evidence),
        stance_classifier=classifier,
    )

    result = results[0]
    ecs = _ecs_result(result)
    assert ecs.status is ECSStatus.NLI_ERROR
    assert ecs.ecs is None
    assert ecs.predictions == ()
    assert ecs.counts == StanceCounts()
    assert result.ecs_error == ECSErrorInfo(
        error_type="StanceInferenceError",
        message="inference boom",
    )
    assert result.predicted_truth == "True"
    assert result.initial_rank == 1
    assert result.evidence == evidence
    assert result.retrieval_error is None


def test_scoring_failure_preserves_legacy_output_and_exposes_error() -> None:
    evidence = _evidence("Claim A", 2)
    classifier = FakeStanceClassifier(
        {"Claim A": _predictions(SUPPORTS, REFUTES)}
    )

    with patch(
        "src.evidence_conflict.scoring.compute_evidence_conflict",
        side_effect=ECSError("scoring boom"),
    ):
        results = prioritize_and_retrieve(
            [Claim("a", "Claim A")],
            [UserRating("a", "True"), UserRating("a", "False")],
            Objective.O1,
            artifact_store=FakeStore(
                FakeModel("True", ("False", "True"))
            ),
            retrieve=Mock(return_value=evidence),
            stance_classifier=classifier,
        )

    result = results[0]
    assert result.ecs_result is None
    assert result.ecs_error == ECSErrorInfo(
        error_type="ECSError",
        message="scoring boom",
    )
    assert result.predicted_truth == "True"
    assert result.initial_rank == 1
    assert result.evidence == evidence
    assert result.retrieval_error is None


def test_nli_failure_for_one_claim_does_not_stop_other_claims() -> None:
    classifier = FakeStanceClassifier(
        {
            "Claim A": _predictions(SUPPORTS, REFUTES),
            "Claim C": _predictions(SUPPORTS, SUPPORTS),
        },
        errors_by_claim={
            "Claim B": StanceInferenceError("inference boom")
        },
    )

    results = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        Objective.O1,
        artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
        retrieve=Mock(side_effect=_retrieve_with_evidence),
        stance_classifier=classifier,
    )

    by_id = {result.claim_id: result for result in results}
    assert by_id["a"].ecs_result is not None
    assert by_id["a"].ecs_result.status is ECSStatus.COMPUTED
    assert by_id["b"].ecs_result is not None
    assert by_id["b"].ecs_result.status is ECSStatus.NLI_ERROR
    assert by_id["c"].ecs_result is not None
    assert by_id["c"].ecs_result.status is ECSStatus.COMPUTED
    assert [call[0] for call in classifier.calls] == [
        "Claim A",
        "Claim B",
        "Claim C",
    ]


def test_multiple_claims_receive_independent_ecs_results() -> None:
    classifier = FakeStanceClassifier(
        {
            "Claim A": _predictions(SUPPORTS, REFUTES),
            "Claim B": _predictions(
                NOT_ENOUGH_INFO,
                NOT_ENOUGH_INFO,
            ),
            "Claim C": _predictions(SUPPORTS, SUPPORTS),
        }
    )

    results = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        Objective.O1,
        artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
        retrieve=Mock(side_effect=_retrieve_with_evidence),
        stance_classifier=classifier,
    )

    by_id = {result.claim_id: result for result in results}
    assert by_id["a"].ecs_result is not None
    assert by_id["a"].ecs_result.status is ECSStatus.COMPUTED
    assert by_id["a"].ecs_result.ecs == 1.0
    assert by_id["b"].ecs_result is not None
    assert by_id["b"].ecs_result.status is ECSStatus.INSUFFICIENT_STANCE
    assert by_id["b"].ecs_result.ecs is None
    assert by_id["c"].ecs_result is not None
    assert by_id["c"].ecs_result.status is ECSStatus.COMPUTED
    assert by_id["c"].ecs_result.ecs == 0.0
    assert len(classifier.calls) == 3


def test_evidence_order_and_prediction_association_are_preserved() -> None:
    evidence = [
        {
            "title": "first",
            "url": "https://example.test/1",
            "content": "first passage",
            "relevance_score": 0.9,
            "rerank_score": 0.8,
        },
        {
            "title": "second",
            "url": "https://example.test/2",
            "content": "second passage",
            "relevance_score": 0.8,
            "rerank_score": 0.7,
        },
        {
            "title": "third",
            "url": "https://example.test/3",
            "content": "third passage",
            "relevance_score": 0.7,
            "rerank_score": 0.6,
        },
    ]
    predictions = _predictions(
        SUPPORTS,
        REFUTES,
        NOT_ENOUGH_INFO,
    )
    classifier = FakeStanceClassifier({"Claim A": predictions})

    results = prioritize_and_retrieve(
        [Claim("a", "Claim A")],
        [UserRating("a", "True"), UserRating("a", "False")],
        "O3",
        retrieve=Mock(return_value=evidence),
        stance_classifier=classifier,
    )

    assert classifier.calls[0][0] == "Claim A"
    assert [
        item["content"] for item in classifier.calls[0][1]
    ] == [
        "first passage",
        "second passage",
        "third passage",
    ]
    ecs = _ecs_result(results[0])
    assert ecs.predictions == tuple(predictions)
    assert [item.evidence_index for item in ecs.predictions] == [0, 1, 2]
    assert [item.label for item in ecs.predictions] == [
        SUPPORTS,
        REFUTES,
        NOT_ENOUGH_INFO,
    ]
    assert ecs.counts == StanceCounts(
        supports=1,
        refutes=1,
        not_enough_info=1,
    )
    assert ecs.ecs == pytest.approx(2.0 / 3.0)


def test_evidence_dictionaries_are_not_mutated_or_augmented() -> None:
    evidence = _evidence("Claim A", 2)
    original = deepcopy(evidence)
    classifier = FakeStanceClassifier(
        {"Claim A": _predictions(SUPPORTS, REFUTES)}
    )

    results = prioritize_and_retrieve(
        [Claim("a", "Claim A")],
        [UserRating("a", "True"), UserRating("a", "False")],
        "O3",
        retrieve=Mock(return_value=evidence),
        stance_classifier=classifier,
    )

    assert results[0].evidence == original
    assert evidence == original
    assert classifier.calls[0][1] == original
    for record in results[0].evidence:
        assert set(record) == EXPECTED_EVIDENCE_KEYS
        assert "stance" not in record
        assert "stance_label" not in record
        assert "stance_probabilities" not in record
        assert "ecs" not in record


@pytest.mark.parametrize(
    ("objective", "store"),
    [
        (
            Objective.O1,
            FakeStore(FakeModel("True", ("False", "True"))),
        ),
        (
            Objective.O2,
            FakeStore(o2_model=FakeModel("Low", ("Low", "High"))),
        ),
        (Objective.O3, None),
    ],
)
def test_ecs_does_not_change_prioritization_or_evidence_fields(
    objective: Objective,
    store: FakeStore | None,
) -> None:
    retrieval = Mock(side_effect=_retrieve_with_evidence)
    baseline = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        objective,
        artifact_store=store,
        retrieve=retrieval,
    )
    classifier = FakeStanceClassifier(
        {
            "Claim A": _predictions(SUPPORTS, REFUTES),
            "Claim B": _predictions(SUPPORTS, SUPPORTS),
            "Claim C": _predictions(
                NOT_ENOUGH_INFO,
                NOT_ENOUGH_INFO,
            ),
        }
    )
    integrated = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        objective,
        artifact_store=store,
        retrieve=retrieval,
        stance_classifier=classifier,
    )

    assert len(baseline) == len(integrated)
    for base, enriched in zip(baseline, integrated):
        assert _legacy_fields(base) == _legacy_fields(enriched)
        assert enriched.ecs_result is not None
    assert [call[0] for call in classifier.calls] == [
        result.claim for result in integrated
    ]


def test_ecs_config_is_passed_to_classifier_and_reused() -> None:
    config = ECSConfig(
        model_name="local/test-nli",
        device="cpu",
        max_length=128,
        batch_size=2,
        confidence_threshold=0.7,
        max_evidence_chars=64,
        local_files_only=True,
        model_revision="rev-123",
    )
    created = ConfigRecordingClassifier(config)

    with patch(
        "src.evidence_conflict.nli_classifier.NLIStanceClassifier",
        return_value=created,
    ) as classifier_cls:
        results = prioritize_and_retrieve(
            _claims(),
            _ratings(),
            Objective.O1,
            artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
            retrieve=Mock(side_effect=_retrieve_with_evidence),
            ecs_config=config,
        )

    classifier_cls.assert_called_once_with(config)
    assert created.config is config
    assert [call[0] for call in created.calls] == [
        "Claim A",
        "Claim B",
        "Claim C",
    ]
    assert all(result.ecs_result is not None for result in results)


def test_pipeline_public_exports_remain_compatible() -> None:
    import src.pipeline as pipeline

    assert {
        "PrioritizationRetrievalResult",
        "RetrievalErrorInfo",
        "prioritize_and_retrieve",
        "ECSErrorInfo",
    } <= set(pipeline.__all__)
    assert pipeline.PrioritizationRetrievalResult is PrioritizationRetrievalResult
    assert pipeline.RetrievalErrorInfo is RetrievalErrorInfo
    assert pipeline.ECSErrorInfo is ECSErrorInfo


def test_invalid_ecs_configuration_objects_are_rejected() -> None:
    with pytest.raises(TypeError, match="ecs_config"):
        prioritize_and_retrieve(
            _claims(),
            _ratings(),
            "O3",
            retrieve=Mock(),
            ecs_config=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="stance_classifier"):
        prioritize_and_retrieve(
            _claims(),
            _ratings(),
            "O3",
            retrieve=Mock(),
            stance_classifier=object(),
        )
