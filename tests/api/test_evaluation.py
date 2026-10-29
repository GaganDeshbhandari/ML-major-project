"""Focused tests for read-only model evaluation endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import evaluation as evaluation_api
from api.main import app

client = TestClient(app)

SUMMARY_URL = "/api/v1/evaluation/summary"
COMPARISON_URL = "/api/v1/evaluation/prioritization-comparison"
RETRIEVAL_URL = "/api/v1/evaluation/retrieval-metrics"


def _write(root: Path, relative: str, data: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def evaluation_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "evaluation"
    root.mkdir()
    monkeypatch.setattr(evaluation_api, "EVALUATION_ROOT", root)
    return root


def _factllama_report() -> dict:
    return {
        "evaluation": "FactLLaMA test",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "dataset": {
            "path": "data/RAWFC/test",
            "file_count": 2,
            "expected_file_count": 2,
            "label_field": "label",
            "evidence_field": "reports[*].content",
            "original_label_field": "original_label",
        },
        "label_mapping": {"true": "true", "mixture": "half", "false": "false"},
        "scoring_policy": {
            "metrics_include": "successfully parsed predictions",
        },
        "counts": {"total_files": 2, "scored": 2, "failed": 0},
        "metrics": {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {
                "true": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 2,
                }
            },
        },
        "confusion_matrix": {
            "labels": ["true"],
            "rows_actual_columns_predicted": {"true": {"true": 0}},
        },
        "model": {
            "base_model_snapshot": "C:/cache/snapshot",
            "adapter_path": "models/factllama/adapter",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "device": "cpu",
        },
    }


def _nli_report() -> dict:
    return {
        "benchmark": {
            "name": "nli_stance_diagnostic_benchmark",
            "purpose": "test",
            "scored_examples": 2,
        },
        "dataset": {
            "path": "evaluation/evidence_conflict/data/nli.json",
            "sha256": "abc123",
            "total_examples": 2,
            "class_counts": {"supports": 1, "refutes": 1},
            "metadata": {"version": "1", "limitations": "small diagnostic set"},
        },
        "generated_at_utc": "2026-01-02T00:00:00+00:00",
        "metrics": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "coverage": 1.0,
            "evaluated_examples": 2,
            "per_class": {},
        },
        "model": {
            "name": "test-nli-model",
            "model_class": "TestClassifier",
            "resolved_device": "cpu",
            "sequence_length": 128,
            "configured_revision": None,
            "confidence_threshold": None,
        },
        "runtime": {"average_warm_inference_seconds": 0.1},
    }


def _pending_comparison() -> dict:
    return {
        "status": "pending_annotation",
        "dataset_id": "claim-ranking-eval",
        "dataset_version": "1",
        "dataset_schema_version": "1.0",
        "labeling_guide_version": "1",
        "dataset_path": None,
        "dataset_sha256": None,
        "expected_dataset_path": "evaluation/claim_ranking/data/dataset.json",
        "k_values": [1, 3],
        "rrf_k": 60.0,
        "ecs_only_missing_policy": "exclude_missing_ecs",
        "num_claims_evaluated": 0,
        "baseline_rankings_available": False,
        "annotation_required": True,
        "methods": {
            method: {"status": "pending_annotation", "metrics": None}
            for method in ("baseline", "ecs_only", "fused")
        },
        "exclusions": [],
        "missing_data": ["No labeled dataset."],
        "reason": "No manually labeled claim-ranking dataset.",
        "notes": ["No metric values are reported."],
        "generated_at": "2026-01-03T00:00:00+00:00",
    }


def _computed_comparison() -> dict:
    return {
        "status": "computed",
        "dataset_id": "claim-ranking-eval",
        "dataset_version": "2",
        "dataset_schema_version": "1.0",
        "labeling_guide_version": "1",
        "dataset_path": "evaluation/claim_ranking/data/dataset.json",
        "dataset_sha256": "def456",
        "k_values": [1, 3],
        "rrf_k": 60.0,
        "relevant_min_grade": 1,
        "ecs_only_missing_policy": "exclude_missing_ecs",
        "num_claims_evaluated": 2,
        "annotation_required": False,
        "objectives": {
            "O1": {
                "num_claims": 2,
                "exclusions": [],
                "methods": {
                    "baseline": {
                        "num_ranked": 2,
                        "exclusions": [],
                        "metrics": {"precision@1": 1.0, "recall@1": 0.5},
                    },
                    "ecs_only": {
                        "num_ranked": 1,
                        "exclusions": [],
                        "metrics": {"precision@1": 0.0, "recall@1": 0.0},
                    },
                    "fused": {
                        "num_ranked": 2,
                        "exclusions": [],
                        "metrics": {"precision@1": 0.5, "recall@1": 0.5},
                    },
                },
            }
        },
        "exclusions": [],
        "missing_data": {"missing_ecs_count": 1},
        "notes": ["Computed from labeled data."],
        "metric_definitions": {
            "baseline": "Existing objective ranking.",
            "ecs_only": "ECS-only ranking.",
            "fused": "RRF combination.",
        },
        "generated_at": "2026-01-04T00:00:00+00:00",
    }


def _retrieval_report() -> dict:
    return {
        "generated_at": "2026-01-05T00:00:00+00:00",
        "dataset": {"name": "retrieval-set", "queries": 2},
        "configuration": {"k": 5},
        "metrics": {
            "Recall@5": 0.0,
            "Precision@5": 0.25,
            "MRR": 0.5,
        },
    }


def test_summary_reports_real_saved_metrics() -> None:
    response = client.get(SUMMARY_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["available"] is True
    assert payload["metrics"]["factllama_fact_verification"]["accuracy"] == 0.615
    assert payload["metrics"]["nli_stance_classification"][
        "accuracy"
    ] == pytest.approx(0.8666666666666667)
    assert payload["metrics"]["claim_ranking_prioritization"] is None
    assert payload["metrics"]["evidence_retrieval"] is None

    names = {item["name"] for item in payload["unavailable"]}
    assert names == {
        "claim_ranking_prioritization",
        "evidence_retrieval",
    }


def test_summary_preserves_zero_values_and_metadata(
    evaluation_root: Path,
) -> None:
    _write(
        evaluation_root,
        "factllama/results/rawfc_test_factllama_20990101.json",
        _factllama_report(),
    )
    _write(
        evaluation_root,
        "evidence_conflict/results/nli_stance_benchmark.json",
        _nli_report(),
    )
    _write(
        evaluation_root,
        "claim_ranking/results/claim_ranking_fusion_pending.json",
        _pending_comparison(),
    )
    _write(
        evaluation_root,
        "retrieval/results/retrieval_metrics.json",
        _retrieval_report(),
    )

    response = client.get(SUMMARY_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["available_experiment_count"] == 3
    assert payload["generated_at"] == "2026-01-05T00:00:00+00:00"

    factllama = payload["metrics"]["factllama_fact_verification"]
    assert factllama["accuracy"] == 0.0
    assert factllama["macro_f1"] == 0.0
    assert factllama["per_class"]["true"]["support"] == 2

    nli = payload["metrics"]["nli_stance_classification"]
    assert nli["accuracy"] == 0.5
    assert nli["macro_f1"] == 0.4

    assert payload["metrics"]["claim_ranking_prioritization"] is None
    assert payload["metrics"]["evidence_retrieval"] == {
        "Recall@5": 0.0,
        "Precision@5": 0.25,
        "MRR": 0.5,
    }

    experiments = {
        experiment["name"]: experiment
        for experiment in payload["experiments"]
    }
    fact = experiments["factllama_fact_verification"]
    assert fact["status"] == "available"
    assert fact["dataset"]["path"] == "data/RAWFC/test"
    assert fact["dataset"]["file_count"] == 2
    assert fact["model"]["adapter_path"] == "models/factllama/adapter"
    assert fact["model"]["base_model_snapshot"] == "snapshot"

    nli_experiment = experiments["nli_stance_classification"]
    assert nli_experiment["dataset"]["total_examples"] == 2
    assert nli_experiment["dataset"]["metadata"]["limitations"] == (
        "small diagnostic set"
    )
    assert nli_experiment["model"]["name"] == "test-nli-model"

    ranking = experiments["claim_ranking_prioritization"]
    assert ranking["status"] == "pending_annotation"
    assert ranking["available"] is False
    assert ranking["dataset"]["num_claims_evaluated"] == 0

    retrieval = experiments["evidence_retrieval"]
    assert retrieval["status"] == "available"
    assert retrieval["metrics"]["MRR"] == 0.5


def test_summary_marks_everything_unavailable_when_artifacts_are_missing(
    evaluation_root: Path,
) -> None:
    response = client.get(SUMMARY_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["available"] is False
    assert payload["available_experiment_count"] == 0
    assert all(value is None for value in payload["metrics"].values())
    assert len(payload["unavailable"]) == 4
    assert payload["artifact_errors"] == []


def test_summary_sanitizes_corrupt_artifact(
    evaluation_root: Path,
) -> None:
    path = (
        evaluation_root
        / "factllama"
        / "results"
        / "rawfc_test_factllama_20990101.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")

    response = client.get(SUMMARY_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_errors"] == [
        {
            "source": "factllama/results/rawfc_test_factllama_20990101.json",
            "message": "Artifact could not be read.",
        }
    ]
    fact = next(
        experiment
        for experiment in payload["experiments"]
        if experiment["name"] == "factllama_fact_verification"
    )
    assert fact["status"] == "error"
    assert "Traceback" not in response.text


def test_comparison_returns_pending_report_without_fabricating_metrics(
    evaluation_root: Path,
) -> None:
    _write(
        evaluation_root,
        "claim_ranking/results/claim_ranking_fusion_pending.json",
        _pending_comparison(),
    )

    response = client.get(COMPARISON_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["status"] == "pending_annotation"
    assert payload["source"] == (
        "claim_ranking/results/claim_ranking_fusion_pending.json"
    )
    assert payload["k_values"] == [1, 3]
    assert payload["rrf_k"] == 60.0
    assert payload["num_claims_evaluated"] == 0
    assert payload["dataset"]["dataset_id"] == "claim-ranking-eval"
    assert payload["configuration"]["annotation_required"] is True
    assert payload["reason"] == "No manually labeled claim-ranking dataset."
    for method in ("baseline", "ecs_only", "fused"):
        entry = payload["methods"][method]
        assert entry["available"] is False
        assert entry["status"] == "pending_annotation"
        assert entry["metrics"] is None
        assert entry["metrics_by_objective"] is None


def test_comparison_preserves_computed_per_objective_metrics(
    evaluation_root: Path,
) -> None:
    _write(
        evaluation_root,
        "claim_ranking/results/claim_ranking_fusion.json",
        _computed_comparison(),
    )

    response = client.get(COMPARISON_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["status"] == "computed"
    assert payload["source"] == (
        "claim_ranking/results/claim_ranking_fusion.json"
    )
    assert payload["dataset"]["dataset_id"] == "claim-ranking-eval"
    assert payload["dataset"]["path"] == (
        "evaluation/claim_ranking/data/dataset.json"
    )
    assert payload["configuration"]["relevant_min_grade"] == 1

    for method in ("baseline", "ecs_only", "fused"):
        entry = payload["methods"][method]
        assert entry["available"] is True
        assert entry["status"] == "computed"
        assert entry["metrics_by_objective"]["O1"] is not None
    assert payload["methods"]["ecs_only"]["metrics_by_objective"]["O1"] == {
        "precision@1": 0.0,
        "recall@1": 0.0,
    }
    assert payload["methods"]["baseline"]["details_by_objective"]["O1"] == {
        "num_ranked": 2,
        "exclusions": [],
    }


def test_comparison_without_artifact_returns_explicit_unavailable_result(
    evaluation_root: Path,
) -> None:
    response = client.get(COMPARISON_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["status"] == "unavailable"
    assert set(payload["methods"]) == {"baseline", "ecs_only", "fused"}
    assert payload["reason"] == (
        "No saved prioritization comparison artifact exists."
    )


def test_comparison_corrupt_artifact_returns_sanitized_503(
    evaluation_root: Path,
) -> None:
    path = (
        evaluation_root
        / "claim_ranking"
        / "results"
        / "claim_ranking_fusion.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    response = client.get(COMPARISON_URL)

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "message": "Evaluation artifact could not be read.",
        "source": "claim_ranking/results/claim_ranking_fusion.json",
    }
    assert "Traceback" not in response.text


def test_retrieval_metrics_without_artifact_are_unavailable(
    evaluation_root: Path,
) -> None:
    response = client.get(RETRIEVAL_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["status"] == "unavailable"
    assert payload["metrics"] is None
    assert payload["expected_metrics"] == [
        "Recall@K",
        "Precision@K",
        "MRR",
    ]
    assert "No saved retrieval evaluation artifact" in payload["note"]


def test_retrieval_metrics_preserve_saved_values(
    evaluation_root: Path,
) -> None:
    _write(
        evaluation_root,
        "retrieval/results/retrieval_metrics.json",
        _retrieval_report(),
    )

    response = client.get(RETRIEVAL_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["status"] == "available"
    assert payload["source"] == "retrieval/results/retrieval_metrics.json"
    assert payload["metrics"] == {
        "Recall@5": 0.0,
        "Precision@5": 0.25,
        "MRR": 0.5,
    }
    assert payload["dataset"] == {"name": "retrieval-set", "queries": 2}
    assert payload["configuration"] == {"k": 5}


def test_retrieval_artifact_without_metrics_is_unavailable(
    evaluation_root: Path,
) -> None:
    _write(
        evaluation_root,
        "retrieval/results/retrieval_metrics.json",
        {"generated_at": "2026-01-01T00:00:00+00:00", "dataset": {"queries": 2}},
    )

    response = client.get(RETRIEVAL_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["status"] == "unavailable"
    assert payload["source"] == "retrieval/results/retrieval_metrics.json"
    assert payload["metrics"] is None
    assert "does not contain metric values" in payload["note"]


def test_retrieval_corrupt_artifact_returns_sanitized_503(
    evaluation_root: Path,
) -> None:
    path = (
        evaluation_root
        / "retrieval"
        / "results"
        / "retrieval_metrics.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("[not a report]", encoding="utf-8")

    response = client.get(RETRIEVAL_URL)

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "message": "Evaluation artifact could not be read.",
        "source": "retrieval/results/retrieval_metrics.json",
    }
    assert "Traceback" not in response.text
