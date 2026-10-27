"""Unit tests for the NLI model comparison report.

The tests use synthetic benchmark-report payloads. They do not load models or
run inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.evidence_conflict.compare_nli_models import (
    ComparisonError,
    build_comparison,
    load_report,
    render_comparison_markdown,
    write_comparison,
)


def _per_class(
    supports_recall: float,
    refutes_recall: float,
    nei_recall: float,
) -> list[dict[str, object]]:
    return [
        {
            "label": "supports",
            "precision": 0.9,
            "recall": supports_recall,
            "f1": 0.8,
            "support": 1,
        },
        {
            "label": "refutes",
            "precision": 0.8,
            "recall": refutes_recall,
            "f1": 0.7,
            "support": 1,
        },
        {
            "label": "not_enough_info",
            "precision": 0.7,
            "recall": nei_recall,
            "f1": 0.6,
            "support": 1,
        },
    ]


def _misclassified(
    example_id: str,
    gold: str,
    predicted: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "id": example_id,
        "claim": f"claim {example_id}",
        "evidence": f"evidence {example_id}",
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": confidence,
        "category": "test",
        "rationale": "test rationale",
    }


def _fake_report(
    *,
    name: str,
    accuracy: float,
    macro_f1: float,
    nei_recall: float,
    warm: float,
    initial: float,
    total: float,
    peak_mb: float,
    sha256: str = "same-sha",
    misclassified: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "dataset": {
            "path": f"{name}.json",
            "sha256": sha256,
            "total_examples": 3,
            "class_counts": {
                "supports": 1,
                "refutes": 1,
                "not_enough_info": 1,
            },
        },
        "model": {
            "name": name,
            "configured_revision": "revision",
            "device_requested": "auto",
            "resolved_device": "cpu",
            "model_dtype": "torch.float32",
            "sequence_length": 512,
            "batch_size": 2,
            "confidence_threshold": None,
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "inference_failures": 0,
            "coverage": 1.0,
            "per_class": _per_class(0.9, 0.8, nei_recall),
            "confusion_matrix_labels": [
                "supports",
                "refutes",
                "not_enough_info",
            ],
            "confusion_matrix": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            "confusions": [],
        },
        "runtime": {
            "average_warm_inference_seconds": warm,
            "initial_call_seconds": initial,
            "total_attempt_seconds": total,
        },
        "environment": {"peak_working_set_mb": peak_mb},
        "misclassified_examples": misclassified or [],
        "inference_failures": [],
    }


def test_build_comparison_calculates_deltas_and_verdict() -> None:
    baseline = _fake_report(
        name="baseline",
        accuracy=0.8,
        macro_f1=0.7,
        nei_recall=0.5,
        warm=0.1,
        initial=1.0,
        total=2.0,
        peak_mb=100.0,
    )
    alternative = _fake_report(
        name="alternative",
        accuracy=0.9,
        macro_f1=0.85,
        nei_recall=0.8,
        warm=0.2,
        initial=2.0,
        total=3.0,
        peak_mb=200.0,
    )

    comparison = build_comparison(
        baseline,
        alternative,
        generated_at="2026-01-01T00:00:00+00:00",
        baseline_note="baseline note",
        extra_notes=["extra note"],
    )

    assert comparison["generated_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert "baseline note" in comparison["measurement_notes"]
    assert "extra note" in comparison["measurement_notes"]
    assert comparison["dataset"]["sha256"] == "same-sha"
    assert comparison["summary"]["verdict"] == "alternative"
    assert comparison["summary"]["accuracy_delta"] == pytest.approx(0.1)
    assert comparison["summary"]["macro_f1_delta"] == pytest.approx(0.15)
    assert comparison["summary"]["not_enough_info_recall_delta"] == pytest.approx(
        0.3
    )
    assert comparison["summary"][
        "warm_runtime_ratio_alternative_over_baseline"
    ] == pytest.approx(2.0)
    assert comparison["summary"]["peak_memory_delta_mb"] == pytest.approx(100.0)

    headline = {item["metric"]: item for item in comparison["headline_metrics"]}
    assert headline["Accuracy"]["delta_alternative_minus_baseline"] == pytest.approx(
        0.1
    )
    assert headline["NOT_ENOUGH_INFO recall"][
        "delta_alternative_minus_baseline"
    ] == pytest.approx(0.3)


def test_build_comparison_rejects_different_datasets() -> None:
    baseline = _fake_report(
        name="baseline",
        accuracy=0.8,
        macro_f1=0.7,
        nei_recall=0.5,
        warm=0.1,
        initial=1.0,
        total=2.0,
        peak_mb=100.0,
    )
    alternative = _fake_report(
        name="alternative",
        accuracy=0.9,
        macro_f1=0.85,
        nei_recall=0.8,
        warm=0.2,
        initial=2.0,
        total=3.0,
        peak_mb=200.0,
        sha256="different-sha",
    )

    with pytest.raises(ComparisonError, match="different datasets"):
        build_comparison(baseline, alternative)


def test_render_and_write_comparison_preserve_ids(tmp_path: Path) -> None:
    baseline = _fake_report(
        name="baseline",
        accuracy=0.8,
        macro_f1=0.7,
        nei_recall=0.5,
        warm=0.1,
        initial=1.0,
        total=2.0,
        peak_mb=100.0,
        misclassified=[
            _misclassified(
                "NEI-001",
                "not_enough_info",
                "supports",
                0.95,
            )
        ],
    )
    alternative = _fake_report(
        name="alternative",
        accuracy=0.9,
        macro_f1=0.85,
        nei_recall=0.8,
        warm=0.2,
        initial=2.0,
        total=3.0,
        peak_mb=200.0,
        misclassified=[
            _misclassified(
                "REF-001",
                "refutes",
                "supports",
                0.91,
            )
        ],
    )
    comparison = build_comparison(
        baseline,
        alternative,
        baseline_label="Existing xsmall",
        alternative_label="Alternative base",
        generated_at="2026-01-01T00:00:00+00:00",
    )

    markdown = render_comparison_markdown(comparison)

    assert "# NLI Stance Model Comparison" in markdown
    assert "Existing xsmall" in markdown
    assert "Alternative base" in markdown
    assert "## Headline comparison" in markdown
    assert "## Confusion matrices" in markdown
    assert "NEI-001" in markdown
    assert "REF-001" in markdown
    assert "NOT_ENOUGH_INFO" in markdown

    json_path, markdown_path = write_comparison(comparison, tmp_path)
    assert json_path.exists()
    assert markdown_path.exists()
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["summary"]["verdict"] == "alternative"
    assert written["dataset"]["sha256"] == "same-sha"


def test_load_report_rejects_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ComparisonError, match="must be a JSON object"):
        load_report(path)
