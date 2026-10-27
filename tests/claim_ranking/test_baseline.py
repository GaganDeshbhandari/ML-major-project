"""Offline tests for claim-ranking baseline report generation.

All datasets are synthetic unit-test fixtures. They are not saved as real
evaluation results and do not imply real-world performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.claim_ranking.baseline import (
    COMPUTED_STATUS,
    DEFAULT_PENDING_REASON,
    PENDING_OUTPUT_PREFIX,
    PENDING_STATUS,
    build_baseline_report,
    build_pending_report,
    main,
    render_markdown,
    write_report,
)
from evaluation.claim_ranking.dataset import parse_dataset


def _dataset_payload() -> dict[str, Any]:
    return {
        "dataset_id": "synthetic-fixture",
        "version": "1",
        "description": "Synthetic unit-test fixture only.",
        "labeling_guide_version": "1",
        "examples": [
            {
                "claim_id": "A",
                "claim": "Synthetic claim A.",
                "relevance_grade": 3,
                "rationale": "Synthetic rationale A.",
                "baseline_ranks": {"O1": 1, "O2": 2, "O3": 1},
                "ecs": 0.5,
            },
            {
                "claim_id": "B",
                "claim": "Synthetic claim B.",
                "relevance_grade": 0,
                "rationale": "Synthetic rationale B.",
                "baseline_ranks": {"O1": 2, "O2": 1, "O3": 2},
                "ecs": None,
            },
        ],
    }


def test_pending_report_has_no_metric_values() -> None:
    report = build_pending_report(
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert report["status"] == PENDING_STATUS
    assert report["num_claims_evaluated"] == 0
    assert report["baseline_metrics"] == {
        "O1": None,
        "O2": None,
        "O3": None,
    }
    assert report["dataset_path"] is None
    assert report["dataset_sha256"] is None
    assert report["baseline_rankings_available"] is False
    assert report["missing_data"] == [
        "No manually labeled claim-ranking dataset.",
        "No baseline rankings recorded for a real claim set.",
    ]
    assert report["annotation_required"] is True
    assert report["nli_benchmark_used_as_claim_ranking_benchmark"] is False
    assert report["generated_at"] == "2026-01-01T00:00:00+00:00"
    assert report["expected_dataset_path"].endswith(
        "evaluation/claim_ranking/data/claim_ranking_dataset_v1.json"
    )


def test_pending_report_states_why_metrics_are_pending() -> None:
    report = build_pending_report(
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert report["reason"] == DEFAULT_PENDING_REASON
    assert "No manually labeled" in report["reason"]
    assert all(
        value is None
        for value in report["baseline_metrics"].values()
    )
    assert "precision_at_k" in report["metric_definitions"]
    assert "missing_ecs" in report["metric_definitions"]


def test_render_pending_markdown_is_explicit() -> None:
    markdown = render_markdown(
        build_pending_report(
            generated_at="2026-01-01T00:00:00+00:00",
        )
    )

    assert "pending annotation" in markdown.lower()
    assert "No metric values are reported" in markdown
    assert "O1" in markdown
    assert "O2" in markdown
    assert "O3" in markdown


def test_write_report_round_trips_json(tmp_path: Path) -> None:
    report = build_pending_report(
        generated_at="2026-01-01T00:00:00+00:00",
    )

    json_path, markdown_path = write_report(
        report,
        output_dir=tmp_path / "nested" / "results",
        prefix="pending_fixture",
    )

    assert json_path.exists()
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "pending annotation" in markdown_path.read_text(encoding="utf-8").lower()


def test_build_baseline_report_with_synthetic_dataset() -> None:
    dataset = parse_dataset(_dataset_payload())
    report = build_baseline_report(
        dataset,
        k_values=[1, 2, 5],
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert report["status"] == COMPUTED_STATUS
    assert report["dataset_id"] == "synthetic-fixture"
    assert report["dataset_version"] == "1"
    assert report["dataset_path"] == "<memory>"
    assert report["dataset_sha256"] is None
    assert report["num_claims_evaluated"] == 2
    assert report["baseline_rankings_available"] is True
    assert report["missing_data"] == {
        "missing_ecs_count": 1,
        "missing_baseline_rank_counts": {
            "O1": 0,
            "O2": 0,
            "O3": 0,
        },
    }
    assert report["baseline_metrics"]["O1"]["num_claims"] == 2
    assert report["baseline_metrics"]["O1"]["metrics"]["1"]["precision"] == pytest.approx(1.0)
    assert report["baseline_metrics"]["O1"]["metrics"]["5"] == {
        "precision": None,
        "recall": None,
        "ndcg": None,
    }


def test_render_computed_markdown_includes_metrics_and_na() -> None:
    dataset = parse_dataset(_dataset_payload())
    report = build_baseline_report(
        dataset,
        k_values=[1, 5],
        generated_at="2026-01-01T00:00:00+00:00",
    )

    markdown = render_markdown(report)

    assert "Status: computed" in markdown
    assert "Precision@K" in markdown
    assert "n/a" in markdown
    assert "O1" in markdown
    assert "O2" in markdown
    assert "O3" in markdown


def test_cli_pending_writes_reports(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--generated-at",
            "2026-01-01T00:00:00+00:00",
        ]
    )

    assert exit_code == 0
    json_path = tmp_path / f"{PENDING_OUTPUT_PREFIX}.json"
    markdown_path = tmp_path / f"{PENDING_OUTPUT_PREFIX}.md"
    assert json_path.exists()
    assert markdown_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["status"] == PENDING_STATUS
    assert report["baseline_metrics"] == {
        "O1": None,
        "O2": None,
        "O3": None,
    }


def test_cli_with_synthetic_dataset_writes_computed_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.json"
    dataset_path.write_text(
        json.dumps(_dataset_payload()),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--k",
            "1",
            "2",
            "--generated-at",
            "2026-01-01T00:00:00+00:00",
        ]
    )

    assert exit_code == 0
    json_path = output_dir / "claim_ranking_baseline.json"
    markdown_path = output_dir / "claim_ranking_baseline.md"
    assert json_path.exists()
    assert markdown_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["status"] == COMPUTED_STATUS
    assert report["dataset_sha256"] is not None
    assert report["baseline_metrics"]["O1"]["num_claims"] == 2


def test_cli_invalid_dataset_returns_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text('{"examples": []}', encoding="utf-8")

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 2
