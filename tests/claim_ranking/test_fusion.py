"""Offline tests for baseline, ECS-only, and RRF-fused evaluation.

All datasets are synthetic unit-test fixtures. They are not saved as empirical
evaluation results and do not imply real-world performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.claim_ranking.fusion import (
    COMPUTED_STATUS,
    ECS_ONLY_MISSING_POLICY,
    PENDING_OUTPUT_PREFIX,
    PENDING_STATUS,
    build_fusion_pending_report,
    build_fusion_report,
    ecs_only_ranking,
    evaluate_fusion,
    fused_ranking,
    main,
    render_fusion_markdown,
    write_fusion_report,
)
from evaluation.claim_ranking.dataset import parse_dataset
from evaluation.claim_ranking.metrics import evaluate_ranking
from src.ranking_fusion import RankingFusionError


def _payload() -> dict[str, Any]:
    return {
        "dataset_id": "synthetic-fusion-fixture",
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
                "relevance_grade": 1,
                "rationale": "Synthetic rationale B.",
                "baseline_ranks": {"O1": 2, "O2": 1, "O3": 2},
                "ecs": 0.5,
            },
            {
                "claim_id": "C",
                "claim": "Synthetic claim C.",
                "relevance_grade": 0,
                "rationale": "Synthetic rationale C.",
                "baseline_ranks": {"O1": 3, "O2": 3, "O3": None},
                "ecs": None,
            },
            {
                "claim_id": "D",
                "claim": "Synthetic claim D.",
                "relevance_grade": 2,
                "rationale": "Synthetic rationale D.",
                "baseline_ranks": {"O1": 4, "O2": 4, "O3": 3},
                "ecs": 0.9,
            },
        ],
    }


def _metrics_json(
    metrics: dict[int, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    return {
        str(k): {
            "precision": values["precision"],
            "recall": values["recall"],
            "ndcg": values["ndcg"],
        }
        for k, values in metrics.items()
    }


def test_ecs_only_ranking_is_descending_and_excludes_missing_ecs() -> None:
    dataset = parse_dataset(_payload())
    ranking = ecs_only_ranking(dataset, "O1")

    assert ranking.claim_ids == ("A", "B", "C", "D")
    assert ranking.ranked_claim_ids == ("D", "A", "B")
    assert ranking.missing_ecs_claim_ids == ("C",)
    assert [item.claim_id for item in ranking.exclusions] == ["C"]
    assert ranking.exclusions[0].reason == (
        "No ECS score for ECS-only O1 ranking."
    )
    assert dict(ranking.relevance) == {
        "A": 3,
        "B": 1,
        "C": 0,
        "D": 2,
    }


def test_ecs_only_ranking_preserves_baseline_order_for_ecs_ties() -> None:
    dataset = parse_dataset(_payload())
    ranking = ecs_only_ranking(dataset, "O1")

    # A and B both have ECS 0.5; A has the better baseline rank.
    assert ranking.ranked_claim_ids.index("A") < ranking.ranked_claim_ids.index(
        "B"
    )


def test_fused_ranking_contains_every_eligible_claim_once() -> None:
    dataset = parse_dataset(_payload())
    ranking = fused_ranking(dataset, "O1", rrf_k=60)

    assert ranking.ranked_claim_ids == ("A", "D", "B", "C")
    assert len(ranking.ranked_claim_ids) == len(set(ranking.ranked_claim_ids))
    assert ranking.ecs_ranks["D"] == 1
    assert ranking.ecs_ranks["A"] == 2
    assert ranking.ecs_ranks["B"] == 3
    assert ranking.ecs_ranks["C"] is None
    assert ranking.rrf_scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert ranking.rrf_scores["D"] == pytest.approx(1 / 64 + 1 / 61)
    assert ranking.rrf_scores["C"] == pytest.approx(1 / 63)


def test_objective_specific_baselines_produce_different_fused_orders() -> None:
    dataset = parse_dataset(_payload())

    assert fused_ranking(dataset, "O1").ranked_claim_ids == (
        "A",
        "D",
        "B",
        "C",
    )
    assert fused_ranking(dataset, "O2").ranked_claim_ids == (
        "B",
        "D",
        "A",
        "C",
    )
    assert fused_ranking(dataset, "O3").ranked_claim_ids == (
        "A",
        "D",
        "B",
    )


def test_evaluate_fusion_compares_all_methods_on_the_same_labels() -> None:
    dataset = parse_dataset(_payload())
    result = evaluate_fusion(dataset, [1, 2])

    assert result["k_values"] == [1, 2]
    assert result["rrf_k"] == pytest.approx(60.0)
    assert result["ecs_only_missing_policy"] == ECS_ONLY_MISSING_POLICY

    o1 = result["objectives"]["O1"]
    assert o1["num_claims"] == 4
    assert o1["exclusions"] == []
    assert o1["methods"]["baseline"]["ranked_claim_ids"] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert o1["methods"]["ecs_only"]["ranked_claim_ids"] == [
        "D",
        "A",
        "B",
    ]
    assert o1["methods"]["ecs_only"]["missing_ecs_claim_ids"] == ["C"]
    assert o1["methods"]["fused"]["ranked_claim_ids"] == [
        "A",
        "D",
        "B",
        "C",
    ]
    assert o1["methods"]["fused"]["ecs_ranks"]["C"] is None

    relevance = {"A": 3, "B": 1, "C": 0, "D": 2}
    assert o1["methods"]["baseline"]["metrics"] == _metrics_json(
        evaluate_ranking(["A", "B", "C", "D"], relevance, [1, 2])
    )
    assert o1["methods"]["ecs_only"]["metrics"] == _metrics_json(
        evaluate_ranking(["D", "A", "B"], relevance, [1, 2])
    )
    assert o1["methods"]["fused"]["metrics"] == _metrics_json(
        evaluate_ranking(["A", "D", "B", "C"], relevance, [1, 2])
    )


def test_evaluate_fusion_reports_o3_exclusions_and_fused_order() -> None:
    dataset = parse_dataset(_payload())
    result = evaluate_fusion(dataset, [1, 2])

    o3 = result["objectives"]["O3"]
    assert o3["num_claims"] == 3
    assert o3["exclusions"] == [
        {
            "claim_id": "C",
            "reason": "No baseline rank for O3.",
        }
    ]
    assert o3["methods"]["baseline"]["ranked_claim_ids"] == [
        "A",
        "B",
        "D",
    ]
    assert o3["methods"]["ecs_only"]["ranked_claim_ids"] == [
        "D",
        "A",
        "B",
    ]
    assert o3["methods"]["ecs_only"]["missing_ecs_claim_ids"] == []
    assert o3["methods"]["fused"]["ranked_claim_ids"] == [
        "A",
        "D",
        "B",
    ]


def test_all_missing_ecs_is_deterministic_and_not_imputed() -> None:
    payload = _payload()
    for example in payload["examples"]:
        example["ecs"] = None
    dataset = parse_dataset(payload)

    result = evaluate_fusion(dataset, [1, 2])
    o1 = result["objectives"]["O1"]

    assert o1["methods"]["ecs_only"]["ranked_claim_ids"] == []
    assert o1["methods"]["ecs_only"]["missing_ecs_claim_ids"] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert o1["methods"]["ecs_only"]["metrics"]["1"]["precision"] == 0.0
    assert o1["methods"]["ecs_only"]["metrics"]["1"]["recall"] == 0.0
    assert o1["methods"]["ecs_only"]["metrics"]["1"]["ndcg"] == 0.0
    assert o1["methods"]["fused"]["ranked_claim_ids"] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert all(
        ecs_rank is None
        for ecs_rank in o1["methods"]["fused"]["ecs_ranks"].values()
    )


def test_evaluate_fusion_is_reproducible() -> None:
    dataset = parse_dataset(_payload())

    first = evaluate_fusion(dataset, [1, 3], rrf_k=10)
    second = evaluate_fusion(dataset, [1, 3], rrf_k=10)

    assert first == second


def test_invalid_rrf_k_rejected() -> None:
    dataset = parse_dataset(_payload())

    with pytest.raises(RankingFusionError):
        evaluate_fusion(dataset, [1], rrf_k=0)
    with pytest.raises(RankingFusionError):
        build_fusion_pending_report(rrf_k=-1)


def test_pending_report_has_no_metric_values() -> None:
    report = build_fusion_pending_report(
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert report["status"] == PENDING_STATUS
    assert report["num_claims_evaluated"] == 0
    assert report["rrf_k"] == pytest.approx(60.0)
    assert report["ecs_only_missing_policy"] == ECS_ONLY_MISSING_POLICY
    assert report["annotation_required"] is True
    assert report["nli_benchmark_used_as_claim_ranking_benchmark"] is False
    assert report["expected_dataset_path"].endswith(
        "evaluation/claim_ranking/data/claim_ranking_dataset_v1.json"
    )
    for method in ("baseline", "ecs_only", "fused"):
        assert report["methods"][method]["status"] == PENDING_STATUS
        assert report["methods"][method]["metrics"] is None


def test_pending_markdown_is_explicit() -> None:
    markdown = render_fusion_markdown(
        build_fusion_pending_report(
            generated_at="2026-01-01T00:00:00+00:00",
        )
    )

    assert "pending annotation" in markdown.lower()
    assert "No metric values are reported" in markdown
    assert "RRF" in markdown
    assert "ECS-only" in markdown


def test_computed_report_and_markdown() -> None:
    dataset = parse_dataset(_payload())
    report = build_fusion_report(
        dataset,
        k_values=[1, 5],
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert report["status"] == COMPUTED_STATUS
    assert report["dataset_id"] == "synthetic-fusion-fixture"
    assert report["dataset_path"] == "<memory>"
    assert report["dataset_sha256"] is None
    assert report["num_claims_evaluated"] == 4
    assert report["missing_data"]["missing_ecs_count"] == 1
    assert report["missing_data"]["missing_baseline_rank_counts"]["O3"] == 1
    assert report["nli_benchmark_used_as_claim_ranking_benchmark"] is False
    assert report["objectives"]["O1"]["methods"]["fused"][
        "ranked_claim_ids"
    ] == ["A", "D", "B", "C"]

    markdown = render_fusion_markdown(report)
    assert "Status: computed" in markdown
    assert "baseline" in markdown
    assert "ecs_only" in markdown
    assert "fused" in markdown
    assert "n/a" in markdown


def test_write_fusion_report_round_trips_json(tmp_path: Path) -> None:
    report = build_fusion_pending_report(
        generated_at="2026-01-01T00:00:00+00:00",
    )

    json_path, markdown_path = write_fusion_report(
        report,
        output_dir=tmp_path / "nested" / "results",
        prefix="pending_fixture",
    )

    assert json_path.exists()
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "pending annotation" in markdown_path.read_text(
        encoding="utf-8"
    ).lower()


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
    assert all(
        report["methods"][method]["metrics"] is None
        for method in ("baseline", "ecs_only", "fused")
    )


def test_cli_with_synthetic_dataset_writes_computed_report(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "synthetic.json"
    dataset_path.write_text(
        json.dumps(_payload()),
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
    json_path = output_dir / "claim_ranking_fusion.json"
    markdown_path = output_dir / "claim_ranking_fusion.md"
    assert json_path.exists()
    assert markdown_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["status"] == COMPUTED_STATUS
    assert report["dataset_sha256"] is not None
    assert report["objectives"]["O1"]["methods"]["fused"][
        "ranked_claim_ids"
    ] == ["A", "D", "B", "C"]


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
