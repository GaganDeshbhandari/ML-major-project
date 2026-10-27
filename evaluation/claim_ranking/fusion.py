"""Compare baseline, ECS-only, and RRF-fused claim rankings.

This module is an offline evaluation harness. It consumes the Phase 7A
validated dataset, which contains human priority labels, recorded O1/O2/O3
baseline ranks, and recorded ECS values. It does not run prioritization,
retrieval, NLI, ECS scoring, or FactLLaMA.

Missing ECS policy:

- Fused ranking: missing ECS keeps only the baseline RRF contribution. The
  claim remains in the fused ranking and receives no fabricated ECS rank.
- ECS-only ranking: claims with missing ECS are excluded and reported. They
  remain in the evaluation relevance set, so ECS-only metrics treat the
  unranked positions as absent rather than inventing a score.

All rankings use the same objective-eligible claim set, human labels, K values,
and relevance threshold for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.ranking_fusion import (
    DEFAULT_RRF_K,
    RankingFusionError,
    fuse_rankings,
    rank_claim_ids_by_ecs,
    validate_rrf_k,
)

from .dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_VERSION,
    DEFAULT_LABELING_GUIDE_VERSION,
    OBJECTIVES,
    ClaimRankingDataset,
    ClaimRankingExample,
    DatasetValidationError,
    Exclusion,
    baseline_ranking,
    dataset_summary,
    load_dataset,
)
from .metrics import (
    DEFAULT_K_VALUES,
    DEFAULT_RELEVANT_MIN_GRADE,
    EvaluationInputError,
    evaluate_ranking,
    validate_k_values,
    validate_relevant_min_grade,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_PREFIX = "claim_ranking_fusion"
PENDING_OUTPUT_PREFIX = "claim_ranking_fusion_pending"

PENDING_STATUS = "pending_annotation"
COMPUTED_STATUS = "computed"
ECS_ONLY_MISSING_POLICY = "exclude_missing_ecs"

DEFAULT_PENDING_REASON = (
    "No manually labeled claim-ranking dataset is present in the repository. "
    "The 90-example NLI stance benchmark is a claim-evidence stance benchmark, "
    "not a claim-level fact-checking priority benchmark. Baseline, ECS-only, "
    "and RRF-fused metrics must wait until independent claim-level annotations "
    "are collected and frozen."
)

FUSION_METRIC_DEFINITIONS: dict[str, str] = {
    "precision_at_k": (
        "Number of claims with relevance_grade >= relevant_min_grade in the "
        "top K, divided by K. A K larger than the number of labeled claims in "
        "the objective's evaluation set is reported as null."
    ),
    "recall_at_k": (
        "Number of claims with relevance_grade >= relevant_min_grade in the "
        "top K, divided by the total number of relevant claims in the "
        "objective's evaluation set. Null when there are no relevant claims "
        "or K exceeds the set size."
    ),
    "ndcg_at_k": (
        "Discounted cumulative gain using exponential gain (2**grade - 1) and "
        "discount 1/log2(position + 1), divided by the ideal DCG over the same "
        "K and labels. Null when ideal DCG is zero or K exceeds the set size."
    ),
    "relevance_threshold": (
        "Precision and recall use relevance_grade >= 1 as binary relevant. "
        "NDCG uses the full 0-3 grade."
    ),
    "k_selection": (
        "K values are pre-registered before inspecting results. Default K "
        "values are 1, 3, 5, and 10. A K larger than the number of labeled "
        "claims in an objective's evaluation set is reported as null rather "
        "than computed on an undersized set."
    ),
    "baseline": (
        "The selected objective's existing O1, O2, or O3 ranking. It is never "
        "replaced by a synthetic combined objective ranking."
    ),
    "ecs_only": (
        "Claims with finite recorded ECS values, ordered by descending ECS. "
        "ECS ties use baseline order. Missing ECS claims are excluded and "
        "reported; they remain in the shared relevance set for metric "
        "denominators."
    ),
    "fused": (
        "Reciprocal Rank Fusion of the selected baseline ranking and the ECS "
        "ranking: 1/(k + baseline_rank) + 1/(k + ecs_rank), using 1-based "
        "ranks. Missing ECS claims receive only the baseline contribution."
    ),
    "rrf_k": (
        "Positive finite RRF smoothing constant. The default is 60.0, matching "
        "the standard RRF convention."
    ),
    "ties": (
        "ECS ties preserve baseline order. Final RRF ties resolve by baseline "
        "rank, then original input order. All rankings are total orders."
    ),
    "missing_ecs": (
        "null ECS is never imputed as 0.0. The fused ranking keeps missing-ECS "
        "claims with a baseline-only contribution. ECS-only evaluation "
        "excludes them and reports the affected claim IDs."
    ),
}


@dataclass(frozen=True, slots=True)
class ECSOnlyEvaluationSet:
    """The ECS-only ranking and its explicit missing-ECS exclusions."""

    objective: str
    claim_ids: tuple[str, ...]
    ranked_claim_ids: tuple[str, ...]
    relevance: Mapping[str, int]
    exclusions: tuple[Exclusion, ...]
    missing_ecs_claim_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FusedEvaluationSet:
    """The RRF-fused ranking and its component ranks and scores."""

    objective: str
    claim_ids: tuple[str, ...]
    ranked_claim_ids: tuple[str, ...]
    relevance: Mapping[str, int]
    exclusions: tuple[Exclusion, ...]
    baseline_ranks: Mapping[str, int]
    ecs_ranks: Mapping[str, int | None]
    rrf_scores: Mapping[str, float]


def _expected_dataset_path() -> str:
    try:
        return DEFAULT_DATASET_PATH.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return DEFAULT_DATASET_PATH.as_posix()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _examples_by_id(
    dataset: ClaimRankingDataset,
) -> dict[str, ClaimRankingExample]:
    return {example.claim_id: example for example in dataset.examples}


def _eligible_maps(
    dataset: ClaimRankingDataset,
    objective: str,
) -> tuple[
    tuple[str, ...],
    Mapping[str, int],
    Mapping[str, float | None],
    Mapping[str, int],
    tuple[Exclusion, ...],
]:
    base = baseline_ranking(dataset, objective)
    examples = _examples_by_id(dataset)
    baseline_ranks: dict[str, int] = {}
    ecs_scores: dict[str, float | None] = {}

    for claim_id in base.claim_ids:
        example = examples[claim_id]
        rank = example.baseline_ranks[base.objective]
        if rank is None:
            raise AssertionError(
                f"Baseline ranking included claim {claim_id!r} without a rank."
            )
        baseline_ranks[claim_id] = rank
        ecs_scores[claim_id] = example.ecs

    return (
        base.claim_ids,
        MappingProxyType(baseline_ranks),
        MappingProxyType(ecs_scores),
        base.relevance,
        base.exclusions,
    )


def ecs_only_ranking(
    dataset: ClaimRankingDataset,
    objective: str,
) -> ECSOnlyEvaluationSet:
    """Build the ECS-only ranking for one objective.

    Only finite recorded ECS values are ranked. Missing ECS claims are excluded
    from the ranking and listed with a reason; they remain in ``relevance`` so
    the shared evaluation set is unchanged.
    """

    claim_ids, baseline_ranks, ecs_scores, relevance, base_exclusions = (
        _eligible_maps(dataset, objective)
    )
    ranked = rank_claim_ids_by_ecs(
        claim_ids,
        baseline_ranks,
        ecs_scores,
    )
    missing = tuple(
        claim_id
        for claim_id in claim_ids
        if ecs_scores[claim_id] is None
    )
    exclusions = base_exclusions + tuple(
        Exclusion(
            claim_id=claim_id,
            reason=(
                f"No ECS score for ECS-only {objective} ranking."
            ),
        )
        for claim_id in missing
    )
    return ECSOnlyEvaluationSet(
        objective=objective,
        claim_ids=claim_ids,
        ranked_claim_ids=ranked,
        relevance=relevance,
        exclusions=exclusions,
        missing_ecs_claim_ids=missing,
    )


def fused_ranking(
    dataset: ClaimRankingDataset,
    objective: str,
    *,
    rrf_k: float = DEFAULT_RRF_K,
) -> FusedEvaluationSet:
    """Build the RRF-fused ranking for one objective.

    Every objective-eligible claim is returned exactly once. Missing ECS claims
    keep only their baseline contribution and receive ``ecs_rank=None``.
    """

    claim_ids, baseline_ranks, ecs_scores, relevance, exclusions = (
        _eligible_maps(dataset, objective)
    )
    fused = fuse_rankings(
        claim_ids,
        baseline_ranks,
        ecs_scores,
        k=rrf_k,
    )
    return FusedEvaluationSet(
        objective=objective,
        claim_ids=claim_ids,
        ranked_claim_ids=tuple(item.claim_id for item in fused),
        relevance=relevance,
        exclusions=exclusions,
        baseline_ranks=MappingProxyType(
            {
                item.claim_id: item.baseline_rank
                for item in fused
            }
        ),
        ecs_ranks=MappingProxyType(
            {
                item.claim_id: item.ecs_rank
                for item in fused
            }
        ),
        rrf_scores=MappingProxyType(
            {
                item.claim_id: item.rrf_score
                for item in fused
            }
        ),
    )


def _metrics_payload(
    metrics: Mapping[int, Mapping[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    return {
        str(k): {
            "precision": values["precision"],
            "recall": values["recall"],
            "ndcg": values["ndcg"],
        }
        for k, values in metrics.items()
    }


def _exclusions_payload(
    exclusions: Sequence[Exclusion],
) -> list[dict[str, str]]:
    return [
        {
            "claim_id": exclusion.claim_id,
            "reason": exclusion.reason,
        }
        for exclusion in exclusions
    ]


def evaluate_fusion(
    dataset: ClaimRankingDataset,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    *,
    rrf_k: float = DEFAULT_RRF_K,
    relevant_min_grade: int = DEFAULT_RELEVANT_MIN_GRADE,
) -> dict[str, Any]:
    """Compare baseline, ECS-only, and fused rankings on one labeled set.

    All three methods use the same objective-eligible claim IDs, human labels,
    K values, and relevance threshold. ECS-only rankings may be shorter because
    missing ECS claims are excluded and reported. Metrics treat unranked
    positions as absent; no scores are imputed.
    """

    validated_k_values = validate_k_values(k_values)
    smoothing = validate_rrf_k(rrf_k)
    threshold = validate_relevant_min_grade(relevant_min_grade)
    objectives: dict[str, Any] = {}

    for objective in OBJECTIVES:
        base = baseline_ranking(dataset, objective)
        ecs_set = ecs_only_ranking(dataset, objective)
        fused_set = fused_ranking(
            dataset,
            objective,
            rrf_k=smoothing,
        )

        baseline_metrics = evaluate_ranking(
            base.ranked_claim_ids,
            base.relevance,
            validated_k_values,
            relevant_min_grade=threshold,
        )
        ecs_metrics = evaluate_ranking(
            ecs_set.ranked_claim_ids,
            base.relevance,
            validated_k_values,
            relevant_min_grade=threshold,
        )
        fused_metrics = evaluate_ranking(
            fused_set.ranked_claim_ids,
            base.relevance,
            validated_k_values,
            relevant_min_grade=threshold,
        )

        objectives[objective] = {
            "num_claims": len(base.claim_ids),
            "exclusions": _exclusions_payload(base.exclusions),
            "methods": {
                "baseline": {
                    "num_ranked": len(base.ranked_claim_ids),
                    "ranked_claim_ids": list(base.ranked_claim_ids),
                    "exclusions": [],
                    "metrics": _metrics_payload(baseline_metrics),
                },
                "ecs_only": {
                    "num_ranked": len(ecs_set.ranked_claim_ids),
                    "ranked_claim_ids": list(ecs_set.ranked_claim_ids),
                    "missing_ecs_claim_ids": list(
                        ecs_set.missing_ecs_claim_ids
                    ),
                    "exclusions": _exclusions_payload(
                        tuple(
                            Exclusion(
                                claim_id=claim_id,
                                reason=(
                                    f"No ECS score for ECS-only "
                                    f"{objective} ranking."
                                ),
                            )
                            for claim_id in ecs_set.missing_ecs_claim_ids
                        )
                    ),
                    "metrics": _metrics_payload(ecs_metrics),
                },
                "fused": {
                    "num_ranked": len(fused_set.ranked_claim_ids),
                    "ranked_claim_ids": list(
                        fused_set.ranked_claim_ids
                    ),
                    "exclusions": [],
                    "rrf_k": smoothing,
                    "baseline_ranks": dict(fused_set.baseline_ranks),
                    "ecs_ranks": dict(fused_set.ecs_ranks),
                    "rrf_scores": dict(fused_set.rrf_scores),
                    "metrics": _metrics_payload(fused_metrics),
                },
            },
        }

    return {
        "k_values": list(validated_k_values),
        "relevant_min_grade": threshold,
        "rrf_k": smoothing,
        "ecs_only_missing_policy": ECS_ONLY_MISSING_POLICY,
        "objectives": objectives,
        "dataset_exclusions": [
            {
                "claim_id": exclusion.claim_id,
                "reason": exclusion.reason,
            }
            for exclusion in dataset.exclusions
        ],
    }


def build_fusion_report(
    dataset: ClaimRankingDataset,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    rrf_k: float = DEFAULT_RRF_K,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a computed comparison report for a labeled dataset."""

    summary = dataset_summary(dataset)
    evaluation = evaluate_fusion(
        dataset,
        k_values,
        rrf_k=rrf_k,
    )
    return {
        "status": COMPUTED_STATUS,
        "dataset_id": dataset.metadata["dataset_id"],
        "dataset_version": dataset.metadata["version"],
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "labeling_guide_version": dataset.metadata[
            "labeling_guide_version"
        ],
        "dataset_path": str(dataset.path),
        "dataset_sha256": dataset.sha256 or None,
        "k_values": evaluation["k_values"],
        "rrf_k": evaluation["rrf_k"],
        "ecs_only_missing_policy": evaluation[
            "ecs_only_missing_policy"
        ],
        "metric_definitions": dict(FUSION_METRIC_DEFINITIONS),
        "num_claims_evaluated": summary["num_examples"],
        "dataset_summary": summary,
        "objectives": evaluation["objectives"],
        "exclusions": evaluation["dataset_exclusions"],
        "missing_data": {
            "missing_ecs_count": summary["missing_ecs_count"],
            "missing_baseline_rank_counts": summary[
                "missing_baseline_rank_counts"
            ],
        },
        "annotation_required": False,
        "nli_benchmark_used_as_claim_ranking_benchmark": False,
        "notes": [
            "All methods use the same human labels, K values, relevance "
            "threshold, and objective-eligible claim set.",
            "Synthetic fixtures in unit tests are not saved as empirical "
            "evaluation results.",
        ],
        "generated_at": generated_at or _now_utc(),
    }


def build_fusion_pending_report(
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    rrf_k: float = DEFAULT_RRF_K,
    generated_at: str | None = None,
    reason: str = DEFAULT_PENDING_REASON,
) -> dict[str, Any]:
    """Build a report that explicitly contains no metric values."""

    validated_k_values = validate_k_values(k_values)
    smoothing = validate_rrf_k(rrf_k)
    return {
        "status": PENDING_STATUS,
        "dataset_id": DEFAULT_DATASET_ID,
        "dataset_version": DEFAULT_DATASET_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "labeling_guide_version": DEFAULT_LABELING_GUIDE_VERSION,
        "dataset_path": None,
        "dataset_sha256": None,
        "expected_dataset_path": _expected_dataset_path(),
        "k_values": list(validated_k_values),
        "rrf_k": smoothing,
        "ecs_only_missing_policy": ECS_ONLY_MISSING_POLICY,
        "metric_definitions": dict(FUSION_METRIC_DEFINITIONS),
        "num_claims_evaluated": 0,
        "baseline_rankings_available": False,
        "methods": {
            method: {
                "status": PENDING_STATUS,
                "metrics": None,
            }
            for method in ("baseline", "ecs_only", "fused")
        },
        "exclusions": [],
        "missing_data": [
            "No manually labeled claim-ranking dataset.",
            "No baseline rankings or ECS values recorded for a real claim set.",
        ],
        "annotation_required": True,
        "nli_benchmark_used_as_claim_ranking_benchmark": False,
        "reason": reason,
        "notes": [
            "No metric values are reported because no manually labeled "
            "claim-ranking dataset exists.",
            "Synthetic fixtures in unit tests are not saved as empirical "
            "evaluation results.",
        ],
        "generated_at": generated_at or _now_utc(),
    }


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_exclusions(
    exclusions: Sequence[Mapping[str, Any]],
) -> list[str]:
    if not exclusions:
        return ["- None."]
    return [
        f"- `{exclusion.get('claim_id')}`: {exclusion.get('reason')}"
        for exclusion in exclusions
    ]


def render_fusion_markdown(report: Mapping[str, Any]) -> str:
    """Render a pending or computed fusion comparison as Markdown."""

    lines: list[str] = ["# Claim-Ranking Fusion", ""]
    status = report.get("status")

    if status == PENDING_STATUS:
        lines.extend(
            [
                "**Status: pending annotation.** No metric values are reported.",
                "",
                "## Dataset",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Dataset ID | `{report.get('dataset_id')}` |",
                f"| Dataset version | `{report.get('dataset_version')}` |",
                f"| Schema version | `{report.get('dataset_schema_version')}` |",
                "| Labeling guide version | "
                f"`{report.get('labeling_guide_version')}` |",
                "| Dataset path | not available |",
                "| Dataset SHA-256 | not available |",
                "| Expected dataset path | "
                f"`{report.get('expected_dataset_path')}` |",
                f"| Claims evaluated | {report.get('num_claims_evaluated')} |",
                f"| RRF k | {report.get('rrf_k')} |",
                "| ECS-only missing policy | "
                f"`{report.get('ecs_only_missing_policy')}` |",
                "",
                "## Why metrics are pending",
                "",
                str(report.get("reason", "")),
                "",
                "## Missing data",
                "",
            ]
        )
        missing_data = report.get("missing_data", [])
        if isinstance(missing_data, list) and missing_data:
            lines.extend(f"- {item}" for item in missing_data)
        else:
            lines.append("- None recorded.")
        lines.extend(
            [
                "",
                "## Methods",
                "",
                "| Method | Status |",
                "|---|---|",
                "| Baseline O1/O2/O3 | pending annotation |",
                "| ECS-only | pending annotation |",
                "| RRF fused | pending annotation |",
                "",
                "## Metric definitions",
                "",
            ]
        )
        for name, definition in report.get("metric_definitions", {}).items():
            lines.append(f"- **{name}**: {definition}")
        lines.extend(
            [
                "",
                "## Reproducibility",
                "",
                "Regenerate this pending report from the project root:",
                "",
                "```powershell",
                ".\\venv312\\Scripts\\python.exe -m evaluation.claim_ranking.fusion",
                "```",
                "",
                "Once a labeled dataset exists, run:",
                "",
                "```powershell",
                ".\\venv312\\Scripts\\python.exe -m evaluation.claim_ranking.fusion "
                "--dataset <path-to-dataset.json> --k 1 3 5 10 --rrf-k 60",
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "**Status: computed from a manually labeled dataset.**",
            "",
            "## Dataset",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Dataset ID | `{report.get('dataset_id')}` |",
            f"| Dataset version | `{report.get('dataset_version')}` |",
            f"| Schema version | `{report.get('dataset_schema_version')}` |",
            f"| Labeling guide version | `{report.get('labeling_guide_version')}` |",
            f"| Dataset path | `{report.get('dataset_path')}` |",
            f"| Dataset SHA-256 | `{report.get('dataset_sha256')}` |",
            f"| Claims evaluated | {report.get('num_claims_evaluated')} |",
            f"| RRF k | {report.get('rrf_k')} |",
            f"| ECS-only missing policy | `{report.get('ecs_only_missing_policy')}` |",
            "",
            "## Missing data",
            "",
        ]
    )

    missing_data = report.get("missing_data", {})
    if isinstance(missing_data, Mapping):
        lines.append(
            "- Claims with missing ECS: "
            f"{missing_data.get('missing_ecs_count', 'n/a')}"
        )
        missing_ranks = missing_data.get(
            "missing_baseline_rank_counts",
            {},
        )
        if isinstance(missing_ranks, Mapping):
            for objective in OBJECTIVES:
                lines.append(
                    f"- Missing baseline ranks for {objective}: "
                    f"{missing_ranks.get(objective, 'n/a')}"
                )
        else:
            lines.append("- Missing baseline ranks: n/a")
    else:
        lines.append("- None recorded.")

    objectives = report.get("objectives", {})
    for objective in OBJECTIVES:
        objective_report = objectives.get(objective)
        lines.extend([f"## {objective}", ""])
        if not objective_report:
            lines.extend(["No eligible claims.", ""])
            continue

        lines.extend(
            [
                f"- Claims in evaluation set: {objective_report.get('num_claims')}",
                "- Exclusions:",
            ]
        )
        lines.extend(
            _render_exclusions(objective_report.get("exclusions", []))
        )
        methods = objective_report.get("methods", {})
        lines.extend(
            [
                "",
                "| Method | Ranked | K | Precision@K | Recall@K | NDCG@K |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in ("baseline", "ecs_only", "fused"):
            method_report = methods.get(method, {})
            metrics = method_report.get("metrics", {})
            if not metrics:
                lines.append(
                    f"| {method} | {method_report.get('num_ranked', 'n/a')} "
                    "| n/a | n/a | n/a | n/a |"
                )
                continue
            for k in sorted(metrics, key=lambda value: int(value)):
                values = metrics[k]
                lines.append(
                    f"| {method} | {method_report.get('num_ranked')} | {k} "
                    f"| {_format_metric(values.get('precision'))} "
                    f"| {_format_metric(values.get('recall'))} "
                    f"| {_format_metric(values.get('ndcg'))} |"
                )
        ecs_only = methods.get("ecs_only", {})
        missing_ids = ecs_only.get("missing_ecs_claim_ids", [])
        if missing_ids:
            lines.extend(
                [
                    "",
                    "ECS-only exclusions for missing ECS:",
                ]
            )
            lines.extend(
                _render_exclusions(ecs_only.get("exclusions", []))
            )
        lines.append("")

    lines.extend(
        [
            "## Dataset exclusions",
            "",
        ]
    )
    lines.extend(_render_exclusions(report.get("exclusions", [])))
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
        ]
    )
    for name, definition in report.get("metric_definitions", {}).items():
        lines.append(f"- **{name}**: {definition}")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "The report records the dataset path and SHA-256. Recompute with:",
            "",
            "```powershell",
            ".\\venv312\\Scripts\\python.exe -m evaluation.claim_ranking.fusion "
            "--dataset <path-to-dataset.json> --k 1 3 5 10 --rrf-k 60",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_fusion_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> tuple[Path, Path]:
    """Write JSON and Markdown fusion reports and return their paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{prefix}.json"
    markdown_path = output_path / f"{prefix}.md"

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_fusion_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the claim-ranking fusion comparison report builder."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline, ECS-only, and RRF-fused claim rankings on a "
            "manually labeled dataset, or write a pending report when no "
            "dataset exists."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to a validated claim-ranking dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON and Markdown reports.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output filename prefix without an extension.",
    )
    parser.add_argument(
        "--k",
        dest="k_values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Pre-registered K values.",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=DEFAULT_RRF_K,
        help="Positive RRF smoothing constant.",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default=None,
        help="Optional ISO timestamp for reproducible report fixtures.",
    )
    args = parser.parse_args(argv)

    try:
        if args.dataset is None:
            report = build_fusion_pending_report(
                k_values=args.k_values,
                rrf_k=args.rrf_k,
                generated_at=args.generated_at,
            )
            prefix = args.output_prefix or PENDING_OUTPUT_PREFIX
        else:
            dataset = load_dataset(args.dataset)
            report = build_fusion_report(
                dataset,
                k_values=args.k_values,
                rrf_k=args.rrf_k,
                generated_at=args.generated_at,
            )
            prefix = args.output_prefix or DEFAULT_OUTPUT_PREFIX
    except (
        DatasetValidationError,
        EvaluationInputError,
        RankingFusionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_path, markdown_path = write_fusion_report(
        report,
        output_dir=args.output_dir,
        prefix=prefix,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPUTED_STATUS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OUTPUT_PREFIX",
    "DEFAULT_PENDING_REASON",
    "ECS_ONLY_MISSING_POLICY",
    "FUSION_METRIC_DEFINITIONS",
    "PENDING_OUTPUT_PREFIX",
    "PENDING_STATUS",
    "ECSOnlyEvaluationSet",
    "FusedEvaluationSet",
    "build_fusion_pending_report",
    "build_fusion_report",
    "ecs_only_ranking",
    "evaluate_fusion",
    "fused_ranking",
    "main",
    "render_fusion_markdown",
    "write_fusion_report",
]
