"""Build reproducible claim-ranking baseline reports.

The runner has two modes:

- No dataset: write a clearly marked pending report. No metric values are
  fabricated.
- With a validated, manually labeled dataset: compute per-objective O1/O2/O3
  baseline metrics and write JSON and Markdown reports.

The runner does not invoke production models, retrieval, ECS, or FactLLaMA.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_VERSION,
    DEFAULT_LABELING_GUIDE_VERSION,
    OBJECTIVES,
    ClaimRankingDataset,
    DatasetValidationError,
    dataset_summary,
    evaluate_baseline,
    load_dataset,
)
from .metrics import (
    DEFAULT_K_VALUES,
    EvaluationInputError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_PREFIX = "claim_ranking_baseline"
PENDING_OUTPUT_PREFIX = "claim_ranking_baseline_pending"

PENDING_STATUS = "pending_annotation"
COMPUTED_STATUS = "computed"

DEFAULT_PENDING_REASON = (
    "No manually labeled claim-ranking dataset is present in the repository. "
    "The 90-example NLI stance benchmark is a claim-evidence stance benchmark, "
    "not a claim-level fact-checking priority benchmark, so it cannot be used "
    "to compute claim-ranking baseline metrics. Baseline metrics must wait "
    "until independent claim-level annotations are collected and frozen."
)

METRIC_DEFINITIONS: dict[str, str] = {
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
    "ties": (
        "A ranking is a total order. Baseline ranks are sorted by "
        "(rank, dataset order). For score-based methods, stable_rank_by_score "
        "sorts by descending score, places null last, and preserves input "
        "order for ties. The evaluation module does not reorder a supplied "
        "ranking."
    ),
    "missing_ecs": (
        "null ECS is preserved as missing and is never imputed as 0.0. "
        "Baseline metrics do not use ECS. Future ECS-sensitive methods must "
        "report how missing ECS values are ordered or excluded."
    ),
    "duplicate_claims": (
        "Duplicate claim IDs are rejected in the dataset and in ranking "
        "inputs. Each claim ID must identify exactly one claim."
    ),
    "insufficient_evidence": (
        "A claim may have ecs=null and/or no evidence. It is not automatically "
        "excluded from baseline metrics if it has a human priority label and a "
        "baseline rank. Future ECS methods must document their handling."
    ),
    "exclusions": (
        "A claim with a null baseline rank for an objective is excluded from "
        "that objective's metrics and listed with a reason. Dataset-level "
        "exclusions are recorded separately. No labels or ranks are imputed."
    ),
}


def _expected_dataset_path() -> str:
    try:
        return DEFAULT_DATASET_PATH.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return DEFAULT_DATASET_PATH.as_posix()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pending_report(
    *,
    generated_at: str | None = None,
    reason: str = DEFAULT_PENDING_REASON,
) -> dict[str, Any]:
    """Build a report that explicitly contains no metric values."""

    return {
        "status": PENDING_STATUS,
        "dataset_id": DEFAULT_DATASET_ID,
        "dataset_version": DEFAULT_DATASET_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "labeling_guide_version": DEFAULT_LABELING_GUIDE_VERSION,
        "dataset_path": None,
        "dataset_sha256": None,
        "expected_dataset_path": _expected_dataset_path(),
        "k_values": list(DEFAULT_K_VALUES),
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "num_claims_evaluated": 0,
        "baseline_rankings_available": False,
        "baseline_metrics": {
            objective: None
            for objective in OBJECTIVES
        },
        "exclusions": [],
        "annotation_required": True,
        "nli_benchmark_used_as_claim_ranking_benchmark": False,
        "reason": reason,
        "notes": [
            "No metric values are reported because no manually labeled "
            "claim-ranking dataset exists.",
            "Synthetic fixtures in unit tests are not saved as evaluation "
            "results.",
        ],
        "generated_at": generated_at or _now_utc(),
    }


def build_baseline_report(
    dataset: ClaimRankingDataset,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a report for a validated, manually labeled dataset."""

    summary = dataset_summary(dataset)
    baseline = evaluate_baseline(dataset, k_values)
    baseline_rankings_available = any(
        count > 0
        for count in summary["baseline_rank_counts"].values()
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
        "k_values": baseline["k_values"],
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "num_claims_evaluated": summary["num_examples"],
        "baseline_rankings_available": baseline_rankings_available,
        "dataset_summary": summary,
        "baseline_metrics": baseline["objectives"],
        "exclusions": baseline["dataset_exclusions"],
        "annotation_required": False,
        "nli_benchmark_used_as_claim_ranking_benchmark": False,
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
    lines: list[str] = []
    for exclusion in exclusions:
        lines.append(
            f"- `{exclusion.get('claim_id')}`: {exclusion.get('reason')}"
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a pending or computed report as Markdown."""

    lines: list[str] = ["# Claim-Ranking Baseline", ""]
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
                f"| Labeling guide version | `{report.get('labeling_guide_version')}` |",
                f"| Dataset path | not available |",
                f"| Dataset SHA-256 | not available |",
                f"| Expected dataset path | `{report.get('expected_dataset_path')}` |",
                f"| Claims evaluated | {report.get('num_claims_evaluated')} |",
                "",
                "## Why metrics are pending",
                "",
                str(report.get("reason", "")),
                "",
                "## Baseline metrics",
                "",
                "| Objective | Status |",
                "|---|---|",
            ]
        )
        for objective in OBJECTIVES:
            lines.append(f"| {objective} | pending annotation |")
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
                "Regenerate this pending report from the project root:",
                "",
                "```powershell",
                ".\\venv312\\Scripts\\python.exe -m evaluation.claim_ranking.baseline",
                "```",
                "",
                "Once a labeled dataset exists, run:",
                "",
                "```powershell",
                ".\\venv312\\Scripts\\python.exe -m evaluation.claim_ranking.baseline "
                "--dataset <path-to-dataset.json> --k 1 3 5 10",
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
            "",
            "## Baseline metrics",
            "",
        ]
    )

    baseline_metrics = report.get("baseline_metrics", {})
    for objective in OBJECTIVES:
        objective_report = baseline_metrics.get(objective)
        lines.extend([f"### {objective}", ""])
        if not objective_report:
            lines.extend(["No eligible claims.", ""])
            continue
        lines.extend(
            [
                f"- Claims in evaluation set: {objective_report.get('num_claims')}",
                f"- Ranked claims: {objective_report.get('num_ranked')}",
                "- Exclusions:",
            ]
        )
        lines.extend(
            _render_exclusions(objective_report.get("exclusions", []))
        )
        lines.extend(
            [
                "",
                "| K | Precision@K | Recall@K | NDCG@K |",
                "|---:|---:|---:|---:|",
            ]
        )
        metrics = objective_report.get("metrics", {})
        for k in sorted(metrics, key=lambda value: int(value)):
            values = metrics[k]
            lines.append(
                f"| {k} | {_format_metric(values.get('precision'))} | "
                f"{_format_metric(values.get('recall'))} | "
                f"{_format_metric(values.get('ndcg'))} |"
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
            ".\\venv312\\Scripts\\python.exe -m evaluation.claim_ranking.baseline "
            "--dataset <path-to-dataset.json> --k 1 3 5 10",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports and return their paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{prefix}.json"
    markdown_path = output_path / f"{prefix}.md"

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the claim-ranking baseline report builder."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a claim-ranking baseline report from a manually labeled "
            "dataset, or write a pending report when no dataset exists."
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
        "--generated-at",
        type=str,
        default=None,
        help="Optional ISO timestamp for reproducible report fixtures.",
    )
    args = parser.parse_args(argv)

    try:
        if args.dataset is None:
            report = build_pending_report(
                generated_at=args.generated_at,
            )
            prefix = args.output_prefix or PENDING_OUTPUT_PREFIX
        else:
            dataset = load_dataset(args.dataset)
            report = build_baseline_report(
                dataset,
                k_values=args.k_values,
                generated_at=args.generated_at,
            )
            prefix = args.output_prefix or DEFAULT_OUTPUT_PREFIX
    except (DatasetValidationError, EvaluationInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_path, markdown_path = write_report(
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
    "METRIC_DEFINITIONS",
    "PENDING_OUTPUT_PREFIX",
    "PENDING_STATUS",
    "build_baseline_report",
    "build_pending_report",
    "main",
    "render_markdown",
    "write_report",
]
