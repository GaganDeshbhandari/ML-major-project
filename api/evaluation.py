"""Read-only evaluation endpoints backed by saved artifacts.

No training, inference, retrieval, or metric recomputation happens here. The
endpoints load JSON reports produced by the offline evaluation runners and
return their measured values. Missing artifacts and missing metrics are marked
as unavailable rather than reported as zero.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = REPO_ROOT / "evaluation"

NLI_BENCHMARK_PATH = "evidence_conflict/results/nli_stance_benchmark.json"
CLAIM_RANKING_FUSION_PATH = "claim_ranking/results/claim_ranking_fusion.json"
CLAIM_RANKING_PENDING_PATH = (
    "claim_ranking/results/claim_ranking_fusion_pending.json"
)
RETRIEVAL_METRICS_PATH = "retrieval/results/retrieval_metrics.json"
FACTLLAMA_PATTERN = "factllama/results/rawfc_test_factllama_*.json"

RETRIEVAL_METRIC_NAMES = (
    "Recall@K",
    "Precision@K",
    "MRR",
    "recall_at_k",
    "precision_at_k",
    "mrr",
)

METHOD_NAMES = {
    "baseline": "Bias-aware-only (O1/O2/O3)",
    "ecs_only": "ECS-only",
    "fused": "Combined (baseline + ECS via RRF)",
}


class EvaluationArtifactError(RuntimeError):
    """A saved evaluation artifact exists but could not be read."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(source)


def _artifact_path(relative: str) -> Path:
    return EVALUATION_ROOT / relative


def _read_json(relative: str) -> dict[str, Any]:
    path = _artifact_path(relative)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read evaluation artifact %s: %s", relative, exc)
        raise EvaluationArtifactError(relative) from exc
    if not isinstance(data, dict):
        logger.warning("Evaluation artifact %s is not a JSON object", relative)
        raise EvaluationArtifactError(relative)
    return data


def _artifact_http_error(exc: EvaluationArtifactError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": "Evaluation artifact could not be read.",
            "source": exc.source,
        },
    )


def _display_path(value: Any) -> Any:
    """Prefer repository-relative paths and avoid leaking absolute paths."""
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _latest_factllama_artifact() -> tuple[Path, str] | None:
    directory = EVALUATION_ROOT / "factllama" / "results"
    paths = list(directory.glob("rawfc_test_factllama_*.json"))
    if not paths:
        return None
    latest = max(paths, key=lambda path: path.name)
    return latest, latest.relative_to(EVALUATION_ROOT).as_posix()


def _claim_ranking_artifact() -> tuple[Path, str] | None:
    for relative in (
        CLAIM_RANKING_FUSION_PATH,
        CLAIM_RANKING_PENDING_PATH,
    ):
        path = _artifact_path(relative)
        if path.is_file():
            return path, relative
    return None


def _unavailable_experiment(
    *,
    name: str,
    task: str,
    note: str,
    status: str = "unavailable",
    source: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "task": task,
        "available": False,
        "status": status,
        "source": source,
        "generated_at": None,
        "dataset": None,
        "model": None,
        "configuration": None,
        "metrics": None,
        "note": note,
    }


def _factllama_experiment(
    data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    metrics = data.get("metrics")
    available = isinstance(metrics, dict) and bool(metrics)
    dataset = data.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    model = data.get("model")
    model = model if isinstance(model, dict) else {}
    note = (
        None
        if available
        else "The saved FactLLaMA artifact does not contain metric values."
    )
    return {
        "name": "factllama_fact_verification",
        "task": "fact_verification",
        "available": available,
        "status": "available" if available else "unavailable",
        "source": source,
        "generated_at": data.get("timestamp_utc"),
        "dataset": {
            "path": _display_path(dataset.get("path")),
            "file_count": dataset.get("file_count"),
            "expected_file_count": dataset.get("expected_file_count"),
            "label_field": dataset.get("label_field"),
            "evidence_field": dataset.get("evidence_field"),
            "original_label_field": dataset.get("original_label_field"),
        },
        "model": {
            "base_model_snapshot": _display_path(
                model.get("base_model_snapshot")
            ),
            "adapter_path": _display_path(model.get("adapter_path")),
            "load_in_4bit": model.get("load_in_4bit"),
            "quant_type": model.get("quant_type"),
            "compute_dtype": model.get("compute_dtype"),
            "double_quant": model.get("double_quant"),
            "device": model.get("device"),
        },
        "configuration": {
            "label_mapping": data.get("label_mapping"),
            "scoring_policy": data.get("scoring_policy"),
        },
        "metrics": metrics if available else None,
        "counts": data.get("counts"),
        "confusion_matrix": data.get("confusion_matrix"),
        "label_distribution": data.get("label_distribution"),
        "prediction_distribution": data.get("prediction_distribution"),
        "note": note,
    }


def _nli_experiment(
    data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    metrics = data.get("metrics")
    available = isinstance(metrics, dict) and bool(metrics)
    dataset = data.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    model = data.get("model")
    model = model if isinstance(model, dict) else {}
    note = (
        None
        if available
        else "The saved NLI stance artifact does not contain metric values."
    )
    return {
        "name": "nli_stance_classification",
        "task": "nli_stance_classification",
        "available": available,
        "status": "available" if available else "unavailable",
        "source": source,
        "generated_at": data.get("generated_at_utc"),
        "benchmark": data.get("benchmark"),
        "dataset": {
            "path": _display_path(dataset.get("path")),
            "sha256": dataset.get("sha256"),
            "total_examples": dataset.get("total_examples"),
            "class_counts": dataset.get("class_counts"),
            "metadata": dataset.get("metadata"),
        },
        "model": {
            "name": model.get("name"),
            "model_class": model.get("model_class"),
            "resolved_device": model.get("resolved_device"),
            "sequence_length": model.get("sequence_length"),
            "configured_revision": model.get("configured_revision"),
            "confidence_threshold": model.get("confidence_threshold"),
        },
        "configuration": {
            "runtime": data.get("runtime"),
        },
        "metrics": metrics if available else None,
        "note": note,
    }


def _claim_ranking_experiment(
    data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    objectives = data.get("objectives")
    metrics_available = (
        data.get("status") == "computed"
        and isinstance(objectives, dict)
        and bool(objectives)
    )
    status = data.get("status")
    return {
        "name": "claim_ranking_prioritization",
        "task": "claim_ranking",
        "available": metrics_available,
        "status": status or ("available" if metrics_available else "unavailable"),
        "source": source,
        "generated_at": data.get("generated_at"),
        "dataset": {
            "dataset_id": data.get("dataset_id"),
            "version": data.get("dataset_version"),
            "schema_version": data.get("dataset_schema_version"),
            "labeling_guide_version": data.get("labeling_guide_version"),
            "path": _display_path(data.get("dataset_path")),
            "sha256": data.get("dataset_sha256"),
            "expected_path": data.get("expected_dataset_path"),
            "num_claims_evaluated": data.get("num_claims_evaluated"),
        },
        "model": None,
        "configuration": {
            "k_values": data.get("k_values"),
            "rrf_k": data.get("rrf_k"),
            "ecs_only_missing_policy": data.get("ecs_only_missing_policy"),
            "baseline_rankings_available": data.get(
                "baseline_rankings_available"
            ),
            "annotation_required": data.get("annotation_required"),
        },
        "metrics": objectives if metrics_available else None,
        "metrics_scope": "per_objective" if metrics_available else None,
        "reason": data.get("reason"),
        "notes": data.get("notes") if isinstance(data.get("notes"), list) else [],
    }


def _retrieval_metrics(data: dict[str, Any]) -> dict[str, Any] | None:
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    fallback = {
        name: data[name]
        for name in RETRIEVAL_METRIC_NAMES
        if name in data
    }
    return fallback or None


def _retrieval_experiment(
    data: dict[str, Any] | None,
    source: str | None,
) -> dict[str, Any]:
    metrics = _retrieval_metrics(data) if data is not None else None
    available = metrics is not None
    note = None
    if data is None:
        note = (
            "No saved retrieval evaluation artifact exists. Retrieval metrics "
            "such as Recall@K, Precision@K, and MRR are unavailable."
        )
    elif not available:
        note = (
            "The saved retrieval artifact does not contain metric values. "
            "Recall@K, Precision@K, and MRR remain unavailable."
        )
    return {
        "name": "evidence_retrieval",
        "task": "evidence_retrieval",
        "available": available,
        "status": "available" if available else "unavailable",
        "source": source,
        "generated_at": (
            data.get("generated_at") or data.get("generated_at_utc")
            if data is not None
            else None
        ),
        "dataset": data.get("dataset") if data is not None else None,
        "model": None,
        "configuration": (
            data.get("configuration") if data is not None else None
        ),
        "metrics": metrics,
        "expected_metrics": ["Recall@K", "Precision@K", "MRR"],
        "note": note,
    }


def _unavailable_comparison() -> dict[str, Any]:
    return {
        "available": False,
        "status": "unavailable",
        "source": None,
        "generated_at": None,
        "dataset": None,
        "configuration": None,
        "k_values": None,
        "rrf_k": None,
        "num_claims_evaluated": None,
        "methods": {
            method: {
                "display_name": display_name,
                "description": None,
                "available": False,
                "status": "unavailable",
                "metrics": None,
                "metrics_by_objective": None,
                "details_by_objective": None,
            }
            for method, display_name in METHOD_NAMES.items()
        },
        "exclusions": [],
        "missing_data": None,
        "reason": "No saved prioritization comparison artifact exists.",
        "notes": [
            "Run the offline claim-ranking fusion evaluation to create "
            "evaluation/claim_ranking/results/claim_ranking_fusion.json."
        ],
        "metric_definitions": None,
    }


def _comparison_payload(
    data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    status = data.get("status") or "unavailable"
    objectives = data.get("objectives")
    objectives = objectives if isinstance(objectives, dict) else {}
    source_methods = data.get("methods")
    source_methods = source_methods if isinstance(source_methods, dict) else {}
    definitions = data.get("metric_definitions")
    definitions = definitions if isinstance(definitions, dict) else {}

    methods: dict[str, Any] = {}
    for method, display_name in METHOD_NAMES.items():
        metrics_by_objective: dict[str, Any] = {}
        details_by_objective: dict[str, Any] = {}
        for objective, objective_data in objectives.items():
            if not isinstance(objective_data, dict):
                continue
            objective_methods = objective_data.get("methods")
            if not isinstance(objective_methods, dict):
                continue
            detail = objective_methods.get(method)
            if not isinstance(detail, dict):
                continue
            metrics = detail.get("metrics")
            if isinstance(metrics, dict):
                metrics_by_objective[objective] = metrics
            details_by_objective[objective] = {
                "num_ranked": detail.get("num_ranked"),
                "exclusions": detail.get("exclusions"),
            }

        source_method = source_methods.get(method)
        source_method = source_method if isinstance(source_method, dict) else {}
        source_metrics = source_method.get("metrics")
        source_metrics = (
            source_metrics if isinstance(source_metrics, dict) else None
        )
        available = source_metrics is not None or bool(metrics_by_objective)
        method_status = (
            "computed"
            if available and status == "computed"
            else source_method.get("status") or status
        )
        methods[method] = {
            "display_name": display_name,
            "description": definitions.get(method),
            "available": available,
            "status": method_status,
            "metrics": source_metrics,
            "metrics_by_objective": metrics_by_objective or None,
            "details_by_objective": details_by_objective or None,
        }

    return {
        "available": any(method["available"] for method in methods.values()),
        "status": status,
        "source": source,
        "generated_at": data.get("generated_at"),
        "dataset": {
            "dataset_id": data.get("dataset_id"),
            "version": data.get("dataset_version"),
            "schema_version": data.get("dataset_schema_version"),
            "labeling_guide_version": data.get("labeling_guide_version"),
            "path": _display_path(data.get("dataset_path")),
            "sha256": data.get("dataset_sha256"),
            "expected_path": data.get("expected_dataset_path"),
            "num_claims_evaluated": data.get("num_claims_evaluated"),
            "summary": data.get("dataset_summary"),
        },
        "configuration": {
            "k_values": data.get("k_values"),
            "rrf_k": data.get("rrf_k"),
            "relevant_min_grade": data.get("relevant_min_grade"),
            "ecs_only_missing_policy": data.get("ecs_only_missing_policy"),
            "baseline_rankings_available": data.get(
                "baseline_rankings_available"
            ),
            "annotation_required": data.get("annotation_required"),
        },
        "k_values": data.get("k_values"),
        "rrf_k": data.get("rrf_k"),
        "num_claims_evaluated": data.get("num_claims_evaluated"),
        "methods": methods,
        "exclusions": data.get("exclusions") or [],
        "missing_data": data.get("missing_data"),
        "reason": data.get("reason"),
        "notes": data.get("notes") if isinstance(data.get("notes"), list) else [],
        "metric_definitions": definitions or None,
    }


@router.get("/summary")
def get_evaluation_summary() -> dict[str, Any]:
    """Return available saved metrics and experiment metadata."""
    experiments: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    artifact_errors: list[dict[str, str]] = []

    factllama = _latest_factllama_artifact()
    if factllama is None:
        experiment = _unavailable_experiment(
            name="factllama_fact_verification",
            task="fact_verification",
            note="No saved FactLLaMA evaluation artifact exists.",
        )
    else:
        _, source = factllama
        try:
            data = _read_json(source)
        except EvaluationArtifactError:
            experiment = _unavailable_experiment(
                name="factllama_fact_verification",
                task="fact_verification",
                status="error",
                source=source,
                note="The saved FactLLaMA evaluation artifact could not be read.",
            )
            artifact_errors.append(
                {"source": source, "message": "Artifact could not be read."}
            )
        else:
            experiment = _factllama_experiment(data, source)
    experiments.append(experiment)
    if not experiment["available"]:
        unavailable.append(
            {
                "name": experiment["name"],
                "status": experiment["status"],
                "reason": experiment.get("note") or "Metrics are unavailable.",
            }
        )

    nli_path = _artifact_path(NLI_BENCHMARK_PATH)
    if not nli_path.is_file():
        experiment = _unavailable_experiment(
            name="nli_stance_classification",
            task="nli_stance_classification",
            note="No saved NLI stance benchmark artifact exists.",
        )
    else:
        try:
            data = _read_json(NLI_BENCHMARK_PATH)
        except EvaluationArtifactError:
            experiment = _unavailable_experiment(
                name="nli_stance_classification",
                task="nli_stance_classification",
                status="error",
                source=NLI_BENCHMARK_PATH,
                note="The saved NLI stance artifact could not be read.",
            )
            artifact_errors.append(
                {
                    "source": NLI_BENCHMARK_PATH,
                    "message": "Artifact could not be read.",
                }
            )
        else:
            experiment = _nli_experiment(data, NLI_BENCHMARK_PATH)
    experiments.append(experiment)
    if not experiment["available"]:
        unavailable.append(
            {
                "name": experiment["name"],
                "status": experiment["status"],
                "reason": experiment.get("note") or "Metrics are unavailable.",
            }
        )

    claim_ranking = _claim_ranking_artifact()
    if claim_ranking is None:
        experiment = _unavailable_experiment(
            name="claim_ranking_prioritization",
            task="claim_ranking",
            note="No saved claim-ranking comparison artifact exists.",
        )
    else:
        _, source = claim_ranking
        try:
            data = _read_json(source)
        except EvaluationArtifactError:
            experiment = _unavailable_experiment(
                name="claim_ranking_prioritization",
                task="claim_ranking",
                status="error",
                source=source,
                note="The saved claim-ranking artifact could not be read.",
            )
            artifact_errors.append(
                {"source": source, "message": "Artifact could not be read."}
            )
        else:
            experiment = _claim_ranking_experiment(data, source)
    experiments.append(experiment)
    if not experiment["available"]:
        unavailable.append(
            {
                "name": experiment["name"],
                "status": experiment["status"],
                "reason": (
                    experiment.get("reason")
                    or experiment.get("note")
                    or "Metrics are unavailable."
                ),
            }
        )

    retrieval_path = _artifact_path(RETRIEVAL_METRICS_PATH)
    retrieval_data: dict[str, Any] | None = None
    retrieval_source: str | None = None
    if retrieval_path.is_file():
        retrieval_source = RETRIEVAL_METRICS_PATH
        try:
            retrieval_data = _read_json(RETRIEVAL_METRICS_PATH)
        except EvaluationArtifactError:
            artifact_errors.append(
                {
                    "source": RETRIEVAL_METRICS_PATH,
                    "message": "Artifact could not be read.",
                }
            )
    experiment = _retrieval_experiment(retrieval_data, retrieval_source)
    if retrieval_data is None and retrieval_source is not None:
        experiment["status"] = "error"
        experiment["note"] = "The saved retrieval artifact could not be read."
    experiments.append(experiment)
    if not experiment["available"]:
        unavailable.append(
            {
                "name": experiment["name"],
                "status": experiment["status"],
                "reason": experiment.get("note") or "Metrics are unavailable.",
            }
        )

    available_count = sum(
        1 for experiment in experiments if experiment["available"]
    )
    if available_count == 0:
        status = "unavailable"
    elif available_count == len(experiments):
        status = "available"
    else:
        status = "partial"

    generated_at = max(
        (
            experiment["generated_at"]
            for experiment in experiments
            if experiment.get("generated_at")
        ),
        default=None,
    )
    return {
        "status": status,
        "available": available_count > 0,
        "generated_at": generated_at,
        "experiment_count": len(experiments),
        "available_experiment_count": available_count,
        "metrics": {
            experiment["name"]: experiment["metrics"]
            for experiment in experiments
        },
        "experiments": experiments,
        "unavailable": unavailable,
        "artifact_errors": artifact_errors,
        "notes": [
            "Only saved evaluation artifacts are reported; no training, "
            "inference, retrieval, or metric recomputation is performed.",
            "Metrics from different tasks and datasets are reported "
            "separately; no composite score is computed.",
            "A null metric means unavailable, not zero.",
        ],
    }


@router.get("/prioritization-comparison")
def get_prioritization_comparison() -> dict[str, Any]:
    """Compare baseline, ECS-only, and combined ranking results if saved."""
    artifact = _claim_ranking_artifact()
    if artifact is None:
        return _unavailable_comparison()
    _, source = artifact
    try:
        data = _read_json(source)
    except EvaluationArtifactError as exc:
        raise _artifact_http_error(exc) from None
    return _comparison_payload(data, source)


@router.get("/retrieval-metrics")
def get_retrieval_metrics() -> dict[str, Any]:
    """Return saved retrieval metrics or an explicit unavailable result."""
    path = _artifact_path(RETRIEVAL_METRICS_PATH)
    if not path.is_file():
        return _retrieval_experiment(None, None)
    try:
        data = _read_json(RETRIEVAL_METRICS_PATH)
    except EvaluationArtifactError as exc:
        raise _artifact_http_error(exc) from None
    return _retrieval_experiment(data, RETRIEVAL_METRICS_PATH)
