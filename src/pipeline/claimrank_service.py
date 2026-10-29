"""Reusable end-to-end service for the existing ClaimRank pipeline.

The service composes, in order:

1. bias-aware claim prioritization (O1/O2/O3),
2. web evidence retrieval,
3. NLI stance classification and the Evidence Conflict Score (ECS),
4. the existing Reciprocal Rank Fusion (RRF) final ranking.

The existing ``PrioritizedClaim`` contract exposes predicted labels for O1/O2
and sample variance for O3, not classifier probabilities. This service
preserves exactly those available values and never synthesizes missing scores.
The final score is the existing RRF score from
:func:`src.pipeline.ranking_fusion.fuse_prioritization_results`; it is a
relative ranking score, not a probability or a verified-fact score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.evidence_conflict import (
    ECSConfig,
    ECSConfigurationError,
    EvidenceConflictResult,
)
from src.evidence_conflict.errors import EvidenceConflictError
from src.prioritization import (
    ArtifactError,
    ArtifactStore,
    Claim,
    ClaimId,
    InputValidationError,
    Objective,
    PrioritizationError,
    UserRating,
)
from src.ranking_fusion import DEFAULT_RRF_K, RankingFusionError, validate_rrf_k
from src.retrieval import EvidenceRetrievalError

from .prioritization_retrieval import (
    ECSErrorInfo,
    PrioritizationRetrievalResult,
    RetrievalErrorInfo,
    RetrieveEvidence,
    prioritize_and_retrieve,
)
from .ranking_fusion import FusedPrioritizationResult, fuse_prioritization_results


class ClaimRankObjective(str, Enum):
    """Public objective names mapped to the existing O1/O2/O3 objectives."""

    REMOVE_FALSE_CLAIMS = "remove_false_claims"
    CORRECT_USER_MISCONCEPTIONS = "correct_user_misconceptions"
    REDUCE_USER_DISAGREEMENT = "reduce_user_disagreement"

    @property
    def pipeline_objective(self) -> Objective:
        """Return the corresponding existing prioritization objective."""

        return _PIPELINE_OBJECTIVES[self]


_PIPELINE_OBJECTIVES = {
    ClaimRankObjective.REMOVE_FALSE_CLAIMS: Objective.O1,
    ClaimRankObjective.CORRECT_USER_MISCONCEPTIONS: Objective.O2,
    ClaimRankObjective.REDUCE_USER_DISAGREEMENT: Objective.O3,
}
_FRIENDLY_BY_PIPELINE = {
    objective: friendly for friendly, objective in _PIPELINE_OBJECTIVES.items()
}

FUSION_METHOD = "reciprocal_rank_fusion"


class ClaimRankServiceError(RuntimeError):
    """A component failure with the pipeline stage that produced it."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True, slots=True)
class OriginalPriority:
    """The original objective-specific rank and available scores."""

    rank: int
    predicted_truth: str | None = None
    predicted_tpb: str | None = None
    disputability: float | None = None


@dataclass(frozen=True, slots=True)
class FinalPriority:
    """The final RRF rank and relative score."""

    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class ClaimRankEntry:
    """One claim's original priority, evidence, conflict, and final priority."""

    claim_id: ClaimId
    claim: str
    objective: ClaimRankObjective
    original: OriginalPriority
    final: FinalPriority
    ecs_score: float | None
    ecs_rank: int | None
    evidence: tuple[dict[str, Any], ...]
    conflict: EvidenceConflictResult | None
    retrieval_error: RetrievalErrorInfo | None
    ecs_error: ECSErrorInfo | None


@dataclass(frozen=True, slots=True)
class ClaimRankRun:
    """The final ranked results for one selected objective."""

    objective: ClaimRankObjective
    results: tuple[ClaimRankEntry, ...]
    fusion_method: str
    rrf_k: float


def parse_objective(
    objective: ClaimRankObjective | Objective | str,
) -> ClaimRankObjective:
    if isinstance(objective, ClaimRankObjective):
        return objective
    if isinstance(objective, Objective):
        return _FRIENDLY_BY_PIPELINE[objective]
    if isinstance(objective, str):
        try:
            return ClaimRankObjective(objective)
        except ValueError:
            for friendly, pipeline_objective in _PIPELINE_OBJECTIVES.items():
                if objective == pipeline_objective.value:
                    return friendly

    supported = ", ".join(item.value for item in ClaimRankObjective)
    raise ClaimRankServiceError(
        "objective",
        f"Unsupported objective {objective!r}; expected one of {supported}, "
        "or the existing codes O1, O2, O3.",
    )


def _wrap_pipeline_error(exc: Exception) -> ClaimRankServiceError:
    if isinstance(exc, ClaimRankServiceError):
        return exc
    if isinstance(exc, ECSConfigurationError):
        stage = "ecs_config"
    elif isinstance(exc, ArtifactError):
        stage = "prioritization_models"
    elif isinstance(exc, InputValidationError):
        stage = "input"
    elif isinstance(exc, EvidenceRetrievalError):
        stage = "evidence_retrieval"
    elif isinstance(exc, EvidenceConflictError):
        stage = "ecs"
    elif isinstance(exc, RankingFusionError):
        stage = "ranking_fusion"
    elif isinstance(exc, PrioritizationError):
        stage = "prioritization"
    else:
        stage = "pipeline"
    return ClaimRankServiceError(stage, str(exc))


def _make_default_classifier(config: ECSConfig) -> Any:
    from src.evidence_conflict.nli_classifier import NLIStanceClassifier

    return NLIStanceClassifier(config)


class ClaimRankService:
    """Reusable orchestration for one ClaimRank run.

    The default NLI classifier is created once and reused across calls. Its
    model weights are loaded lazily only when a claim has evidence, so
    constructing this service does not download or load a model.

    Args:
        artifact_store: Optional local prioritization artifact store.
        retrieve: Optional evidence retrieval callable for tests/wiring.
        stance_classifier: Optional classifier exposing
            ``predict(claim, evidence) -> list[StancePrediction]``.
        ecs_config: Optional ECS configuration. When omitted, the existing
            ``ECS_*`` environment configuration is used.
        top_k: Maximum evidence records per claim.
        reranker: Optional reranker passed to evidence retrieval.
        use_reranker: Whether retrieval applies its reranker.
        rrf_k: Existing RRF smoothing constant; defaults to 60.
    """

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore | None = None,
        retrieve: RetrieveEvidence | None = None,
        stance_classifier: Any | None = None,
        ecs_config: ECSConfig | None = None,
        top_k: int = 5,
        reranker: Any | None = None,
        use_reranker: bool = True,
        rrf_k: float = DEFAULT_RRF_K,
    ) -> None:
        if ecs_config is not None and not isinstance(ecs_config, ECSConfig):
            raise ClaimRankServiceError(
                "ecs_config",
                "ecs_config must be an ECSConfig instance or None.",
            )
        if retrieve is not None and not callable(retrieve):
            raise ClaimRankServiceError(
                "evidence_retrieval",
                "retrieve must be a callable matching retrieve_evidence().",
            )
        if stance_classifier is not None and not callable(
            getattr(stance_classifier, "predict", None)
        ):
            raise ClaimRankServiceError(
                "ecs",
                "stance_classifier must expose a callable predict(claim, evidence) method.",
            )

        try:
            self._rrf_k = validate_rrf_k(rrf_k)
        except RankingFusionError as exc:
            raise ClaimRankServiceError("ranking_fusion", str(exc)) from exc

        self._artifact_store = artifact_store
        self._retrieve = retrieve
        self._top_k = top_k
        self._reranker = reranker
        self._use_reranker = use_reranker

        if stance_classifier is not None:
            self._classifier = stance_classifier
            self._ecs_config = ecs_config
        else:
            try:
                config = ecs_config if ecs_config is not None else ECSConfig.from_env()
            except ECSConfigurationError as exc:
                raise ClaimRankServiceError("ecs_config", str(exc)) from exc
            self._classifier = _make_default_classifier(config)
            self._ecs_config = config

    def rank_claims(
        self,
        claims: Iterable[Claim],
        ratings: Iterable[UserRating],
        objective: ClaimRankObjective | Objective | str,
    ) -> ClaimRankRun:
        """Run all four stages and return the final ranked claim entries.

        Retrieval and NLI failures that the existing pipeline already models
        are returned per claim in ``retrieval_error``/``ecs_error`` without
        fabricated evidence or ECS values. Fatal prioritization, configuration,
        or fusion failures raise :class:`ClaimRankServiceError` with a stage.
        """

        selected = parse_objective(objective)
        try:
            pipeline_results = prioritize_and_retrieve(
                claims,
                ratings,
                selected.pipeline_objective,
                artifact_store=self._artifact_store,
                retrieve=self._retrieve,
                top_k=self._top_k,
                reranker=self._reranker,
                use_reranker=self._use_reranker,
                ecs_config=self._ecs_config,
                stance_classifier=self._classifier,
            )
        except Exception as exc:
            raise _wrap_pipeline_error(exc) from exc

        try:
            fused_results = fuse_prioritization_results(
                pipeline_results,
                k=self._rrf_k,
            )
        except RankingFusionError as exc:
            raise ClaimRankServiceError("ranking_fusion", str(exc)) from exc

        return ClaimRankRun(
            objective=selected,
            results=tuple(
                self._to_entry(selected, item)
                for item in fused_results
            ),
            fusion_method=FUSION_METHOD,
            rrf_k=self._rrf_k,
        )

    @staticmethod
    def _to_entry(
        objective: ClaimRankObjective,
        fused: FusedPrioritizationResult,
    ) -> ClaimRankEntry:
        result: PrioritizationRetrievalResult = fused.result
        return ClaimRankEntry(
            claim_id=result.claim_id,
            claim=result.claim,
            objective=objective,
            original=OriginalPriority(
                rank=result.initial_rank,
                predicted_truth=result.predicted_truth,
                predicted_tpb=result.predicted_tpb,
                disputability=result.disputability,
            ),
            final=FinalPriority(
                rank=fused.fused_rank,
                score=fused.rrf_score,
            ),
            ecs_score=fused.ecs_score,
            ecs_rank=fused.ecs_rank,
            evidence=tuple(result.evidence),
            conflict=result.ecs_result,
            retrieval_error=result.retrieval_error,
            ecs_error=result.ecs_error,
        )


__all__ = [
    "ClaimRankEntry",
    "ClaimRankObjective",
    "ClaimRankRun",
    "ClaimRankService",
    "ClaimRankServiceError",
    "FinalPriority",
    "FUSION_METHOD",
    "OriginalPriority",
    "parse_objective",
]
