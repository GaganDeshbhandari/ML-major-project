"""Orchestration for claim prioritization, evidence retrieval, and optional ECS.

The pipeline preserves the existing prioritization and retrieval contracts.
Evidence Conflict Score (ECS) processing is opt-in: it runs only when an
:class:`ECSConfig` or a preconfigured stance classifier is supplied. Importing
this module does not import PyTorch, Transformers, or load an NLI model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from src.evidence_conflict import (
    ECSError,
    ECSConfig,
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceInferenceError,
    StanceInputError,
    StanceModelError,
    StancePrediction,
)
from src.prioritization import (
    ArtifactStore,
    Claim,
    Objective,
    PrioritizedClaim,
    UserRating,
    prioritize_claims,
)
from src.retrieval import EvidenceRetrievalError, retrieve_evidence

EvidenceRecord = dict[str, Any]
RetrieveEvidence = Callable[..., list[EvidenceRecord]]


@dataclass(frozen=True, slots=True)
class RetrievalErrorInfo:
    """Structured information about an expected per-claim retrieval failure."""

    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ECSErrorInfo:
    """Structured information about an expected per-claim ECS failure."""

    error_type: str
    message: str


@dataclass(slots=True)
class PrioritizationRetrievalResult:
    """A prioritized claim, retrieved evidence, and optional ECS result."""

    claim_id: str | int
    claim: str
    objective: Objective
    initial_rank: int
    evidence: list[EvidenceRecord]
    predicted_truth: str | None = None
    predicted_tpb: str | None = None
    disputability: float | None = None
    retrieval_error: RetrievalErrorInfo | None = None
    ecs_result: EvidenceConflictResult | None = None
    ecs_error: ECSErrorInfo | None = None


def _to_result(
    prioritized: PrioritizedClaim,
    evidence: list[EvidenceRecord],
    retrieval_error: RetrievalErrorInfo | None,
    *,
    ecs_result: EvidenceConflictResult | None = None,
    ecs_error: ECSErrorInfo | None = None,
) -> PrioritizationRetrievalResult:
    return PrioritizationRetrievalResult(
        claim_id=prioritized.claim_id,
        claim=prioritized.claim,
        objective=prioritized.objective,
        initial_rank=prioritized.rank,
        predicted_truth=prioritized.predicted_truth,
        predicted_tpb=prioritized.predicted_tpb,
        disputability=prioritized.disputability,
        evidence=evidence,
        retrieval_error=retrieval_error,
        ecs_result=ecs_result,
        ecs_error=ecs_error,
    )


def _empty_ecs_result(claim: str) -> EvidenceConflictResult:
    """Score a claim with no retrieved evidence."""

    from src.evidence_conflict.scoring import compute_evidence_conflict

    return compute_evidence_conflict([], claim_text=claim)


def _nli_error_result(claim: str) -> EvidenceConflictResult:
    """Build the schema-valid result used when NLI inference fails."""

    return EvidenceConflictResult(
        claim=claim,
        predictions=(),
        counts=StanceCounts(),
        status=ECSStatus.NLI_ERROR,
        ecs=None,
    )


def _validate_stance_predictions(
    predictions: Any,
    evidence: list[EvidenceRecord],
) -> list[StancePrediction]:
    """Validate one-to-one prediction/evidence association and order."""

    if isinstance(predictions, (str, bytes, bytearray)):
        raise StanceInferenceError(
            "NLI classifier must return an iterable of StancePrediction "
            "records, not a string or byte sequence."
        )
    try:
        prediction_list = list(predictions)
    except TypeError as exc:
        raise StanceInferenceError(
            "NLI classifier must return an iterable of StancePrediction "
            "records."
        ) from exc

    if len(prediction_list) != len(evidence):
        raise StanceInferenceError(
            "NLI classifier returned "
            f"{len(prediction_list)} predictions for {len(evidence)} "
            "evidence passages."
        )
    for index, prediction in enumerate(prediction_list):
        if not isinstance(prediction, StancePrediction):
            raise StanceInferenceError(
                f"NLI classifier returned {type(prediction).__name__} at "
                f"index {index}; expected StancePrediction."
            )
        if prediction.evidence_index != index:
            raise StanceInferenceError(
                "NLI classifier returned predictions out of evidence order: "
                f"expected evidence_index={index}, got "
                f"{prediction.evidence_index}."
            )
    return prediction_list


def _score_claim(
    claim: str,
    evidence: list[EvidenceRecord],
    classifier: Any,
) -> tuple[EvidenceConflictResult | None, ECSErrorInfo | None]:
    """Run one claim's NLI and ECS steps, keeping failures observable."""

    try:
        raw_predictions = classifier.predict(claim, evidence)
        predictions = _validate_stance_predictions(raw_predictions, evidence)
    except (StanceModelError, StanceInferenceError, StanceInputError) as exc:
        return _nli_error_result(claim), ECSErrorInfo(
            error_type=type(exc).__name__,
            message=str(exc),
        )

    try:
        from src.evidence_conflict.scoring import compute_evidence_conflict

        result = compute_evidence_conflict(predictions, claim_text=claim)
    except (StanceInputError, ECSError) as exc:
        return None, ECSErrorInfo(
            error_type=type(exc).__name__,
            message=str(exc),
        )
    return result, None


def _make_classifier(config: ECSConfig | None) -> Any:
    """Construct the default classifier lazily on first use."""

    if config is None:
        raise TypeError(
            "ecs_config is required when constructing the default "
            "stance classifier."
        )
    from src.evidence_conflict.nli_classifier import NLIStanceClassifier

    return NLIStanceClassifier(config)


def prioritize_and_retrieve(
    claims: Iterable[Claim],
    ratings: Iterable[UserRating],
    objective: Objective | str,
    *,
    artifact_store: ArtifactStore | None = None,
    retrieve: RetrieveEvidence | None = None,
    top_k: int = 5,
    reranker: Any | None = None,
    use_reranker: bool = True,
    ecs_config: ECSConfig | None = None,
    stance_classifier: Any | None = None,
) -> list[PrioritizationRetrievalResult]:
    """Prioritize claims, retrieve evidence, and optionally compute ECS.

    Existing prioritization and retrieval behavior is preserved. ECS is
    disabled by default. It is enabled when ``ecs_config`` is supplied or when
    a preconfigured ``stance_classifier`` is injected. The classifier is
    created once, lazily, on the first non-empty evidence list and reused for
    the remaining claims. Its model loads lazily on the first inference call.

    Expected retrieval failures are recorded in ``retrieval_error`` while
    processing continues. When ECS is enabled, successful retrieval with no
    evidence yields a ``NO_EVIDENCE`` result. Expected NLI failures yield an
    ``NLI_ERROR`` result plus ``ecs_error``. Expected scoring failures leave
    ``ecs_result`` as ``None`` and populate ``ecs_error``. Unexpected
    exceptions propagate. Evidence dictionaries are passed to the classifier
    unchanged and are never augmented with stance metadata.

    Args:
        claims: Claims with stable IDs and non-empty text.
        ratings: User perception ratings associated with those claim IDs.
        objective: ``"O1"``, ``"O2"``, or ``"O3"`` (or an ``Objective``).
        artifact_store: Optional preconfigured local artifact store.
        retrieve: Optional retrieval callable for application wiring and tests.
        top_k: Maximum evidence records requested per claim.
        reranker: Optional reranker passed through to retrieval.
        use_reranker: Whether retrieval should apply its reranker.
        ecs_config: Optional ECS configuration. Supplying it enables ECS and
            constructs the default NLI classifier with these settings. Use
            :meth:`ECSConfig.from_env` to read the existing ``ECS_*``
            environment values. The default is disabled.
        stance_classifier: Optional preconfigured classifier exposing
            ``predict(claim, evidence) -> list[StancePrediction]``. Supplying
            it enables ECS and takes precedence over ``ecs_config`` for
            inference. It must return predictions in evidence order.

    Returns:
        Results in prioritized order. Legacy fields are unchanged. When ECS is
        enabled, ``ecs_result`` contains the claim-level
        :class:`EvidenceConflictResult`; ``ecs_error`` records an expected NLI
        or scoring failure without discarding prioritization or retrieval
        output.

    Raises:
        TypeError: If ``ecs_config`` or ``stance_classifier`` has an invalid
            type or interface.
        Exception: Unexpected retrieval, NLI, or scoring exceptions propagate.
            Expected retrieval failures are recorded in ``retrieval_error``.
    """

    if ecs_config is not None and not isinstance(ecs_config, ECSConfig):
        raise TypeError("ecs_config must be an ECSConfig instance or None.")
    if stance_classifier is not None and not callable(
        getattr(stance_classifier, "predict", None)
    ):
        raise TypeError(
            "stance_classifier must expose a callable predict(claim, evidence) "
            "method."
        )

    prioritized_claims = prioritize_claims(
        claims,
        ratings,
        objective,
        artifact_store=artifact_store,
    )
    retrieve_fn = retrieve or retrieve_evidence
    results: list[PrioritizationRetrievalResult] = []
    ecs_enabled = ecs_config is not None or stance_classifier is not None
    classifier = stance_classifier

    for prioritized in prioritized_claims:
        try:
            evidence = retrieve_fn(
                prioritized.claim,
                top_k=top_k,
                reranker=reranker,
                use_reranker=use_reranker,
            )
        except EvidenceRetrievalError as exc:
            results.append(
                _to_result(
                    prioritized,
                    [],
                    RetrievalErrorInfo(
                        error_type=type(exc).__name__,
                        message=str(exc),
                    ),
                )
            )
            continue

        ecs_result: EvidenceConflictResult | None = None
        ecs_error: ECSErrorInfo | None = None
        if ecs_enabled:
            if not evidence:
                try:
                    ecs_result = _empty_ecs_result(prioritized.claim)
                except (StanceInputError, ECSError) as exc:
                    ecs_error = ECSErrorInfo(
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
            else:
                if classifier is None:
                    classifier = _make_classifier(ecs_config)
                ecs_result, ecs_error = _score_claim(
                    prioritized.claim,
                    evidence,
                    classifier,
                )

        results.append(
            _to_result(
                prioritized,
                evidence,
                None,
                ecs_result=ecs_result,
                ecs_error=ecs_error,
            )
        )

    return results
