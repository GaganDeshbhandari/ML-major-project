"""Orchestration for claim prioritization followed by evidence retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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


@dataclass(slots=True)
class PrioritizationRetrievalResult:
    """A prioritized claim and the evidence retrieved for that claim."""

    claim_id: str | int
    claim: str
    objective: Objective
    initial_rank: int
    evidence: list[EvidenceRecord]
    predicted_truth: str | None = None
    predicted_tpb: str | None = None
    disputability: float | None = None
    retrieval_error: RetrievalErrorInfo | None = None


def _to_result(
    prioritized: PrioritizedClaim,
    evidence: list[EvidenceRecord],
    retrieval_error: RetrievalErrorInfo | None,
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
    )


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
) -> list[PrioritizationRetrievalResult]:
    """Prioritize every claim and retrieve evidence in the resulting order.

    Expected retrieval failures are recorded on the affected result while
    processing continues for the remaining claims. The prioritizer performs
    all input validation, and unexpected retrieval exceptions are allowed to
    propagate.
    """

    prioritized_claims = prioritize_claims(
        claims,
        ratings,
        objective,
        artifact_store=artifact_store,
    )
    retrieve_fn = retrieve or retrieve_evidence
    results: list[PrioritizationRetrievalResult] = []

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
        else:
            results.append(_to_result(prioritized, evidence, None))

    return results
