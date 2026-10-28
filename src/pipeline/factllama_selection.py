"""User-selected FactLLaMA verification at the pipeline boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.factllama import FactLLaMAConfig, FactLLaMAError, FactLLaMAPredictor
from src.factllama.predictor import Prediction
from src.prioritization import ClaimId

from .prioritization_retrieval import PrioritizationRetrievalResult


class FactLLaMASelectionError(ValueError):
    """Raised when a user selection cannot be resolved or verified."""


@dataclass(frozen=True, slots=True)
class FactLLaMAErrorInfo:
    """An explicit verification failure that did not produce a verdict."""

    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class FactLLaMAVerificationResult:
    """The selected claim, its existing evidence, and FactLLaMA output."""

    claim_id: ClaimId
    claim: str
    evidence: tuple[dict[str, Any], ...]
    verdict: str | None
    raw_output: str | None
    prediction: Prediction | None = None
    error: FactLLaMAErrorInfo | None = None


class FactLLaMASelectionService:
    """Verify only an explicitly selected ranked claim.

    The predictor is created once and owns the lazy model loader. Reusing this
    service for later selections therefore reuses the loaded model rather than
    loading one model per request.
    """

    def __init__(
        self,
        *,
        predictor: FactLLaMAPredictor | None = None,
        config: FactLLaMAConfig | None = None,
    ) -> None:
        if predictor is not None and config is not None:
            raise ValueError("Pass either predictor or config, not both")
        self._predictor = predictor or FactLLaMAPredictor(config=config)

    @property
    def predictor(self) -> FactLLaMAPredictor:
        """Return the reusable predictor owned by this service."""

        return self._predictor

    def verify_selected_claim(
        self,
        results: Iterable[PrioritizationRetrievalResult],
        claim_id: ClaimId,
    ) -> FactLLaMAVerificationResult:
        """Verify the one result whose stable ID matches ``claim_id``."""

        selected = self._find_selected(results, claim_id)
        if selected.retrieval_error is not None:
            raise FactLLaMASelectionError(
                f"Evidence retrieval failed for claim {claim_id!r}: "
                f"{selected.retrieval_error.message}"
            )
        if not selected.evidence:
            raise FactLLaMASelectionError(
                f"No evidence is available for claim {claim_id!r}; "
                "FactLLaMA was not invoked."
            )

        evidence = tuple(selected.evidence)
        try:
            prediction = self._predictor.predict(selected.claim, evidence)
        except (FactLLaMAError, RuntimeError) as exc:
            return FactLLaMAVerificationResult(
                claim_id=selected.claim_id,
                claim=selected.claim,
                evidence=evidence,
                verdict=None,
                raw_output=None,
                error=FactLLaMAErrorInfo(type(exc).__name__, str(exc)),
            )

        if prediction.verdict is None:
            return FactLLaMAVerificationResult(
                claim_id=selected.claim_id,
                claim=selected.claim,
                evidence=evidence,
                verdict=None,
                raw_output=prediction.raw_output,
                prediction=prediction,
                error=FactLLaMAErrorInfo(
                    "VerdictParseError",
                    "FactLLaMA output did not contain one unambiguous "
                    "supported verdict.",
                ),
            )
        return FactLLaMAVerificationResult(
            claim_id=selected.claim_id,
            claim=selected.claim,
            evidence=evidence,
            verdict=prediction.verdict,
            raw_output=prediction.raw_output,
            prediction=prediction,
        )

    @staticmethod
    def _find_selected(
        results: Iterable[PrioritizationRetrievalResult],
        claim_id: ClaimId,
    ) -> PrioritizationRetrievalResult:
        if isinstance(claim_id, bool) or not isinstance(claim_id, (str, int)):
            raise FactLLaMASelectionError(
                "claim_id must be a non-boolean string or integer"
            )
        if isinstance(claim_id, str) and not claim_id.strip():
            raise FactLLaMASelectionError("claim_id must not be empty")
        try:
            values = list(results)
        except TypeError as exc:
            raise FactLLaMASelectionError(
                "results must be an iterable of ranked pipeline results"
            ) from exc
        matches = [result for result in values if result.claim_id == claim_id]
        if not matches:
            raise FactLLaMASelectionError(
                f"Claim ID {claim_id!r} was not found in ranked results"
            )
        if len(matches) > 1:
            raise FactLLaMASelectionError(
                f"Claim ID {claim_id!r} is duplicated in ranked results"
            )
        return matches[0]


__all__ = [
    "FactLLaMAErrorInfo",
    "FactLLaMASelectionError",
    "FactLLaMASelectionService",
    "FactLLaMAVerificationResult",
]
