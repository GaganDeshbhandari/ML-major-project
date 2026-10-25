"""Public bias-aware prioritization function and objective implementations."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import cast

import numpy as np

from ..errors import InferenceError, InvalidObjectiveError
from ..features.extraction import compute_claim_features, compute_disputability
from ..preprocessing.ratings import PreparedRatings, prepare_ratings
from ..schemas import (
    Claim,
    Objective,
    PrioritizedClaim,
    TPBLabel,
    TruthLabel,
    UserRating,
)
from .artifacts import O1_CLASSES, O2_CLASSES, ArtifactStore, get_default_artifact_store

_VALID_OBJECTIVES = ", ".join(objective.value for objective in Objective)


def _parse_objective(objective: Objective | str) -> Objective:
    if isinstance(objective, Objective):
        return objective
    if isinstance(objective, str):
        try:
            return Objective(objective)
        except ValueError as exc:
            raise InvalidObjectiveError(
                f"Unsupported objective {objective!r}: expected one of "
                f"{_VALID_OBJECTIVES}."
            ) from exc
    raise InvalidObjectiveError(
        f"Unsupported objective {objective!r}: expected one of {_VALID_OBJECTIVES}."
    )


def _feature_matrix(prepared: PreparedRatings) -> np.ndarray:
    rows = [
        compute_claim_features(prepared.ptl_by_claim[claim.claim_id]).to_vector()
        for claim in prepared.claims
    ]
    return np.vstack(rows)


def _transform_features(scaler: object, features: np.ndarray) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            # The scaler was fitted with named columns. We pass an ordered
            # NumPy matrix and validate its column order when the artifact is
            # loaded, so the missing-DataFrame-columns warning is redundant.
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
                module="sklearn",
            )
            transformed = scaler.transform(features)  # type: ignore[attr-defined]
        scaled = np.asarray(transformed, dtype=np.float64)
    except Exception as exc:
        raise InferenceError(
            "The saved scaler could not transform the extracted features."
        ) from exc

    if scaled.shape != features.shape:
        raise InferenceError(
            "The saved scaler returned an unexpected shape: "
            f"expected {features.shape}, found {scaled.shape}."
        )
    if not np.all(np.isfinite(scaled)):
        raise InferenceError("The saved scaler returned non-finite features.")
    return scaled


def _predict_labels_and_scores(
    store: ArtifactStore,
    objective: Objective,
    features: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    scaled = _transform_features(store.scaler, features)

    if objective is Objective.O1:
        model = store.o1_model
        allowed = O1_CLASSES
        model_name = "O1"
        prioritized_label = "False"
    else:
        model = store.o2_model
        allowed = O2_CLASSES
        model_name = "O2"
        prioritized_label = "High"

    try:
        raw_predictions = model.predict(scaled)
        labels = [str(label) for label in np.asarray(raw_predictions).reshape(-1)]
    except Exception as exc:
        raise InferenceError(
            f"The saved {model_name} model could not produce predictions."
        ) from exc

    if len(labels) != features.shape[0]:
        raise InferenceError(
            f"The saved {model_name} model returned {len(labels)} predictions "
            f"for {features.shape[0]} claims."
        )

    invalid = sorted(set(labels) - allowed)
    if invalid:
        raise InferenceError(
            f"The saved {model_name} model returned unsupported labels: "
            f"{invalid}. Expected labels from {sorted(allowed)}."
        )

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise InferenceError(
            f"The saved {model_name} model does not support predict_proba(); "
            "confidence-ranked prioritization cannot be performed."
        )

    try:
        raw_probabilities = predict_proba(scaled)
        probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    except Exception as exc:
        raise InferenceError(
            f"The saved {model_name} model could not produce probability "
            "predictions."
        ) from exc

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise InferenceError(
            f"The saved {model_name} model does not expose classes_; "
            "the target probability column cannot be identified."
        )
    class_labels = [str(label) for label in np.asarray(classes).reshape(-1)]
    if prioritized_label not in class_labels:
        raise InferenceError(
            f"The saved {model_name} model does not expose the required "
            f"{prioritized_label!r} probability class."
        )
    target_column = class_labels.index(prioritized_label)

    expected_shape = (features.shape[0], len(class_labels))
    if probabilities.shape != expected_shape:
        raise InferenceError(
            f"The saved {model_name} model returned probabilities with shape "
            f"{probabilities.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise InferenceError(
            f"The saved {model_name} model returned non-finite probabilities."
        )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise InferenceError(
            f"The saved {model_name} model returned values outside the "
            "[0, 1] probability range."
        )

    return labels, probabilities[:, target_column]


def _prioritize_supervised(
    prepared: PreparedRatings,
    objective: Objective,
    store: ArtifactStore,
) -> list[PrioritizedClaim]:
    features = _feature_matrix(prepared)
    labels, scores = _predict_labels_and_scores(store, objective, features)
    ranked = list(zip(prepared.claims, labels, scores, strict=True))
    ranked.sort(key=lambda item: -item[2])

    results: list[PrioritizedClaim] = []

    for rank, (claim, label, _) in enumerate(ranked, start=1):
        results.append(
            PrioritizedClaim(
                claim_id=claim.claim_id,
                claim=claim.claim,
                objective=objective,
                rank=rank,
                predicted_truth=(
                    cast(TruthLabel, label) if objective is Objective.O1 else None
                ),
                predicted_tpb=(
                    cast(TPBLabel, label) if objective is Objective.O2 else None
                ),
            )
        )

    return results


def _disputability_sort_key(item: tuple[Claim, float]) -> tuple[int, float]:
    value = item[1]
    if math.isnan(value):
        return (1, 0.0)
    return (0, -value)


def _prioritize_by_disputability(prepared: PreparedRatings) -> list[PrioritizedClaim]:
    items = [
        (
            claim,
            compute_disputability(prepared.ptl_by_claim[claim.claim_id]),
        )
        for claim in prepared.claims
    ]
    items.sort(key=_disputability_sort_key)

    results: list[PrioritizedClaim] = []
    for claim, disputability in items:
        results.append(
            PrioritizedClaim(
                claim_id=claim.claim_id,
                claim=claim.claim,
                objective=Objective.O3,
                rank=len(results) + 1,
                disputability=(
                    None if math.isnan(disputability) else float(disputability)
                ),
            )
        )
    return results


def prioritize_claims(
    claims: Iterable[Claim],
    ratings: Iterable[UserRating],
    objective: Objective | str,
    *,
    artifact_store: ArtifactStore | None = None,
) -> list[PrioritizedClaim]:
    """Rank claims according to one bias-aware objective.

    ``O1`` returns every claim sorted by descending predicted probability of
    ``False``. ``O2`` returns every claim sorted by descending predicted
    probability of ``High`` TPB. ``O3`` returns every rated claim sorted by
    descending sample variance of PTL, with undefined variance (one rating)
    last.

    Model predictions are classifier outputs, not verified facts. The
    function does not calculate GTL or TPB for incoming claims.

    Args:
        claims: Claims with stable IDs and non-empty text.
        ratings: User perception ratings associated with those claim IDs.
        objective: ``"O1"``, ``"O2"``, or ``"O3"`` (or an ``Objective``).
        artifact_store: Optional preconfigured local artifact store. This is
            for application wiring and tests; the public API never accepts a
            path or URL here.

    Raises:
        InvalidObjectiveError: If the objective is not supported.
        InputValidationError: If claims or ratings are malformed, duplicated,
            unknown, or a claim has no perception records.
        ArtifactError: If required local artifacts are missing or invalid.
        InferenceError: If a model returns unusable predictions.
    """

    selected_objective = _parse_objective(objective)
    prepared = prepare_ratings(claims, ratings)

    if not prepared.claims:
        return []

    if selected_objective is Objective.O3:
        return _prioritize_by_disputability(prepared)

    store = (
        artifact_store if artifact_store is not None else get_default_artifact_store()
    )
    return _prioritize_supervised(prepared, selected_objective, store)
