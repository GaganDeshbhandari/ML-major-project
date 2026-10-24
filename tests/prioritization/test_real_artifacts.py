"""Local real-artifact smoke tests.

These tests use the actual trusted model files when present. They are not an
accuracy evaluation and do not validate real-world generalization.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.prioritization import (
    ArtifactPaths,
    ArtifactStore,
    Claim,
    UserRating,
    compute_claim_features,
    prioritize_claims,
)

ARTIFACT_PATHS = ArtifactPaths.default()
ARTIFACTS_PRESENT = all(
    path.is_file()
    for path in (
        ARTIFACT_PATHS.o1_model,
        ARTIFACT_PATHS.o2_model,
        ARTIFACT_PATHS.scaler,
    )
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="Local prioritization model artifacts are not present.",
)


@pytest.fixture(scope="module")
def store() -> ArtifactStore:
    return ArtifactStore(ARTIFACT_PATHS)


def _transform(store: ArtifactStore, features: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        # The production path validates feature order before calling transform.
        # This smoke test uses the same ordered NumPy matrix.
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )
        return np.asarray(store.scaler.transform(features), dtype=np.float64)


def test_all_artifacts_load_and_scaler_accepts_features(store: ArtifactStore) -> None:
    features = np.vstack(
        [
            compute_claim_features([1.0, 0.5]).to_vector(),
            compute_claim_features([-1.0, -0.5]).to_vector(),
            compute_claim_features([0.0, 0.0, 0.0]).to_vector(),
        ]
    )
    scaled = _transform(store, features)

    assert store.o1_model is not None
    assert store.o2_model is not None
    assert store.scaler.n_features_in_ == 4
    assert tuple(str(name) for name in store.scaler.feature_names_in_) == (
        "mean",
        "median",
        "variance",
        "skewness",
    )
    assert scaled.shape == features.shape
    assert np.all(np.isfinite(scaled))


def test_real_o1_and_o2_predictions_use_expected_labels(store: ArtifactStore) -> None:
    features = np.vstack(
        [
            compute_claim_features([1.0, 1.0, 0.5]).to_vector(),
            compute_claim_features([-1.0, -1.0, -0.5]).to_vector(),
            compute_claim_features([0.0, 1.0, -1.0]).to_vector(),
        ]
    )
    scaled = _transform(store, features)

    o1_predictions = {str(label) for label in store.o1_model.predict(scaled)}
    o2_predictions = {str(label) for label in store.o2_model.predict(scaled)}

    assert o1_predictions <= {"True", "False"}
    assert o2_predictions <= {"High", "Low"}


def test_public_api_runs_against_real_artifacts(store: ArtifactStore) -> None:
    claims = [
        Claim("a", "Claim A"),
        Claim("b", "Claim B"),
        Claim("c", "Claim C"),
    ]
    ratings = [
        UserRating("a", "True"),
        UserRating("a", "Mostly True"),
        UserRating("b", "False"),
        UserRating("b", "Mostly False"),
        UserRating("c", "Mixture"),
        UserRating("c", "True"),
        UserRating("c", "False"),
    ]

    o1_results = prioritize_claims(claims, ratings, "O1", artifact_store=store)
    o2_results = prioritize_claims(claims, ratings, "O2", artifact_store=store)

    assert all(result.predicted_truth == "False" for result in o1_results)
    assert all(result.predicted_tpb == "High" for result in o2_results)
    assert [result.rank for result in o1_results] == list(range(1, len(o1_results) + 1))
    assert [result.rank for result in o2_results] == list(range(1, len(o2_results) + 1))


def test_real_o3_orders_known_variances(store: ArtifactStore) -> None:
    claims = [
        Claim("high", "High variance"),
        Claim("mid", "Middle variance"),
        Claim("low", "Zero variance"),
        Claim("single", "Single rating"),
    ]
    ratings = [
        UserRating("high", "True"),
        UserRating("high", "False"),
        UserRating("mid", "Mostly True"),
        UserRating("mid", "Mixture"),
        UserRating("low", "True"),
        UserRating("low", "True"),
        UserRating("single", "True"),
    ]

    results = prioritize_claims(claims, ratings, "O3", artifact_store=store)

    assert [result.claim_id for result in results] == ["high", "mid", "low", "single"]
    assert results[0].disputability == pytest.approx(2.0)
    assert results[1].disputability == pytest.approx(0.125)
    assert results[2].disputability == pytest.approx(0.0)
    assert results[3].disputability is None
