"""Tests for the public prioritization API and objective behavior."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.prioritization import (
    FEATURE_COLUMNS,
    Claim,
    Objective,
    UserRating,
    compute_claim_features,
    prioritize_claims,
)
from src.prioritization.errors import (
    DuplicateClaimError,
    InferenceError,
    InvalidObjectiveError,
    InvalidRatingError,
    MissingPerceptionError,
    UnknownClaimError,
)


class IdentityScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def __init__(self) -> None:
        self.transform_calls: list[np.ndarray] = []
        self.fit_calls = 0
        self.fit_transform_calls = 0

    def transform(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        self.transform_calls.append(values.copy())
        return values

    def fit(self, X: object, y: object = None) -> None:
        self.fit_calls += 1
        raise AssertionError("fit() must never be called during inference")

    def fit_transform(self, X: object, y: object = None) -> None:
        self.fit_transform_calls += 1
        raise AssertionError("fit_transform() must never be called during inference")


class RuleModel:
    def __init__(
        self,
        rule: Callable[[np.ndarray], str],
        classes: tuple[str, ...],
        probability_rule: Callable[[np.ndarray], tuple[float, ...]] | None = None,
    ) -> None:
        self._rule = rule
        self._probability_rule = probability_rule
        self.classes_ = np.asarray(classes, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.predict_calls: list[np.ndarray] = []
        self.predict_proba_calls: list[np.ndarray] = []

    def predict(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        self.predict_calls.append(values.copy())
        return np.asarray([self._rule(row) for row in values], dtype=object)

    def predict_proba(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        self.predict_proba_calls.append(values.copy())
        if self._probability_rule is not None:
            return np.asarray([self._probability_rule(row) for row in values])

        return np.asarray(
            [
                [
                    0.9 if label == predicted else 0.1
                    for label in self.classes_
                ]
                for predicted in (self._rule(row) for row in values)
            ],
            dtype=np.float64,
        )


class WrongCountModel:
    classes_ = np.asarray(("False", "True"), dtype=object)
    n_features_in_ = len(FEATURE_COLUMNS)

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(["False"], dtype=object)


class FakeStore:
    """Duck-typed stand-in for ArtifactStore, with access tracking."""

    def __init__(
        self,
        scaler: IdentityScaler | None,
        *,
        o1_model: RuleModel | WrongCountModel | None = None,
        o2_model: RuleModel | None = None,
    ) -> None:
        self._scaler = scaler
        self._o1_model = o1_model
        self._o2_model = o2_model
        self.scaler_accesses = 0
        self.o1_accesses = 0
        self.o2_accesses = 0

    @property
    def scaler(self) -> IdentityScaler:
        self.scaler_accesses += 1
        if self._scaler is None:
            raise AssertionError("The scaler must not be loaded for this objective")
        return self._scaler

    @property
    def o1_model(self) -> RuleModel | WrongCountModel:
        self.o1_accesses += 1
        if self._o1_model is None:
            raise AssertionError("The O1 model must not be loaded for this objective")
        return self._o1_model

    @property
    def o2_model(self) -> RuleModel:
        self.o2_accesses += 1
        if self._o2_model is None:
            raise AssertionError("The O2 model must not be loaded for this objective")
        return self._o2_model


def _three_claims() -> list[Claim]:
    return [
        Claim("a", "Claim A"),
        Claim("b", "Claim B"),
        Claim("c", "Claim C"),
    ]


def _three_claim_ratings() -> list[UserRating]:
    return [
        UserRating("a", "True"),
        UserRating("a", "Mostly True"),
        UserRating("b", "False"),
        UserRating("b", "Mostly False"),
        UserRating("c", "Mixture"),
    ]


def test_o1_returns_all_claims_sorted_by_false_probability() -> None:
    scaler = IdentityScaler()
    model = RuleModel(
        lambda row: "False" if row[0] < 0.2 else "True",
        ("True", "False"),
        lambda row: (
            (0.8, 0.2)
            if row[0] > 0.5
            else (0.1, 0.9)
            if row[0] < 0
            else (0.3, 0.7)
        ),
    )
    store = FakeStore(scaler, o1_model=model)

    results = prioritize_claims(
        _three_claims(), _three_claim_ratings(), "O1", artifact_store=store
    )

    assert [result.claim_id for result in results] == ["b", "c", "a"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.objective is Objective.O1 for result in results)
    assert [result.predicted_truth for result in results] == ["False", "False", "True"]
    assert all(result.predicted_tpb is None for result in results)
    assert all(result.disputability is None for result in results)
    assert store.o1_accesses == 1
    assert store.o2_accesses == 0
    assert store.scaler_accesses == 1
    assert scaler.fit_calls == 0
    assert scaler.fit_transform_calls == 0


def test_o1_passes_features_in_expected_order_to_scaler() -> None:
    scaler = IdentityScaler()
    model = RuleModel(lambda row: "True", ("False", "True"))
    store = FakeStore(scaler, o1_model=model)

    prioritize_claims(
        _three_claims(), _three_claim_ratings(), "O1", artifact_store=store
    )

    expected_first_row = compute_claim_features([1.0, 0.5]).to_vector()
    np.testing.assert_allclose(scaler.transform_calls[0][0], expected_first_row)
    assert scaler.transform_calls[0].shape == (3, 4)


def test_o2_returns_all_claims_sorted_by_high_probability() -> None:
    scaler = IdentityScaler()
    model = RuleModel(
        lambda row: "High" if row[2] > 0.2 else "Low",
        ("Low", "High"),
        lambda row: (
            (0.2, 0.8)
            if row[2] > 0.2
            else (0.7, 0.3)
        ),
    )
    store = FakeStore(scaler, o2_model=model)
    ratings = [
        UserRating("a", "True"),
        UserRating("a", "True"),
        UserRating("b", "True"),
        UserRating("b", "False"),
        UserRating("c", "Mostly True"),
        UserRating("c", "Mixture"),
    ]

    results = prioritize_claims(_three_claims(), ratings, "O2", artifact_store=store)

    assert [result.claim_id for result in results] == ["b", "a", "c"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.objective is Objective.O2 for result in results)
    assert [result.predicted_tpb for result in results] == ["High", "Low", "Low"]
    assert all(result.predicted_truth is None for result in results)
    assert all(result.disputability is None for result in results)
    assert store.o2_accesses == 1
    assert store.o1_accesses == 0
    assert store.scaler_accesses == 1
    assert scaler.fit_calls == 0
    assert scaler.fit_transform_calls == 0


def test_o3_sorts_descending_with_single_rating_last() -> None:
    claims = [
        *_three_claims(),
        Claim("d", "Claim D"),
    ]
    ratings = [
        UserRating("a", "True"),
        UserRating("a", "False"),
        UserRating("b", "True"),
        UserRating("b", "True"),
        UserRating("c", "Mostly True"),
        UserRating("c", "Mixture"),
        UserRating("d", "True"),
    ]
    store = FakeStore(scaler=None)

    results = prioritize_claims(claims, ratings, "O3", artifact_store=store)

    assert [result.claim_id for result in results] == ["a", "c", "b", "d"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert results[0].disputability == pytest.approx(2.0)
    assert results[1].disputability == pytest.approx(0.125)
    assert results[2].disputability == pytest.approx(0.0)
    assert results[3].disputability is None
    assert all(result.objective is Objective.O3 for result in results)
    assert all(result.predicted_truth is None for result in results)
    assert all(result.predicted_tpb is None for result in results)
    assert store.scaler_accesses == 0
    assert store.o1_accesses == 0
    assert store.o2_accesses == 0


def test_o3_ties_preserve_input_order() -> None:
    claims = [Claim("b", "B"), Claim("a", "A"), Claim("c", "C")]
    ratings = [
        UserRating("a", "True"),
        UserRating("a", "False"),
        UserRating("b", "False"),
        UserRating("b", "True"),
        UserRating("c", "True"),
        UserRating("c", "True"),
    ]

    results = prioritize_claims(claims, ratings, "O3", artifact_store=FakeStore(None))

    assert [result.claim_id for result in results] == ["b", "a", "c"]
    assert [result.rank for result in results] == [1, 2, 3]


def test_shuffled_ratings_associate_by_claim_id() -> None:
    claims = [Claim("c", "C"), Claim("a", "A"), Claim("b", "B")]
    ratings = [
        UserRating("a", "True"),
        UserRating("b", "False"),
        UserRating("c", "Mixture"),
    ]
    scaler = IdentityScaler()
    model = RuleModel(
        lambda row: "False" if row[0] < 0.5 else "True",
        ("False", "True"),
    )
    store = FakeStore(scaler, o1_model=model)

    results = prioritize_claims(claims, ratings, "O1", artifact_store=store)

    # c and b are predicted False with equal scores, so their input order is
    # preserved; a is predicted True and has the lower False probability.
    assert [result.claim_id for result in results] == ["c", "b", "a"]
    assert [result.claim for result in results] == ["C", "B", "A"]


def test_empty_claims_return_empty_without_loading_artifacts() -> None:
    store = FakeStore(scaler=None)

    assert prioritize_claims([], [], "O1", artifact_store=store) == []
    assert store.scaler_accesses == 0
    assert store.o1_accesses == 0
    assert store.o2_accesses == 0


def test_invalid_objective_is_rejected_before_artifacts() -> None:
    store = FakeStore(scaler=None)

    with pytest.raises(InvalidObjectiveError):
        prioritize_claims(
            [Claim("1", "Claim")],
            [UserRating("1", "True")],
            "O4",
            artifact_store=store,
        )

    assert store.scaler_accesses == 0
    assert store.o1_accesses == 0
    assert store.o2_accesses == 0


def test_duplicate_claim_ids_are_rejected() -> None:
    with pytest.raises(DuplicateClaimError):
        prioritize_claims(
            [Claim("1", "First"), Claim("1", "Second")],
            [UserRating("1", "True")],
            "O3",
            artifact_store=FakeStore(None),
        )


def test_unknown_claim_ids_are_rejected() -> None:
    with pytest.raises(UnknownClaimError):
        prioritize_claims(
            [Claim("1", "First")],
            [UserRating("1", "True"), UserRating("2", "False")],
            "O3",
            artifact_store=FakeStore(None),
        )


def test_missing_perception_records_are_rejected() -> None:
    with pytest.raises(MissingPerceptionError, match="'2'"):
        prioritize_claims(
            [Claim("1", "First"), Claim("2", "Second")],
            [UserRating("1", "True")],
            "O3",
            artifact_store=FakeStore(None),
        )


def test_malformed_rating_labels_are_rejected() -> None:
    with pytest.raises(InvalidRatingError):
        prioritize_claims(
            [Claim("1", "First")],
            [UserRating("1", "TRUE")],
            "O3",
            artifact_store=FakeStore(None),
        )


def test_invalid_model_labels_are_rejected() -> None:
    model = RuleModel(lambda row: "Maybe", ("False", "True"))
    store = FakeStore(IdentityScaler(), o1_model=model)

    with pytest.raises(InferenceError, match="unsupported labels"):
        prioritize_claims(
            [Claim("1", "First")], [UserRating("1", "True")], "O1", artifact_store=store
        )


def test_wrong_prediction_count_is_rejected() -> None:
    store = FakeStore(IdentityScaler(), o1_model=WrongCountModel())

    with pytest.raises(InferenceError, match="returned 1 predictions"):
        prioritize_claims(
            [Claim("1", "First"), Claim("2", "Second")],
            [UserRating("1", "True"), UserRating("2", "False")],
            "O1",
            artifact_store=store,
        )
