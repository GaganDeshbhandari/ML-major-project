"""Tests for PTL mapping, rating validation, and claim association."""

from __future__ import annotations

import pytest

from src.prioritization import Claim, UserRating
from src.prioritization.errors import (
    DuplicateClaimError,
    InvalidClaimError,
    InvalidRatingError,
    MissingPerceptionError,
    UnknownClaimError,
)
from src.prioritization.preprocessing.ratings import (
    PTL_MAPPING,
    map_rating_to_ptl,
    prepare_ratings,
)

PTL_CASES = [
    ("True", 1.0),
    ("Mostly True", 0.5),
    ("Mixture", 0.0),
    ("Mostly False", -0.5),
    ("False", -1.0),
]


@pytest.mark.parametrize(("label", "expected"), PTL_CASES)
def test_ptl_mapping_is_exact(label: str, expected: float) -> None:
    assert map_rating_to_ptl(label) == expected


def test_ptl_mapping_has_exact_labels_and_order() -> None:
    assert tuple(PTL_MAPPING.items()) == (
        ("True", 1.0),
        ("Mostly True", 0.5),
        ("Mixture", 0.0),
        ("Mostly False", -0.5),
        ("False", -1.0),
    )


@pytest.mark.parametrize("label", ["true", "TRUE", "", "Very True", None, 1, True])
def test_invalid_rating_labels_are_rejected(label: object) -> None:
    with pytest.raises(InvalidRatingError):
        map_rating_to_ptl(label)  # type: ignore[arg-type]


def test_prepare_ratings_groups_by_exact_claim_id() -> None:
    claims = [Claim(1, "First claim"), Claim("2", "Second claim")]
    ratings = [
        UserRating("2", "True"),
        UserRating(1, "False"),
        UserRating("2", "Mostly True"),
    ]

    prepared = prepare_ratings(claims, ratings)

    assert prepared.claims == tuple(claims)
    assert prepared.ptl_by_claim[1] == (-1.0,)
    assert prepared.ptl_by_claim["2"] == (1.0, 0.5)


def test_claim_and_rating_ids_must_match_exactly() -> None:
    with pytest.raises(UnknownClaimError):
        prepare_ratings(
            [Claim(1, "Numeric ID")],
            [UserRating("1", "True")],
        )


def test_duplicate_claim_ids_are_rejected() -> None:
    with pytest.raises(DuplicateClaimError, match="1"):
        prepare_ratings(
            [Claim("1", "First"), Claim("1", "Duplicate")],
            [UserRating("1", "True")],
        )


def test_unknown_claim_ids_are_rejected() -> None:
    with pytest.raises(UnknownClaimError, match="999"):
        prepare_ratings(
            [Claim("1", "Known")],
            [UserRating("1", "True"), UserRating("999", "False")],
        )


def test_missing_perception_records_are_reported() -> None:
    with pytest.raises(MissingPerceptionError, match="'2'"):
        prepare_ratings(
            [Claim("1", "First"), Claim("2", "Second")],
            [UserRating("1", "True")],
        )


@pytest.mark.parametrize("bad_claim", [object(), "not a Claim"])
def test_non_claim_records_are_rejected(bad_claim: object) -> None:
    with pytest.raises(InvalidClaimError):
        prepare_ratings([bad_claim], [])  # type: ignore[list-item]


def test_empty_claim_text_is_rejected() -> None:
    with pytest.raises(InvalidClaimError, match="claim text"):
        prepare_ratings([Claim("1", "   ")], [UserRating("1", "True")])


def test_boolean_claim_id_is_rejected() -> None:
    with pytest.raises(InvalidClaimError):
        prepare_ratings([Claim(True, "Boolean ID")], [UserRating(True, "True")])


def test_invalid_rating_record_is_rejected() -> None:
    with pytest.raises(InvalidRatingError):
        prepare_ratings(
            [Claim("1", "Claim")],
            ["not a UserRating"],  # type: ignore[list-item]
        )
