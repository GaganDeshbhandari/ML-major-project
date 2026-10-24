"""Rating validation, PTL mapping, and claim-associated grouping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..errors import (
    DuplicateClaimError,
    InvalidClaimError,
    InvalidRatingError,
    MissingPerceptionError,
    UnknownClaimError,
)
from ..schemas import Claim, ClaimId, UserRating

PTL_MAPPING: Mapping[str, float] = MappingProxyType(
    {
        "True": 1.0,
        "Mostly True": 0.5,
        "Mixture": 0.0,
        "Mostly False": -0.5,
        "False": -1.0,
    }
)


@dataclass(frozen=True, slots=True)
class PreparedRatings:
    """Claims in input order and validated PTL values grouped by claim ID."""

    claims: tuple[Claim, ...]
    ptl_by_claim: Mapping[ClaimId, tuple[float, ...]]


def map_rating_to_ptl(rating: str) -> float:
    """Map a perception label to its PTL value.

    Labels are matched exactly, as in the notebook's ``Series.map`` call.
    """

    if not isinstance(rating, str):
        raise InvalidRatingError(
            f"Unsupported rating label {rating!r}: expected one of "
            f"{', '.join(PTL_MAPPING)}."
        )
    try:
        return PTL_MAPPING[rating]
    except KeyError as exc:
        raise InvalidRatingError(
            f"Unsupported rating label {rating!r}: expected one of "
            f"{', '.join(PTL_MAPPING)}."
        ) from exc


def _validate_claim_id(claim_id: object, *, context: str) -> None:
    if isinstance(claim_id, bool) or not isinstance(claim_id, (str, int)):
        raise InvalidClaimError(
            f"{context} must use a string or integer claim_id; got "
            f"{type(claim_id).__name__}."
        )
    if isinstance(claim_id, str) and not claim_id:
        raise InvalidClaimError(f"{context} must not use an empty claim_id.")


def _format_ids(ids: Iterable[ClaimId]) -> str:
    return ", ".join(repr(claim_id) for claim_id in ids)


def index_claims(
    claims: Iterable[Claim],
) -> tuple[tuple[Claim, ...], dict[ClaimId, Claim]]:
    """Validate claims and build an ordered ID lookup."""

    if claims is None:
        raise InvalidClaimError("claims must be an iterable of Claim records.")

    validated: list[Claim] = []
    by_id: dict[ClaimId, Claim] = {}
    duplicates: list[ClaimId] = []

    for position, claim in enumerate(claims):
        if not isinstance(claim, Claim):
            raise InvalidClaimError(
                f"Claim at position {position} must be a Claim record; got "
                f"{type(claim).__name__}."
            )
        _validate_claim_id(claim.claim_id, context=f"Claim at position {position}")
        if not isinstance(claim.claim, str) or not claim.claim.strip():
            raise InvalidClaimError(
                f"Claim {claim.claim_id!r} must contain non-empty claim text."
            )
        if claim.claim_id in by_id:
            if claim.claim_id not in duplicates:
                duplicates.append(claim.claim_id)
            continue
        by_id[claim.claim_id] = claim
        validated.append(claim)

    if duplicates:
        raise DuplicateClaimError(
            f"Duplicate claim IDs are not allowed: {_format_ids(duplicates)}."
        )

    return tuple(validated), by_id


def group_ratings_by_claim(
    claim_index: Mapping[ClaimId, Claim],
    ratings: Iterable[UserRating],
) -> dict[ClaimId, tuple[float, ...]]:
    """Validate ratings and group their PTL values by exact claim ID."""

    if ratings is None:
        raise InvalidRatingError("ratings must be an iterable of UserRating records.")

    grouped: dict[ClaimId, list[float]] = {claim_id: [] for claim_id in claim_index}
    unknown: list[ClaimId] = []

    for position, rating in enumerate(ratings):
        if not isinstance(rating, UserRating):
            raise InvalidRatingError(
                f"Rating at position {position} must be a UserRating record; got "
                f"{type(rating).__name__}."
            )

        claim_id = rating.claim_id
        if (
            isinstance(claim_id, bool)
            or not isinstance(claim_id, (str, int))
            or (isinstance(claim_id, str) and not claim_id)
        ):
            raise InvalidRatingError(
                f"Rating at position {position} has an invalid claim_id: {claim_id!r}."
            )

        if claim_id not in claim_index:
            if claim_id not in unknown:
                unknown.append(claim_id)
            continue

        grouped[claim_id].append(map_rating_to_ptl(rating.rating))

    if unknown:
        raise UnknownClaimError(
            "Ratings reference claim IDs that were not supplied: "
            f"{_format_ids(unknown)}."
        )

    missing = [claim_id for claim_id, values in grouped.items() if not values]
    if missing:
        raise MissingPerceptionError(
            "Claims without perception records: "
            f"{_format_ids(missing)}. Every claim must have at least one rating."
        )

    return {claim_id: tuple(values) for claim_id, values in grouped.items()}


def prepare_ratings(
    claims: Iterable[Claim],
    ratings: Iterable[UserRating],
) -> PreparedRatings:
    """Validate and associate all claim and rating input by claim ID."""

    validated_claims, claim_index = index_claims(claims)
    grouped = group_ratings_by_claim(claim_index, ratings)
    return PreparedRatings(
        claims=validated_claims,
        ptl_by_claim=MappingProxyType(grouped),
    )
