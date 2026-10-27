"""Typed inputs and outputs for bias-aware prioritization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

ClaimId: TypeAlias = str | int
TruthLabel: TypeAlias = Literal["True", "False"]
TPBLabel: TypeAlias = Literal["High", "Low"]


class Objective(str, Enum):
    """The three bias-aware prioritization objectives."""

    O1 = "O1"
    O2 = "O2"
    O3 = "O3"


@dataclass(frozen=True, slots=True)
class Claim:
    """A claim with a stable identifier and its text."""

    claim_id: ClaimId
    claim: str


@dataclass(frozen=True, slots=True)
class UserRating:
    """A single user's perception rating for a claim."""

    claim_id: ClaimId
    rating: str


@dataclass(frozen=True, slots=True)
class PrioritizedClaim:
    """One claim in an objective-specific prioritized result list."""

    claim_id: ClaimId
    claim: str
    objective: Objective
    rank: int
    predicted_truth: TruthLabel | None = None
    predicted_tpb: TPBLabel | None = None
    disputability: float | None = None
