"""Prompt construction and evidence budgeting for FactLLaMA verification.

The prompt mirrors the fine-tuning input:

* a system instruction that asks for evidence-only verification and exactly
  one of ``true``, ``half`` or ``false``;
* a user message with ``Claim: <claim>``, a numbered evidence list using
  ``[1]``, ``[2]``, ... and a final ``Verdict:`` marker.

Nothing here imports a model, tokenizer or ML dependency, so the module is
safe to import in unit tests and lightweight callers.

Evidence can be plain strings or retrieval records (mappings with a
``content``/``text``/``evidence`` field). Only that text field is sent to the
model; titles, URLs and other source metadata are ignored. Evidence order is
preserved. A configurable :class:`EvidenceBudget` prevents unbounded evidence
from entering the prompt; the budget is estimated from characters, not from a
tokenizer, so it is deliberately conservative and approximate.

The claim itself is never truncated; only its surrounding whitespace is
trimmed. Truncation/exclusion is reported by :class:`EvidenceSelection`, which
:func:`build_prompt` exposes alongside the messages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_EVIDENCE_BUDGET",
    "DEFAULT_MAX_EVIDENCE_TOKENS",
    "DEFAULT_MIN_TRUNCATE_TOKENS",
    "SYSTEM_PROMPT",
    "VERDICT_LABELS",
    "BuiltPrompt",
    "EvidenceBudget",
    "EvidenceSelection",
    "PromptBuildError",
    "build_messages",
    "build_prompt",
    "build_user_prompt",
    "normalise_evidence",
    "render_chat_fallback",
    "select_evidence",
    "truncate_to_budget",
]

#: Labels the model is allowed to produce. Order is intentional.
VERDICT_LABELS = ("true", "half", "false")

#: Exact system instruction used during fine-tuning (notebook cells 8 and 10).
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Evaluate the claim using "
    "only the provided evidence. Output exactly one verdict: "
    "true, half, or false."
)

#: Default evidence budget, in approximate tokens.
DEFAULT_MAX_EVIDENCE_TOKENS = 1024

#: Conservative character-to-token estimate (English text is usually 3-4).
DEFAULT_CHARS_PER_TOKEN = 4.0

#: Do not truncate a passage to fewer than this many approximate tokens.
DEFAULT_MIN_TRUNCATE_TOKENS = 32

#: Field names accepted from retrieval-style evidence records.
_EVIDENCE_TEXT_KEYS = ("content", "text", "evidence")


class PromptBuildError(ValueError):
    """Raised when a claim, evidence list or budget is invalid."""


@dataclass(frozen=True)
class EvidenceBudget:
    """Approximate token budget for evidence text.

    Args:
        max_tokens: Maximum estimated tokens across all evidence passages.
        chars_per_token: Characters per token used for the estimate. Four is
            conservative for English but is not tokenizer-exact.
        min_truncate_tokens: A passage that does not fit is only truncated
            when at least this many estimated tokens remain; otherwise it and
            all following passages are excluded. The value is capped at
            ``max_tokens`` so small budgets remain usable. Keeping this above
            zero avoids sending tiny, meaningless fragments.

    The estimate is intentionally approximate because the prompt builder is
    tokenizer-free. Callers that need exact counts can size the budget
    conservatively.
    """

    max_tokens: int = DEFAULT_MAX_EVIDENCE_TOKENS
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    min_truncate_tokens: int = DEFAULT_MIN_TRUNCATE_TOKENS

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise PromptBuildError(
                "EvidenceBudget.max_tokens must be an integer >= 1"
            )
        if (
            isinstance(self.chars_per_token, bool)
            or not isinstance(self.chars_per_token, (int, float))
            or self.chars_per_token <= 0
        ):
            raise PromptBuildError(
                "EvidenceBudget.chars_per_token must be a number > 0"
            )
        if (
            isinstance(self.min_truncate_tokens, bool)
            or not isinstance(self.min_truncate_tokens, int)
            or self.min_truncate_tokens < 0
        ):
            raise PromptBuildError(
                "EvidenceBudget.min_truncate_tokens must be an integer >= 0"
            )

    def estimate_tokens(self, text: str) -> int:
        """Return a conservative token estimate for ``text``."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))


#: Budget used when the caller does not provide one.
DEFAULT_EVIDENCE_BUDGET = EvidenceBudget()


@dataclass(frozen=True)
class EvidenceSelection:
    """Evidence after applying a budget, plus explicit budget metadata."""

    texts: tuple[str, ...]
    total_items: int
    estimated_tokens: int
    truncated: bool
    excluded: int

    @property
    def included(self) -> int:
        """Number of passages included in the prompt."""
        return len(self.texts)

    @property
    def budget_exhausted(self) -> bool:
        """Whether at least one passage was truncated or excluded."""
        return self.truncated or self.excluded > 0


@dataclass(frozen=True)
class BuiltPrompt:
    """Chat messages plus the evidence-selection metadata."""

    messages: tuple[dict[str, str], ...]
    selection: EvidenceSelection


def _validate_claim(claim: Any) -> str:
    """Return the trimmed claim or raise :class:`PromptBuildError`."""
    if not isinstance(claim, str):
        raise PromptBuildError(
            f"claim must be a string, got {type(claim).__name__}"
        )
    trimmed = claim.strip()
    if not trimmed:
        raise PromptBuildError("claim must not be empty")
    return trimmed


def _evidence_text(item: Any) -> str:
    """Extract evidence text from a string or a retrieval-style mapping."""
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise PromptBuildError("evidence items must not be empty")
        return text

    if isinstance(item, Mapping):
        for key in _EVIDENCE_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise PromptBuildError(
            "evidence mappings must contain a non-empty "
            f"'content', 'text' or 'evidence' field; got keys {sorted(item)}"
        )

    raise PromptBuildError(
        "evidence items must be strings or mappings, got "
        f"{type(item).__name__}"
    )


def normalise_evidence(evidence: Any) -> list[str]:
    """Validate and normalise evidence into a list of non-empty strings.

    Only the evidence text is kept; source metadata such as ``title`` or
    ``url`` is never added to the prompt. A single string is rejected on
    purpose: callers must pass a sequence so one item does not silently
    become a sequence of characters.
    """
    if isinstance(evidence, (str, bytes)):
        raise PromptBuildError(
            "evidence must be a sequence of items, not a single string"
        )
    if not isinstance(evidence, Sequence):
        raise PromptBuildError(
            "evidence must be a sequence of strings or retrieval records, "
            f"got {type(evidence).__name__}"
        )

    items = [_evidence_text(item) for item in evidence]
    if not items:
        raise PromptBuildError("at least one evidence item is required")
    return items


def truncate_to_budget(
    text: str,
    max_tokens: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> str:
    """Truncate ``text`` to at most ``max_tokens`` estimated tokens.

    The cut is moved back to a word/line boundary when one exists in the
    second half of the allowed span, so passages are not normally cut in the
    middle of a word. No ellipsis is added: the evidence text is not modified
    beyond shortening, and truncation is reported separately.
    """
    if max_tokens < 1:
        raise PromptBuildError("max_tokens must be >= 1")
    if chars_per_token <= 0:
        raise PromptBuildError("chars_per_token must be > 0")
    if not text:
        return ""

    max_chars = int(max_tokens * chars_per_token)
    if max_chars < 1:
        return ""
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    boundary = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
    if boundary >= max(1, max_chars // 2):
        cut = cut[:boundary]
    return cut.rstrip()


def select_evidence(
    evidence: Any,
    *,
    budget: EvidenceBudget | None = None,
) -> EvidenceSelection:
    """Select evidence in order under an approximate token budget.

    Behaviour:
        * Passages are considered in their original order.
        * A passage is included whole when it fits the remaining budget.
        * The first passage that does not fit is truncated to the remaining
          budget when at least ``min_truncate_tokens`` remains; later
          passages are then excluded because the budget is exhausted.
        * Otherwise that passage and all following passages are excluded.
        * The claim is not part of this budget and is never truncated.

    The returned :class:`EvidenceSelection` makes every truncation/exclusion
    explicit through ``truncated``, ``excluded`` and ``estimated_tokens``.
    """
    active_budget = DEFAULT_EVIDENCE_BUDGET if budget is None else budget
    if not isinstance(active_budget, EvidenceBudget):
        raise PromptBuildError("budget must be an EvidenceBudget instance")

    items = normalise_evidence(evidence)
    selected: list[str] = []
    used_tokens = 0
    truncated = False
    excluded = 0
    minimum_truncate = min(
        active_budget.min_truncate_tokens, active_budget.max_tokens
    )

    for offset, text in enumerate(items):
        cost = active_budget.estimate_tokens(text)
        remaining = active_budget.max_tokens - used_tokens

        if cost <= remaining:
            selected.append(text)
            used_tokens += cost
            continue

        if remaining >= minimum_truncate:
            shortened = truncate_to_budget(
                text, remaining, active_budget.chars_per_token
            )
            shortened_cost = active_budget.estimate_tokens(shortened)
            if shortened and shortened_cost <= remaining:
                selected.append(shortened)
                used_tokens += shortened_cost
                truncated = True
                excluded += len(items) - offset - 1
            else:
                excluded += len(items) - offset
        else:
            excluded += len(items) - offset
        break

    return EvidenceSelection(
        texts=tuple(selected),
        total_items=len(items),
        estimated_tokens=used_tokens,
        truncated=truncated,
        excluded=excluded,
    )


def _format_user_prompt(claim: str, evidence_texts: Sequence[str]) -> str:
    lines = [f"Claim: {claim}", "", "Evidence:"]
    lines.extend(
        f"[{index}] {text}"
        for index, text in enumerate(evidence_texts, start=1)
    )
    lines.extend(["", "Verdict:"])
    return "\n".join(lines)


def build_prompt(
    claim: str,
    evidence: Any,
    *,
    budget: EvidenceBudget | None = None,
) -> BuiltPrompt:
    """Build chat messages and return the evidence-selection metadata."""
    claim_text = _validate_claim(claim)
    selection = select_evidence(evidence, budget=budget)
    messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _format_user_prompt(claim_text, selection.texts),
        },
    )
    return BuiltPrompt(messages=messages, selection=selection)


def build_user_prompt(
    claim: str,
    evidence: Any,
    *,
    budget: EvidenceBudget | None = None,
) -> str:
    """Build the user message for one verification request.

    The format is ``Claim: <claim>``, a blank line, ``Evidence:``, numbered
    ``[n] <text>`` lines, a blank line and ``Verdict:``. Use
    :func:`build_prompt` when truncation/exclusion metadata is needed.
    """
    return build_prompt(claim, evidence, budget=budget).messages[1]["content"]


def build_messages(
    claim: str,
    evidence: Any,
    *,
    budget: EvidenceBudget | None = None,
) -> list[dict[str, str]]:
    """Return chat messages for ``tokenizer.apply_chat_template``."""
    return list(build_prompt(claim, evidence, budget=budget).messages)


def render_chat_fallback(messages: Sequence[Mapping[str, str]]) -> str:
    """Render chat messages when a tokenizer has no chat template.

    This is a plain-text fallback for base checkpoints that ship without a
    chat template. Llama-3.2-Instruct normally provides one, so this path is
    rarely used.
    """
    if not messages:
        raise PromptBuildError("messages must not be empty")

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if not role or not isinstance(content, str):
            raise PromptBuildError("each message needs a role and string content")
        parts.append(f"### {role.capitalize()}:\n{content}")
    return "\n\n".join(parts) + "\n\n### Assistant:\n"
