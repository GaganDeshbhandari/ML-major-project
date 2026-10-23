"""Prompt construction for FactLLaMA verification.

The prompt mirrors the format used for fine-tuning:

* a system instruction that requires evidence-based verification, forbids
  outside knowledge, and asks for exactly one of ``true``, ``half`` or
  ``false``;
* a user message containing the claim, a numbered evidence list and the
  ``Verdict:`` marker that cues the model's answer.

Nothing here imports a model, a tokenizer or any ML dependency, so the
module is safe to import in unit tests and lightweight callers.

Evidence can be passed as plain strings or as retrieval records (mappings
with a ``content``/``text``/``evidence`` field). The text itself is never
rewritten; only surrounding whitespace is trimmed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "SYSTEM_PROMPT",
    "PromptBuildError",
    "VERDICT_LABELS",
    "build_messages",
    "build_user_prompt",
    "normalise_evidence",
    "render_chat_fallback",
]

#: Labels the model is allowed to produce. Order is intentional.
VERDICT_LABELS = ("true", "half", "false")

#: System instruction used for every verification request.
SYSTEM_PROMPT = (
    "You are a fact-verification assistant. Decide whether the claim is "
    "supported by the evidence supplied by the user.\n"
    "Rules:\n"
    "- Use only the evidence in the user message. Do not use outside "
    "knowledge, memory, or assumptions.\n"
    "- Answer true when the evidence fully supports the claim.\n"
    "- Answer half when the evidence only partially supports the claim or "
    "the evidence is mixed.\n"
    "- Answer false when the evidence contradicts the claim or does not "
    "support it.\n"
    "- Respond with exactly one of these three words: true, half, false.\n"
    "- Do not explain, quote the evidence, or add any other text."
)

#: Field names accepted from retrieval-style evidence records.
_EVIDENCE_TEXT_KEYS = ("content", "text", "evidence")


class PromptBuildError(ValueError):
    """Raised when a claim or evidence list cannot be turned into a prompt."""


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

    A single string is rejected on purpose: callers must pass a sequence so
    that one evidence item does not silently become five characters.
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


def build_user_prompt(claim: str, evidence: Any) -> str:
    """Build the user message for one verification request.

    The output contains the claim, numbered evidence text and a trailing
    ``Verdict:`` marker, matching the fine-tuning input layout.
    """
    claim_text = _validate_claim(claim)
    evidence_items = normalise_evidence(evidence)

    lines = ["Claim:", claim_text, "", "Evidence:"]
    lines.extend(
        f"{index}. {text}" for index, text in enumerate(evidence_items, start=1)
    )
    lines.extend(["", "Verdict:"])
    return "\n".join(lines)


def build_messages(claim: str, evidence: Any) -> list[dict[str, str]]:
    """Return chat messages for ``tokenizer.apply_chat_template``."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(claim, evidence)},
    ]


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
