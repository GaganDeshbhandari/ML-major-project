"""Parsing of FactLLaMA verdict output.

The model is instructed to answer with exactly one of ``true``, ``half`` or
``false``. Real generations are sometimes less tidy (surrounding text,
capitalisation, markdown fences, a ``Verdict:`` prefix), so the parser
extracts the first label found rather than requiring an exact match.

Parsing never guesses: when no supported label is present,
:func:`parse_verdict` raises :class:`VerdictParseError` and
:func:`try_parse_verdict` returns ``None``. The raw model output is always
preserved by the caller.
"""

from __future__ import annotations

import re

from .prompt_builder import VERDICT_LABELS

__all__ = [
    "VERDICT_LABELS",
    "VerdictParseError",
    "parse_verdict",
    "try_parse_verdict",
]

_LABEL_PATTERN = re.compile(r"\b(true|half|false)\b", re.IGNORECASE)
_VERDICT_MARKER_PATTERN = re.compile(r"verdict\s*:", re.IGNORECASE)
_CODE_FENCE_START = re.compile(r"^```[A-Za-z0-9_-]*\s*")
_CODE_FENCE_END = re.compile(r"\s*```$")


class VerdictParseError(ValueError):
    """Raised when generated text contains no supported verdict label."""


def _strip_code_fences(text: str) -> str:
    """Remove a single surrounding markdown code fence, if present."""
    stripped = text.strip()
    stripped = _CODE_FENCE_START.sub("", stripped, count=1)
    stripped = _CODE_FENCE_END.sub("", stripped, count=1)
    return stripped.strip()


def parse_verdict(text: str) -> str:
    """Return ``"true"``, ``"half"`` or ``"false"`` from generated text.

    The search is limited to the text after the last ``Verdict:`` marker when
    one is present, which avoids picking a label out of the model's
    explanation. Matching is case-insensitive and word-bounded, so
    ``truthful`` or ``falsehood`` are not treated as labels.

    Raises:
        VerdictParseError: the text is empty or contains no supported label.
    """
    if not isinstance(text, str):
        raise VerdictParseError(
            f"model output must be a string, got {type(text).__name__}"
        )

    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise VerdictParseError("model output is empty")

    markers = list(_VERDICT_MARKER_PATTERN.finditer(cleaned))
    if markers:
        cleaned = cleaned[markers[-1].end():]

    match = _LABEL_PATTERN.search(cleaned)
    if match is None:
        preview = text.strip().replace("\n", " ")[:120]
        raise VerdictParseError(
            f"no verdict label (true, half, false) in model output: {preview!r}"
        )
    return match.group(1).lower()


def try_parse_verdict(text: str) -> str | None:
    """Return the parsed verdict, or ``None`` when parsing fails."""
    try:
        return parse_verdict(text)
    except VerdictParseError:
        return None
