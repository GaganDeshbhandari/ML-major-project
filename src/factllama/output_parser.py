"""Parsing of FactLLaMA verdict output.

The model is instructed to answer with exactly one of ``true``, ``half`` or
``false``. Real generations are sometimes less tidy (surrounding text,
capitalisation, markdown fences, a ``Verdict:`` prefix), so the parser
tolerates those forms.

The parser is deliberately conservative: exactly one distinct label must be
present. Explanations containing multiple conflicting labels (for example
``Verdict: true or false``) are rejected instead of picking the first one.
When parsing fails, :func:`parse_verdict` raises :class:`VerdictParseError`
and :func:`try_parse_verdict` returns ``None``. The raw output is always kept
by the caller.
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
    """Raised when generated text has no unambiguous supported verdict."""


def _strip_code_fences(text: str) -> str:
    """Remove a single surrounding markdown code fence, if present."""
    stripped = text.strip()
    stripped = _CODE_FENCE_START.sub("", stripped, count=1)
    stripped = _CODE_FENCE_END.sub("", stripped, count=1)
    return stripped.strip()


def parse_verdict(text: str) -> str:
    """Return ``"true"``, ``"half"`` or ``"false"`` from generated text.

    Rules:
        * Case and surrounding whitespace are ignored.
        * A ``Verdict:`` marker limits the search to the text after it, so a
          label mentioned in an explanation before it is ignored. Multiple
          markers are rejected as ambiguous.
        * Labels must appear as whole words; ``truthful`` and ``falsehood``
          do not count.
        * If zero labels are found, or more than one distinct label is found,
          parsing fails. Conflicting labels are never resolved by position.

    Raises:
        VerdictParseError: the text is empty, has no label after the marker,
            has no label at all, or contains conflicting labels.
    """
    if not isinstance(text, str):
        raise VerdictParseError(
            f"model output must be a string, got {type(text).__name__}"
        )

    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise VerdictParseError("model output is empty")

    markers = list(_VERDICT_MARKER_PATTERN.finditer(cleaned))
    if len(markers) > 1:
        raise VerdictParseError(
            "model output contains multiple 'Verdict:' markers and is ambiguous"
        )
    if markers:
        cleaned = cleaned[markers[-1].end():]
        if not cleaned.strip():
            raise VerdictParseError(
                "model output has a 'Verdict:' marker but no label after it"
            )

    labels = [
        match.group(1).lower()
        for match in _LABEL_PATTERN.finditer(cleaned)
    ]
    if not labels:
        preview = text.strip().replace("\n", " ")[:120]
        raise VerdictParseError(
            f"no verdict label (true, half, false) in model output: {preview!r}"
        )

    distinct = sorted(set(labels))
    if len(distinct) > 1:
        preview = text.strip().replace("\n", " ")[:120]
        raise VerdictParseError(
            "conflicting verdict labels "
            f"({', '.join(distinct)}) in model output: {preview!r}"
        )

    return labels[0]


def try_parse_verdict(text: str) -> str | None:
    """Return the parsed verdict, or ``None`` when parsing fails."""
    try:
        return parse_verdict(text)
    except VerdictParseError:
        return None
