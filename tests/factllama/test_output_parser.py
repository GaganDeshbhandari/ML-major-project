"""Unit tests for src.factllama.output_parser."""

import pytest

from src.factllama.output_parser import (
    VerdictParseError,
    parse_verdict,
    try_parse_verdict,
)


@pytest.mark.parametrize("label", ["true", "half", "false"])
def test_exact_labels(label):
    assert parse_verdict(label) == label


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  TRUE  ", "true"),
        ("Half", "half"),
        ("FALSE.", "false"),
        ('"true"', "true"),
        ("Verdict: half", "half"),
        ("verdict : FALSE", "false"),
        ("The claim is supported.\nVerdict: true", "true"),
        ("```\nfalse\n```", "false"),
        ("```text\ntrue\n```", "true"),
    ],
)
def test_tolerant_formats(text, expected):
    assert parse_verdict(text) == expected


def test_marker_limits_search_to_the_answer():
    text = "A false start, but the evidence supports it.\nVerdict: true"
    assert parse_verdict(text) == "true"


def test_word_boundaries_do_not_match_incidental_words():
    with pytest.raises(VerdictParseError):
        parse_verdict("The statement is truthful and not a falsehood.")


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_empty_or_non_string_output_rejected(text):
    with pytest.raises(VerdictParseError):
        parse_verdict(text)


def test_unknown_output_raises_with_preview():
    with pytest.raises(VerdictParseError) as excinfo:
        parse_verdict("I cannot decide based on this evidence")
    assert "no verdict label" in str(excinfo.value)


def test_try_parse_verdict_returns_none_on_failure():
    assert try_parse_verdict("unparseable") is None
    assert try_parse_verdict("Verdict: false") == "false"
