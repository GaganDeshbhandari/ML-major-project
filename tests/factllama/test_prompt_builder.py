"""Unit tests for src.factllama.prompt_builder.

No model, tokenizer, network or GPU is involved.
"""

import pytest

from src.factllama.prompt_builder import (
    SYSTEM_PROMPT,
    PromptBuildError,
    build_messages,
    build_user_prompt,
    normalise_evidence,
    render_chat_fallback,
)


def test_system_prompt_has_required_constraints():
    lowered = SYSTEM_PROMPT.lower()
    assert "evidence" in lowered
    assert "outside" in lowered and "knowledge" in lowered
    for label in ("true", "half", "false"):
        assert label in lowered


def test_user_prompt_contains_claim_numbered_evidence_and_marker():
    prompt = build_user_prompt(
        "The Eiffel Tower is in Paris.",
        ["Paris is in France.", "The tower is in Paris, France."],
    )

    assert "The Eiffel Tower is in Paris." in prompt
    assert "1. Paris is in France." in prompt
    assert "2. The tower is in Paris, France." in prompt
    assert prompt.rstrip().endswith("Verdict:")
    assert prompt.index("Evidence:") < prompt.index("1.")
    assert prompt.index("Verdict:") > prompt.index("2.")


def test_claim_is_trimmed():
    prompt = build_user_prompt("  claim text  ", ["evidence"])
    assert "claim text" in prompt
    assert "  claim text  " not in prompt


def test_retrieval_records_are_supported():
    evidence = [
        {"title": "A", "url": "https://a", "content": "Evidence A"},
        {"title": "B", "url": "https://b", "text": "Evidence B"},
        {"title": "C", "url": "https://c", "evidence": "Evidence C"},
    ]
    assert normalise_evidence(evidence) == [
        "Evidence A",
        "Evidence B",
        "Evidence C",
    ]


def test_evidence_text_is_preserved_apart_from_whitespace():
    assert normalise_evidence(["  keep  internal   spacing  "]) == [
        "keep  internal   spacing"
    ]


def test_build_messages_roles_and_contents():
    messages = build_messages("claim", ["evidence"])

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "claim" in messages[1]["content"]
    assert "Verdict:" in messages[1]["content"]


@pytest.mark.parametrize("claim", ["", "   ", None, 123])
def test_invalid_claim_rejected(claim):
    with pytest.raises(PromptBuildError):
        build_user_prompt(claim, ["evidence"])


@pytest.mark.parametrize("evidence", [[], (), ["   "], [None], [123]])
def test_invalid_evidence_rejected(evidence):
    with pytest.raises(PromptBuildError):
        build_user_prompt("claim", evidence)


def test_single_string_evidence_is_rejected():
    with pytest.raises(PromptBuildError):
        normalise_evidence("one big evidence blob")


def test_mapping_without_text_field_is_rejected():
    with pytest.raises(PromptBuildError):
        normalise_evidence([{"title": "no text here"}])


def test_render_chat_fallback_ends_with_assistant_turn():
    rendered = render_chat_fallback(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
    )
    assert "### System:\nsys" in rendered
    assert "### User:\nusr" in rendered
    assert rendered.rstrip().endswith("### Assistant:")


def test_render_chat_fallback_rejects_empty_messages():
    with pytest.raises(PromptBuildError):
        render_chat_fallback([])
