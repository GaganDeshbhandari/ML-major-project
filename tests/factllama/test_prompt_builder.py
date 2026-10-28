"""Unit tests for src.factllama.prompt_builder.

No model, tokenizer, network or GPU is involved.
"""

import pytest

from src.factllama.prompt_builder import (
    DEFAULT_MAX_EVIDENCE_TOKENS,
    SYSTEM_PROMPT,
    EvidenceBudget,
    PromptBuildError,
    build_messages,
    build_prompt,
    build_user_prompt,
    normalise_evidence,
    render_chat_fallback,
    select_evidence,
    truncate_to_budget,
)

EXACT_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Evaluate the claim using "
    "only the provided evidence. Output exactly one verdict: "
    "true, half, or false."
)


def test_system_prompt_is_exact_training_prompt():
    assert SYSTEM_PROMPT == EXACT_SYSTEM_PROMPT


def test_user_prompt_exact_format():
    prompt = build_user_prompt(
        "The Eiffel Tower is in Paris.",
        ["Paris is in France.", "The tower is in Paris, France."],
    )

    expected = (
        "Claim: The Eiffel Tower is in Paris.\n"
        "\n"
        "Evidence:\n"
        "[1] Paris is in France.\n"
        "[2] The tower is in Paris, France.\n"
        "\n"
        "Verdict:"
    )
    assert prompt == expected


def test_claim_is_trimmed_but_not_otherwise_changed():
    prompt = build_user_prompt("  claim text  ", ["evidence"])
    assert prompt.startswith("Claim: claim text\n")
    assert "  claim text  " not in prompt


def test_long_claim_is_not_truncated_by_evidence_budget():
    claim = "claim " * 30
    budget = EvidenceBudget(
        max_tokens=5, chars_per_token=1.0, min_truncate_tokens=1
    )

    prompt = build_user_prompt(claim, ["x" * 200], budget=budget)

    assert prompt.startswith(f"Claim: {claim.strip()}\n")


def test_retrieval_records_supply_text_only():
    evidence = [
        {
            "title": "Secret title",
            "url": "https://example.test/source",
            "content": "Evidence A",
        },
        {"title": "B", "url": "https://b", "text": "Evidence B"},
        {"title": "C", "url": "https://c", "evidence": "Evidence C"},
    ]

    assert normalise_evidence(evidence) == [
        "Evidence A",
        "Evidence B",
        "Evidence C",
    ]

    built = build_prompt("claim", evidence)
    user_content = built.messages[1]["content"]
    assert "Secret title" not in user_content
    assert "https://example.test/source" not in user_content
    assert "[1] Evidence A" in user_content


def test_evidence_text_is_preserved_apart_from_whitespace():
    assert normalise_evidence(["  keep  internal   spacing  "]) == [
        "keep  internal   spacing"
    ]


def test_evidence_order_is_preserved():
    budget = EvidenceBudget(
        max_tokens=100, chars_per_token=1.0, min_truncate_tokens=1
    )

    selection = select_evidence(
        ["first evidence", "second evidence", "third evidence"],
        budget=budget,
    )

    assert selection.texts == (
        "first evidence",
        "second evidence",
        "third evidence",
    )
    assert selection.truncated is False
    assert selection.excluded == 0


def test_build_messages_roles_and_contents():
    messages = build_messages("claim", ["evidence"])

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"].startswith("Claim: claim\n")
    assert messages[1]["content"].endswith("Verdict:")


def test_build_prompt_exposes_selection_metadata():
    budget = EvidenceBudget(
        max_tokens=10, chars_per_token=1.0, min_truncate_tokens=2
    )

    built = build_prompt("claim", ["a" * 8, "b" * 8], budget=budget)

    assert built.selection.texts == ("a" * 8, "bb")
    assert built.selection.truncated is True
    assert built.selection.excluded == 0
    assert built.selection.total_items == 2
    assert built.selection.estimated_tokens == 10
    assert "[2] bb" in built.messages[1]["content"]


def test_oversized_passage_is_truncated_at_word_boundary():
    budget = EvidenceBudget(
        max_tokens=8, chars_per_token=1.0, min_truncate_tokens=1
    )

    selection = select_evidence(["alpha beta gamma"], budget=budget)

    assert selection.texts == ("alpha",)
    assert selection.truncated is True
    assert selection.estimated_tokens == 5


def test_later_passages_are_excluded_when_budget_is_exhausted():
    budget = EvidenceBudget(
        max_tokens=4, chars_per_token=1.0, min_truncate_tokens=4
    )

    selection = select_evidence(["abc", "defgh", "ijk"], budget=budget)

    assert selection.texts == ("abc",)
    assert selection.truncated is False
    assert selection.excluded == 2
    assert selection.estimated_tokens == 3


def test_tiny_remaining_budget_does_not_create_fragments():
    budget = EvidenceBudget(
        max_tokens=10, chars_per_token=1.0, min_truncate_tokens=4
    )

    selection = select_evidence(["123456789", "abcdefgh"], budget=budget)

    # First passage consumes 9; only 1 token remains, below the minimum.
    assert selection.texts == ("123456789",)
    assert selection.truncated is False
    assert selection.excluded == 1


def test_default_budget_limits_very_long_evidence():
    budget_default = EvidenceBudget()
    assert budget_default.max_tokens == DEFAULT_MAX_EVIDENCE_TOKENS

    selection = select_evidence(["x" * 5000])

    assert selection.truncated is True
    assert selection.excluded == 0
    assert selection.estimated_tokens <= DEFAULT_MAX_EVIDENCE_TOKENS
    assert len(selection.texts[0]) < 5000


def test_truncate_to_budget_returns_short_text_unchanged():
    assert truncate_to_budget("short", 100, 4.0) == "short"


def test_truncate_to_budget_does_not_cut_mid_word_when_possible():
    assert truncate_to_budget("alpha beta gamma", 8, 1.0) == "alpha"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": 10, "chars_per_token": 0},
        {"max_tokens": 10, "min_truncate_tokens": -1},
    ],
)
def test_invalid_budget_rejected(kwargs):
    with pytest.raises(PromptBuildError):
        EvidenceBudget(**kwargs)


def test_small_budget_caps_truncation_threshold():
    selection = select_evidence(
        ["abcdefgh"],
        budget=EvidenceBudget(max_tokens=5, chars_per_token=1.0),
    )

    assert selection.texts == ("abcde",)
    assert selection.truncated is True
    assert selection.estimated_tokens == 5


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
