"""Evidence retrieval via Tavily Search API.

Public entry point:
    retrieve_evidence(claim: str, top_k: int = 5) -> list[dict]

Each returned record has exactly:
    - title: str (may be "" if the source omits it)
    - url: str (non-empty; records without a URL are dropped)
    - content: str (non-empty evidence text, preserved verbatim apart
      from surrounding-whitespace trimming; never invented or rewritten)
    - relevance_score: float (Tavily ``score`` coerced to float,
      0.0 when missing or non-numeric)

Behavior:
    - Validates ``claim`` (non-empty string) and ``top_k`` (int in 1..N)
      with ``ValueError`` on bad input.
    - Loads ``TAVILY_API_KEY`` from the project-root ``.env`` via
      ``python-dotenv`` (existing environment variables take precedence).
      Raises :class:`MissingAPIKeyError` when absent or blank. The key
      value is never logged or printed.
    - Wraps any Tavily/API failure in :class:`EvidenceSearchError`
      (chained with ``from``) instead of returning fabricated evidence.
    - Drops records with missing URLs or blank content, deduplicates by
      URL (keeping the highest score), sorts by score descending, and
      returns at most ``min(top_k, MAX_RESULTS)`` records.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

MAX_RESULTS = 5


class EvidenceRetrievalError(Exception):
    """Base error for this module."""


class MissingAPIKeyError(EvidenceRetrievalError):
    """Raised when TAVILY_API_KEY is not configured."""


class EvidenceSearchError(EvidenceRetrievalError):
    """Raised when the Tavily search request or response handling fails."""


def _project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _get_api_key() -> str:
    """Load and return TAVILY_API_KEY, or raise MissingAPIKeyError.

    Loads ``<project-root>/.env`` if present (without overriding
    existing environment variables), then reads the environment.
    The key value is never printed or logged.
    """
    try:
        load_dotenv(dotenv_path=_project_root() / ".env", override=False)
    except Exception:
        # A broken .env must not mask the environment; fall through to
        # the getenv check below.
        pass
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key is None or not api_key.strip():
        raise MissingAPIKeyError(
            "TAVILY_API_KEY is not configured. "
            "Set it in the environment or in the project-root .env file."
        )
    return api_key.strip()


def _validate_inputs(claim: str, top_k: int) -> tuple[str, int]:
    """Validate inputs; return (cleaned_claim, effective_top_k)."""
    if not isinstance(claim, str):
        raise ValueError("claim must be a string.")
    cleaned = claim.strip()
    if not cleaned:
        raise ValueError("claim must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    return cleaned, min(top_k, MAX_RESULTS)


def _coerce_score(value: Any) -> float:
    """Coerce a Tavily score to float; return 0.0 when unusable."""
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_results(raw_results: Any, limit: int) -> list[dict]:
    """Filter, deduplicate, sort, and limit raw Tavily results.

    - Skips non-dict entries.
    - Drops entries with a missing/blank URL or blank content.
    - Deduplicates by stripped URL, keeping the highest score.
    - Sorts by relevance_score descending; stable for ties.
    - Returns at most ``limit`` records.
    """
    if not isinstance(raw_results, list):
        raise EvidenceSearchError(
            f"Unexpected search response shape: expected list, got {type(raw_results).__name__}."
        )
    best_by_url: dict[str, dict] = {}
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()
        content = entry.get("content")
        # Fall back to raw_content only when content is absent; the text
        # itself is preserved, never rewritten.
        if content is None:
            content = entry.get("raw_content")
        if not isinstance(content, str) or not content.strip():
            continue
        title = entry.get("title")
        record = {
            "title": title.strip() if isinstance(title, str) else "",
            "url": url,
            "content": content.strip(),
            "relevance_score": _coerce_score(entry.get("score")),
        }
        existing = best_by_url.get(url)
        if existing is None or record["relevance_score"] > existing["relevance_score"]:
            best_by_url[url] = record
    ranked = sorted(best_by_url.values(), key=lambda r: r["relevance_score"], reverse=True)
    return ranked[:limit]


def retrieve_evidence(claim: str, top_k: int = 5) -> list[dict]:
    """Retrieve web evidence for a factual claim via Tavily.

    Args:
        claim: Factual claim to search for. Must be a non-empty string.
        top_k: Maximum records to return. Capped at 5 for this MVP.

    Returns:
        List of at most ``min(top_k, 5)`` dicts with keys ``title``,
        ``url``, ``content``, ``relevance_score``, sorted by score
        descending. May be shorter (or empty) after cleaning.

    Raises:
        ValueError: If ``claim`` is empty/not a string, or ``top_k``
            is not an integer >= 1.
        MissingAPIKeyError: If ``TAVILY_API_KEY`` is not configured.
        EvidenceSearchError: If client creation, the API call, or
            response handling fails. The original exception is chained.
    """
    cleaned_claim, effective_top_k = _validate_inputs(claim, top_k)
    api_key = _get_api_key()
    try:
        client = TavilyClient(api_key=api_key)
    except MissingAPIKeyError:
        raise
    except Exception as exc:
        raise EvidenceSearchError(f"Tavily client initialization failed: {exc}") from exc
    try:
        response = client.search(query=cleaned_claim, max_results=effective_top_k)
    except Exception as exc:
        raise EvidenceSearchError(f"Tavily search failed: {exc}") from exc
    if isinstance(response, dict):
        raw_results = response.get("results", [])
    elif isinstance(response, list):
        raw_results = response
    else:
        raise EvidenceSearchError(
            "Unexpected search response shape: "
            f"expected dict or list, got {type(response).__name__}."
        )
    return _clean_results(raw_results, effective_top_k)
