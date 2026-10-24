"""Evidence retrieval via Tavily Search API + cross-encoder reranking.

Public entry point:
    retrieve_evidence(claim: str, top_k: int = 5, *, reranker=None, use_reranker=True)

Stage 2 flow (default):
    1. Fetch ``CANDIDATE_LIMIT`` (15) candidates from Tavily.
    2. Clean and deduplicate them (URL dedupe, blank filtering).
    3. Score each claim–candidate pair with the cross-encoder
       (``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default).
    4. Sort by the cross-encoder score and return the top ``top_k``
       records, capped at ``MAX_RESULTS`` (5).

Each reranked record contains:
    - title: str (may be "" if the source omits it)
    - url: str (non-empty)
    - content: str (evidence text, preserved verbatim apart from
      surrounding-whitespace trimming; never invented or rewritten)
    - relevance_score: float (Tavily ``score`` coerced to float,
      0.0 when missing or non-numeric — Tavily's own ranking signal)
    - rerank_score: float (cross-encoder claim–evidence score; the two
      scores live on unrelated scales and must not be treated as
      equivalent — final ordering uses ``rerank_score``)

With ``use_reranker=False`` the module keeps its Stage 1 behavior:
fetch ``min(top_k, 5)`` from Tavily, clean/dedupe, sort by
``relevance_score`` — records then have exactly the four original keys.

Behavior:
    - Validates ``claim`` (non-empty string) and ``top_k`` (int >= 1,
      bool rejected) with ``ValueError`` on bad input.
    - Loads ``TAVILY_API_KEY`` from the project-root ``.env`` via
      ``python-dotenv`` (existing environment variables take precedence).
      Raises :class:`MissingAPIKeyError` when absent or blank. The key
      value is never logged or printed.
    - Wraps any Tavily/API failure in :class:`EvidenceSearchError`
      (chained with ``from``) instead of returning fabricated evidence.
    - Reranker load/inference failures raise :class:`RerankerError`
      (chained); there is no silent fallback to Tavily ordering.
    - An injected ``reranker`` object exposing
      ``score(claim, texts) -> list[float]`` is used as-is (for tests
      and future replacements); otherwise a lazily-loaded, cached
      default cross-encoder is used. Importing this module never loads
      model weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

#: Final records returned at most (MVP cap).
MAX_RESULTS = 5

#: Tavily candidates fetched per claim before reranking.
CANDIDATE_LIMIT = 15


class EvidenceRetrievalError(Exception):
    """Base error for this module."""


class MissingAPIKeyError(EvidenceRetrievalError):
    """Raised when TAVILY_API_KEY is not configured."""


class EvidenceSearchError(EvidenceRetrievalError):
    """Raised when the Tavily search request or response handling fails."""


class RerankerError(EvidenceRetrievalError):
    """Raised when cross-encoder loading or scoring fails."""


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


def _clean_results(raw_results: Any, limit: int | None) -> list[dict]:
    """Filter, deduplicate, sort, and optionally limit raw Tavily results.

    - Skips non-dict entries.
    - Drops entries with a missing/blank URL or blank content.
    - Deduplicates by stripped URL, keeping the highest relevance_score.
    - Sorts by relevance_score descending; stable for ties.
    - Returns at most ``limit`` records; ``None`` means no slicing
      (used for the pre-rerank candidate pool).
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
    if limit is None:
        return ranked
    return ranked[:limit]


def _apply_reranking(candidates: list[dict], claim: str, reranker: Any) -> list[dict]:
    """Attach ``rerank_score`` and sort candidates by it, highest first.

    The injected ``reranker`` must expose
    ``score(claim, texts) -> list[float]``. Non-``RerankerError``
    failures are wrapped in :class:`RerankerError` (chained); a
    length-mismatched score list also raises :class:`RerankerError`.
    """
    texts = [c["content"] for c in candidates]
    try:
        raw_scores = reranker.score(claim, texts)
    except RerankerError:
        raise
    except Exception as exc:
        raise RerankerError(f"Reranker inference failed: {exc}") from exc
    try:
        scores = [float(s) for s in raw_scores]
    except (TypeError, ValueError) as exc:
        raise RerankerError(f"Reranker returned non-numeric scores: {exc}") from exc
    if len(scores) != len(candidates):
        raise RerankerError(
            f"Reranker returned {len(scores)} scores for {len(candidates)} candidates."
        )
    for record, score in zip(candidates, scores):
        record["rerank_score"] = score
    return sorted(candidates, key=lambda r: r["rerank_score"], reverse=True)


def retrieve_evidence(
    claim: str,
    top_k: int = 5,
    *,
    reranker: Any | None = None,
    use_reranker: bool = True,
) -> list[dict]:
    """Retrieve and rerank web evidence for a factual claim.

    Args:
        claim: Factual claim to search for. Must be a non-empty string.
        top_k: Maximum records to return. Capped at 5 for this MVP.
        reranker: Optional scorer exposing
            ``score(claim, texts) -> list[float]``. Injected fakes keep
            unit tests offline. When ``None`` and ``use_reranker`` is
            true, a lazily-loaded cached default cross-encoder is used.
        use_reranker: When false, skip reranking and preserve the exact
            Stage 1 behavior (Tavily order, four record keys) for
            baseline comparisons.

    Returns:
        Reranked mode: at most ``min(top_k, 5)`` dicts with keys
        ``title``, ``url``, ``content``, ``relevance_score``,
        ``rerank_score``, sorted by ``rerank_score`` descending.
        Baseline mode: the four original keys sorted by
        ``relevance_score``. Either may be shorter (or empty) after
        cleaning; empty candidates never trigger a model load.

    Raises:
        ValueError: If ``claim`` is empty/not a string, or ``top_k``
            is not an integer >= 1.
        MissingAPIKeyError: If ``TAVILY_API_KEY`` is not configured.
        EvidenceSearchError: If client creation, the API call, or
            response handling fails. The original exception is chained.
        RerankerError: If reranker loading or scoring fails.
    """
    cleaned_claim, effective_top_k = _validate_inputs(claim, top_k)
    api_key = _get_api_key()
    try:
        client = TavilyClient(api_key=api_key)
    except MissingAPIKeyError:
        raise
    except Exception as exc:
        raise EvidenceSearchError(f"Tavily client initialization failed: {exc}") from exc
    fetch_n = CANDIDATE_LIMIT if use_reranker else effective_top_k
    try:
        response = client.search(query=cleaned_claim, max_results=fetch_n)
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
    if not use_reranker:
        return _clean_results(raw_results, effective_top_k)
    candidates = _clean_results(raw_results, None)
    if not candidates:
        return []
    scorer = reranker
    if scorer is None:
        # Local import: keeps module import side-effect free (no model
        # load) and avoids a hard dependency at import time.
        from src.retrieval.reranker import get_default_reranker

        scorer = get_default_reranker()
    reranked = _apply_reranking(candidates, cleaned_claim, scorer)
    return reranked[:effective_top_k]
