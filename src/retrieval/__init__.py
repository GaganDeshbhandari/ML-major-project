"""Evidence retrieval package (Member 2).

Fetches web evidence for a factual claim via the Tavily Search API.
Scope is intentionally narrow: search + clean + select. No stance
classification, embeddings, reranking, or model training here.
"""

from src.retrieval.evidence_retriever import (
    EvidenceRetrievalError,
    EvidenceSearchError,
    MissingAPIKeyError,
    retrieve_evidence,
)

__all__ = [
    "EvidenceRetrievalError",
    "EvidenceSearchError",
    "MissingAPIKeyError",
    "retrieve_evidence",
]
