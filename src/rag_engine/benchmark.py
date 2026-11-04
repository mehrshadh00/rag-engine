"""Helpers for evaluating chunk retrieval at document level."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class DocumentLevelRetriever:
    """Optionally rerank chunks, then collapse them into unique source documents."""

    def __init__(
        self, retriever: Any, candidate_k: int = 1000, reranker: Any | None = None
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.retriever = retriever
        self.candidate_k = candidate_k
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunk_matches = self.retriever.retrieve(query, top_k=self.candidate_k)
        if self.reranker is not None and chunk_matches:
            # Rerank chunks before document deduplication so each source's best
            # evidence chunk determines its position in the document ranking.
            chunk_matches = self.reranker.rerank(query, chunk_matches)

        documents = []
        seen_sources = set()
        for match in chunk_matches:
            source = Path(match.get("metadata", {}).get("source", "unknown")).name
            if source in seen_sources:
                continue
            seen_sources.add(source)
            documents.append({
                "id": source,
                "distance": match.get("distance"),
                "rerank_score": match.get("rerank_score"),
            })
            if len(documents) >= top_k:
                break

        return documents
