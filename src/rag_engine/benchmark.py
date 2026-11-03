"""Helpers for evaluating retrieval results at document level."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class DocumentLevelRetriever:
    """Collapse ranked chunks into unique source documents.

    The first occurrence of a source represents its highest-ranked chunk.
    This makes relevance labels document-level rather than chunk-level.
    """

    def __init__(self, retriever: Any, candidate_k: int = 1000):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.retriever = retriever
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunk_matches = self.retriever.retrieve(query, top_k=self.candidate_k)
        documents = []
        seen_sources = set()

        for match in chunk_matches:
            source = Path(match.get("metadata", {}).get("source", "unknown")).name
            if source in seen_sources:
                continue
            seen_sources.add(source)
            documents.append({"id": source, "distance": match.get("distance")})
            if len(documents) >= top_k:
                break

        return documents
