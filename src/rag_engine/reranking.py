"""Cross-encoder reranking for retrieved document chunks."""
from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    """Rerank retrieved chunks by jointly scoring each query-document pair."""

    def __init__(self, model_name: str, *, encoder: Any | None = None):
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if encoder is None:
            from sentence_transformers import CrossEncoder

            encoder = CrossEncoder(model_name)
        self.model_name = model_name
        self.encoder = encoder

    def rerank(
        self, query: str, documents: list[dict], *, top_k: int | None = None
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if not documents:
            return []

        pairs = [(query, str(item.get("text", ""))) for item in documents]
        scores = self.encoder.predict(pairs)
        if len(scores) != len(documents):
            raise ValueError("Cross-encoder returned an unexpected number of scores")

        ranked = [
            (float(score), index, {**document, "rerank_score": float(score)})
            for index, (score, document) in enumerate(zip(scores, documents))
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        results = [item[2] for item in ranked]
        return results if top_k is None else results[:top_k]
