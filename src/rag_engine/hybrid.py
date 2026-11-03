"""Rank-fusion retrieval combining dense and sparse search."""
from __future__ import annotations
from typing import Any

class HybridRetriever:
    """Combine dense and sparse ranked lists with Reciprocal Rank Fusion."""
    def __init__(
        self, dense_retriever: Any, sparse_retriever: Any, *,
        candidate_k: int = 50, dense_weight: float = 1.0,
        sparse_weight: float = 1.0, rrf_k: int = 60,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if dense_weight <= 0 or sparse_weight <= 0:
            raise ValueError("retrieval weights must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_k = candidate_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        dense = self.dense_retriever.retrieve(query, top_k=self.candidate_k)
        sparse = self.sparse_retriever.retrieve(query, top_k=self.candidate_k)
        fused: dict[str, dict] = {}
        for channel, matches, weight in (
            ("dense", dense, self.dense_weight),
            ("sparse", sparse, self.sparse_weight),
        ):
            for rank, match in enumerate(matches, start=1):
                item_id = match["id"]
                if item_id not in fused:
                    fused[item_id] = {
                        **match, "rrf_score": 0.0,
                        "dense_rank": None, "sparse_rank": None,
                    }
                fused[item_id]["rrf_score"] += weight / (self.rrf_k + rank)
                fused[item_id][f"{channel}_rank"] = rank
        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                min(rank for rank in (item["dense_rank"], item["sparse_rank"]) if rank is not None),
                item["id"],
            ),
        )
        return ranked[:top_k]
