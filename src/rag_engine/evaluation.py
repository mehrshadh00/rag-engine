"""Metrics and a small evaluation harness for retrieval quality."""

from collections.abc import Iterable, Mapping
from typing import Any


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the returned top-k items that are relevant."""
    _validate_k(k)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(item in relevant for item in top_k) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all known relevant items found in the top-k results."""
    _validate_k(k)
    if not relevant:
        return 0.0
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Return 1 when at least one relevant item appears in the top-k."""
    _validate_k(k)
    return float(bool(set(retrieved[:k]) & relevant))


def evaluate_retrieval(
    retriever: Any,
    dataset: Iterable[Mapping[str, Any]],
    k: int = 5,
) -> dict:
    """Evaluate a retriever on cases with query and relevant_ids fields.

    Each dataset item must contain:
      - query: the natural-language query
      - relevant_ids: an iterable of relevant document/chunk IDs
    """
    _validate_k(k)
    cases = []
    for index, item in enumerate(dataset):
        query = str(item["query"]).strip()
        if not query:
            raise ValueError(f"Evaluation query at index {index} is empty")

        relevant = set(item["relevant_ids"])
        matches = retriever.retrieve(query, top_k=k)
        retrieved_ids = [match["id"] for match in matches]

        cases.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": sorted(relevant),
            "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
            "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
            "hit_rate_at_k": hit_rate_at_k(retrieved_ids, relevant, k),
        })

    if not cases:
        return {"k": k, "query_count": 0, "metrics": {}, "results": []}

    metric_names = (
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "hit_rate_at_k",
    )
    averages = {
        name: sum(case[name] for case in cases) / len(cases)
        for name in metric_names
    }
    return {
        "k": k,
        "query_count": len(cases),
        "metrics": averages,
        "results": cases,
    }
