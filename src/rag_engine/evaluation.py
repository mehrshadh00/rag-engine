"""Metrics and a small evaluation harness for retrieval quality."""

from collections import defaultdict
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
    """Return 1 when at least one relevant result appears in the top-k."""
    _validate_k(k)
    return float(bool(set(retrieved[:k]) & relevant))


def _summarize(cases: list[dict], k: int) -> dict:
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    multi_hop = [case for case in cases if case["required_ids"]]

    metric_names = (
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "hit_rate_at_k",
    )
    metrics = {
        name: sum(case[name] for case in answerable) / len(answerable)
        for name in metric_names
    } if answerable else {}

    diagnostics = {
        "unanswerable_false_positive_rate_at_k": (
            sum(bool(case["retrieved_ids"][:k]) for case in unanswerable) / len(unanswerable)
            if unanswerable else None
        ),
        "multi_hop_complete_evidence_rate_at_k": (
            sum(case["complete_evidence_at_k"] for case in multi_hop) / len(multi_hop)
            if multi_hop else None
        ),
    }
    return {
        "query_count": len(cases),
        "answerable_query_count": len(answerable),
        "unanswerable_query_count": len(unanswerable),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def evaluate_retrieval(
    retriever: Any,
    dataset: Iterable[Mapping[str, Any]],
    k: int = 5,
) -> dict:
    """Evaluate document retrieval with answerability and evidence-set labels.

    Each item requires query and relevant_ids. Optional fields:
      - answerable: whether the corpus contains sufficient evidence
      - category: query category, used for grouped reporting
      - required_ids: all sources required for a complete multi-hop result

    Conventional relevance metrics are averaged over answerable queries only.
    Unanswerable queries are measured separately by the fraction for which any
    result was returned. Multi-hop completeness requires every required source
    to appear in the top-k result list.
    """
    _validate_k(k)
    cases = []
    for index, item in enumerate(dataset):
        query = str(item["query"]).strip()
        if not query:
            raise ValueError(f"Evaluation query at index {index} is empty")

        relevant = set(item["relevant_ids"])
        answerable = bool(item.get("answerable", bool(relevant)))
        category = str(item.get("category", "unspecified"))
        required = set(item.get("required_ids", []))

        if answerable and not relevant:
            raise ValueError(f"Answerable evaluation query at index {index} has no relevant_ids")
        if not answerable and relevant:
            raise ValueError(f"Unanswerable evaluation query at index {index} has relevant_ids")
        if not required.issubset(relevant):
            raise ValueError(f"Required evidence at index {index} must be relevant")
        if category == "multi_hop" and len(required) < 2:
            raise ValueError(f"Multi-hop evaluation query at index {index} needs at least two required_ids")

        matches = retriever.retrieve(query, top_k=k)
        retrieved_ids = [match["id"] for match in matches]
        complete = bool(required) and required.issubset(set(retrieved_ids[:k]))

        case = {
            "query": query,
            "category": category,
            "answerable": answerable,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": sorted(relevant),
            "required_ids": sorted(required),
            "complete_evidence_at_k": complete if required else None,
            "precision_at_k": precision_at_k(retrieved_ids, relevant, k) if answerable else None,
            "recall_at_k": recall_at_k(retrieved_ids, relevant, k) if answerable else None,
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant) if answerable else None,
            "hit_rate_at_k": hit_rate_at_k(retrieved_ids, relevant, k) if answerable else None,
        }
        cases.append(case)

    if not cases:
        return {"k": k, "query_count": 0, "metrics": {}, "results": []}

    summary = _summarize(cases, k)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["category"]].append(case)
    by_category = {
        category: _summarize(category_cases, k)
        for category, category_cases in sorted(grouped.items())
    }

    return {
        "k": k,
        **summary,
        "by_category": by_category,
        "results": cases,
    }
