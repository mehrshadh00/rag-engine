import pytest

from rag_engine.evaluation import (
    evaluate_retrieval,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_recall_and_reciprocal_rank():
    retrieved = ["a", "x", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_precision_uses_actual_number_of_returned_results():
    assert precision_at_k(["a", "x"], {"a"}, 5) == pytest.approx(0.5)
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_no_relevant_results():
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert recall_at_k(["x"], set(), 3) == 0.0
    assert hit_rate_at_k(["x"], {"a"}, 3) == 0.0


def test_hit_rate_when_relevant_result_is_found():
    assert hit_rate_at_k(["x", "b"], {"a", "b"}, 2) == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_non_positive_k(k):
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["a"], {"a"}, k)


class FakeRetriever:
    def __init__(self, responses):
        self.responses = responses

    def retrieve(self, query, top_k):
        return [{"id": item} for item in self.responses[query][:top_k]]


def test_evaluate_retrieval_returns_per_query_and_averages():
    retriever = FakeRetriever({
        "question one": ["doc-a", "doc-x"],
        "question two": ["doc-y", "doc-b"],
    })
    dataset = [
        {"query": "question one", "relevant_ids": ["doc-a"]},
        {"query": "question two", "relevant_ids": ["doc-b"]},
    ]

    result = evaluate_retrieval(retriever, dataset, k=2)

    assert result["query_count"] == 2
    assert result["metrics"]["precision_at_k"] == pytest.approx(0.5)
    assert result["metrics"]["recall_at_k"] == pytest.approx(1.0)
    assert result["metrics"]["reciprocal_rank"] == pytest.approx(0.75)
    assert result["metrics"]["hit_rate_at_k"] == pytest.approx(1.0)


def test_evaluate_retrieval_handles_empty_dataset():
    result = evaluate_retrieval(FakeRetriever({}), [], k=3)
    assert result == {"k": 3, "query_count": 0, "metrics": {}, "results": []}


def test_evaluate_retrieval_rejects_empty_query():
    with pytest.raises(ValueError, match="query.*empty"):
        evaluate_retrieval(
            FakeRetriever({}),
            [{"query": " ", "relevant_ids": []}],
        )
