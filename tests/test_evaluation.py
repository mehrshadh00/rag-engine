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


def test_unanswerable_queries_are_excluded_from_relevance_metrics():
    retriever = FakeRetriever({"not in corpus": ["doc-a"]})
    result = evaluate_retrieval(
        retriever,
        [{
            "query": "not in corpus",
            "relevant_ids": [],
            "answerable": False,
            "category": "unanswerable",
        }],
        k=2,
    )

    assert result["metrics"] == {}
    assert result["diagnostics"]["unanswerable_false_positive_rate_at_k"] == 1.0
    assert result["answerable_query_count"] == 0
    assert result["unanswerable_query_count"] == 1
    assert result["results"][0]["precision_at_k"] is None


def test_multi_hop_complete_evidence_requires_every_required_source():
    retriever = FakeRetriever({
        "multi-hop": ["doc-a", "doc-x"],
        "complete": ["doc-a", "doc-b"],
    })
    dataset = [
        {
            "query": "multi-hop",
            "relevant_ids": ["doc-a", "doc-b"],
            "required_ids": ["doc-a", "doc-b"],
            "answerable": True,
            "category": "multi_hop",
        },
        {
            "query": "complete",
            "relevant_ids": ["doc-a", "doc-b"],
            "required_ids": ["doc-a", "doc-b"],
            "answerable": True,
            "category": "multi_hop",
        },
    ]

    result = evaluate_retrieval(retriever, dataset, k=2)

    assert result["diagnostics"]["multi_hop_complete_evidence_rate_at_k"] == 0.5
    assert result["by_category"]["multi_hop"]["query_count"] == 2


def test_answerability_labels_are_validated():
    with pytest.raises(ValueError, match="no relevant_ids"):
        evaluate_retrieval(
            FakeRetriever({}),
            [{"query": "bad label", "relevant_ids": [], "answerable": True}],
        )
    with pytest.raises(ValueError, match="must have relevant_ids"):
        evaluate_retrieval(
            FakeRetriever({}),
            [{"query": "bad label", "relevant_ids": ["doc-a"], "answerable": False}],
        )


def test_multi_hop_requires_at_least_two_required_sources():
    with pytest.raises(ValueError, match="at least two"):
        evaluate_retrieval(
            FakeRetriever({"question": ["doc-a"]}),
            [{
                "query": "question",
                "relevant_ids": ["doc-a"],
                "required_ids": ["doc-a"],
                "answerable": True,
                "category": "multi_hop",
            }],
        )
