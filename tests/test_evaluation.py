import pytest
from rag_engine.evaluation import precision_at_k, recall_at_k, reciprocal_rank


def test_precision_recall_and_reciprocal_rank():
    retrieved = ["a", "x", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_no_relevant_results():
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert recall_at_k(["x"], set(), 3) == 0.0
