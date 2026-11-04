import pytest

from rag_engine.reranking import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def sample_documents():
    return [
        {"id": "a", "text": "A broad overview.", "metadata": {"source": "a.md"}},
        {"id": "b", "text": "A precise answer about BM25.", "metadata": {"source": "b.md"}},
        {"id": "c", "text": "Unrelated content.", "metadata": {"source": "c.md"}},
    ]


def test_cross_encoder_reranks_by_relevance_score_and_preserves_metadata():
    encoder = FakeCrossEncoder([0.1, 0.9, 0.3])
    reranker = CrossEncoderReranker("test-model", encoder=encoder)

    results = reranker.rerank("What is BM25?", sample_documents(), top_k=2)

    assert [item["id"] for item in results] == ["b", "c"]
    assert results[0]["rerank_score"] == pytest.approx(0.9)
    assert results[0]["metadata"]["source"] == "b.md"
    assert encoder.pairs[1] == ("What is BM25?", "A precise answer about BM25.")


def test_cross_encoder_keeps_original_order_when_scores_tie():
    reranker = CrossEncoderReranker(
        "test-model", encoder=FakeCrossEncoder([0.5, 0.5, 0.5])
    )

    assert [item["id"] for item in reranker.rerank("query", sample_documents())] == [
        "a", "b", "c"
    ]


def test_cross_encoder_validates_inputs_and_score_count():
    with pytest.raises(ValueError, match="model_name"):
        CrossEncoderReranker(" ", encoder=FakeCrossEncoder([]))
    reranker = CrossEncoderReranker("test-model", encoder=FakeCrossEncoder([0.1]))
    with pytest.raises(ValueError, match="query"):
        reranker.rerank(" ", sample_documents())
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("query", sample_documents(), top_k=0)
    with pytest.raises(ValueError, match="unexpected number"):
        reranker.rerank("query", sample_documents())
