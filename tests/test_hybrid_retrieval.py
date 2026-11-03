import pytest
from rag_engine.bm25 import BM25Retriever, tokenize
from rag_engine.hybrid import HybridRetriever

def sample_docs():
    return [
        {"id": "a", "text": "BM25 uses term frequency and inverse document frequency.", "metadata": {"source": "a.md"}},
        {"id": "b", "text": "Dense retrieval represents text with semantic embedding vectors.", "metadata": {"source": "b.md"}},
        {"id": "c", "text": "Reciprocal rank fusion combines ranked search result lists.", "metadata": {"source": "c.md"}},
    ]

def test_tokenize_is_case_insensitive_and_ignores_punctuation():
    assert tokenize("BM25, RAG-engine!") == ["bm25", "rag", "engine"]

def test_bm25_ranks_exact_term_match_first():
    results = BM25Retriever(sample_docs()).retrieve("inverse document frequency", top_k=3)
    assert results[0]["id"] == "a"
    assert results[0]["bm25_score"] > results[1]["bm25_score"]

def test_bm25_returns_empty_when_no_terms_match():
    assert BM25Retriever(sample_docs()).retrieve("quantum banana", top_k=3) == []

def test_bm25_validates_parameters():
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(sample_docs(), k1=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        BM25Retriever(sample_docs(), b=1.5)
    with pytest.raises(ValueError, match="top_k"):
        BM25Retriever(sample_docs()).retrieve("BM25", top_k=0)

class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids
    def retrieve(self, query, top_k):
        return [{"id": item, "text": item, "metadata": {}} for item in self.ids[:top_k]]

def test_rrf_rewards_consensus_between_ranked_lists():
    dense = FakeRetriever(["dense-first", "shared", "dense-third"])
    sparse = FakeRetriever(["shared", "sparse-second", "dense-first"])
    results = HybridRetriever(dense, sparse, candidate_k=3, rrf_k=0).retrieve("q", top_k=3)
    assert results[0]["id"] == "shared"
    assert results[0]["dense_rank"] == 2
    assert results[0]["sparse_rank"] == 1
    assert results[1]["id"] == "dense-first"

def test_hybrid_validates_configuration():
    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), candidate_k=0)
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), dense_weight=0)
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), rrf_k=-1)

def test_hybrid_returns_union_without_duplicates():
    retriever = HybridRetriever(FakeRetriever(["a", "b"]), FakeRetriever(["b", "c"]), candidate_k=3)
    results = retriever.retrieve("q", top_k=5)
    assert len(results) == 3
    assert {item["id"] for item in results} == {"a", "b", "c"}
