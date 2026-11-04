from rag_engine.benchmark import DocumentLevelRetriever


class FakeChunkRetriever:
    def __init__(self, matches):
        self.matches = matches
        self.requested_k = None

    def retrieve(self, query, top_k):
        self.requested_k = top_k
        return self.matches[:top_k]


class ReverseChunkReranker:
    def rerank(self, query, documents):
        return [
            {**item, "rerank_score": float(index)}
            for index, item in reversed(list(enumerate(documents)))
        ]


def test_document_level_retriever_deduplicates_sources_in_rank_order():
    chunks = FakeChunkRetriever([
        {"id": "a-0", "metadata": {"source": "/corpus/a.md"}, "distance": 0.1},
        {"id": "a-1", "metadata": {"source": "/corpus/a.md"}, "distance": 0.2},
        {"id": "b-0", "metadata": {"source": "/corpus/b.md"}, "distance": 0.3},
        {"id": "c-0", "metadata": {"source": "/corpus/c.md"}, "distance": 0.4},
    ])
    retriever = DocumentLevelRetriever(chunks, candidate_k=4)
    results = retriever.retrieve("question", top_k=2)
    assert [item["id"] for item in results] == ["a.md", "b.md"]
    assert chunks.requested_k == 4


def test_document_level_retriever_returns_available_unique_sources():
    chunks = FakeChunkRetriever([
        {"id": "a-0", "metadata": {"source": "/corpus/a.md"}, "distance": 0.1},
        {"id": "a-1", "metadata": {"source": "/corpus/a.md"}, "distance": 0.2},
    ])
    retriever = DocumentLevelRetriever(chunks, candidate_k=2)
    assert [item["id"] for item in retriever.retrieve("question", top_k=3)] == ["a.md"]


def test_reranking_happens_before_document_deduplication():
    chunks = FakeChunkRetriever([
        {"id": "a-0", "text": "weak evidence", "metadata": {"source": "/corpus/a.md"}},
        {"id": "b-0", "text": "strong evidence", "metadata": {"source": "/corpus/b.md"}},
        {"id": "a-1", "text": "strongest evidence", "metadata": {"source": "/corpus/a.md"}},
    ])
    retriever = DocumentLevelRetriever(
        chunks, candidate_k=3, reranker=ReverseChunkReranker()
    )

    results = retriever.retrieve("question", top_k=2)

    assert [item["id"] for item in results] == ["a.md", "b.md"]
    assert results[0]["rerank_score"] == 0.0
