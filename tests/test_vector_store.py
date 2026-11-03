from rag_engine.vector_store import ChromaVectorStore

class FakeCollection:
    def __init__(self, records):
        self.records = records
    def count(self):
        return len(self.records)
    def get(self, limit, include):
        assert limit == len(self.records)
        assert include == ["documents", "metadatas"]
        return {
            "ids": [item["id"] for item in self.records],
            "documents": [item["text"] for item in self.records],
            "metadatas": [item["metadata"] for item in self.records],
        }

def test_get_all_returns_persisted_chunks():
    store = object.__new__(ChromaVectorStore)
    store.collection = FakeCollection([
        {"id": "c1", "text": "First", "metadata": {"source": "a.md"}},
        {"id": "c2", "text": "Second", "metadata": {"source": "b.md"}},
    ])
    assert store.get_all() == [
        {"id": "c1", "text": "First", "metadata": {"source": "a.md"}},
        {"id": "c2", "text": "Second", "metadata": {"source": "b.md"}},
    ]

def test_get_all_handles_empty_collection():
    store = object.__new__(ChromaVectorStore)
    store.collection = FakeCollection([])
    assert store.get_all() == []
