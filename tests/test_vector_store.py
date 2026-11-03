import pytest

from rag_engine.vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self, count=0, records=None):
        self.records = records or []
        self.item_count = len(self.records) if records is not None else count
        self.deleted = []
        self.query_called = False

    def count(self):
        return self.item_count

    def query(self, **kwargs):
        self.query_called = True
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, **kwargs):
        self.deleted.append(kwargs)

    def get(self, limit, include):
        assert limit == len(self.records)
        assert include == ["documents", "metadatas"]
        return {
            "ids": [item["id"] for item in self.records],
            "documents": [item["text"] for item in self.records],
            "metadatas": [item["metadata"] for item in self.records],
        }


def make_store(count=0):
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = FakeCollection(count)
    return store


def test_empty_collection_returns_no_matches_without_querying():
    store = make_store()
    assert store.query([0.1, 0.2]) == []
    assert store.collection.query_called is False


def test_non_positive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be positive"):
        make_store().query([0.1], top_k=0)


def test_delete_sources_removes_each_unique_source():
    store = make_store()
    store.delete_sources(["a.txt", "b.txt", "a.txt"])
    assert store.collection.deleted == [
        {"where": {"source": "a.txt"}},
        {"where": {"source": "b.txt"}},
    ]


def test_add_rejects_mismatched_embedding_count():
    store = make_store()
    chunks = [{"id": "one", "text": "hello", "metadata": {"source": "a.txt"}}]
    with pytest.raises(ValueError, match="exactly one embedding"):
        store.add(chunks, [])


def test_get_all_returns_persisted_chunks():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = FakeCollection(records=[
        {"id": "c1", "text": "First", "metadata": {"source": "a.md"}},
        {"id": "c2", "text": "Second", "metadata": {"source": "b.md"}},
    ])
    assert store.get_all() == [
        {"id": "c1", "text": "First", "metadata": {"source": "a.md"}},
        {"id": "c2", "text": "Second", "metadata": {"source": "b.md"}},
    ]


def test_get_all_handles_empty_collection():
    store = make_store()
    assert store.get_all() == []
