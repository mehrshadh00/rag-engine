import pytest
from rag_engine.chunking import chunk_documents, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []


def test_short_text_returns_one_chunk():
    assert chunk_text("hello world", chunk_size=20, overlap=2) == ["hello world"]


def test_long_text_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(40))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(chunks)
    assert any(set(a.split()) & set(b.split()) for a, b in zip(chunks, chunks[1:]))


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)


def test_chunk_ids_are_stable_and_source_aware():
    documents = [
        {"text": "alpha beta gamma", "metadata": {"source": "a.txt"}},
        {"text": "alpha beta gamma", "metadata": {"source": "b.txt"}},
    ]
    first = chunk_documents(documents, chunk_size=30, overlap=5)
    second = chunk_documents(list(reversed(documents)), chunk_size=30, overlap=5)

    assert first[0]["id"] != first[1]["id"]
    assert {item["id"] for item in first} == {item["id"] for item in second}
