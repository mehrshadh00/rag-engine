import pytest
from rag_engine.chunking import chunk_text


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
