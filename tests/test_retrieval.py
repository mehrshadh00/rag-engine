from rag_engine.retrieval import build_context


def test_context_includes_source_and_page():
    context = build_context([{
        "id": "chunk-1", "text": "Evidence text",
        "metadata": {"source": "paper.pdf", "page": 3},
    }])
    assert "[1] Source: paper.pdf, page 3" in context
    assert "Evidence text" in context


def test_empty_context_is_empty_string():
    assert build_context([]) == ""
