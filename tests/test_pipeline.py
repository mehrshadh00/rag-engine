from types import SimpleNamespace

import pytest

from rag_engine.pipeline import RAGPipeline


class EmptyEmbedder:
    def embed_query(self, question):
        return [0.1, 0.2]


class EmptyVectorStore:
    def query(self, embedding, top_k=5):
        return []


def make_pipeline():
    config = SimpleNamespace(
        embedding_model="test-model",
        chroma_path="unused",
        chroma_collection="unused",
        chunk_size=100,
        chunk_overlap=10,
        top_k=5,
        openai_api_key=None,
        openai_model="test-model",
    )
    return RAGPipeline(
        config=config,
        embedder=EmptyEmbedder(),
        vector_store=EmptyVectorStore(),
    )


def test_empty_retrieval_does_not_require_openai_key():
    result = make_pipeline().ask("What is in the documents?")
    assert result["sources"] == []
    assert "could not find relevant information" in result["answer"]


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        make_pipeline().ask("   ")


class PopulatedVectorStore:
    def get_all(self):
        return [
            {
                "id": "chunk-1",
                "text": "Python is a programming language.",
                "metadata": {"source": "python.md", "page": 1},
            }
        ]


class StaticGenerator:
    def answer(self, question, context):
        return "Python is a programming language."


def test_bm25_mode_uses_sparse_retriever():
    config = SimpleNamespace(
        embedding_model="test-model",
        chroma_path="unused",
        chroma_collection="unused",
        chunk_size=100,
        chunk_overlap=10,
        top_k=1,
        retrieval_mode="bm25",
        openai_api_key=None,
        openai_model="test-model",
    )
    pipeline = RAGPipeline(
        config=config,
        embedder=EmptyEmbedder(),
        vector_store=PopulatedVectorStore(),
        generator=StaticGenerator(),
    )

    result = pipeline.ask("Which programming language is Python?")

    assert result["answer"] == "Python is a programming language."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "python.md"

