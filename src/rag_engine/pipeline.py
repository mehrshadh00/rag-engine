"""End-to-end ingestion and question-answering orchestration."""

from .chunking import chunk_documents
from .config import settings
from .embeddings import SentenceTransformerEmbedder
from .generation import OpenAIAnswerGenerator
from .ingestion import load_directory
from .retrieval import Retriever, build_context
from .vector_store import ChromaVectorStore


class RAGPipeline:
    def __init__(self, config=settings, embedder=None, vector_store=None, generator=None):
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbedder(config.embedding_model)
        self.vector_store = vector_store or ChromaVectorStore(
            config.chroma_path, config.chroma_collection
        )
        self.generator = generator

    def ingest(self, directory: str) -> dict:
        documents = load_directory(directory)
        chunks = chunk_documents(
            documents, self.config.chunk_size, self.config.chunk_overlap
        )
        if not chunks:
            return {"documents": len(documents), "chunks": 0}

        sources = list({
            str(item.get("metadata", {}).get("source", "unknown"))
            for item in chunks
        })
        if hasattr(self.vector_store, "delete_sources"):
            self.vector_store.delete_sources(sources)

        embeddings = self.embedder.embed_documents([item["text"] for item in chunks])
        stored = self.vector_store.add(chunks, embeddings)
        return {"documents": len(documents), "chunks": stored}

    def ask(self, question: str, top_k: int | None = None) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        retriever = Retriever(self.embedder, self.vector_store)
        matches = retriever.retrieve(question, top_k or self.config.top_k)
        context = build_context(matches)
        sources = [
            {"id": item["id"], "source": item.get("metadata", {}).get("source"),
             "page": item.get("metadata", {}).get("page"), "distance": item.get("distance")}
            for item in matches
        ]

        if not context:
            return {
                "answer": "I could not find relevant information in the indexed documents.",
                "sources": [],
            }

        generator = self.generator
        if generator is None:
            generator = OpenAIAnswerGenerator(
                self.config.openai_api_key, self.config.openai_model
            )
        answer = generator.answer(question, context)
        return {"answer": answer, "sources": sources}
