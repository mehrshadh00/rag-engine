"""End-to-end ingestion and question-answering orchestration."""
from .bm25 import BM25Retriever
from .chunking import chunk_documents
from .config import settings
from .embeddings import SentenceTransformerEmbedder
from .generation import OpenAIAnswerGenerator
from .hybrid import HybridRetriever
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
        existing = self.vector_store.get_all() if hasattr(self.vector_store, "get_all") else []
        self.sparse_retriever = BM25Retriever(existing)

    def _refresh_sparse_index(self, fallback_chunks: list[dict] | None = None) -> None:
        documents = (
            self.vector_store.get_all()
            if hasattr(self.vector_store, "get_all")
            else (fallback_chunks or [])
        )
        self.sparse_retriever = BM25Retriever(documents)

    def ingest(self, directory: str) -> dict:
        documents = load_directory(directory)
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            return {"documents": len(documents), "chunks": 0}
        sources = list({
            str(item.get("metadata", {}).get("source", "unknown")) for item in chunks
        })
        if hasattr(self.vector_store, "delete_sources"):
            self.vector_store.delete_sources(sources)
        embeddings = self.embedder.embed_documents([item["text"] for item in chunks])
        stored = self.vector_store.add(chunks, embeddings)
        self._refresh_sparse_index(chunks)
        return {"documents": len(documents), "chunks": stored}

    def ask(self, question: str, top_k: int | None = None) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        result_count = top_k or self.config.top_k
        dense_retriever = Retriever(self.embedder, self.vector_store)
        mode = getattr(self.config, "retrieval_mode", "hybrid")
        if mode == "dense":
            retriever = dense_retriever
            matches = retriever.retrieve(question, top_k=result_count)
        elif mode == "bm25":
            retriever = self.sparse_retriever
            matches = retriever.retrieve(question, top_k=result_count)
        else:
            retriever = HybridRetriever(
                dense_retriever, self.sparse_retriever,
                candidate_k=getattr(self.config, "hybrid_candidate_k", 50),
                dense_weight=getattr(self.config, "dense_weight", 1.0),
                sparse_weight=getattr(self.config, "sparse_weight", 1.0),
                rrf_k=getattr(self.config, "rrf_k", 60),
            )
            matches = retriever.retrieve(
                question,
                top_k=max(result_count, getattr(self.config, "hybrid_candidate_k", 50)),
            )[:result_count]
        context = build_context(matches)
        sources = [
            {
                "id": item["id"],
                "source": item.get("metadata", {}).get("source"),
                "page": item.get("metadata", {}).get("page"),
                "distance": item.get("distance"),
                "rrf_score": item.get("rrf_score"),
            }
            for item in matches
        ]
        if not context:
            return {
                "answer": "I could not find relevant information in the indexed documents.",
                "sources": [],
            }
        generator = self.generator
        if generator is None:
            generator = OpenAIAnswerGenerator(self.config.openai_api_key, self.config.openai_model)
        answer = generator.answer(question, context)
        return {"answer": answer, "sources": sources}
