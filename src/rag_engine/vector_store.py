"""ChromaDB-backed persistent vector storage."""
class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to use the persistent vector store") from exc
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def delete_sources(self, sources: list[str]) -> None:
        """Remove previously indexed records for the supplied source paths."""
        for source in dict.fromkeys(sources):
            self.collection.delete(where={"source": source})

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        self.collection.upsert(
            ids=[item["id"] for item in chunks],
            documents=[item["text"] for item in chunks],
            metadatas=[item["metadata"] for item in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def get_all(self) -> list[dict]:
        """Return all persisted chunks to rebuild the BM25 index after restart."""
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.get(limit=count, include=["documents", "metadatas"])
        return [
            {"id": item_id, "text": document, "metadata": metadata or {}}
            for item_id, document, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for doc_id, document, metadata, distance in zip(ids, docs, metas, distances):
            matches.append({
                "id": doc_id, "text": document, "metadata": metadata or {},
                "distance": float(distance),
            })
        return matches
