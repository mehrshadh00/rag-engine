"""ChromaDB-backed persistent vector storage."""

import chromadb


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[item["id"] for item in chunks],
            documents=[item["text"] for item in chunks],
            metadatas=[item["metadata"] for item in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for doc_id, document, metadata, distance in zip(ids, docs, metas, distances):
            matches.append({
                "id": doc_id,
                "text": document,
                "metadata": metadata or {},
                "distance": float(distance),
            })
        return matches
