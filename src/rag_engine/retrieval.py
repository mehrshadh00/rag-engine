"""Semantic retrieval and source-aware context formatting."""


class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedder.embed_query(question)
        return self.vector_store.query(query_vector, top_k=top_k)


def build_context(matches: list[dict]) -> str:
    sections = []
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        source = metadata.get("source", "unknown source")
        page = metadata.get("page")
        location = f"{source}, page {page}" if page else source
        sections.append(f"[{index}] Source: {location}\n{match['text']}")
    return "\n\n".join(sections)
