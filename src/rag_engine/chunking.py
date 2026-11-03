"""Deterministic character-based chunking with overlap."""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a nearby word boundary without creating tiny chunks.
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    chunks = []
    for document_index, document in enumerate(documents):
        for chunk_index, chunk in enumerate(chunk_text(document["text"], chunk_size, overlap)):
            chunks.append({
                "id": f"{document_index}:{chunk_index}",
                "text": chunk,
                "metadata": {**document.get("metadata", {}), "chunk_index": chunk_index},
            })
    return chunks
