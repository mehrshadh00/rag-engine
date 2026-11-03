"""Environment-based application configuration."""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    hybrid_candidate_k: int = int(os.getenv("HYBRID_CANDIDATE_K", "50"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "1.0"))
    sparse_weight: float = float(os.getenv("SPARSE_WEIGHT", "1.0"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive")
        if self.retrieval_mode not in {"dense", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be 'dense' or 'hybrid'")
        if self.hybrid_candidate_k <= 0:
            raise ValueError("HYBRID_CANDIDATE_K must be positive")
        if self.rrf_k < 0:
            raise ValueError("RRF_K cannot be negative")
        if self.dense_weight <= 0 or self.sparse_weight <= 0:
            raise ValueError("DENSE_WEIGHT and SPARSE_WEIGHT must be positive")

settings = Settings()
settings.validate()
