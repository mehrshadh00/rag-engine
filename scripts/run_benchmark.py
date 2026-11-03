"""Run the versioned, document-level retrieval benchmark.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --k 2

The benchmark uses local embeddings and ChromaDB; it does not call an LLM API.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from rag_engine.config import settings
from rag_engine.pipeline import RAGPipeline
from rag_engine.retrieval import Retriever


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "benchmarks" / "corpus"
DATASET_PATH = ROOT / "benchmarks" / "datasets" / "rag_fundamentals_v1.json"


class DocumentLevelRetriever:
    """Collapse ranked chunks into unique source documents.

    The first occurrence of a source represents its highest-ranked chunk.
    This makes relevance labels document-level rather than chunk-level.
    """

    def __init__(self, retriever: Retriever, candidate_k: int = 1000):
        self.retriever = retriever
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        chunk_matches = self.retriever.retrieve(query, top_k=self.candidate_k)
        documents = []
        seen_sources = set()

        for match in chunk_matches:
            source = Path(match.get("metadata", {}).get("source", "unknown")).name
            if source in seen_sources:
                continue
            seen_sources.add(source)
            documents.append({"id": source, "distance": match.get("distance")})
            if len(documents) >= top_k:
                break

        return documents


def load_dataset() -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset.get("queries"):
        raise ValueError("Benchmark dataset contains no queries")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG Fundamentals retrieval benchmark")
    parser.add_argument("--k", type=int, default=3, help="Number of unique source documents to evaluate")
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")

    dataset = load_dataset()
    available_sources = {path.name for path in CORPUS_DIR.glob("*.md")}
    for case in dataset["queries"]:
        unknown = set(case["relevant_sources"]) - available_sources
        if unknown:
            raise ValueError(f"Unknown relevance sources for {case['id']}: {sorted(unknown)}")

    benchmark_config = replace(
        settings,
        chroma_path=str(ROOT / ".benchmark_chroma"),
        chroma_collection="rag_fundamentals_v1",
    )
    pipeline = RAGPipeline(config=benchmark_config)
    ingestion = pipeline.ingest(str(CORPUS_DIR))

    evaluation_cases = [
        {
            "query": case["query"],
            "relevant_ids": case["relevant_sources"],
        }
        for case in dataset["queries"]
    ]

    retriever = DocumentLevelRetriever(
        Retriever(pipeline.embedder, pipeline.vector_store),
        candidate_k=max(1, ingestion["chunks"]),
    )
    from rag_engine.evaluation import evaluate_retrieval

    report = evaluate_retrieval(retriever, evaluation_cases, k=args.k)
    for case, result in zip(dataset["queries"], report["results"]):
        result["id"] = case["id"]

    output = {
        "benchmark": dataset["name"],
        "version": dataset["version"],
        "evaluation_level": "document",
        "embedding_model": benchmark_config.embedding_model,
        "k": report["k"],
        "corpus_documents": len(available_sources),
        "indexed_chunks": ingestion["chunks"],
        "query_count": report["query_count"],
        "metrics": report["metrics"],
        "results": report["results"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
