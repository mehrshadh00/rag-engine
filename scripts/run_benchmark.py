"""Compare dense-only and hybrid retrieval on the versioned benchmark."""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
from rag_engine.benchmark import DocumentLevelRetriever
from rag_engine.config import settings
from rag_engine.evaluation import evaluate_retrieval
from rag_engine.hybrid import HybridRetriever
from rag_engine.pipeline import RAGPipeline
from rag_engine.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "benchmarks" / "corpus"
DATASET_PATH = ROOT / "benchmarks" / "datasets" / "rag_fundamentals_v1.json"

def load_dataset() -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset.get("queries"):
        raise ValueError("Benchmark dataset contains no queries")
    return dataset

def evaluate_mode(retriever, cases, k: int, candidate_k: int) -> dict:
    adapter = DocumentLevelRetriever(retriever, candidate_k=candidate_k)
    return evaluate_retrieval(adapter, cases, k=k)

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense and hybrid RAG retrieval")
    parser.add_argument("--k", type=int, default=2, help="Unique source documents to evaluate")
    parser.add_argument(
        "--mode", choices=("both", "dense", "hybrid"), default="both",
        help="Retrieval mode to evaluate (default: both)",
    )
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
        settings, chroma_path=str(ROOT / ".benchmark_chroma"),
        chroma_collection="rag_fundamentals_v1",
    )
    pipeline = RAGPipeline(config=benchmark_config)
    ingestion = pipeline.ingest(str(CORPUS_DIR))
    candidate_k = max(1, ingestion["chunks"])
    cases = [
        {"query": item["query"], "relevant_ids": item["relevant_sources"]}
        for item in dataset["queries"]
    ]
    dense = Retriever(pipeline.embedder, pipeline.vector_store)
    hybrid = HybridRetriever(
        dense, pipeline.sparse_retriever, candidate_k=candidate_k,
        dense_weight=benchmark_config.dense_weight,
        sparse_weight=benchmark_config.sparse_weight,
        rrf_k=benchmark_config.rrf_k,
    )

    modes = {}
    if args.mode in {"both", "dense"}:
        modes["dense"] = evaluate_mode(dense, cases, args.k, candidate_k)
    if args.mode in {"both", "hybrid"}:
        modes["hybrid"] = evaluate_mode(hybrid, cases, args.k, candidate_k)
    for report in modes.values():
        for source_case, result in zip(dataset["queries"], report["results"]):
            result["id"] = source_case["id"]

    output = {
        "benchmark": dataset["name"],
        "version": dataset["version"],
        "evaluation_level": "document",
        "embedding_model": benchmark_config.embedding_model,
        "retrieval_modes": list(modes),
        "k": args.k,
        "corpus_documents": len(available_sources),
        "indexed_chunks": ingestion["chunks"],
        "query_count": len(dataset["queries"]),
        "metrics": {mode: report["metrics"] for mode, report in modes.items()},
        "results": {mode: report["results"] for mode, report in modes.items()},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
