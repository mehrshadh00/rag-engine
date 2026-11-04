"""Compare dense, BM25, hybrid, and reranked retrieval on a versioned dataset."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from rag_engine.benchmark import DocumentLevelRetriever
from rag_engine.config import settings
from rag_engine.evaluation import evaluate_retrieval
from rag_engine.hybrid import HybridRetriever
from rag_engine.pipeline import RAGPipeline
from rag_engine.reranking import CrossEncoderReranker
from rag_engine.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "benchmarks" / "corpus_v2"
DEFAULT_DATASET_PATH = ROOT / "benchmarks" / "datasets" / "rag_fundamentals_v2.json"
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset.get("queries"), list) or not dataset["queries"]:
        raise ValueError("Benchmark dataset contains no queries")
    return dataset


def validate_dataset(dataset: dict, available_sources: set[str]) -> None:
    """Validate IDs, labels, splits, and source references before indexing."""
    seen_ids = set()
    allowed_splits = {"dev", "test"}
    for index, item in enumerate(dataset["queries"]):
        query_id = item.get("id")
        if not query_id or query_id in seen_ids:
            raise ValueError(f"Missing or duplicate query id at index {index}: {query_id!r}")
        seen_ids.add(query_id)
        if not str(item.get("query", "")).strip():
            raise ValueError(f"Empty query text for {query_id}")
        if item.get("split") not in allowed_splits:
            raise ValueError(f"Invalid split for {query_id}: {item.get('split')!r}")
        if not isinstance(item.get("answerable"), bool):
            raise ValueError(f"Missing boolean answerable label for {query_id}")
        relevant = set(item.get("relevant_sources", []))
        required = set(item.get("required_sources", []))
        unknown = (relevant | required) - available_sources
        if unknown:
            raise ValueError(f"Unknown relevance sources for {query_id}: {sorted(unknown)}")
        if not required.issubset(relevant):
            raise ValueError(f"Required sources must be relevant for {query_id}")
        if item["answerable"] and not relevant:
            raise ValueError(f"Answerable query has no relevant sources: {query_id}")
        if not item["answerable"] and (relevant or required):
            raise ValueError(f"Unanswerable query must have empty source labels: {query_id}")
        if item.get("category") == "multi_hop" and len(required) < 2:
            raise ValueError(f"Multi-hop query needs at least two required sources: {query_id}")


def evaluate_mode(
    retriever, cases: list[dict], k: int, candidate_k: int, reranker=None
) -> dict:
    adapter = DocumentLevelRetriever(
        retriever, candidate_k=candidate_k, reranker=reranker
    )
    return evaluate_retrieval(adapter, cases, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense, BM25, hybrid, and cross-encoder reranked retrieval"
    )
    parser.add_argument("--k", type=int, default=3,
                        help="Number of unique source documents in the final ranking")
    parser.add_argument(
        "--mode",
        choices=("all", "dense", "bm25", "hybrid", "hybrid-reranked"),
        default="all",
        help="Retrieval method to evaluate; all includes reranked hybrid",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    dataset_path = args.dataset.resolve()
    corpus_dir = args.corpus_dir.resolve()
    if not dataset_path.is_file():
        parser.error(f"Dataset file does not exist: {dataset_path}")
    if not corpus_dir.is_dir():
        parser.error(f"Corpus directory does not exist: {corpus_dir}")

    dataset = load_dataset(dataset_path)
    corpus_files = sorted(
        path for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    available_sources = {path.name for path in corpus_files}
    if not corpus_files:
        parser.error(f"No supported source files found in {corpus_dir}")
    if len(available_sources) != len(corpus_files):
        parser.error("Corpus contains duplicate filenames; document-level labels require unique basenames")
    validate_dataset(dataset, available_sources)

    selected = [
        item for item in dataset["queries"]
        if args.split == "all" or item["split"] == args.split
    ]
    if not selected:
        parser.error(f"No queries found for split: {args.split}")
    cases = [
        {
            "query": item["query"],
            "relevant_ids": item["relevant_sources"],
            "required_ids": item.get("required_sources", []),
            "answerable": item["answerable"],
            "category": item["category"],
        }
        for item in selected
    ]

    dataset_slug = re.sub(r"[^a-z0-9_-]+", "_", dataset_path.stem.lower()).strip("_")
    collection_name = f"bench_{dataset_slug}"[:63]
    benchmark_config = replace(
        settings,
        chroma_path=str(ROOT / ".benchmark_chroma"),
        chroma_collection=collection_name,
    )
    pipeline = RAGPipeline(config=benchmark_config)
    pipeline.vector_store.clear()
    ingestion = pipeline.ingest(str(corpus_dir))
    if ingestion["chunks"] <= 0:
        raise RuntimeError("Benchmark corpus produced no indexed chunks")

    candidate_k = ingestion["chunks"]
    dense = Retriever(pipeline.embedder, pipeline.vector_store)
    bm25 = pipeline.sparse_retriever
    hybrid = HybridRetriever(
        dense, bm25, candidate_k=candidate_k,
        dense_weight=benchmark_config.dense_weight,
        sparse_weight=benchmark_config.sparse_weight,
        rrf_k=benchmark_config.rrf_k,
    )
    available_modes = {"dense": dense, "bm25": bm25, "hybrid": hybrid}
    selected_modes = (
        ["dense", "bm25", "hybrid", "hybrid-reranked"]
        if args.mode == "all" else [args.mode]
    )

    reranker = None
    if "hybrid-reranked" in selected_modes:
        # Model loading is deferred unless the reranked benchmark is requested.
        reranker = CrossEncoderReranker(benchmark_config.reranker_model)

    reports = {}
    for mode in selected_modes:
        reports[mode] = evaluate_mode(
            hybrid if mode == "hybrid-reranked" else available_modes[mode],
            cases, args.k, candidate_k,
            reranker=reranker if mode == "hybrid-reranked" else None,
        )

    for report in reports.values():
        for source_case, result in zip(selected, report["results"]):
            result["id"] = source_case["id"]
            result["split"] = source_case["split"]

    output = {
        "benchmark": dataset.get("name", dataset_path.stem),
        "version": dataset.get("version"),
        "dataset_path": str(dataset_path),
        "corpus_path": str(corpus_dir),
        "split": args.split,
        "evaluation_level": dataset.get("evaluation_level", "document"),
        "embedding_model": benchmark_config.embedding_model,
        "reranker_model": benchmark_config.reranker_model if reranker else None,
        "retrieval_modes": selected_modes,
        "k": args.k,
        "corpus_documents": len(corpus_files),
        "indexed_chunks": ingestion["chunks"],
        "query_count": len(selected),
        "category_counts": dict(Counter(item["category"] for item in selected)),
        "metrics": {mode: report["metrics"] for mode, report in reports.items()},
        "diagnostics": {mode: report["diagnostics"] for mode, report in reports.items()},
        "by_category": {mode: report["by_category"] for mode, report in reports.items()},
        "results": {mode: report["results"] for mode, report in reports.items()},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
