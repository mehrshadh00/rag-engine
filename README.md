# RAG Engine

A modular Retrieval-Augmented Generation (RAG) system for grounded question answering over PDF, Markdown, and text documents.

## What it does

1. Loads local PDF, Markdown, and TXT files.
2. Splits documents into overlapping chunks while preserving source metadata.
3. Encodes chunks with a local Sentence Transformers embedding model.
4. Stores vectors in a persistent ChromaDB collection.
5. Combines dense semantic search and sparse BM25 search with Reciprocal Rank Fusion (RRF).
6. Optionally reranks retrieved chunks with a Cross-Encoder before context assembly.
7. Builds source-attributed context and generates answers with an OpenAI-compatible chat model.
8. Exposes ingestion and question-answering endpoints through FastAPI.

## Architecture

```text
Files -> Loaders -> Chunker -> Embeddings -> ChromaDB
                              |                 |
                              +-> BM25 Index <--+
                                      |
Question -> Dense Search + BM25 Search
                    |
             Reciprocal Rank Fusion
                    |
             Candidate Chunks
                    |
       Optional Cross-Encoder Reranking
                    |
       Ranked Chunks -> Context -> Chat Model
                                      |
                               Answer + Citations
```

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/mehrshadh00/rag-engine.git
cd rag-engine
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows CMD: .venv\\Scripts\\activate.bat
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to enable answer generation. Ingestion and retrieval use local models and do not require an LLM API key.

Add documents under `data/`, then ingest them:

```bash
python -m rag_engine.cli ingest data
python -m rag_engine.cli ask "What do the documents say about artificial intelligence?"
uvicorn rag_engine.api:app --reload
```

- `POST /ingest` with JSON: `{"directory": "data"}`
- `POST /ask` with JSON: `{"question": "Your question", "top_k": 5}`
- `GET /health`

The first run downloads the configured embedding model from Hugging Face. Enabling reranking also downloads the configured Cross-Encoder model on first use.

## Configuration

See `.env.example`.

| Variable | Purpose |
|---|---|
| `EMBEDDING_MODEL` | Sentence Transformers embedding model |
| `CHROMA_PATH` / `CHROMA_COLLECTION` | Persistent vector store location and collection |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters |
| `TOP_K` | Number of final passages retrieved |
| `RETRIEVAL_MODE` | `hybrid` (default), `dense`, or `bm25` |
| `HYBRID_CANDIDATE_K` | Candidate pool size per retriever before fusion |
| `RRF_K` | Reciprocal Rank Fusion smoothing constant |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | Relative contribution of each ranked list |
| `RERANK_ENABLED` | Enable optional Cross-Encoder reranking (`false` by default) |
| `RERANKER_MODEL` | Cross-Encoder model identifier |
| `RERANK_CANDIDATE_K` | Candidate chunk pool reranked before selecting final top-k |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Answer generation credentials and model |

## Retrieval and reranking

- **Dense retrieval:** semantic nearest-neighbor search over ChromaDB embeddings.
- **Sparse retrieval:** dependency-free Okapi BM25 over persisted chunk text.
- **Fusion:** Reciprocal Rank Fusion combines rank positions rather than incomparable raw similarity and BM25 scores.
- **Reranking:** a Cross-Encoder jointly scores each query/chunk pair and reorders the retrieved candidate pool. It is applied after retrieval and before context assembly.

RRF adds `weight / (rrf_k + rank)` for each ranked list. Default weights are equal and `rrf_k=60`. The sparse index is rebuilt from the persistent Chroma collection at startup and refreshed after ingestion.

Reranking is disabled by default. To enable it, set `RERANK_ENABLED=true`; the default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Reranking can improve ordering but cannot recover relevant chunks absent from the candidate pool, and adds model latency.

## Evaluation and tests

Run unit tests:

```bash
pytest
```

Tests use lightweight fakes and do not require API credentials or model downloads.

### Versioned retrieval benchmark

The v2 benchmark contains 15 technical documents and 100 manually labeled, document-level queries. It is stratified into five categories: direct factual, semantic paraphrase, exact terminology, multi-hop, and unanswerable. Each category has 12 development queries and 8 held-out test queries (60 dev / 40 test overall).

The runner compares dense-only, BM25-only, hybrid RRF, and hybrid RRF followed by Cross-Encoder reranking. For the reranked mode, the first-stage dense and sparse retrievers each use `HYBRID_CANDIDATE_K`, RRF fuses their results, and only the top `RERANK_CANDIDATE_K` chunks are passed to the Cross-Encoder. This matches the runtime candidate-generation settings. The reranker scores chunks before document-level deduplication, so the evaluation cutoff remains a number of unique source documents. The report records both candidate settings. By default only the development split is evaluated.

```bash
# Compare all four methods on dev (downloads the Cross-Encoder on first run)
python scripts/run_benchmark.py

# Compare only the reranked hybrid method
python scripts/run_benchmark.py --mode hybrid-reranked

# Compare retrieval baselines without loading the Cross-Encoder
python scripts/run_benchmark.py --mode hybrid

# Change the final number of unique source documents
python scripts/run_benchmark.py --k 1
python scripts/run_benchmark.py --k 3

# Evaluate the held-out test split only after decisions are finalized
python scripts/run_benchmark.py --split test
```

The report includes Precision@K, Recall@K, MRR, Hit Rate@K, multi-hop complete-evidence rate, unanswerable false-positive rate, and category-level breakdowns. The unanswerable diagnostic measures whether any result was returned, not whether generation can abstain.

Each run clears its dataset-specific Chroma collection before ingestion. The benchmark does not call an LLM API. It is a reproducible diagnostic dataset, not a general-purpose leaderboard; conclusions are limited by this corpus and its human relevance labels.

## Current scope and limitations

- BM25 currently uses a lowercase alphanumeric tokenizer; stemming and language-specific analyzers are not included.
- PDF extraction is text-based; scanned PDFs require OCR, which is not included yet.
- Reranking only reorders retrieved candidates; candidate recall remains dependent on the first-stage retriever.
- A local demonstration is not automatically a production-ready knowledge system.

## Project structure

```text
src/rag_engine/
  api.py
  benchmark.py
  bm25.py
  chunking.py
  config.py
  embeddings.py
  evaluation.py
  generation.py
  hybrid.py
  ingestion.py
  pipeline.py
  reranking.py
  retrieval.py
  vector_store.py
benchmarks/
  corpus_v2/
  datasets/
scripts/
  run_benchmark.py
tests/
.github/workflows/tests.yml
```

## License

MIT
