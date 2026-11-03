# RAG Engine

A modular Retrieval-Augmented Generation (RAG) system for grounded question answering over PDF, Markdown, and text documents.

## What it does

1. Loads local PDF, Markdown, and TXT files.
2. Splits documents into overlapping chunks while preserving source metadata.
3. Encodes chunks with a local Sentence Transformers embedding model.
4. Stores vectors in a persistent ChromaDB collection.
5. Combines dense semantic search and sparse BM25 search with Reciprocal Rank Fusion (RRF).
6. Builds source-attributed context and generates answers with an OpenAI-compatible chat model.
7. Exposes ingestion and question-answering endpoints through FastAPI.

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
# Windows CMD: .venv\Scripts\activate.bat
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to enable answer generation. Ingestion and retrieval use a local embedding model and do not require an LLM API key.

Add documents under `data/`, then ingest them:

```bash
python -m rag_engine.cli ingest data
```

Ask a question:

```bash
python -m rag_engine.cli ask "What do the documents say about artificial intelligence?"
```

Run the API:

```bash
uvicorn rag_engine.api:app --reload
```

- `POST /ingest` with JSON: `{"directory": "data"}`
- `POST /ask` with JSON: `{"question": "Your question", "top_k": 5}`
- `GET /health`

The first run downloads the configured embedding model from Hugging Face.

## Configuration

See `.env.example`.

| Variable | Purpose |
|---|---|
| `EMBEDDING_MODEL` | Sentence Transformers model |
| `CHROMA_PATH` | Persistent vector database directory |
| `CHROMA_COLLECTION` | Chroma collection name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters |
| `TOP_K` | Number of final passages retrieved |
| `RETRIEVAL_MODE` | `hybrid` (default) or `dense` |
| `HYBRID_CANDIDATE_K` | Candidate pool size per retriever before fusion |
| `RRF_K` | Reciprocal Rank Fusion smoothing constant |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | Relative contribution of each ranked list |
| `OPENAI_API_KEY` | API key for answer generation |
| `OPENAI_MODEL` | Chat completion model |

## Hybrid retrieval

- **Dense retrieval:** semantic nearest-neighbor search over ChromaDB embeddings.
- **Sparse retrieval:** dependency-free Okapi BM25 over persisted chunk text.
- **Fusion:** Reciprocal Rank Fusion combines rank positions rather than incomparable raw similarity and BM25 scores.

For a result at rank `r`, RRF adds `weight / (rrf_k + r)`. Default weights are equal and `rrf_k=60`. The sparse index is rebuilt from the persistent Chroma collection when the pipeline starts and refreshed after ingestion, so it remains available across CLI/API process restarts.

Set `RETRIEVAL_MODE=dense` to use the original dense-only baseline. The default is `RETRIEVAL_MODE=hybrid`.

## Evaluation and tests

Run unit tests:

```bash
pytest
```

The evaluation module reports Precision@K, Recall@K, Reciprocal Rank, and Hit Rate@K. Unit tests use lightweight fakes and do not require API credentials or downloading the embedding model.

### Dense vs. hybrid benchmark

The manually labeled benchmark contains three technical documents and 20 natural-language queries. Labels are document-level. The runner compares dense-only and hybrid retrieval on the same corpus, labels, and K.

```bash
# Compare both modes (default)
python scripts/run_benchmark.py

# Compare at a different document cutoff
python scripts/run_benchmark.py --k 1
python scripts/run_benchmark.py --k 3

# Run one mode only
python scripts/run_benchmark.py --mode dense
python scripts/run_benchmark.py --mode hybrid
```

The benchmark uses a separate Chroma collection under `.benchmark_chroma/`, prints aggregate and per-query metrics for each mode, and does not call an LLM API. This small dataset is a diagnostic baseline, not a general-purpose leaderboard.

## Current scope and limitations

- BM25 currently uses a lowercase alphanumeric tokenizer; stemming and language-specific analyzers are not included.
- PDF extraction is text-based; scanned PDFs require OCR, which is not included yet.
- Hybrid fusion combines chunk-level rankings; document-level evaluation deduplicates chunks by source.
- Cross-encoder reranking is a planned extension.
- A local demonstration is not automatically a production-ready knowledge system.

## Project structure

```text
src/rag_engine/
  api.py          # FastAPI endpoints
  bm25.py         # Dependency-free BM25 sparse retrieval
  benchmark.py    # Document-level benchmark adapter
  chunking.py     # Overlap-aware text chunking
  config.py       # Environment-based settings
  embeddings.py   # Sentence Transformers adapter
  evaluation.py   # Retrieval metrics
  generation.py   # Grounded answer generation
  hybrid.py       # RRF-based dense/sparse fusion
  ingestion.py    # PDF/TXT/Markdown loaders
  pipeline.py     # End-to-end orchestration
  retrieval.py    # Dense retrieval and context formatting
  vector_store.py # Persistent ChromaDB storage
benchmarks/
  corpus/
  datasets/
scripts/
  run_benchmark.py
tests/
.github/workflows/tests.yml
```

## License

MIT
