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

### Versioned retrieval benchmark (v2)

The v2 benchmark contains 15 technical documents and 100 manually labeled, document-level queries. It is stratified into five categories: direct factual, semantic paraphrase, exact terminology, multi-hop, and unanswerable. Each category has 12 development queries and 8 held-out test queries (60 dev / 40 test overall).

The runner compares four methods:

1. Dense-only retrieval
2. BM25-only retrieval
3. Hybrid retrieval with weighted Reciprocal Rank Fusion
4. Hybrid retrieval followed by Cross-Encoder reranking

For the reranked mode, the first-stage dense and sparse retrievers each use `HYBRID_CANDIDATE_K`, RRF fuses their results, and only the top `RERANK_CANDIDATE_K` chunks are passed to the Cross-Encoder. This matches the runtime candidate-generation settings. The reranker scores chunks before document-level deduplication, so the evaluation cutoff is a number of unique source documents. The report records both candidate settings.

### Benchmark results

All results below use **K=3 unique source documents**. Precision@K, Recall@K, MRR, and Hit Rate@K are macro-averaged over **answerable queries only** (48 Dev queries and 32 Test queries); unanswerable queries are reported separately through the false-positive diagnostic. Multi-hop complete-evidence rate is averaged over the multi-hop subset (12 Dev and 8 Test queries). The corpus contained 43 chunks at evaluation time. The configured first-stage candidate limit was 50, so each baseline retriever returned up to the 43 available chunks; the reranked path passed at most 20 fused candidates to the Cross-Encoder.

**Development split — 60 queries**

| Metric | Dense | BM25 | Hybrid | Hybrid + Reranker |
|---|---:|---:|---:|---:|
| Precision@3 | 35.42% | 36.11% | 37.50% | **39.58%** |
| Recall@3 | 88.54% | 89.58% | 91.67% | **96.88%** |
| MRR | 88.54% | 84.38% | 87.15% | **90.63%** |
| Hit Rate@3 | 95.83% | 95.83% | 95.83% | **100.00%** |
| Multi-hop complete evidence | 41.67% | 50.00% | 66.67% | **75.00%** |

**Held-out test split — 40 queries**

| Metric | Dense | BM25 | Hybrid | Hybrid + Reranker |
|---|---:|---:|---:|---:|
| Precision@3 | 35.42% | 35.42% | 37.50% | **39.58%** |
| Recall@3 | 87.50% | 87.50% | 92.19% | **96.88%** |
| MRR | 83.85% | 76.56% | 84.90% | **89.58%** |
| Hit Rate@3 | 93.75% | 93.75% | 96.88% | **100.00%** |
| Multi-hop complete evidence | 50.00% | 50.00% | 62.50% | **75.00%** |

On this dataset, Hybrid + Reranker achieved the highest value for every reported retrieval metric on both splits. Compared with Hybrid on the held-out test split, reranking improved Precision@3 by 2.08 percentage points, Recall@3 by 4.69 points, MRR by 4.69 points, Hit Rate@3 by 3.12 points, and multi-hop complete evidence by 12.50 points.

### Interpretation and limitations

- These are results from one fixed corpus and one manually labeled dataset. They are useful for comparing implementations under the same conditions, not evidence of universal RAG performance or a statistically conclusive improvement across domains.
- The test split contains 40 queries (8 per category); small sample sizes can make metric changes sensitive to individual examples.
- The `unanswerable_false_positive_rate_at_k` was 1.00 for all four methods on both splits. This diagnostic means retrieval returned at least one document for every unanswerable query. It does **not** measure whether the answer-generation layer can abstain. A separate answerability / abstention mechanism and evaluation are still needed.
- Reranking only reorders retrieved candidates. It cannot recover relevant chunks that first-stage retrieval did not return, and it adds inference latency.
- The benchmark evaluates document retrieval, not answer factuality, citation correctness, latency, throughput, or end-to-end generation quality.

Run the benchmark:

```bash
# Development split (default)
python scripts/run_benchmark.py --split dev --k 3

# Held-out test split; run after development decisions are finalized
python scripts/run_benchmark.py --split test --k 3

# Compare only the reranked hybrid method
python scripts/run_benchmark.py --mode hybrid-reranked --split dev

# Compare retrieval baselines without loading the Cross-Encoder
python scripts/run_benchmark.py --mode hybrid --split dev
```

Each run clears its dataset-specific Chroma collection before ingestion. The benchmark does not call an LLM API. Keep the test split held out when tuning retrieval parameters.

## Current scope and limitations

- BM25 currently uses a lowercase alphanumeric tokenizer; stemming and language-specific analyzers are not included.
- PDF extraction is text-based; scanned PDFs require OCR, which is not included yet.
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
