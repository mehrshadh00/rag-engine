# RAG Engine

A modular Retrieval-Augmented Generation (RAG) system for grounded question answering over PDF, Markdown, and text documents.

## What it does

1. Loads local PDF, Markdown, and TXT files.
2. Splits documents into overlapping chunks while preserving source metadata.
3. Encodes chunks with a local Sentence Transformers embedding model.
4. Stores vectors in a persistent ChromaDB collection.
5. Retrieves relevant passages and builds a source-attributed context.
6. Generates an answer with an OpenAI-compatible chat model, with instructions to abstain when evidence is insufficient.
7. Exposes ingestion and question-answering endpoints through FastAPI.

## Architecture

```text
Files -> Loaders -> Chunker -> Embeddings -> ChromaDB
                                           |
Question -> Embedding -> Retriever -> Context + Sources
                                           |
                                      Chat Model
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
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to enable answer generation. Ingestion and retrieval use a local embedding model and persistent ChromaDB storage.

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

See `.env.example`. Important settings include:

| Variable | Purpose |
|---|---|
| `EMBEDDING_MODEL` | Sentence Transformers model used for document and query vectors |
| `CHROMA_PATH` | Persistent vector database directory |
| `CHROMA_COLLECTION` | Chroma collection name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters |
| `TOP_K` | Number of passages retrieved |
| `OPENAI_API_KEY` | API key for answer generation |
| `OPENAI_MODEL` | Chat completion model |

## Evaluation and tests

Run unit tests:

```bash
pytest
```

A retrieval evaluation module provides Precision@K, Recall@K, Reciprocal Rank, and Hit Rate@K. `evaluate_retrieval()` runs a labeled query dataset against any compatible retriever and returns per-query results plus aggregate metrics. Tests use lightweight fakes and do not require API credentials or downloading the embedding model.

### Reproducible retrieval benchmark

The repository includes a small, manually labeled benchmark with three technical documents and 20 natural-language queries. Labels are document-level: a source is relevant when it contains evidence needed to answer the query. The runner collapses ranked chunks into unique source documents, using each document's highest-ranked chunk.

```bash
python scripts/run_benchmark.py
# Optional: evaluate a different number of unique source documents
python scripts/run_benchmark.py --k 2
```

The runner indexes the benchmark corpus in a separate local ChromaDB collection under `.benchmark_chroma/`, uses the configured local embedding model, and prints JSON containing aggregate and per-query metrics. It does not call an LLM API; the embedding model may be downloaded on first run. The default `k=2` keeps the ranking meaningful for a three-document corpus; use `--k 1` or `--k 3` to inspect other cutoffs. This is a transparent regression baseline, not a general-purpose leaderboard. See `benchmarks/datasets/rag_fundamentals_v1.json` for the queries and relevance labels.

## Current scope and limitations

- PDF extraction is text-based; scanned PDFs require OCR, which is not included yet.
- Retrieval currently uses dense semantic similarity. Hybrid search and cross-encoder reranking are planned extensions.
- The answer-generation layer currently targets the OpenAI chat completions API.
- This is an actively developed portfolio project, not a claim of production certification.

## Project structure

```text
src/rag_engine/
  api.py          # FastAPI endpoints
  cli.py          # Command-line interface
  config.py       # Environment-based settings
  ingestion.py    # PDF/TXT/Markdown loaders
  chunking.py     # Overlap-aware text chunking
  embeddings.py   # Sentence Transformers adapter
  vector_store.py # ChromaDB persistence
  retrieval.py    # Similarity retrieval
  generation.py   # Grounded answer generation
  pipeline.py     # End-to-end orchestration
  evaluation.py   # Retrieval metrics
benchmarks/
  corpus/         # Three benchmark source documents
  datasets/       # Versioned labeled queries
scripts/
  run_benchmark.py # Reproducible retrieval benchmark runner
tests/
.github/workflows/tests.yml
```

## فارسی

این پروژه یک موتور ماژولار RAG برای پرسش‌وپاسخ مستند بر اساس فایل‌های PDF، Markdown و TXT است. اسناد را بارگذاری و قطعه‌بندی می‌کند، embedding می‌سازد، بردارها را در ChromaDB ذخیره می‌کند، مرتبط‌ترین بخش‌ها را بازیابی می‌کند و پاسخ را همراه با ارجاع به منبع تولید می‌کند. برای توضیح کامل تنظیمات و معماری، همین README را دنبال کنید.

## License

MIT
