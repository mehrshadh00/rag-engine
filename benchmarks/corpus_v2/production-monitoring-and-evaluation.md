# Production Monitoring and RAG Evaluation

## Quality monitoring
Offline benchmarks provide controlled comparisons, but production queries may differ from the evaluation set. Monitor retrieval quality using sampled queries, human relevance judgments, user feedback, and task-specific success signals. Avoid treating clicks or engagement as definitive proof that a retrieved passage was relevant.

## Latency and reliability
Measure latency for embedding, dense search, lexical search, fusion, reranking, context assembly, and generation separately. End-to-end latency alone can hide which component causes a slowdown. Track error rates, timeouts, index availability, and ingestion freshness.

## Cost and capacity
Embedding models, rerankers, and language models have different compute and API costs. Record token usage, model calls, and resource consumption where appropriate. Cache only when the cache key includes all inputs that affect the result, including authorization scope and relevant configuration.

## Regression testing
Run a stable offline evaluation set after changes to embeddings, chunking, filters, ranking, or prompts. Version the corpus and labels, record configuration, and inspect per-query regressions. Aggregate metrics should be paired with latency, cost, and safety checks before deployment.
