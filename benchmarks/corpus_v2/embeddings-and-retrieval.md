# Embeddings and Semantic Retrieval

## What an embedding represents
An embedding model converts text into a dense numerical vector. The vector is intended to preserve useful semantic information: texts with related meanings should have vectors that are close under an appropriate similarity measure. An embedding is not a literal summary and individual dimensions usually do not have a simple human-readable meaning.

A document embedding represents a passage that may later be retrieved. A query embedding represents the user's search request. For retrieval to work, document and query vectors must be produced in a compatible vector space, normally by the same model or by a model designed with paired query and document encoders.

## Similarity measures
Cosine similarity measures the cosine of the angle between two non-zero vectors. It is often used to compare semantic embeddings because it focuses on direction rather than vector magnitude. A larger cosine similarity generally indicates greater alignment. Cosine distance is commonly defined as one minus cosine similarity, so smaller distances indicate closer vectors.

Dot product is another common scoring function. When vectors are normalized to unit length, their dot product equals cosine similarity. Without normalization, vector magnitude also affects the dot product. Euclidean distance measures straight-line distance between vectors; smaller values indicate closer points.

A system must understand the scoring convention of its vector database. Some databases return distances rather than similarities. Sorting a distance score in descending order by mistake can reverse the ranking.

## Chunking and overlap
Long documents are usually split into smaller passages called chunks before embedding. Chunk size creates a trade-off. Very small chunks may omit context needed to understand a fact; very large chunks may combine unrelated topics and make retrieval less focused. Overlap repeats a small portion of neighboring text so that a fact near a chunk boundary is less likely to be separated from its context.

A chunking strategy should preserve useful boundaries where possible, such as paragraphs, headings, or sentences. Character-based chunking is simple and deterministic, but character count is not the same as token count. A production system may use tokenizer-aware chunking and evaluate several sizes instead of assuming one setting is universally best.

## Dense retrieval
Dense retrieval embeds a query, compares it with stored document vectors, and returns the nearest passages. It can retrieve semantically related wording even when the query does not contain the exact terms in the document. Its weaknesses include difficulty with rare identifiers, exact product codes, unusual names, and some precise keyword constraints.

The top-k parameter controls how many passages are returned. A small k reduces context size but may omit useful evidence. A large k can improve coverage while adding irrelevant passages, increasing latency, and consuming more language-model context.

## Hybrid retrieval and reranking
Hybrid retrieval combines dense semantic search with lexical search, such as BM25. Lexical search is useful for exact terms, identifiers, and rare keywords; dense search is useful for paraphrases and semantic relationships. A fusion method combines the two ranked lists. Reciprocal Rank Fusion (RRF) is one common approach: it gives each result a score based on its rank in each list, rather than requiring the raw scores to be directly comparable.

A reranker is a second-stage model that receives a query and candidate passages and scores their relevance together. A cross-encoder can attend to both query and passage at once, often improving ordering among a small candidate set. Reranking costs additional computation, so a common architecture retrieves a wider candidate set first and reranks only those candidates.

## Retrieval evaluation
A retrieval benchmark contains queries and labels identifying which documents or passages are relevant to each query. Precision@k is the fraction of returned top-k results that are relevant. Recall@k is the fraction of all labeled relevant items found in the top k. Reciprocal Rank is the inverse rank of the first relevant result, or zero if none is found. Hit Rate@k records whether at least one relevant item appears in the top k.

Metrics answer different questions. Precision focuses on result cleanliness; recall focuses on finding known relevant evidence; reciprocal rank rewards placing the first useful result early; hit rate measures whether a query found any useful evidence. Results depend on the quality and coverage of relevance labels. A benchmark with only one labeled relevant chunk can unfairly penalize another passage that is also useful but was not labeled.

A fair comparison keeps the corpus, queries, relevance labels, chunking, and evaluation settings fixed while changing one retrieval component at a time. Report the model name, configuration, k, dataset version, and metrics. A small hand-built benchmark is useful for regression checks, but it is not proof of general performance across domains.
