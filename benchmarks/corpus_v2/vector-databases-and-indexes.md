# Vector Databases and Approximate Nearest-Neighbor Indexes

## Vector storage
A vector database stores embedding vectors together with identifiers and metadata. A retrieval request supplies a query vector and asks the database to find stored vectors that are close under a configured distance metric. The database may also support metadata filters, persistence, updates, and deletion.

The vector record should retain enough metadata to map a retrieved vector back to its source document and chunk. Without stable identifiers and provenance, it becomes difficult to cite results, update individual documents, or remove obsolete content.

## Exact and approximate search
Exact nearest-neighbor search compares a query against every stored vector. It can be practical for small collections and provides a straightforward reference result, but its computational cost grows with the number of vectors.

Approximate nearest-neighbor (ANN) indexes trade some retrieval accuracy for lower latency at scale. Graph-based and inverted-file approaches organize vectors to avoid exhaustive comparison. The appropriate index depends on corpus size, update patterns, memory limits, latency targets, and acceptable recall loss.

## Operational choices
Distance metric, vector normalization, index configuration, and embedding model must be treated as a coordinated design. Changing the embedding model requires re-embedding the corpus; vectors from incompatible spaces should not be mixed. A vector database is an indexing and retrieval component, not a substitute for relevance evaluation or data governance.
