# Chunking Strategies for Retrieval

## Why chunking matters
Chunking divides long source material into passages that can be embedded and retrieved independently. The goal is to make each passage focused enough to retrieve accurately while retaining enough context to interpret its facts. Chunk boundaries therefore influence both retrieval recall and the usefulness of the context passed to a language model.

Fixed-size chunking is easy to implement and reproduce. It can, however, split a sentence, table, list, or definition across boundaries. Structure-aware chunking uses headings, paragraphs, sentences, or document-specific elements to create more meaningful units.

## Size and overlap
Small chunks reduce unrelated content but may omit definitions or surrounding conditions. Large chunks preserve context but can dilute the embedding with multiple topics and increase prompt size. Overlap repeats text between adjacent chunks so facts near a boundary remain available with neighboring context.

Overlap also creates duplicate or near-duplicate retrieval results. A system may need to deduplicate chunks by source or section before constructing context. The best chunk size and overlap should be measured on representative queries rather than selected from a universal rule.

## Evaluation
When chunking changes, rebuild the index and keep the query set and relevance policy fixed. Passage-level labels are preferable when the benchmark evaluates chunks. If evaluation is document-level, document aggregation and deduplication rules should be explicit because multiple retrieved chunks can map to one source document.
