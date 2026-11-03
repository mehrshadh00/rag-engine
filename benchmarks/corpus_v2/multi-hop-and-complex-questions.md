# Multi-Hop and Complex Questions in RAG

## Multi-hop information needs
A multi-hop question requires evidence from two or more passages or documents. Each passage may provide only one part of the answer, so retrieving a single highly similar chunk may be insufficient. The system must identify and combine the separate evidence pieces.

## Retrieval strategies
One approach decomposes a complex question into smaller sub-questions, retrieves evidence for each, and combines the results. Another approach retrieves broadly and uses a model to identify which passages jointly support the answer. Both approaches can increase recall but may add latency and introduce errors during decomposition or evidence combination.

## Evaluation
A benchmark should label every source needed to answer a multi-hop question. A query is not fully supported if only one of its required evidence sources is retrieved. Evaluation should distinguish finding at least one relevant source from retrieving the complete evidence set. The answer-generation stage should also be checked for whether it combines the evidence correctly.
