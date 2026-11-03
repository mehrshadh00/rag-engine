# Reranking and Cross-Encoders

## Two-stage retrieval
A retrieval system can use a fast first stage to produce a candidate set and a more expensive second stage to refine the order. The first stage should prioritize recall so that relevant passages are likely to enter the candidate pool. The reranker then prioritizes precision by moving the most relevant candidates toward the top.

## Cross-encoder scoring
A cross-encoder receives the query and a candidate passage together and produces a relevance score. Because the model processes both texts jointly, it can model interactions that independent query and document embeddings may not capture. Cross-encoder scores are generally useful for ranking candidates, not as calibrated probabilities unless a separate calibration process has been performed.

## Trade-offs
Reranking adds model inference latency and computational cost. It is commonly applied only to a limited candidate pool rather than the entire corpus. Candidate-pool size is important: a reranker cannot recover a relevant passage that the first-stage retriever failed to retrieve.

## Evaluation
Compare first-stage ranking with reranked ranking using the same queries, relevance labels, candidate pool, and final k. Report retrieval metrics as well as latency. Improvements in ranking quality should be weighed against the added runtime cost.
