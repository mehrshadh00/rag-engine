# Evaluation Methods for Information Retrieval Systems

## Building a labeled dataset
A retrieval evaluation dataset consists of representative user queries and relevance judgments. Each query should be paired with one or more document or passage identifiers that contain evidence needed to answer it. Labels should be created by reading the corpus, not by assuming that the current retriever's top results are correct.

Queries should vary in wording and difficulty. Include direct factual questions, paraphrases, questions using synonyms, questions about exact terminology, and questions that combine two related concepts. Avoid making every query copy a sentence from the source document, because that can overstate performance on natural user questions.

## Precision and recall at k
Precision@k is the number of relevant results among the returned top k divided by the number of results actually returned, when fewer than k results are available. It measures how concentrated the result list is with relevant items.

Recall@k is the number of relevant items retrieved in the top k divided by the total number of labeled relevant items for the query. It measures coverage of the known relevant set. If no relevant items have been labeled for a query, recall is undefined in principle; a benchmark implementation should document its convention, commonly returning zero or excluding that query from the aggregate.

## Reciprocal rank and hit rate
Reciprocal Rank is 1 divided by the rank of the first relevant result. If the first relevant result is at rank one, the score is 1.0; at rank two, it is 0.5; if no relevant result is retrieved, it is 0. Mean Reciprocal Rank (MRR) is the average reciprocal rank across queries.

Hit Rate@k is 1 when at least one relevant item appears in the top k, and 0 otherwise. The average hit rate is the fraction of queries for which the system retrieved at least one known relevant result. Hit rate does not measure how many relevant items were found or where they appeared within the top k.

## Designing useful comparisons
A baseline is a simple, documented system against which later changes can be compared. To evaluate an embedding model fairly, keep the corpus, chunking strategy, query set, relevance judgments, distance metric, and k constant. Rebuild the index whenever the embedding model changes, because vectors from incompatible embedding spaces must not be compared.

For hybrid retrieval, compare dense-only, lexical-only, and fused results on the same benchmark. For reranking, compare the initial candidate ranking with the reranked top-k list, while keeping the candidate pool size fixed. Change one major component at a time so that any metric difference has an interpretable cause.

## Interpreting results responsibly
A small benchmark is a diagnostic instrument, not a universal leaderboard. Its scores are sensitive to domain, corpus size, query wording, label quality, and the number of relevant passages annotated. A high score on a narrow dataset does not establish broad language understanding or production reliability.

Inspect per-query results as well as aggregate metrics. Averages can hide failures on exact identifiers, multi-hop questions, or queries with ambiguous wording. Record the dataset version and model configuration so results can be reproduced. If the corpus or labels change, treat the benchmark as a new version and avoid presenting scores as a direct apples-to-apples comparison without qualification.
