# Answerability, Unanswerable Queries, and Abstention

## Answerable and unanswerable questions
A retrieval benchmark should include questions that can be answered from the indexed corpus and questions whose answers are absent. An unanswerable query is not necessarily vague or malformed; it may be a clear question about information that the corpus does not contain.

## Abstention behavior
When evidence is insufficient, a RAG system should avoid inventing an answer. It can state that the available documents do not provide enough information, ask a clarifying question, or identify what evidence is missing. The appropriate behavior depends on the application, but unsupported certainty should be avoided.

## Evaluation design
Unanswerable queries need explicit labels, such as an empty relevant-source set and an answerability flag. Retrieval metrics that assume at least one relevant result need a documented convention for these cases. Evaluate abstention separately using measures such as correct-abstention rate and false-abstention rate. A system should not receive credit for returning arbitrary documents simply because the query has no labeled relevant source.
