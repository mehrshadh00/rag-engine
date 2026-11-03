# Metadata Filtering and Access Control in RAG

## Metadata filters
Metadata filters restrict retrieval to records that satisfy structured conditions, such as document type, language, date, department, or tenant. Filtering can improve relevance when the user has supplied a valid constraint and can reduce the candidate search space.

Filters should be applied consistently across dense and lexical retrieval in a hybrid system. If one retriever searches a restricted subset while another searches the entire corpus, fusion can produce misleading rankings and may expose content outside the intended scope.

## Authorization
Access control is not merely a relevance feature. A system must verify that the requesting user is allowed to access a document before its content is returned or included in a language-model prompt. Client-supplied metadata filters must not be trusted as the only authorization check.

## Testing
Create tests for users with different permissions, missing metadata, and documents whose access changes. Verify that unauthorized passages never appear in retrieved results, logs, citations, or generated context. Deletion and permission updates should propagate to all indexes, including sparse indexes and caches.
