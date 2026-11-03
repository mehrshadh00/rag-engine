# Retrieval-Augmented Generation Architecture

## The basic RAG workflow
Retrieval-Augmented Generation (RAG) combines an information retrieval step with a generative language model. During indexing, source documents are loaded, normalized, split into chunks, converted into embeddings, and stored with metadata in a searchable index. During question answering, the system embeds the query, retrieves relevant chunks, builds a context, and asks a language model to answer using that context.

The indexing stage is often performed before a user asks a question. The query stage runs for each question. Separating these stages avoids recomputing document embeddings for every request and allows the vector index to be reused.

## Metadata and provenance
Each chunk should retain useful metadata such as source filename, page number, section title, and chunk identifier. Metadata supports source attribution, filtering, debugging, and deletion of outdated content. Stable identifiers make it possible to update a document without leaving duplicate or stale chunks behind.

A generated answer should distinguish evidence from inference and cite the retrieved sources when the application requires traceability. A citation is useful only if it points to the passage that actually supports the associated claim. Returning a source list alone does not guarantee that every sentence in an answer is supported.

## Grounding and abstention
A RAG prompt should instruct the language model to answer from the supplied evidence and to say when the evidence is insufficient. This reduces unsupported claims, but a prompt cannot guarantee factual correctness. The retrieved context may be incomplete, irrelevant, contradictory, or maliciously written.

The system should avoid treating retrieved documents as trusted instructions. Retrieved content is data to be analyzed, not a replacement for system-level policy. This is important because a document can contain prompt-injection text that attempts to change the assistant's behavior.

## Failure modes
Retrieval failure occurs when the relevant passage is not present in the candidate results. Causes include poor chunk boundaries, weak embeddings, mismatched query wording, missing documents, stale indexes, or overly restrictive filters. Generation failure occurs when the model misreads evidence, combines unrelated passages, invents details, or produces an invalid citation.

These failure types should be measured separately. If the right evidence is absent from the retrieved context, changing the generation prompt alone is unlikely to solve the underlying retrieval problem. If the evidence is present but the answer is wrong, inspect context formatting, prompt instructions, model behavior, and citation validation.

## Operational considerations
A useful RAG service exposes health and ingestion operations separately from question answering. Ingestion should report how many documents and chunks were processed. Errors should identify the failing file or stage without exposing secrets. Persistent storage allows the index to survive process restarts.

For production use, teams should consider access control, document-level permissions, deletion guarantees, ingestion idempotency, monitoring, latency, cost, backups, and protection against prompt injection. A local demonstration that answers questions correctly is not automatically a production-ready knowledge system.

## Evaluation layers
Retrieval evaluation checks whether relevant evidence appears near the top of the ranked results. Answer evaluation checks whether the generated response is correct, complete, grounded, and properly cited. Operational evaluation checks latency, availability, failure rates, and cost. These layers should not be collapsed into one score because a system can retrieve well but generate poorly, or generate plausible answers despite weak retrieval.
