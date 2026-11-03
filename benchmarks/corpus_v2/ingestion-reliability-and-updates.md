# Ingestion Reliability and Document Updates

## Idempotent ingestion
Ingestion is idempotent when processing the same source repeatedly does not create duplicate logical records or change the result unexpectedly. Stable document and chunk identifiers help the pipeline recognize previously indexed content.

## Updates and deletion
When a source document changes, the system should detect the update and replace or refresh its indexed chunks. Deleting a source should remove its vectors, sparse-index entries, cached passages, and any derived references according to the application's retention policy. Leaving stale chunks can cause the system to answer from outdated information.

## Failure handling
An ingestion job should report which files succeeded, failed, or were skipped. Partial failures should not be mistaken for complete success. Depending on the application, updates can use staging and atomic switching so that users do not query a half-updated index.

## Validation
Test repeated ingestion, modified files, deleted files, malformed documents, and interrupted jobs. Verify document counts and chunk counts before and after each operation. Monitoring should make it possible to identify stale or incomplete indexes.
