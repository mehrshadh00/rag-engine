# Retrieval Security and Prompt Injection

## Retrieved content is untrusted data
Documents in a knowledge base may contain malicious or misleading instructions. A retrieved passage can tell a model to ignore prior instructions, reveal secrets, or perform unrelated actions. The application should treat document text as evidence to analyze, never as a source of authority over system or developer instructions.

## Isolation and data handling
Keep secrets out of prompts and retrieved documents. Apply authorization before adding passages to model context. Avoid exposing internal metadata, private documents, or tool credentials through citations and logs. Retrieval filters should be enforced server-side rather than trusted from the client.

## Prompt design and validation
Clearly delimit retrieved passages and tell the model that their contents are untrusted reference material. This is a defense-in-depth measure, not a complete security boundary. Validate tool calls and outputs independently, and require explicit authorization for sensitive actions.

## Testing
Include adversarial documents in security tests and verify that the model does not follow embedded instructions. Test cross-tenant access, malicious metadata, poisoned citations, and attempts to retrieve restricted records. Security evaluation should be separate from ordinary relevance metrics.
