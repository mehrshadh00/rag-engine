# Context Assembly and Source Citations

## Building the context
After retrieval, a RAG application selects and formats passages for the language model. Context assembly may include deduplication, ordering, source labels, section titles, and limits on total tokens. The objective is to provide sufficient evidence without filling the prompt with repetitive or unrelated text.

A larger retrieved set is not always better. Irrelevant passages can distract generation, while duplicate chunks waste context space. A context builder should preserve the relationship between each passage and its metadata.

## Citation integrity
A citation should identify the source passage that supports a claim. Attaching every retrieved source to every sentence creates the appearance of traceability without demonstrating support. Citation validation can check that cited identifiers were actually retrieved and that the cited passage contains evidence relevant to the associated statement.

## Ordering and truncation
Ordering can affect which evidence a language model attends to, especially when the prompt has a strict context limit. Truncation should not silently remove the only passage supporting an answer. Systems should record which passages were included in the final prompt, not only which candidates were initially retrieved.
