# BM25 and Lexical Search

## Term-based retrieval
Lexical retrieval ranks documents using the terms shared by a query and a document. Unlike dense retrieval, it does not need to infer semantic similarity from embedding vectors. This makes lexical search useful for product identifiers, error codes, uncommon names, and exact technical vocabulary.

A basic term-frequency method can overvalue a word that appears many times in one document. BM25 addresses this with term-frequency saturation and inverse document frequency. Repeating a term increases its contribution, but each additional occurrence has a diminishing effect.

## BM25 components
BM25 uses the frequency of a query term in a document, the length of that document, the average document length in the corpus, and the term's rarity across the corpus. Rare terms generally provide more discriminative evidence than common terms. Length normalization reduces the advantage that long documents might otherwise receive simply because they contain more words.

The parameters k1 and b control term-frequency saturation and document-length normalization. They are hyperparameters, not universal constants. Their best values depend on the corpus and should be selected through evaluation rather than copied blindly.

## Limitations
Lexical search can fail when a query uses synonyms or paraphrases that do not occur in the source. Tokenization also matters: punctuation, hyphens, morphology, casing, and language-specific word forms can affect matching. A simple alphanumeric tokenizer may split identifiers or fail to connect inflected words.

BM25 scores are not probabilities and should not be interpreted as confidence that a passage answers the question. Their raw scale is also not directly comparable to cosine similarity. Hybrid systems commonly fuse rankings instead of adding raw BM25 and dense scores.
