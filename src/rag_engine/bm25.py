"""Dependency-free Okapi BM25 sparse retrieval."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric terms."""
    return _TOKEN_PATTERN.findall(text.lower())

class BM25Retriever:
    """Okapi BM25 retriever over an in-memory collection of text chunks."""
    def __init__(self, documents: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1, self.b = k1, b
        self.documents = list(documents)
        self._tokens = [tokenize(str(doc.get("text", ""))) for doc in self.documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        frequencies: Counter[str] = Counter()
        for tokens in self._tokens:
            frequencies.update(set(tokens))
        total = len(self.documents)
        self._idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = set(tokenize(query))
        if not terms or not self.documents or self._average_length == 0:
            return []
        scored = []
        for index, document in enumerate(self.documents):
            tf = self._term_frequencies[index]
            length = self._lengths[index]
            score = 0.0
            for term in terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self._average_length
                )
                score += self._idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                scored.append((score, index, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": document["id"],
                "text": document.get("text", ""),
                "metadata": document.get("metadata", {}),
                "bm25_score": float(score),
            }
            for score, _, document in scored[:top_k]
        ]
