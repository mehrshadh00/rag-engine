"""Threshold-based answerability decisions from Cross-Encoder relevance scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def is_answerable(score: float | None, threshold: float = 0.0) -> bool:
    """Return whether a reranker score clears the configured relevance threshold.

    Cross-Encoder scores are model- and domain-dependent, not calibrated
    probabilities. The threshold must be validated on a representative dev set.
    """
    if score is None:
        return False
    return float(score) >= threshold


def evaluate_answerability(
    cases: Iterable[Mapping], threshold: float = 0.0
) -> dict:
    """Evaluate threshold-based answerability predictions against boolean labels."""
    tp = fp = tn = fn = 0
    for case in cases:
        actual = bool(case["answerable"])
        predicted = is_answerable(case.get("score"), threshold)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "query_count": total,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }
