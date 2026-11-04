import pytest

from rag_engine.answerability import evaluate_answerability, is_answerable


def test_answerability_uses_inclusive_threshold_and_rejects_missing_score():
    assert is_answerable(0.0, 0.0)
    assert not is_answerable(-0.01, 0.0)
    assert not is_answerable(None, 0.0)


def test_answerability_metrics_report_false_positives_and_false_negatives():
    cases = [
        {"answerable": True, "score": 0.8},
        {"answerable": True, "score": -0.1},
        {"answerable": False, "score": 0.3},
        {"answerable": False, "score": -0.4},
    ]

    report = evaluate_answerability(cases, threshold=0.0)

    assert report["true_positive"] == 1
    assert report["false_negative"] == 1
    assert report["false_positive"] == 1
    assert report["true_negative"] == 1
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(0.5)
    assert report["false_positive_rate"] == pytest.approx(0.5)
