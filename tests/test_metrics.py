import numpy as np
import pytest
from sklearn.metrics import fowlkes_mallows_score

from py_puml.metrics import compute_metrics


def test_compute_metrics_basic() -> None:
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])

    row = compute_metrics(y_true, proba)

    assert 0.0 <= row.f1 <= 1.0
    assert 0.0 <= row.auc <= 1.0
    assert 0.0 <= row.pr_auc <= 1.0
    assert row.tp + row.tn + row.fp + row.fn == len(y_true)
    assert row.fmi == pytest.approx(fowlkes_mallows_score(y_true, (proba >= 0.5).astype(int)))


def test_compute_metrics_threshold_changes_confusion_counts() -> None:
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.2, 0.6, 0.55, 0.95])

    row_default = compute_metrics(y_true, proba, threshold=0.5)
    row_strict = compute_metrics(y_true, proba, threshold=0.7)

    assert (row_default.tp, row_default.tn, row_default.fp, row_default.fn) == (2, 1, 1, 0)
    assert (row_strict.tp, row_strict.tn, row_strict.fp, row_strict.fn) == (1, 2, 0, 1)
