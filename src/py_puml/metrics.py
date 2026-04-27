from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fowlkes_mallows_score,
    roc_auc_score,
)


@dataclass
class MetricsRow:
    auc: float
    pr_auc: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    fmi: float


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> MetricsRow:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    # Enforce binary layout so confusion_matrix always returns 2x2.
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, proba))
        pr_auc = float(average_precision_score(y_true, proba))

    f1 = float(f1_score(y_true, pred, zero_division=0))
    fmi = float(fowlkes_mallows_score(y_true, pred))
    return MetricsRow(
        auc=auc,
        pr_auc=pr_auc,
        f1=f1,
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        fmi=fmi,
    )
