from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


@dataclass
class MethodOutput:
    proba: np.ndarray
    label: np.ndarray
    meta: dict[str, Any]


def _feature_columns(df: pd.DataFrame, y_col: str = "Y", s_col: str = "S") -> list[str]:
    return [c for c in df.columns if c not in {y_col, s_col}]


def _to_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "S",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = feature_cols or [c for c in df.columns if c != target_col]
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = pd.to_numeric(df[target_col], errors="raise").to_numpy().astype(int)
    return x, y, cols


def _labels_from_proba(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (proba >= threshold).astype(int)


def _make_l1_selector(cv: int = 5) -> LogisticRegressionCV:
    # scikit-learn deprecated explicit `penalty` in LogisticRegressionCV; use l1_ratios for L1.
    return LogisticRegressionCV(
        solver="saga",
        l1_ratios=[1.0],
        cv=cv,
        max_iter=10000,
        tol=1e-3,
        scoring="roc_auc",
        use_legacy_attributes=False,
    )


def fit_predict_naive(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_col: str = "Y",
    s_col: str = "S",
) -> MethodOutput:
    feature_cols = _feature_columns(train_df, y_col=y_col, s_col=s_col)
    x_train, y_train, _ = _to_xy(train_df, feature_cols=feature_cols, target_col=s_col)
    x_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    if np.unique(y_train).size < 2:
        p = float(y_train.mean()) if y_train.size else 0.0
        proba = np.full(shape=len(test_df), fill_value=p, dtype=float)
        return MethodOutput(proba=proba, label=_labels_from_proba(proba), meta={"constant": p})

    model = LogisticRegression(max_iter=4000, solver="lbfgs")
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return MethodOutput(proba=proba, label=_labels_from_proba(proba), meta={"feature_cols": feature_cols})


def estimate_clust_coefficients(
    train_df: pd.DataFrame,
    pecked_part: float = 1.0,
    n_of_pecking: int = 5,
    lasso: bool = False,
    strict: bool = False,
    keep_original_1: bool = True,
    y_col: str = "Y",
    s_col: str = "S",
    random_state: int | None = None,
) -> tuple[float, np.ndarray, list[str]]:
    feature_cols = _feature_columns(train_df, y_col=y_col, s_col=s_col)
    s = pd.to_numeric(train_df[s_col], errors="raise").astype(int)

    s1_idx = np.flatnonzero(s.to_numpy() == 1)
    s0_idx = np.flatnonzero(s.to_numpy() == 0)
    if len(s1_idx) == 0 or len(s0_idx) == 0:
        raise ValueError("Both S=1 and S=0 are required for clust methods.")

    rng = np.random.default_rng(random_state)
    coeffs: list[np.ndarray] = []

    for k in range(1, n_of_pecking + 1):
        sample_size = max(1, int(round(len(s1_idx) * pecked_part)))
        sample_size = min(sample_size, len(s1_idx))
        sampled_s1 = rng.choice(s1_idx, size=sample_size, replace=False)
        c1_minus_q_idx = np.setdiff1d(s1_idx, sampled_s1, assume_unique=False)

        c0_plus_q = train_df.iloc[np.concatenate([s0_idx, sampled_s1])].copy()
        x_raw = c0_plus_q[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

        # Standaryzacja Z-score przed klasteryzacją — odpowiednik R's scale(X) w find_glm_coef2.
        # Zapewnia równy wkład każdej cechy do odległości euklidesowych w K-Means,
        # niezależnie od wstępnego skalowania (np. Min-Max z preprocessingu).
        scaler = StandardScaler()
        x_cluster = scaler.fit_transform(x_raw)

        cluster = KMeans(n_clusters=2, n_init=5, random_state=int(rng.integers(0, 2**31 - 1)))
        c0_plus_q["cluster"] = cluster.fit_predict(x_cluster)

        like1 = (
            c0_plus_q.groupby("cluster")[s_col]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

        c1 = c0_plus_q[c0_plus_q["cluster"] == like1].drop(columns=["cluster"])
        c0 = c0_plus_q[c0_plus_q["cluster"] != like1].drop(columns=["cluster"])

        c1_minus_q = train_df.iloc[c1_minus_q_idx]
        if keep_original_1:
            pos_df = pd.concat([c1_minus_q, c1, c0[c0[s_col] == 1]], axis=0)
            neg_df = c0[c0[s_col] == 0]
        else:
            pos_df = pd.concat([c1_minus_q, c1], axis=0)
            neg_df = c0

        to_model = pd.concat(
            [
                pos_df.assign(YY=1),
                neg_df.assign(YY=0),
            ],
            axis=0,
        )

        # Regresja logistyczna na oryginalnych (niezskalowanych) cechach.
        x = to_model[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        y = to_model["YY"].to_numpy().astype(int)

        if np.unique(y).size < 2:
            continue

        if lasso:
            clf = _make_l1_selector(cv=5)
            clf.fit(x, y)
            coef = np.concatenate([clf.intercept_, clf.coef_.ravel()])
            coef[np.abs(coef) < 1e-8] = 0.0
        else:
            clf = LogisticRegression(max_iter=3000, solver="lbfgs")
            clf.fit(x, y)
            coef = np.concatenate([clf.intercept_, clf.coef_.ravel()])

        coeffs.append(coef)

    if not coeffs:
        raise RuntimeError("No valid pecking iterations were completed.")

    mat = np.vstack(coeffs)
    if strict:
        final = np.where((mat == 0).any(axis=0), 0.0, mat.mean(axis=0))
    else:
        final = np.zeros(mat.shape[1], dtype=float)
        for idx in range(mat.shape[1]):
            non_zero = mat[:, idx][mat[:, idx] != 0.0]
            final[idx] = non_zero.mean() if non_zero.size else 0.0

    intercept = float(final[0])
    weights = final[1:]
    return intercept, weights, feature_cols


def fit_predict_clust(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lasso: bool = False,
    strict: bool = False,
    y_col: str = "Y",
    s_col: str = "S",
    random_state: int | None = None,
) -> MethodOutput:
    intercept, weights, feature_cols = estimate_clust_coefficients(
        train_df,
        lasso=lasso,
        strict=strict,
        y_col=y_col,
        s_col=s_col,
        random_state=random_state,
    )

    x_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    logits = intercept + x_test @ weights
    proba = _sigma(logits)
    return MethodOutput(
        proba=proba,
        label=_labels_from_proba(proba),
        meta={"intercept": intercept, "feature_cols": feature_cols},
    )


# ---------------------------------------------------------------------------
# Joint log-likelihood helpers (replicating R's logistic_fit_joint from mdai26)
# ---------------------------------------------------------------------------

def _sigma(s: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    result = np.empty_like(s, dtype=float)
    pos = s >= 0
    exp_neg = np.exp(-s[pos])
    result[pos] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(s[~pos])
    result[~pos] = exp_pos / (1.0 + exp_pos)
    return result


def _joint_neg_log_likelihood(
    par: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
) -> float:
    """Negative joint log-likelihood R(b, c) for PU data.

    Minimises  -sum[ s*log(c*sigma(x^T b)) + (1-s)*log(1 - c*sigma(x^T b)) ]
    with par = [beta0, beta1, ..., betap, c].
    Mirrors logLike_joint from mdai26/scoring_methods.R.
    """
    beta0 = par[0]
    beta = par[1:-1]
    c = par[-1]
    # clip c to a valid probability range to avoid log(0)
    c = float(np.clip(c, 1e-6, 1.0 - 1e-6))
    eta = x @ beta + beta0
    sigma_eta = _sigma(eta)
    term1 = np.clip(c * sigma_eta, 1e-15, None)
    term2 = np.clip(1.0 - c * sigma_eta, 1e-15, None)
    return -float(np.sum(s * np.log(term1) + (1.0 - s) * np.log(term2)))


def _joint_gradient(
    par: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Gradient of the joint negative log-likelihood w.r.t. par.

    Mirrors gr_joint from mdai26/scoring_methods.R.
    """
    beta0 = par[0]
    beta = par[1:-1]
    c = float(np.clip(par[-1], 1e-6, 1.0 - 1e-6))
    eta = x @ beta + beta0
    sigma_eta = _sigma(eta)
    var_eta = sigma_eta * (1.0 - sigma_eta)
    denom = np.clip(sigma_eta * (1.0 - c * sigma_eta), 1e-15, None)
    a = var_eta * ((-s + c * sigma_eta) / denom)
    x1 = np.column_stack([np.ones(len(x)), x])
    grad_beta = x1.T @ a
    grad_c = float(np.sum(-s / c + (1.0 - s) * sigma_eta / np.clip(1.0 - c * sigma_eta, 1e-15, None)))
    return np.append(grad_beta, grad_c)


def _fit_joint(
    x: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit the joint PU logistic model.

    Returns (beta_full, c_hat) where beta_full = [beta0, beta1, ..., betap].
    Mirrors logistic_fit_joint from mdai26/scoring_methods.R.
    """
    p = x.shape[1]
    par0 = np.zeros(p + 2)   # [beta0, beta1..betap, c]
    par0[-1] = 0.5            # initialise c at 0.5

    result = minimize(
        _joint_neg_log_likelihood,
        par0,
        jac=_joint_gradient,
        args=(x, s),
        method="BFGS",
        options={"maxiter": 10_000, "gtol": 1e-6},
    )
    par = result.x
    beta_full = par[:-1]   # [beta0, beta1, ..., betap]
    c_hat = float(np.clip(par[-1], 1e-6, 1.0 - 1e-6))
    return beta_full, c_hat


# ---------------------------------------------------------------------------
# LassoJoint
# ---------------------------------------------------------------------------

def fit_predict_lassojoint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_col: str = "Y",
    s_col: str = "S",
    nfolds: int | None = 10,
    delta_to_lambda: float = 0.5,
) -> MethodOutput:
    """LassoJoint method.

    Step 1 – Lasso feature selection:
      If nfolds is None: theoretical lambda = (log(p)/n)^(1/3).
      Otherwise: lambda selected via nfolds-fold CV (lambda.min), mirroring
      the mdai26 R implementation (cv.glmnet with nfolds=10, lambda.min).

    Step 2 – Thresholded support:
      Retain features with |coef| > delta, where delta = lambda * delta_to_lambda
      (default delta_to_lambda=0.5 matches the mdai26 R default).

    Step 3 – Joint logistic regression on support:
      Fit the PU joint model R(b,c) on the selected features,
      mirroring logistic_fit_joint from mdai26/scoring_methods.R.
    """
    feature_cols = _feature_columns(train_df, y_col=y_col, s_col=s_col)
    x_train, s_train, _ = _to_xy(train_df, feature_cols=feature_cols, target_col=s_col)
    x_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    n, p = x_train.shape

    if np.unique(s_train).size < 2:
        prob = float(s_train.mean()) if s_train.size else 0.0
        proba = np.full(shape=len(test_df), fill_value=prob, dtype=float)
        return MethodOutput(proba=proba, label=_labels_from_proba(proba), meta={"constant": prob})

    # ------------------------------------------------------------------
    # Step 1: Lasso – select lambda and fit
    # ------------------------------------------------------------------
    if nfolds is None:
        # Theoretical lambda matching R default: (log(p)/n)^(1/3)
        lam = (np.log(max(p, 2)) / n) ** (1.0 / 3.0)
        selector = LogisticRegression(
            penalty="l1",
            C=1.0 / (n * lam),
            solver="liblinear",
            max_iter=10_000,
            fit_intercept=True,
        )
        selector.fit(x_train, s_train)
        lam_used = lam
    else:
        # CV-based lambda (mirrors cv.glmnet lambda.min with nfolds=10)
        selector = _make_l1_selector(cv=nfolds)
        selector.fit(x_train, s_train)
        # Recover the effective regularisation strength as lambda
        # LogisticRegressionCV stores best C; lambda = 1/(n*C)
        best_C = float(selector.C_) if np.isscalar(selector.C_) else float(selector.C_[0])
        lam_used = 1.0 / (n * best_C)

    coef = selector.coef_.ravel()

    # ------------------------------------------------------------------
    # Step 2: Thresholded support  (|coef| > delta)
    # ------------------------------------------------------------------
    delta = lam_used * delta_to_lambda
    support = np.flatnonzero(np.abs(coef) > delta)

    # Fallback: if no feature survives the threshold, keep all features
    if support.size == 0:
        support = np.arange(p)

    x_train_s = x_train[:, support]
    x_test_s = x_test[:, support]
    selected_cols = [feature_cols[i] for i in support]

    # ------------------------------------------------------------------
    # Step 3: Joint logistic model on the selected support
    # ------------------------------------------------------------------
    try:
        beta_full, c_hat = _fit_joint(x_train_s, s_train)
    except Exception:
        # Fallback to plain LR if joint optimisation fails
        fallback = LogisticRegression(max_iter=3000, solver="lbfgs")
        fallback.fit(x_train_s, s_train)
        proba = fallback.predict_proba(x_test_s)[:, 1]
        return MethodOutput(
            proba=proba,
            label=_labels_from_proba(proba),
            meta={"selected_features": selected_cols, "fallback": "plain_lr"},
        )

    beta0 = beta_full[0]
    beta = beta_full[1:]
    # Posterior score: sigma(x^T b + b0)  — analogous to R est computation
    logits = x_test_s @ beta + beta0
    proba = _sigma(logits)

    return MethodOutput(
        proba=proba,
        label=_labels_from_proba(proba),
        meta={
            "selected_features": selected_cols,
            "c_hat": c_hat,
            "lambda": lam_used,
            "delta": delta,
        },
    )


def fit_predict_spy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spy_frac: float = 0.2,
    noise_level: float = 0.15,
    em1_iter: int = 5,
    em2_iter: int = 10,
    y_col: str = "Y",
    s_col: str = "S",
    random_state: int | None = None,
) -> MethodOutput:
    feature_cols = _feature_columns(train_df, y_col=y_col, s_col=s_col)
    rng = np.random.default_rng(random_state)

    train = train_df.copy()
    p_df = train[train[s_col] == 1].copy()
    m_df = train[train[s_col] == 0].copy()

    if len(p_df) == 0 or len(m_df) == 0:
        base = fit_predict_naive(train_df, test_df, y_col=y_col, s_col=s_col)
        base.meta["fallback"] = "naive"
        return base

    n_spy = int(np.ceil(spy_frac * len(p_df)))
    n_spy = max(1, min(n_spy, len(p_df)))
    spy_idx = rng.choice(p_df.index.to_numpy(), size=n_spy, replace=False)

    spies = p_df.loc[spy_idx].copy()
    p_clean = p_df.drop(index=spy_idx).copy()

    spies["spy"] = True
    m_df["spy"] = False
    p_clean["spy"] = False

    ms = pd.concat([m_df, spies], axis=0)
    ms["Pr_pos"] = 0.0
    p_clean["Pr_pos"] = 1.0

    model_em1: GaussianNB | None = None
    for _ in range(em1_iter):
        train_em1 = pd.concat(
            [
                p_clean.assign(Class=1, weight=1.0),
                ms.assign(Class=0, weight=1.0 - ms["Pr_pos"]),
            ],
            axis=0,
        )
        x = train_em1[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        y = train_em1["Class"].to_numpy().astype(int)
        w = train_em1["weight"].to_numpy(dtype=float)

        model_em1 = GaussianNB()
        model_em1.fit(x, y, sample_weight=w)

        x_ms = ms[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        ms["Pr_pos"] = model_em1.predict_proba(x_ms)[:, 1]

    spy_probs = ms.loc[ms["spy"], "Pr_pos"].sort_values().to_numpy()
    threshold_idx = int(np.ceil(noise_level * len(spy_probs))) - 1
    threshold_idx = int(np.clip(threshold_idx, 0, max(0, len(spy_probs) - 1)))
    threshold = float(spy_probs[threshold_idx]) if len(spy_probs) else 0.5

    n_df = ms[(~ms["spy"]) & (ms["Pr_pos"] < threshold)].copy()
    u_df = ms[(~ms["spy"]) & (ms["Pr_pos"] >= threshold)].copy()
    if len(n_df) > 0:
        n_df["Pr_pos"] = 0.0
    if len(u_df) > 0:
        u_df["Pr_pos"] = 0.5

    p_full = pd.concat([p_clean, spies], axis=0)
    p_full["Pr_pos"] = 1.0
    all_data = pd.concat([p_full, n_df, u_df], axis=0)

    model_em2: GaussianNB | None = None
    em2_success = True
    for _ in range(em2_iter):
        work = all_data.copy()
        work["Class"] = (work["Pr_pos"] >= 0.5).astype(int)
        work["weight"] = np.where(work["Class"] == 1, work["Pr_pos"], 1.0 - work["Pr_pos"])
        work = work[work["weight"] > 0]

        if len(work) == 0 or work["Class"].nunique() < 2:
            em2_success = False
            break

        x = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        y = work["Class"].to_numpy().astype(int)
        w = work["weight"].to_numpy(dtype=float)

        model_em2 = GaussianNB()
        model_em2.fit(x, y, sample_weight=w)

        x_all = all_data[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        all_data["Pr_pos"] = model_em2.predict_proba(x_all)[:, 1]

    final_model = model_em2 if (em2_success and model_em2 is not None) else model_em1
    if final_model is None:
        base = fit_predict_naive(train_df, test_df, y_col=y_col, s_col=s_col)
        base.meta["fallback"] = "naive"
        return base

    x_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    proba = final_model.predict_proba(x_test)[:, 1]
    return MethodOutput(
        proba=proba,
        label=_labels_from_proba(proba),
        meta={
            "threshold": threshold,
            "model_type": "em2" if (em2_success and model_em2 is not None) else "em1",
        },
    )


def run_all_methods(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_col: str = "Y",
    s_col: str = "S",
    random_state: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, MethodOutput]:
    outputs: dict[str, MethodOutput] = {}
    methods: list[tuple[str, Callable[[], MethodOutput]]] = [
        ("naive", lambda: fit_predict_naive(train_df, test_df, y_col=y_col, s_col=s_col)),
        (
            "clust",
            lambda: fit_predict_clust(
                train_df,
                test_df,
                lasso=False,
                strict=False,
                y_col=y_col,
                s_col=s_col,
                random_state=random_state,
            ),
        ),
        (
            "strict-lassclust",
            lambda: fit_predict_clust(
                train_df,
                test_df,
                lasso=True,
                strict=True,
                y_col=y_col,
                s_col=s_col,
                random_state=random_state,
            ),
        ),
        (
            "non-strict-lassclust",
            lambda: fit_predict_clust(
                train_df,
                test_df,
                lasso=True,
                strict=False,
                y_col=y_col,
                s_col=s_col,
                random_state=random_state,
            ),
        ),
        ("lassojoint", lambda: fit_predict_lassojoint(train_df, test_df, y_col=y_col, s_col=s_col)),
        (
            "spy",
            lambda: fit_predict_spy(
                train_df,
                test_df,
                y_col=y_col,
                s_col=s_col,
                random_state=random_state,
            ),
        ),
    ]

    for method_name, method_runner in methods:
        if progress_callback is not None:
            progress_callback(method_name)
        started_at = perf_counter()
        output = method_runner()
        elapsed_seconds = perf_counter() - started_at
        output.meta["runtime_seconds"] = float(elapsed_seconds)
        outputs[method_name] = output

    return outputs
