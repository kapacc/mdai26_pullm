from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _ensure_numeric_y(df: pd.DataFrame, y_col: str) -> pd.Series:
    if y_col not in df.columns:
        raise KeyError(f"Missing required column: {y_col}")
    y = pd.to_numeric(df[y_col], errors="raise")
    if not y.isin([0, 1]).all():
        raise ValueError(f"Column {y_col} must be binary (0/1).")
    return y.astype(int)


def non_scar_labelling_mvc(
    df: pd.DataFrame,
    target_c_calc: float,
    n_vars: int = 1,
    y_col: str = "Y",
    s_col: str = "S",
) -> pd.DataFrame:
    """Create non-SCAR labels translated from non_scar_labelling.mvc in mdai26.

    MVC stands for Most Variable Columns.

    The function ranks rows by the sum of top-variance columns, estimates a
    linear relation between rank fraction and observed c values, and then
    generates S labels that target the requested c value.

    Parameters
    ----------
    n_vars : int, default=1
        Number of top-variance features used to build ranking scores.
        Effective value is bounded by available features with the project rule
        to avoid using all features at once: use at most p-1 features when
        p > 1, where p is the number of candidate feature columns. For p == 1,
        fallback keeps n_vars == 1 so ranking remains defined.
    """
    if not 0 < target_c_calc < 1:
        raise ValueError("target_c_calc must be in (0, 1).")
    if n_vars < 1:
        raise ValueError("n_vars must be >= 1.")

    out = df.copy()
    y = _ensure_numeric_y(out, y_col)

    feature_cols = [c for c in out.columns if c not in {y_col, s_col}]
    if not feature_cols:
        raise ValueError("No feature columns available after excluding Y/S.")

    num_df = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    if num_df.isna().all(axis=None):
        raise ValueError("Feature matrix became all-NaN after numeric coercion.")
    num_df = num_df.fillna(0.0)

    if n_vars >= num_df.shape[1]:
        # Intentional contract: cap to p-1 when possible, but keep 1 feature
        # for p == 1 so score/rank computation is always possible.
        n_vars = max(1, num_df.shape[1] - 1)

    variances = num_df.var(axis=0)
    top_cols = variances.sort_values(ascending=False).index[:n_vars]

    score = num_df[top_cols].sum(axis=1)
    rn = score.rank(method="first", ascending=True)
    rn_frac = rn / len(out)

    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
    c_values = []
    y_sum = max(int(y.sum()), 1)
    for lvl in levels:
        s_lvl = ((rn_frac < lvl).astype(int) * y).sum()
        c_values.append(float(s_lvl) / float(y_sum))

    slope, intercept = np.polyfit(levels, np.array(c_values), deg=1)
    if abs(slope) < 1e-12:
        target_rank_fraction = target_c_calc
    else:
        target_rank_fraction = (target_c_calc - intercept) / slope

    target_rank_fraction = float(np.clip(target_rank_fraction, 0.0, 1.0))
    out[s_col] = ((rn_frac < target_rank_fraction).astype(int) * y).astype(int)
    out[y_col] = y
    return out


def scar_labelling(
    df: pd.DataFrame,
    target_c_calc: float,
    y_col: str = "Y",
    s_col: str = "S",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Create SCAR labels translated from scar_labelling in mdai26."""
    if not 0 <= target_c_calc <= 1:
        raise ValueError("target_c_calc must be in [0, 1].")

    out = df.copy()
    y = _ensure_numeric_y(out, y_col)

    rng = np.random.default_rng(random_state)
    draws = rng.binomial(1, target_c_calc, size=len(out))
    out[s_col] = (y.to_numpy() * draws).astype(int)

    # R implementation coerces all columns to numeric in scar_labelling.
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def non_scar_labelling_classic(
    df: pd.DataFrame,
    target_c_calc: float,
    y_col: str = "Y",
    s_col: str = "S",
    random_state: int | None = None,
    alpha_start: float = -10.0,
    alpha_stop: float = 10.0,
    alpha_points: int = 401,
    feature_cols: list[str] | None = None,
    theta: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create non-SCAR labels with classic logistic propensity calibration.

    This implementation follows a common non-SCAR simulation scheme:
    - define e(x) = sigmoid(alpha + theta^T x),
    - choose alpha on a dense grid so that mean(e(x) | Y=1) is close to target_c_calc,
    - sample S ~ Bernoulli(e(x)) only for Y=1.
    """
    if not 0 < target_c_calc < 1:
        raise ValueError("target_c_calc must be in (0, 1).")
    if alpha_points < 2:
        raise ValueError("alpha_points must be >= 2.")

    out = df.copy()
    y = _ensure_numeric_y(out, y_col)
    pos_mask = y == 1
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        raise ValueError("Cannot create PU labels: no positive samples (Y=1).")

    if feature_cols is None:
        feature_cols = [c for c in out.columns if c not in {y_col, s_col}]
    if not feature_cols:
        raise ValueError("No feature columns available after excluding Y/S.")

    num_df = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    if num_df.isna().all(axis=None):
        raise ValueError("Feature matrix became all-NaN after numeric coercion.")
    num_df = num_df.fillna(0.0)

    x = num_df.to_numpy(dtype=float)
    p = x.shape[1]
    if theta is None:
        theta_vec = np.ones(p, dtype=float)
    else:
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if theta_vec.shape[0] != p:
            raise ValueError(
                f"theta length ({theta_vec.shape[0]}) does not match number of features ({p})."
            )

    lin_pred = x @ theta_vec
    alpha_grid = np.linspace(alpha_start, alpha_stop, num=alpha_points)

    # Numerically stable sigmoid.
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    pos_idx = pos_mask.to_numpy()
    best_alpha = alpha_grid[0]
    best_dist = np.inf

    for alpha in alpha_grid:
        e = _sigmoid(alpha + lin_pred)
        dist = abs(float(e[pos_idx].mean()) - target_c_calc)
        if dist < best_dist:
            best_dist = dist
            best_alpha = alpha

    e_opt = _sigmoid(best_alpha + lin_pred)

    rng = np.random.default_rng(random_state)
    s = np.zeros(len(out), dtype=int)
    s[pos_idx] = rng.binomial(1, e_opt[pos_idx]).astype(int)

    if int(s.sum()) < 2 and n_pos >= 2:
        promoted = rng.choice(np.where(pos_idx)[0], size=2, replace=False)
        s[promoted] = 1
        warnings.warn(
            "<2 observations with S=1. Two positive instances were assigned S=1.",
            RuntimeWarning,
            stacklevel=2,
        )

    out[y_col] = y
    out[s_col] = s
    return out
