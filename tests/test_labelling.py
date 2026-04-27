import numpy as np
import pandas as pd
import pytest

from py_puml.labelling import (
    non_scar_labelling_classic,
    non_scar_labelling_mvc,
    scar_labelling,
)


def _toy_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(scale=2.0, size=n)
    y = (x1 + x2 > 0.2).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "Y": y})


def test_non_scar_adds_binary_s_column() -> None:
    df = _toy_df()
    out = non_scar_labelling_mvc(df, target_c_calc=0.5, n_vars=2)

    assert "S" in out.columns
    assert set(out["S"].unique()).issubset({0, 1})
    assert (out["S"] <= out["Y"]).all()


def test_non_scar_mvc_rejects_non_positive_n_vars() -> None:
    df = _toy_df()

    with pytest.raises(ValueError, match="n_vars must be >= 1"):
        non_scar_labelling_mvc(df, target_c_calc=0.5, n_vars=0)


def test_non_scar_mvc_single_feature_frame_is_supported() -> None:
    df = _toy_df()[["x1", "Y"]].copy()

    out = non_scar_labelling_mvc(df, target_c_calc=0.5, n_vars=5)

    assert "S" in out.columns
    assert set(out["S"].unique()).issubset({0, 1})
    assert (out["S"] <= out["Y"]).all()


def test_non_scar_mvc_caps_n_vars_to_p_minus_one_when_p_gt_1() -> None:
    base = _toy_df(n=300)
    # Two feature columns -> contract says n_vars>=2 should effectively use 1.
    df_two_features = base[["x1", "x2", "Y"]].copy()
    # Add near-constant third feature so p becomes 3 and cap allows 2 features.
    df_three_features = df_two_features.copy()
    df_three_features["x3"] = 1e-12 * np.arange(len(df_three_features), dtype=float)

    out_two = non_scar_labelling_mvc(df_two_features, target_c_calc=0.5, n_vars=2)
    out_three = non_scar_labelling_mvc(df_three_features, target_c_calc=0.5, n_vars=2)

    # With p=2, n_vars=2 is capped to 1 feature; with p=3, two features are used.
    # The induced rankings should therefore differ for this crafted dataset.
    assert not out_two["S"].equals(out_three["S"])


def test_scar_reproducible_with_seed() -> None:
    df = _toy_df()
    a = scar_labelling(df, target_c_calc=0.4, random_state=123)
    b = scar_labelling(df, target_c_calc=0.4, random_state=123)

    assert (a["S"] == b["S"]).all()
    assert set(a["S"].unique()).issubset({0, 1})


def test_non_scar_classic_adds_binary_s_column_and_respects_y() -> None:
    df = _toy_df()
    out = non_scar_labelling_classic(df, target_c_calc=0.5, random_state=123)

    assert "S" in out.columns
    assert set(out["S"].unique()).issubset({0, 1})
    assert (out["S"] <= out["Y"]).all()


def test_non_scar_classic_reaches_target_c_calc_on_average() -> None:
    df = _toy_df(n=1200)
    out = non_scar_labelling_classic(df, target_c_calc=0.35, random_state=11)

    pos = out["Y"] == 1
    observed = float(out.loc[pos, "S"].mean())
    assert abs(observed - 0.35) < 0.08
