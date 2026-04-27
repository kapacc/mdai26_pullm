from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from py_puml.labelling import non_scar_labelling_mvc
from py_puml.methods import run_all_methods


def make_toy_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(scale=1.5, size=n)
    x3 = rng.normal(scale=0.5, size=n)
    logits = 1.2 * x1 - 0.9 * x2 + 0.3 * x3
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "Y": y})


def main() -> None:
    df = make_toy_dataset()
    df = non_scar_labelling_mvc(df, target_c_calc=0.5, n_vars=2)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=7, stratify=df["S"])
    outputs = run_all_methods(train_df, test_df)

    print("Smoke benchmark finished. Methods and mean predicted probability:")
    for name, out in outputs.items():
        print(f"{name:22s} mean_proba={out.proba.mean():.4f}")


if __name__ == "__main__":
    main()
