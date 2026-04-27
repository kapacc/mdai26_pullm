import numpy as np
import pandas as pd

from py_puml.labelling import non_scar_labelling_mvc
from py_puml.methods import run_all_methods


def _data(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 - 0.3 * x2 > 0.0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "Y": y})


def test_methods_return_probabilities_for_all_targets() -> None:
    df = non_scar_labelling_mvc(_data(), target_c_calc=0.5, n_vars=2)
    train = df.iloc[:120].copy()
    test = df.iloc[120:].copy()

    outputs = run_all_methods(train, test)
    expected = {
        "naive",
        "clust",
        "strict-lassclust",
        "non-strict-lassclust",
        "lassojoint",
        "spy",
    }
    assert set(outputs.keys()) == expected

    for out in outputs.values():
        assert len(out.proba) == len(test)
        assert len(out.label) == len(test)
        assert np.isfinite(out.proba).all()
