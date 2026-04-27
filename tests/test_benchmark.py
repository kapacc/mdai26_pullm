import numpy as np
import pandas as pd
import pytest
from os import PathLike

import py_puml.benchmark as benchmark_module
import scripts.run_benchmark as run_benchmark_module
from py_puml.benchmark import (
    BENCHMARK_MODE_PER_DATASET,
    BENCHMARK_MODE_UNION_TRAIN,
    _sample_preprocessed_dataset,
    _split_train_test_by_y,
    benchmark_many_datasets,
    benchmark_many_datasets_for_strategies,
    benchmark_many_datasets_union_train,
    benchmark_single_dataset,
)
from py_puml.methods import MethodOutput


def _make_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (0.7 * x1 - 0.4 * x2 > 0.0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "Y": y})


def _make_preprocessed_df(n_positive: int, n_negative: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive = pd.DataFrame(
        {
            "x1": rng.normal(size=n_positive),
            "x2": rng.normal(size=n_positive),
            "Y": np.ones(n_positive, dtype=int),
        }
    )
    negative = pd.DataFrame(
        {
            "x1": rng.normal(size=n_negative),
            "x2": rng.normal(size=n_negative),
            "Y": np.zeros(n_negative, dtype=int),
        }
    )
    return pd.concat([positive, negative], axis=0, ignore_index=True)


def test_sample_preprocessed_dataset_samples_both_classes_per_seed() -> None:
    df = _make_preprocessed_df(n_positive=3_000, n_negative=9_000, seed=1)

    sample_a = _sample_preprocessed_dataset(df=df, seed=1, sample_size=10_000)
    sample_b = _sample_preprocessed_dataset(df=df, seed=2, sample_size=10_000)

    assert len(sample_a) == 10_000
    assert len(sample_b) == 10_000
    assert sample_a["Y"].sum() == 2_500
    assert sample_b["Y"].sum() == 2_500
    assert set(sample_a.index[sample_a["Y"] == 1]) != set(sample_b.index[sample_b["Y"] == 1])
    assert set(sample_a.index[sample_a["Y"] == 0]) != set(sample_b.index[sample_b["Y"] == 0])


def test_sample_preprocessed_dataset_is_reproducible_for_same_seed() -> None:
    df = _make_preprocessed_df(n_positive=3_000, n_negative=9_000, seed=3)

    sample_a = _sample_preprocessed_dataset(df=df, seed=11, sample_size=10_000)
    sample_b = _sample_preprocessed_dataset(df=df, seed=11, sample_size=10_000)

    assert sample_a.equals(sample_b)
    assert set(sample_a.index[sample_a["Y"] == 1]) == set(sample_b.index[sample_b["Y"] == 1])
    assert set(sample_a.index[sample_a["Y"] == 0]) == set(sample_b.index[sample_b["Y"] == 0])


def test_sample_preprocessed_dataset_skips_small_frames() -> None:
    df = _make_preprocessed_df(n_positive=1_500, n_negative=2_500, seed=2)

    sampled = _sample_preprocessed_dataset(df=df, seed=1, sample_size=10_000)

    assert sampled is df
    assert sampled.equals(df)


def test_split_train_test_by_y_preserves_class_ratio() -> None:
    df = _make_preprocessed_df(n_positive=300, n_negative=700, seed=11)

    train_df, test_df = _split_train_test_by_y(df=df, test_size=0.3, seed=7)

    train_pos_rate = float(train_df["Y"].mean())
    test_pos_rate = float(test_df["Y"].mean())
    full_pos_rate = float(df["Y"].mean())

    assert set(train_df["Y"].unique()) == {0, 1}
    assert set(test_df["Y"].unique()) == {0, 1}
    assert abs(train_pos_rate - full_pos_rate) < 0.01
    assert abs(test_pos_rate - full_pos_rate) < 0.01


def test_benchmark_applies_labelling_only_to_train(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_preprocessed_df(n_positive=60, n_negative=60, seed=13)

    calls = {"labelling_rows": 0, "train_has_s": False, "test_has_s": True}

    def fake_apply_labelling_strategy(
        df: pd.DataFrame,
        strategy: str,
        c_calc: float,
        seed: int,
    ) -> pd.DataFrame:
        out = df.copy()
        calls["labelling_rows"] = len(out)
        out["S"] = out["Y"].astype(int)
        return out

    def fake_run_all_methods(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_col: str = "Y",
        s_col: str = "S",
        random_state: int | None = None,
        progress_callback=None,
    ) -> dict[str, MethodOutput]:
        calls["train_has_s"] = "S" in train_df.columns
        calls["test_has_s"] = "S" in test_df.columns
        proba = np.full(len(test_df), 0.5, dtype=float)
        return {
            "dummy": MethodOutput(
                proba=proba,
                label=(proba >= 0.5).astype(int),
                meta={},
            )
        }

    monkeypatch.setattr(benchmark_module, "_apply_labelling_strategy", fake_apply_labelling_strategy)
    monkeypatch.setattr(benchmark_module, "run_all_methods", fake_run_all_methods)

    runs = benchmark_single_dataset(
        df=df,
        dataset_name="toy",
        seeds=(1,),
        c_calc=0.5,
        labelling_strategy="mvc",
        test_size=0.3,
    )

    assert not runs.empty
    assert calls["labelling_rows"] == 84
    assert calls["train_has_s"] is True
    assert calls["test_has_s"] is False


def test_benchmark_single_dataset_uses_seeded_sampling_for_both_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_preprocessed_df(n_positive=3_000, n_negative=9_000, seed=21)

    sampled_indices_by_seed: dict[int, list[set[int]]] = {}

    def fake_split_train_test_by_y(
        df: pd.DataFrame,
        test_size: float,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sampled_indices_by_seed.setdefault(seed, []).append(set(df.index))
        split_at = int(round(len(df) * (1.0 - test_size)))
        return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()

    def fake_apply_labelling_strategy(
        df: pd.DataFrame,
        strategy: str,
        c_calc: float,
        seed: int,
    ) -> pd.DataFrame:
        out = df.copy()
        out["S"] = out["Y"].astype(int)
        return out

    def fake_run_all_methods(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_col: str = "Y",
        s_col: str = "S",
        random_state: int | None = None,
        progress_callback=None,
    ) -> dict[str, MethodOutput]:
        proba = np.zeros(len(test_df), dtype=float)
        return {
            "dummy": MethodOutput(
                proba=proba,
                label=np.zeros(len(test_df), dtype=int),
                meta={},
            )
        }

    monkeypatch.setattr(benchmark_module, "_split_train_test_by_y", fake_split_train_test_by_y)
    monkeypatch.setattr(benchmark_module, "_apply_labelling_strategy", fake_apply_labelling_strategy)
    monkeypatch.setattr(benchmark_module, "run_all_methods", fake_run_all_methods)

    benchmark_single_dataset(
        df=df,
        dataset_name="toy",
        seeds=(1, 2, 1),
        c_calc=0.5,
        labelling_strategy="mvc",
        sample_size=10_000,
    )

    sampled_seed_1_first = sampled_indices_by_seed[1][0]
    sampled_seed_1_second = sampled_indices_by_seed[1][1]
    sampled_seed_2 = sampled_indices_by_seed[2][0]
    assert sampled_seed_1_first == sampled_seed_1_second
    assert sampled_seed_1_first != sampled_seed_2


@pytest.mark.parametrize("labelling_strategy", ["mvc", "classic"])
def test_benchmark_many_datasets_returns_tables(labelling_strategy: str) -> None:
    datasets = {
        "a": _make_df(120, 1),
        "b": _make_df(140, 2),
    }

    runs, summary = benchmark_many_datasets(
        datasets=datasets,
        seeds=(1, 2),
        c_calc=0.5,
        labelling_strategy=labelling_strategy,
    )

    assert not runs.empty
    assert not summary.empty
    assert {
        "dataset",
        "seed",
        "method",
        "c_calc",
        "labelling_strategy",
        "benchmark_mode",
        "run_time_min",
        "auc",
        "pr_auc",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "fmi",
    }.issubset(runs.columns)
    assert {
        "dataset",
        "method",
        "c_calc",
        "labelling_strategy",
        "benchmark_mode",
        "run_time_min_mean",
        "run_time_min_std",
        "auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "tp_mean",
        "tn_mean",
        "fp_mean",
        "fn_mean",
        "fmi_mean",
    }.issubset(summary.columns)
    assert set(runs["labelling_strategy"]) == {labelling_strategy}
    assert set(summary["labelling_strategy"]) == {labelling_strategy}
    assert set(runs["benchmark_mode"]) == {BENCHMARK_MODE_PER_DATASET}
    assert set(summary["benchmark_mode"]) == {BENCHMARK_MODE_PER_DATASET}
    assert runs["run_time_min"].notna().all()
    assert (runs["run_time_min"] >= 0.0).all()
    assert summary["run_time_min_mean"].notna().all()
    assert (summary["run_time_min_mean"] >= 0.0).all()


def test_benchmark_many_datasets_for_strategies_returns_combined_tables() -> None:
    datasets = {
        "a": _make_df(120, 1),
        "b": _make_df(140, 2),
    }

    runs, summary = benchmark_many_datasets_for_strategies(
        datasets=datasets,
        seeds=(1, 2),
        c_calc=0.5,
        labelling_strategies=("mvc", "classic"),
    )

    assert not runs.empty
    assert not summary.empty
    assert {
        "dataset",
        "seed",
        "method",
        "c_calc",
        "labelling_strategy",
        "benchmark_mode",
        "run_time_min",
        "auc",
        "pr_auc",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "fmi",
    }.issubset(runs.columns)
    assert {
        "dataset",
        "method",
        "c_calc",
        "labelling_strategy",
        "benchmark_mode",
        "run_time_min_mean",
        "run_time_min_std",
        "auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "tp_mean",
        "tn_mean",
        "fp_mean",
        "fn_mean",
        "fmi_mean",
    }.issubset(summary.columns)
    assert set(runs["labelling_strategy"]) == {"mvc", "classic"}
    assert set(summary["labelling_strategy"]) == {"mvc", "classic"}
    assert set(runs["benchmark_mode"]) == {BENCHMARK_MODE_PER_DATASET}
    assert set(summary["benchmark_mode"]) == {BENCHMARK_MODE_PER_DATASET}
    assert runs["run_time_min"].notna().all()
    assert (runs["run_time_min"] >= 0.0).all()
    assert summary["run_time_min_mean"].notna().all()
    assert (summary["run_time_min_mean"] >= 0.0).all()


def test_union_train_benchmark_reports_per_dataset_and_mode() -> None:
    datasets = {
        "a": _make_df(120, 1),
        "b": _make_df(140, 2),
    }

    runs, summary = benchmark_many_datasets_union_train(
        datasets=datasets,
        seeds=(1, 2),
        c_calc=0.5,
        labelling_strategy="mvc",
    )

    assert not runs.empty
    assert not summary.empty
    assert set(runs["dataset"]) == {"a", "b"}
    assert set(summary["dataset"]) == {"a", "b"}
    assert set(runs["benchmark_mode"]) == {BENCHMARK_MODE_UNION_TRAIN}
    assert set(summary["benchmark_mode"]) == {BENCHMARK_MODE_UNION_TRAIN}


def test_union_train_labels_once_and_reuses_single_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = {
        "a": _make_df(120, 1),
        "b": _make_df(140, 2),
    }
    calls = {"label_rows": [], "fit_calls": 0, "test_rows": []}

    def fake_apply_labelling_strategy(
        df: pd.DataFrame,
        strategy: str,
        c_calc: float,
        seed: int,
    ) -> pd.DataFrame:
        out = df.copy()
        calls["label_rows"].append(len(out))
        out["S"] = out["Y"].astype(int)
        return out

    def fake_run_all_methods(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_col: str = "Y",
        s_col: str = "S",
        random_state: int | None = None,
        progress_callback=None,
    ) -> dict[str, MethodOutput]:
        calls["fit_calls"] += 1
        calls["test_rows"].append(len(test_df))
        assert "S" in train_df.columns
        assert "S" not in test_df.columns
        proba = np.full(len(test_df), 0.5, dtype=float)
        return {"dummy": MethodOutput(proba=proba, label=(proba >= 0.5).astype(int), meta={})}

    monkeypatch.setattr(benchmark_module, "_apply_labelling_strategy", fake_apply_labelling_strategy)
    monkeypatch.setattr(benchmark_module, "run_all_methods", fake_run_all_methods)

    runs, _ = benchmark_many_datasets_union_train(
        datasets=datasets,
        seeds=(1,),
        c_calc=0.5,
        labelling_strategy="mvc",
    )

    assert set(runs["dataset"]) == {"a", "b"}
    assert calls["fit_calls"] == 1
    assert len(calls["label_rows"]) == 1
    assert calls["test_rows"][0] > 0


def test_union_train_raises_on_feature_schema_mismatch() -> None:
    datasets = {
        "a": _make_df(120, 1),
        "b": _make_df(140, 2).assign(x3=np.ones(140)),
    }

    with pytest.raises(ValueError, match="feature columns"):
        benchmark_many_datasets_union_train(
            datasets=datasets,
            seeds=(1,),
            c_calc=0.5,
            labelling_strategy="mvc",
        )


def test_run_benchmark_csv_defaults_to_preprocessed_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_datasets(
        source: str = "speakleash",
        dataset_names: list[str] | None = None,
        data_dir: str | PathLike[str] = "data",
        max_rows: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        captured["source"] = source
        captured["dataset_names"] = dataset_names
        captured["data_dir"] = str(data_dir)
        captured["max_rows"] = max_rows
        return {"toy": _make_df(40, 5)}

    def fake_benchmark_many_datasets(
        datasets: dict[str, pd.DataFrame],
        seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
        c_calc: float = 0.5,
        threshold: float = 0.5,
        labelling_strategy: str = "mvc",
        sample_size: int | None = None,
        use_smote: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured["benchmark_call"] = {
            "datasets": list(datasets.keys()),
            "seeds": seeds,
            "c_calc": c_calc,
            "threshold": threshold,
            "labelling_strategy": labelling_strategy,
            "sample_size": sample_size,
            "use_smote": use_smote,
        }
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(run_benchmark_module, "load_datasets", fake_load_datasets)
    monkeypatch.setattr(run_benchmark_module, "benchmark_many_datasets", fake_benchmark_many_datasets)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *args, **kwargs: None, raising=False)

    run_benchmark_module.main(source="csv")

    assert captured["source"] == "csv"
    assert captured["data_dir"] == "data_preprocessed"
    assert captured["benchmark_call"]["datasets"] == ["toy"]
    assert captured["benchmark_call"]["threshold"] == 0.5
    assert captured["benchmark_call"]["sample_size"] == benchmark_module.DEFAULT_PRETRAIN_SAMPLE_SIZE


def test_run_benchmark_csv_rejects_missing_y(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_datasets(
        source: str = "speakleash",
        dataset_names: list[str] | None = None,
        data_dir: str = "data",
        max_rows: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        return {"toy": pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})}

    monkeypatch.setattr(run_benchmark_module, "load_datasets", fake_load_datasets)

    with pytest.raises(ValueError, match="CSV benchmark input must already contain a 'Y' column"):
        run_benchmark_module.main(source="csv")
