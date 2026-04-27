from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .labelling import non_scar_labelling_classic, non_scar_labelling_mvc
from .methods import run_all_methods
from .metrics import compute_metrics


DEFAULT_PRETRAIN_SAMPLE_SIZE = 10_000
BENCHMARK_MODE_PER_DATASET = "per-dataset"
BENCHMARK_MODE_UNION_TRAIN = "union-train"


def _log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [benchmark] {message}", flush=True)


def _apply_smote(
    df: pd.DataFrame,
    seed: int,
    y_col: str = "Y",
) -> pd.DataFrame:
    """Apply SMOTE oversampling to the minority class defined by *y_col*.

    Mirrors the R pipeline in mdai26/main.R::

        smote_result <- SMOTE(X=X_train, target=target_train, K=5, dup_size=0)

    The column *y_col* is used as the target; all other columns are treated as
    features. The returned DataFrame has the same column layout as *df* and a
    reset integer index.

    Requires ``imbalanced-learn`` (``pip install imbalanced-learn``).
    """
    try:
        from imblearn.over_sampling import SMOTE  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "SMOTE requires imbalanced-learn. Install it with: pip install imbalanced-learn"
        ) from exc

    feature_cols = [c for c in df.columns if c != y_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[y_col], errors="raise").astype(int)

    if y.nunique() < 2:
        return df.copy()

    sm = SMOTE(k_neighbors=5, random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)

    result = pd.DataFrame(X_res, columns=feature_cols)
    result[y_col] = y_res
    return result


def _apply_labelling_strategy(
    df: pd.DataFrame,
    strategy: str,
    c_calc: float,
    seed: int,
) -> pd.DataFrame:
    if strategy == "mvc":
        # Benchmark configuration uses n_vars=2; final effective value follows
        # non_scar_labelling_mvc contract (including p-1 cap and p=1 fallback).
        return non_scar_labelling_mvc(df, target_c_calc=c_calc, n_vars=2)
    if strategy == "classic":
        return non_scar_labelling_classic(df, target_c_calc=c_calc, random_state=seed)
    raise ValueError(
        f"Unsupported labelling scheme: {strategy}. Expected one of: ['mvc', 'classic']."
    )


def _sample_preprocessed_dataset(
    df: pd.DataFrame,
    seed: int,
    sample_size: int = DEFAULT_PRETRAIN_SAMPLE_SIZE,
) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df

    if "Y" not in df.columns:
        raise ValueError("Sampling requires a 'Y' column in the preprocessed dataset.")

    y = pd.to_numeric(df["Y"], errors="raise").astype(int)
    positive_df = df.loc[y == 1]
    negative_df = df.loc[y == 0]

    if positive_df.empty or negative_df.empty:
        return df.sample(n=sample_size, random_state=seed)

    positive_rate = len(positive_df) / len(df)
    desired_positive = int(round(sample_size * positive_rate))
    desired_negative = sample_size - desired_positive

    positive_target = min(desired_positive, len(positive_df))
    negative_target = min(desired_negative, len(negative_df))

    remaining = sample_size - positive_target - negative_target
    if remaining > 0:
        positive_room = len(positive_df) - positive_target
        positive_extra = min(remaining, positive_room)
        positive_target += positive_extra
        remaining -= positive_extra

    if remaining > 0:
        negative_room = len(negative_df) - negative_target
        negative_extra = min(remaining, negative_room)
        negative_target += negative_extra

    positive_sample = (
        positive_df.sample(n=positive_target, random_state=seed)
        if positive_target > 0
        else positive_df.iloc[0:0]
    )
    negative_sample = (
        negative_df.sample(n=negative_target, random_state=seed)
        if negative_target > 0
        else negative_df.iloc[0:0]
    )

    sampled = pd.concat([positive_sample, negative_sample], axis=0)
    return sampled.sample(frac=1, random_state=seed)


def _split_train_test_by_y(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Y" not in df.columns:
        raise ValueError("Train/test split requires a 'Y' column.")

    y = pd.to_numeric(df["Y"], errors="raise").astype(int)
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in {"Y", "S"}]


def _validate_common_feature_schema(datasets: dict[str, pd.DataFrame]) -> list[str]:
    dataset_items = tuple(datasets.items())
    if not dataset_items:
        return []

    reference_name, reference_df = dataset_items[0]
    reference_cols = _feature_columns(reference_df)
    reference_set = set(reference_cols)

    mismatches: list[str] = []
    for name, df in dataset_items[1:]:
        current_cols = _feature_columns(df)
        current_set = set(current_cols)
        if current_set != reference_set:
            missing = sorted(reference_set - current_set)
            extra = sorted(current_set - reference_set)
            mismatches.append(
                f"{name}: missing={missing or '-'} extra={extra or '-'}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "All datasets must share the same feature columns for union-train mode. "
            f"Reference dataset={reference_name}. Details: {mismatch_text}"
        )

    return reference_cols


def _align_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    ordered_cols = [*feature_cols, "Y"]
    if "S" in df.columns:
        ordered_cols.append("S")
    return df.loc[:, ordered_cols].copy()


def _runtime_minutes_from_meta(meta: dict[str, object]) -> float:
    runtime_seconds = meta.get("runtime_seconds")
    if runtime_seconds is None:
        return float("nan")
    return float(runtime_seconds) / 60.0


def _prepare_union_seed_data(
    datasets: dict[str, pd.DataFrame],
    seed: int,
    sample_size: int | None,
    test_size: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    feature_cols = _validate_common_feature_schema(datasets)
    train_parts: list[pd.DataFrame] = []
    test_by_dataset: dict[str, pd.DataFrame] = {}

    for dataset_name, df in datasets.items():
        working_df = (
            _sample_preprocessed_dataset(df=df, seed=seed, sample_size=sample_size)
            if sample_size is not None
            else df
        )
        train_df, test_df = _split_train_test_by_y(
            df=working_df,
            test_size=test_size,
            seed=seed,
        )
        train_parts.append(_align_columns(train_df, feature_cols))
        test_by_dataset[dataset_name] = _align_columns(test_df, feature_cols)

    union_train = pd.concat(train_parts, axis=0, ignore_index=True) if train_parts else pd.DataFrame()
    return union_train, test_by_dataset


def benchmark_single_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    c_calc: float = 0.5,
    threshold: float = 0.5,
    labelling_strategy: str = "mvc",
    test_size: float = 0.3,
    sample_size: int | None = None,
    use_smote: bool = False,
) -> pd.DataFrame:
    """Run benchmark for a single dataset.

    Parameters
    ----------
    use_smote:
        When True, apply SMOTE oversampling to the training fold (on the *Y*
        column) before the labelling step. Mirrors the SMOTE preprocessing
        used in mdai26/main.R with ``K=5, dup_size=0``.
    """
    rows: list[dict[str, object]] = []
    seed_list = tuple(seeds)

    for seed_index, seed in enumerate(seed_list, start=1):
        _log_progress(
            f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: preparing data"
        )
        working_df = (
            _sample_preprocessed_dataset(df=df, seed=seed, sample_size=sample_size)
            if sample_size is not None
            else df
        )
        if sample_size is not None:
            _log_progress(
                f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: sampled {len(working_df)} rows"
            )
        _log_progress(
            f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: splitting train/test"
        )
        train_df, test_df = _split_train_test_by_y(
            df=working_df,
            test_size=test_size,
            seed=seed,
        )

        # SMOTE is applied to the training fold before labelling, on the Y column.
        # This mirrors mdai26/main.R where SMOTE(X=X_train, target=S_train) is called
        # before find_glm_coef2. Applying it on Y (true labels) enriches the positive
        # class prior to PU label generation, improving cluster quality.
        if use_smote:
            _log_progress(
                f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: applying SMOTE"
            )
            train_df = _apply_smote(train_df, seed=seed, y_col="Y")

        _log_progress(
            f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: applying labelling"
        )
        train_df = _apply_labelling_strategy(
            df=train_df,
            strategy=labelling_strategy,
            c_calc=c_calc,
            seed=seed,
        )

        _log_progress(
            f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: fitting methods"
        )
        outputs = run_all_methods(
            train_df,
            test_df,
            random_state=seed,
            progress_callback=lambda method_name: _log_progress(
                f"dataset={dataset_name} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: method={method_name}"
            ),
        )
        y_true = pd.to_numeric(test_df["Y"], errors="raise").to_numpy().astype(int)

        for method_name, out in outputs.items():
            m = compute_metrics(y_true, out.proba, threshold=threshold)
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": method_name,
                    "c_calc": c_calc,
                    "labelling_strategy": labelling_strategy,
                    "smote": use_smote,
                    "benchmark_mode": BENCHMARK_MODE_PER_DATASET,
                    "run_time_min": _runtime_minutes_from_meta(out.meta),
                    **asdict(m),
                }
            )

    return pd.DataFrame(rows)


def benchmark_many_datasets_union_train(
    datasets: dict[str, pd.DataFrame],
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    c_calc: float = 0.5,
    threshold: float = 0.5,
    labelling_strategy: str = "mvc",
    sample_size: int | None = None,
    test_size: float = 0.3,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    seed_list = tuple(seeds)

    for seed_index, seed in enumerate(seed_list, start=1):
        _log_progress(
            f"mode={BENCHMARK_MODE_UNION_TRAIN} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: splitting datasets"
        )
        union_train_df, test_by_dataset = _prepare_union_seed_data(
            datasets=datasets,
            seed=seed,
            sample_size=sample_size,
            test_size=test_size,
        )

        if use_smote:
            _log_progress(
                f"mode={BENCHMARK_MODE_UNION_TRAIN} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: applying SMOTE to union train"
            )
            union_train_df = _apply_smote(union_train_df, seed=seed, y_col="Y")

        _log_progress(
            f"mode={BENCHMARK_MODE_UNION_TRAIN} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: applying labelling to union train"
        )
        union_train_df = _apply_labelling_strategy(
            df=union_train_df,
            strategy=labelling_strategy,
            c_calc=c_calc,
            seed=seed,
        )

        dataset_names = tuple(test_by_dataset.keys())
        test_offsets: dict[str, tuple[int, int]] = {}
        test_union_parts: list[pd.DataFrame] = []
        cursor = 0
        for dataset_name in dataset_names:
            dataset_test = test_by_dataset[dataset_name]
            length = len(dataset_test)
            test_offsets[dataset_name] = (cursor, cursor + length)
            test_union_parts.append(dataset_test)
            cursor += length

        test_union_df = pd.concat(test_union_parts, axis=0, ignore_index=True)

        _log_progress(
            f"mode={BENCHMARK_MODE_UNION_TRAIN} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: fitting methods once on union train"
        )
        outputs = run_all_methods(
            union_train_df,
            test_union_df,
            random_state=seed,
            progress_callback=lambda method_name: _log_progress(
                f"mode={BENCHMARK_MODE_UNION_TRAIN} seed={seed_index}/{len(seed_list)} ({seed}) scheme={labelling_strategy} smote={use_smote}: method={method_name}"
            ),
        )

        for dataset_name in dataset_names:
            test_df = test_by_dataset[dataset_name]
            start, end = test_offsets[dataset_name]
            y_true = pd.to_numeric(test_df["Y"], errors="raise").to_numpy().astype(int)
            for method_name, out in outputs.items():
                m = compute_metrics(y_true, out.proba[start:end], threshold=threshold)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method_name,
                        "c_calc": c_calc,
                        "labelling_strategy": labelling_strategy,
                        "smote": use_smote,
                        "benchmark_mode": BENCHMARK_MODE_UNION_TRAIN,
                        "run_time_min": _runtime_minutes_from_meta(out.meta),
                        **asdict(m),
                    }
                )

    all_runs = pd.DataFrame(rows)
    agg = aggregate_results(all_runs) if not all_runs.empty else pd.DataFrame()
    return all_runs, agg


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    group_columns = ["dataset", "method", "c_calc", "labelling_strategy"]
    if "smote" in results.columns:
        group_columns.append("smote")
    if "benchmark_mode" in results.columns:
        group_columns.append("benchmark_mode")

    return results.groupby(group_columns, as_index=False).agg(
        run_time_min_mean=("run_time_min", "mean"),
        run_time_min_std=("run_time_min", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        tp_mean=("tp", "mean"),
        tp_std=("tp", "std"),
        tn_mean=("tn", "mean"),
        tn_std=("tn", "std"),
        fp_mean=("fp", "mean"),
        fp_std=("fp", "std"),
        fn_mean=("fn", "mean"),
        fn_std=("fn", "std"),
        fmi_mean=("fmi", "mean"),
        fmi_std=("fmi", "std"),
    )


def benchmark_many_datasets(
    datasets: dict[str, pd.DataFrame],
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    c_calc: float = 0.5,
    threshold: float = 0.5,
    labelling_strategy: str = "mvc",
    sample_size: int | None = None,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = []
    dataset_items = tuple(datasets.items())
    for dataset_index, (name, df) in enumerate(dataset_items, start=1):
        _log_progress(f"dataset={dataset_index}/{len(dataset_items)} name={name}: starting")
        per_run.append(
            benchmark_single_dataset(
                df=df,
                dataset_name=name,
                seeds=seeds,
                c_calc=c_calc,
                threshold=threshold,
                labelling_strategy=labelling_strategy,
                sample_size=sample_size,
                use_smote=use_smote,
            )
        )

    all_runs = pd.concat(per_run, axis=0, ignore_index=True) if per_run else pd.DataFrame()
    agg = aggregate_results(all_runs) if not all_runs.empty else pd.DataFrame()
    return all_runs, agg


def benchmark_many_datasets_for_strategies(
    datasets: dict[str, pd.DataFrame],
    labelling_strategies: Iterable[str] = ("mvc", "classic"),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    c_calc: float = 0.5,
    threshold: float = 0.5,
    sample_size: int | None = None,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = []
    strategy_list = tuple(labelling_strategies)
    for strategy_index, labelling_strategy in enumerate(strategy_list, start=1):
        _log_progress(
            f"scheme={strategy_index}/{len(strategy_list)} name={labelling_strategy}: starting"
        )
        runs, _ = benchmark_many_datasets(
            datasets=datasets,
            seeds=seeds,
            c_calc=c_calc,
            threshold=threshold,
            labelling_strategy=labelling_strategy,
            sample_size=sample_size,
            use_smote=use_smote,
        )
        per_run.append(runs)

    all_runs = pd.concat(per_run, axis=0, ignore_index=True) if per_run else pd.DataFrame()
    agg = aggregate_results(all_runs) if not all_runs.empty else pd.DataFrame()
    return all_runs, agg


def benchmark_many_datasets_union_train_for_strategies(
    datasets: dict[str, pd.DataFrame],
    labelling_strategies: Iterable[str] = ("mvc", "classic"),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    c_calc: float = 0.5,
    threshold: float = 0.5,
    sample_size: int | None = None,
    test_size: float = 0.3,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = []
    strategy_list = tuple(labelling_strategies)
    for strategy_index, labelling_strategy in enumerate(strategy_list, start=1):
        _log_progress(
            f"mode={BENCHMARK_MODE_UNION_TRAIN} scheme={strategy_index}/{len(strategy_list)} name={labelling_strategy}: starting"
        )
        runs, _ = benchmark_many_datasets_union_train(
            datasets=datasets,
            seeds=seeds,
            c_calc=c_calc,
            threshold=threshold,
            labelling_strategy=labelling_strategy,
            sample_size=sample_size,
            test_size=test_size,
            use_smote=use_smote,
        )
        per_run.append(runs)

    all_runs = pd.concat(per_run, axis=0, ignore_index=True) if per_run else pd.DataFrame()
    agg = aggregate_results(all_runs) if not all_runs.empty else pd.DataFrame()
    return all_runs, agg


def benchmark_many_datasets_for_c_calc_values(
    datasets: dict[str, pd.DataFrame],
    c_calc_values: Iterable[float] = (0.3, 0.5, 0.8),
    labelling_strategies: Iterable[str] = ("mvc",),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    threshold: float = 0.5,
    sample_size: int | None = None,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = []
    c_calc_list = tuple(c_calc_values)
    strategy_list = tuple(labelling_strategies)

    for c_calc_index, c_calc in enumerate(c_calc_list, start=1):
        _log_progress(
            f"c_calc={c_calc_index}/{len(c_calc_list)} value={c_calc}: starting"
        )
        runs, _ = benchmark_many_datasets_for_strategies(
            datasets=datasets,
            labelling_strategies=strategy_list,
            seeds=seeds,
            c_calc=c_calc,
            threshold=threshold,
            sample_size=sample_size,
            use_smote=use_smote,
        )
        per_run.append(runs)

    all_runs = pd.concat(per_run, axis=0, ignore_index=True) if per_run else pd.DataFrame()
    agg = aggregate_results(all_runs) if not all_runs.empty else pd.DataFrame()
    return all_runs, agg


def benchmark_many_datasets_union_train_for_c_calc_values(
    datasets: dict[str, pd.DataFrame],
    c_calc_values: Iterable[float] = (0.3, 0.5, 0.8),
    labelling_strategies: Iterable[str] = ("mvc",),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    threshold: float = 0.5,
    sample_size: int | None = None,
    test_size: float = 0.3,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = []
    c_calc_list = tuple(c_calc_values)
    strategy_list = tuple(labelling_strategies)

    for c_calc_index, c_calc in enumerate(c_calc_list, start=1):
        _log_progress(
            f"mode={BENCHMARK_MODE_UNION_TRAIN} c_calc={c_calc_index}/{len(c_calc_list)} value={c_calc}: starting"
        )
        runs, _ = benchmark_many_datasets_union_train_for_strategies(
            datasets=datasets,
            labelling_strategies=strategy_list,
            seeds=seeds,
            c_calc=c_calc,
            threshold=threshold,
            sample_size=sample_size,
            test_size=test_size,
            use_smote=use_smote,
        )
        per_run.append(runs)

    all_runs = pd.concat(per_run, axis=0, ignore_index=True) if per_run else pd.DataFrame()
    agg = aggregate_results(all_runs) if not all_runs.empty else pd.DataFrame()
    return all_runs, agg
