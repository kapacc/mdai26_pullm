"""Data loading utilities for SpeakLeash datasets and local CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

import pandas as pd


def get_reader(project_name: str, speakleash_class) -> Generator:
    """Return SpeakLeash reader for a specific dataset/project name."""
    project = speakleash_class.get(project_name)
    if project is None:
        raise ValueError(f"SpeakLeash dataset not found: {project_name}")

    reader = project.ext_data
    if reader is None:
        raise ValueError(f"Unable to read data for dataset: {project_name}")
    return reader


def get_dataframe(project_name: str, speakleash_class) -> pd.DataFrame:
    """Get a DataFrame from a project's data using a SpeakLeash instance."""
    reader = get_reader(project_name, speakleash_class)
    return pd.DataFrame({"text": s[0]} | s[1] for s in reader)


def load_datasets_from_speakleash(
    dataset_names: list[str],
    data_dir: str | Path = "data/speakleash_object",
    max_rows: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Load selected datasets from SpeakLeash as DataFrames (example_6_pandas pattern)."""
    try:
        from speakleash import Speakleash
    except ImportError as exc:
        raise ImportError("speakleash is not installed. Install with: pip install speakleash") from exc

    if not dataset_names:
        raise ValueError("dataset_names must be provided for speakleash source")

    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    sl = Speakleash(str(base_dir))
    datasets: dict[str, pd.DataFrame] = {}

    for name in dataset_names:
        print(f"Loading SpeakLeash dataset: {name}")
        df = get_dataframe(project_name=name, speakleash_class=sl)
        if max_rows is not None:
            df = df.head(max_rows).copy()
        datasets[name] = df
        print(f"  Loaded rows: {len(df)}")

    return datasets


def load_datasets_from_csv(
    data_dir: str | Path = "data",
    pattern: str = "*.csv",
    dataset_names: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load datasets from CSV files in a directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing CSV files.
    pattern : str
        Glob pattern for CSV files (default: "*.csv").
    dataset_names : list[str], optional
        Specific CSV stems to load. When provided, only matching files are read.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping dataset names (filename without extension) to DataFrames.

    Raises
    ------
    FileNotFoundError
        If data_dir does not exist or contains no matching files.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    csv_files = sorted(data_path.glob(pattern))

    if dataset_names:
        wanted = set(dataset_names)
        csv_files = [csv_file for csv_file in csv_files if csv_file.stem in wanted]

    if not csv_files:
        if dataset_names:
            raise FileNotFoundError(
                f"No CSV files matching '{pattern}' and dataset_names={dataset_names} in {data_path}"
            )
        raise FileNotFoundError(f"No CSV files matching '{pattern}' in {data_path}")

    datasets = {}
    for csv_file in csv_files:
        name = csv_file.stem
        print(f"Loading {name} from {csv_file}...")
        df = pd.read_csv(csv_file)
        datasets[name] = df
        print(f"  Shape: {df.shape}")

    return datasets


def load_datasets(
    source: str = "speakleash",
    dataset_names: Optional[list[str]] = None,
    data_dir: str | Path = "data",
    max_rows: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load datasets from specified source.

    Parameters
    ----------
    source : str
        Data source: "speakleash" or "csv" (default: "speakleash").
    dataset_names : list[str], optional
        For speakleash: specific dataset names to load.
        For csv: ignored (loads all CSV files from data_dir).
    data_dir : str or Path
        For csv: directory containing CSV files (default: "data").
    max_rows : int, optional
        For speakleash: max number of rows per dataset to load.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of loaded datasets.

    Raises
    ------
    ValueError
        If source is not "speakleash" or "csv".
    """
    if source == "speakleash":
        return load_datasets_from_speakleash(
            dataset_names=dataset_names or [],
            data_dir=data_dir,
            max_rows=max_rows,
        )
    if source == "csv":
        return load_datasets_from_csv(data_dir=data_dir, dataset_names=dataset_names)
    raise ValueError(f"Unknown source: {source}. Choose 'speakleash' or 'csv'.")
