from __future__ import annotations

import argparse
from pathlib import Path

from speakleash import Speakleash

from py_puml.data_loader import get_dataframe


DEFAULT_DATASETS = [
    "plwiki",
    "open_subtitles_corpus",
    "job_offers_pl_corpus",
    "forum_cdaction_pl_corpus",
]

def _validate_dataset_names(dataset_names: list[str], available_names: set[str]) -> None:
    invalid = sorted(name for name in dataset_names if name not in available_names)
    if not invalid:
        return

    invalid_text = ", ".join(invalid)
    raise ValueError(
        "Unknown SpeakLeash dataset name(s): "
        f"{invalid_text}. "
        "Check available names in speakleash_datasets_catalog.md."
    )


def download_datasets(
    dataset_names: list[str],
    object_dir: Path,
    csv_dir: Path,
    max_rows: int | None = None,
) -> tuple[list[str], list[str]]:
    object_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Speakleash initialization path: {object_dir.resolve()}")
    sl = Speakleash(str(object_dir))
    available_names = {ds.name for ds in (sl.datasets or [])}
    _validate_dataset_names(dataset_names=dataset_names, available_names=available_names)

    ok: list[str] = []
    failed: list[str] = []

    for name in dataset_names:
        print(f"\nProcessing dataset: {name}")
        try:
            project = sl.get(name)
            if project is None:
                print("  -> dataset not found in SpeakLeash catalog")
                failed.append(name)
                continue

            df = get_dataframe(project_name=name, speakleash_class=sl)
            if max_rows is not None:
                df = df.head(max_rows).copy()
            csv_path = csv_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  -> saved CSV: {csv_path} (rows={len(df)})")

            ok.append(name)
        except Exception as exc:
            print(f"  -> failed: {exc}")
            failed.append(name)

    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected SpeakLeash datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to download",
    )
    parser.add_argument(
        "--object-dir",
        default="data/speakleash_object",
        help="Path passed to Speakleash(...) during initialization",
    )
    parser.add_argument(
        "--csv-dir",
        default="data",
        help="Directory for ready CSV files",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap per dataset before CSV export",
    )

    args = parser.parse_args()
    object_dir = Path(args.object_dir)
    csv_dir = Path(args.csv_dir)

    ok, failed = download_datasets(
        dataset_names=args.datasets,
        object_dir=object_dir,
        csv_dir=csv_dir,
        max_rows=args.max_rows,
    )

    print("\nDownload summary:")
    print(f"  ok: {len(ok)} -> {ok}")
    print(f"  failed: {len(failed)} -> {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
