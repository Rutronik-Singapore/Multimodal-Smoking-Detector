#!/usr/bin/env python3
"""Normalize and merge SEN66 CSV logs for model training."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FOLDERS = [
    REPOSITORY_ROOT / "logs" / "csv_log" / "legacy",
    REPOSITORY_ROOT / "logs" / "csv_log" / "pse84",
]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "logs" / "csv_merge"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "merge_manifest.csv"

CANONICAL_COLUMNS = [
    "dataset_id",
    "elapsed_seconds",
    "pm1_0",
    "pm2_5",
    "pm4_0",
    "pm10_0",
    "voc_index",
    "nox_index",
    "co2_ppm",
    "label",
]
SENSOR_COLUMNS = ["pm1_0", "pm2_5", "pm4_0", "pm10_0", "voc_index", "nox_index", "co2_ppm"]

LEGACY_REQUIRED = {
    "elapsed_time",
    "uptime_ms",
    "pm1_0",
    "pm2_5",
    "pm4_0",
    "pm10_0",
    "voc_index",
    "nox_index",
    "co2_ppm",
    "label",
}
NEWER_REQUIRED = {
    "Sample",
    "Date",
    "Time",
    "TimestampISO8601",
    "ElapsedSeconds",
    "PM1_0",
    "PM2_5",
    "PM4_0",
    "PM10",
    "Humidity",
    "Temperature",
    "VOC",
    "NOx",
    "CO2",
    "distance",
    "ManualLabel",
    "ManualLabelRaw",
}


class MergeError(Exception):
    """Raised for validation failures that should stop the merge."""


@dataclass(frozen=True)
class InputCsv:
    folder: Path
    path: Path


@dataclass(frozen=True)
class ManifestRow:
    dataset_id: int
    input_folder: str
    source_filename: str
    detected_schema: str
    source_row_count: int
    accepted_output_row_count: int
    min_elapsed_seconds: float
    max_elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover, validate, normalize and merge SEN66 CSV logs."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        help="Input folder(s) containing .csv/.CSV files. Defaults to the current SEN66 new and pse84 folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--date-prefix",
        default=datetime.now().strftime("%d%m"),
        help="Output filename date prefix. Default: local date as DDMM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged and manifest files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folders recursively. Default is non-recursive.",
    )
    parser.add_argument(
        "--verify-existing-order",
        action="store_true",
        help="Read-only: compare discovered recording order with the bundled merge manifest.",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest used by --verify-existing-order. Default: {DEFAULT_MANIFEST_PATH}",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def discover_csv_files(folders: Iterable[Path], recursive: bool) -> list[InputCsv]:
    discovered: list[InputCsv] = []
    normalized_folders = sorted((normalize_path(folder) for folder in folders), key=path_sort_key)
    print("Selected input folders:")
    for folder in normalized_folders:
        print(f"  {folder}")
        if not folder.exists():
            raise MergeError(f"Input folder does not exist: {folder}")
        if not folder.is_dir():
            raise MergeError(f"Input path is not a folder: {folder}")
        iterator = folder.rglob("*") if recursive else folder.iterdir()
        for path in iterator:
            if path.is_file() and path.suffix.lower() == ".csv":
                discovered.append(InputCsv(folder=folder, path=path.resolve()))

    discovered.sort(key=lambda item: (path_sort_key(item.folder), path_sort_key(item.path.name)))
    print("Selected CSV files:")
    for item in discovered:
        print(f"  {item.path}")
    if not discovered:
        raise MergeError("No .csv/.CSV files were discovered in the selected folders.")
    return discovered


def path_sort_key(value: Path | str) -> str:
    return str(value).casefold()


def logical_source_group(path: Path | str) -> str:
    name = Path(path).name.casefold()
    return "legacy" if name in {"new", "legacy"} else name


def verify_existing_order(inputs: list[InputCsv], manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise MergeError(f"Manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"dataset_id", "input_folder", "source_filename"}
    missing = required.difference(manifest.columns)
    if missing:
        raise MergeError(f"Manifest missing columns: {sorted(missing)}")
    manifest = manifest.sort_values("dataset_id", kind="mergesort").reset_index(drop=True)
    if len(manifest) != len(inputs):
        raise MergeError(f"Recording count differs: manifest={len(manifest)} discovered={len(inputs)}")
    for dataset_id, (item, row) in enumerate(zip(inputs, manifest.itertuples(index=False)), start=1):
        expected_id = int(getattr(row, "dataset_id"))
        expected_name = str(getattr(row, "source_filename"))
        expected_group = logical_source_group(getattr(row, "input_folder"))
        actual_group = logical_source_group(item.folder)
        if expected_id != dataset_id or expected_name.casefold() != item.path.name.casefold() or expected_group != actual_group:
            raise MergeError(
                "Recording order mismatch at dataset_id "
                f"{dataset_id}: manifest=({expected_name}, {expected_group}), "
                f"discovered=({item.path.name}, {actual_group})"
            )
    print(f"Verified {len(inputs)} recording IDs against {manifest_path}; legacy/new treated as one logical group.")


def reject_output_among_inputs(inputs: list[InputCsv], output_paths: Iterable[Path]) -> None:
    input_paths = {item.path.resolve() for item in inputs}
    for output_path in output_paths:
        if output_path.resolve() in input_paths:
            raise MergeError(f"Output file appears among discovered inputs: {output_path}")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas provides many parser exceptions
        raise MergeError(f"Failed to read CSV file {path}: {exc}") from exc


def detect_schema(columns: Iterable[str]) -> str:
    column_set = set(columns)
    if LEGACY_REQUIRED.issubset(column_set):
        return "legacy"
    if NEWER_REQUIRED.issubset(column_set):
        if {"Predicted", "PredictedRaw"}.issubset(column_set):
            return "newer_with_predictions"
        return "newer_without_predictions"
    raise MergeError(
        "Unsupported schema. Columns found: "
        f"{list(columns)}. Expected legacy columns including uptime_ms/label, "
        "or newer columns including ElapsedSeconds/ManualLabelRaw."
    )


def require_columns(path: Path, schema: str, df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise MergeError(
            f"{path}: detected_schema={schema}; missing required columns {missing}; "
            f"expected columns {sorted(required)}"
        )


def csv_row(index: int) -> int:
    return index + 2


def fail_bad_values(path: Path, schema: str, column: str, mask: pd.Series, values: pd.Series, expected: str) -> None:
    bad = mask[mask].index[:5]
    details = []
    for idx in bad:
        details.append(f"row {csv_row(int(idx))} value={values.loc[idx]!r}")
    raise MergeError(
        f"{path}: detected_schema={schema}; column={column}; invalid value(s): "
        f"{'; '.join(details)}; expected {expected}"
    )


def numeric_series(path: Path, schema: str, df: pd.DataFrame, source_column: str, target_column: str) -> pd.Series:
    values = pd.to_numeric(df[source_column], errors="coerce")
    bad = values.isna() | df[source_column].isna()
    if bad.any():
        fail_bad_values(
            path,
            schema,
            source_column,
            bad,
            df[source_column],
            f"numeric value for canonical column {target_column}",
        )
    return values


def normalize_legacy(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    schema = "legacy"
    require_columns(path, schema, df, LEGACY_REQUIRED)
    normalized = pd.DataFrame()
    normalized["elapsed_seconds"] = numeric_series(path, schema, df, "uptime_ms", "elapsed_seconds") / 1000.0
    for column in SENSOR_COLUMNS:
        normalized[column] = numeric_series(path, schema, df, column, column)

    label_text = df["label"].astype("string").str.strip().str.casefold()
    label_map = {"no smoking": 0, "smoking": 1}
    label = label_text.map(label_map)
    bad = label.isna()
    if bad.any():
        fail_bad_values(path, schema, "label", bad, df["label"], "text label 'smoking' or 'no smoking'")
    normalized["label"] = label.astype("int64")
    return normalized


def normalize_newer(path: Path, df: pd.DataFrame, schema: str) -> pd.DataFrame:
    require_columns(path, schema, df, NEWER_REQUIRED)
    normalized = pd.DataFrame()
    source_to_target = {
        "ElapsedSeconds": "elapsed_seconds",
        "PM1_0": "pm1_0",
        "PM2_5": "pm2_5",
        "PM4_0": "pm4_0",
        "PM10": "pm10_0",
        "VOC": "voc_index",
        "NOx": "nox_index",
        "CO2": "co2_ppm",
    }
    for source, target in source_to_target.items():
        normalized[target] = numeric_series(path, schema, df, source, target)

    raw_label = numeric_series(path, schema, df, "ManualLabelRaw", "label")
    invalid_label = ~raw_label.isin([0, 1])
    if invalid_label.any():
        fail_bad_values(path, schema, "ManualLabelRaw", invalid_label, df["ManualLabelRaw"], "0 or 1")

    manual_text = df["ManualLabel"].astype("string").str.strip().str.casefold()
    expected_text = raw_label.map({0: "no smoking", 1: "smoking"})
    conflict = manual_text != expected_text
    if conflict.any():
        details = []
        for idx in conflict[conflict].index[:5]:
            details.append(
                f"row {csv_row(int(idx))} ManualLabelRaw={df['ManualLabelRaw'].loc[idx]!r} "
                f"ManualLabel={df['ManualLabel'].loc[idx]!r}"
            )
        raise MergeError(
            f"{path}: detected_schema={schema}; column=ManualLabelRaw/ManualLabel; "
            f"conflicting value(s): {'; '.join(details)}; expected 0=No Smoking and 1=Smoking"
        )

    normalized["label"] = raw_label.astype("int64")
    return normalized


def ensure_no_duplicate_columns(path: Path, schema: str, df: pd.DataFrame) -> None:
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise MergeError(
            f"{path}: detected_schema={schema}; duplicate canonical columns after normalization: {duplicates}"
        )


def validate_canonical(path: Path, schema: str, df: pd.DataFrame) -> None:
    ensure_no_duplicate_columns(path, schema, df)
    expected_without_dataset = CANONICAL_COLUMNS[1:]
    if list(df.columns) != expected_without_dataset:
        raise MergeError(
            f"{path}: detected_schema={schema}; canonical columns {list(df.columns)} "
            f"do not match expected {expected_without_dataset}"
        )
    if df.empty:
        raise MergeError(f"{path}: detected_schema={schema}; file contains no valid data rows.")

    for column in expected_without_dataset:
        missing = df[column].isna()
        if missing.any():
            fail_bad_values(path, schema, column, missing, df[column], "no missing required values")

    elapsed_negative = df["elapsed_seconds"] < 0
    if elapsed_negative.any():
        fail_bad_values(path, schema, "elapsed_seconds", elapsed_negative, df["elapsed_seconds"], "non-negative time")

    elapsed_decrease = df["elapsed_seconds"].diff().fillna(0) < 0
    if elapsed_decrease.any():
        fail_bad_values(
            path,
            schema,
            "elapsed_seconds",
            elapsed_decrease,
            df["elapsed_seconds"],
            "nondecreasing time within the source file",
        )

    invalid_label = ~df["label"].isin([0, 1])
    if invalid_label.any():
        fail_bad_values(path, schema, "label", invalid_label, df["label"], "0 or 1")


def normalize_file(input_csv: InputCsv, dataset_id: int) -> tuple[pd.DataFrame, ManifestRow, str]:
    source_df = read_csv(input_csv.path)
    try:
        schema = detect_schema(source_df.columns)
    except MergeError as exc:
        raise MergeError(f"{input_csv.path}: {exc}") from exc

    if schema == "legacy":
        normalized = normalize_legacy(input_csv.path, source_df)
    else:
        normalized = normalize_newer(input_csv.path, source_df, schema)
    validate_canonical(input_csv.path, schema, normalized)

    normalized.insert(0, "dataset_id", dataset_id)
    normalized = normalized[CANONICAL_COLUMNS]
    manifest = ManifestRow(
        dataset_id=dataset_id,
        input_folder=str(input_csv.folder),
        source_filename=input_csv.path.name,
        detected_schema=schema,
        source_row_count=len(source_df),
        accepted_output_row_count=len(normalized),
        min_elapsed_seconds=float(normalized["elapsed_seconds"].min()),
        max_elapsed_seconds=float(normalized["elapsed_seconds"].max()),
    )
    return normalized, manifest, schema


def write_outputs(
    merged: pd.DataFrame,
    manifest_rows: list[ManifestRow],
    merged_path: Path,
    manifest_path: Path,
    overwrite: bool,
) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in [merged_path, manifest_path] if path.exists()]
    if existing and not overwrite:
        raise MergeError(
            "Output file already exists: "
            f"{', '.join(str(path) for path in existing)}. Re-run with --overwrite to replace it."
        )

    manifest_df = pd.DataFrame([row.__dict__ for row in manifest_rows])
    try:
        merged.to_csv(merged_path, index=False)
        manifest_df.to_csv(manifest_path, index=False)
    except OSError as exc:
        raise MergeError(f"Failed to write output files under {merged_path.parent}: {exc}") from exc
    verify_outputs(merged, manifest_rows, merged_path, manifest_path)


def verify_outputs(
    expected_merged: pd.DataFrame,
    manifest_rows: list[ManifestRow],
    merged_path: Path,
    manifest_path: Path,
) -> None:
    read_merged = pd.read_csv(merged_path)
    read_manifest = pd.read_csv(manifest_path)
    if list(read_merged.columns) != CANONICAL_COLUMNS:
        raise MergeError(
            f"Written merged file has columns {list(read_merged.columns)}, expected {CANONICAL_COLUMNS}"
        )
    if len(read_merged) != len(expected_merged):
        raise MergeError(f"Written merged row count {len(read_merged)} does not match expected {len(expected_merged)}")
    if not set(read_merged["label"].unique()).issubset({0, 1}):
        raise MergeError("Written merged file contains labels outside 0 and 1.")
    manifest_counts = read_manifest.groupby("dataset_id")["source_filename"].nunique()
    if not (manifest_counts == 1).all():
        raise MergeError("Manifest does not map every dataset_id to exactly one source file.")
    if len(read_manifest) != len(manifest_rows):
        raise MergeError("Written manifest row count does not match the processed source file count.")


def print_summary(
    folder_count: int,
    inputs: list[InputCsv],
    manifest_rows: list[ManifestRow],
    schema_counts: dict[str, int],
    merged: pd.DataFrame,
    merged_path: Path,
    manifest_path: Path,
) -> None:
    print("\nDataset-ID manifest:")
    for row in manifest_rows:
        print(
            f"  {row.dataset_id}: folder={row.input_folder} file={row.source_filename} "
            f"source_rows={row.source_row_count} accepted_rows={row.accepted_output_row_count}"
        )

    print("\nProcessing summary:")
    print(f"  folders scanned: {folder_count}")
    print(f"  CSV files discovered: {len(inputs)}")
    for schema in sorted(schema_counts):
        print(f"  files processed as {schema}: {schema_counts[schema]}")
    print(f"  merged row count: {len(merged)}")
    print(f"  smoking row count: {int((merged['label'] == 1).sum())}")
    print(f"  no-smoking row count: {int((merged['label'] == 0).sum())}")
    print("  elapsed_seconds per dataset:")
    for row in manifest_rows:
        print(f"    dataset_id={row.dataset_id}: min={row.min_elapsed_seconds} max={row.max_elapsed_seconds}")
    print(f"  final merged columns: {CANONICAL_COLUMNS}")
    print(f"  merged output path: {merged_path}")
    print(f"  manifest output path: {manifest_path}")


def run(args: argparse.Namespace) -> None:
    folders = args.folders or DEFAULT_INPUT_FOLDERS
    inputs = discover_csv_files(folders, args.recursive)
    if args.verify_existing_order:
        verify_existing_order(inputs, normalize_path(args.verify_manifest))
        return
    output_dir = normalize_path(args.output_dir)
    merged_path = output_dir / f"{args.date_prefix}_merge.csv"
    manifest_path = output_dir / f"{args.date_prefix}_merge_manifest.csv"
    reject_output_among_inputs(inputs, [merged_path, manifest_path])

    normalized_frames: list[pd.DataFrame] = []
    manifest_rows: list[ManifestRow] = []
    schema_counts: dict[str, int] = {}
    for dataset_id, input_csv in enumerate(inputs, start=1):
        normalized, manifest, schema = normalize_file(input_csv, dataset_id)
        normalized_frames.append(normalized)
        manifest_rows.append(manifest)
        schema_counts[schema] = schema_counts.get(schema, 0) + 1

    merged = pd.concat(normalized_frames, ignore_index=True)
    write_outputs(merged, manifest_rows, merged_path, manifest_path, args.overwrite)
    print_summary(len(folders), inputs, manifest_rows, schema_counts, merged, merged_path, manifest_path)


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except MergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
