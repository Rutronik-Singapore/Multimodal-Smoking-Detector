#!/usr/bin/env python3
"""Shared SEN66 CSV preparation helpers for DEEPCRAFT Studio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


TRAINING_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TRAINING_ROOT.parents[1]
DATA_DIR = TRAINING_ROOT / "data"
MANIFEST_PATH = TRAINING_ROOT / "processing_manifest.csv"
LEGACY_SOURCE_DIR = REPOSITORY_ROOT / "logs" / "csv_log" / "legacy"
PSE84_SOURCE_DIR = REPOSITORY_ROOT / "logs" / "csv_log" / "pse84"
SPLIT_SOURCE_DIR = TRAINING_ROOT / "split"

SENSOR_FILENAME = "sensor.csv"
LABEL_FILENAME = "label.label"
SENSOR_COLUMNS = [
    "elapsed_seconds",
    "pm1_0",
    "pm2_5",
    "pm4_0",
    "pm10_0",
    "voc_index",
    "nox_index",
    "co2_ppm",
]
MEASUREMENT_COLUMNS = SENSOR_COLUMNS[1:]
LABEL_COLUMNS = ["Time", "Length", "Label", "Confidence", "Comment"]
CLASS_NAMES = {0: "No Smoking", 1: "Smoking"}
SESSION_RE = re.compile(r"^session_(\d+)$")

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
NEWEST_REQUIRED = NEWER_REQUIRED | {"Predicted", "PredictedRaw"}

MANIFEST_COLUMNS = [
    "session_id",
    "session_number",
    "session_dir_path",
    "sensor_file_path",
    "label_file_path",
    "original_source_filename",
    "original_full_source_path",
    "source_format_version",
    "source_file_hash",
    "source_size_bytes",
    "source_modified_time",
    "source_row_count",
    "output_row_count",
    "original_first_elapsed_time",
    "original_final_elapsed_time",
    "offset_applied",
    "offset_output_start_time",
    "offset_output_end_time",
    "median_positive_sample_interval",
    "label_interval_count",
    "smoking_sample_count",
    "no_smoking_sample_count",
    "processing_datetime",
]


class ProcessingError(Exception):
    """Raised for validation or processing failures."""


@dataclass(frozen=True)
class SourceFile:
    path: Path
    schema: str
    content_hash: str
    size_bytes: int
    modified_time: str


@dataclass(frozen=True)
class NormalizedRecording:
    source: SourceFile
    sensor: pd.DataFrame
    label_intervals: pd.DataFrame
    source_row_count: int
    interval_count: int
    smoking_sample_count: int
    no_smoking_sample_count: int
    original_first_elapsed_time: float
    original_final_elapsed_time: float
    offset_applied: float
    offset_output_start_time: float
    offset_output_end_time: float
    median_positive_sample_interval: float


@dataclass(frozen=True)
class ProcessResult:
    source_path: Path
    schema: str | None
    status: str
    message: str
    session_number: int | None = None
    session_id: str | None = None
    session_path: Path | None = None
    source_rows: int | None = None
    interval_count: int | None = None
    smoking_count: int | None = None
    no_smoking_count: int | None = None
    original_range: tuple[float, float] | None = None
    offset_applied: float | None = None
    exported_range: tuple[float, float] | None = None


def path_key(path: Path | str) -> str:
    return str(path).casefold()


def csv_row(index: int) -> int:
    return index + 2


def session_id(number: int) -> str:
    return f"session_{number:03d}"


def session_path(number: int) -> Path:
    return DATA_DIR / session_id(number)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_csv_files(source_dir: Path, recursive: bool = False) -> list[Path]:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise ProcessingError(f"Source folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ProcessingError(f"Source path is not a folder: {source_dir}")
    iterator = source_dir.rglob("*") if recursive else source_dir.iterdir()
    return sorted(
        [path.resolve() for path in iterator if path.is_file() and path.suffix.lower() == ".csv"],
        key=path_key,
    )


def detect_schema(columns: Sequence[str]) -> str:
    column_set = set(columns)
    if LEGACY_REQUIRED.issubset(column_set):
        return "legacy"
    if NEWEST_REQUIRED.issubset(column_set):
        return "newer_with_predictions"
    if NEWER_REQUIRED.issubset(column_set):
        return "newer_without_predictions"
    raise ProcessingError(
        f"unsupported schema; columns={list(columns)}; expected legacy, newer_without_predictions, "
        "or newer_with_predictions"
    )


def ensure_schema_allowed(path: Path, schema: str, allowed_schemas: set[str]) -> None:
    if schema not in allowed_schemas:
        raise ProcessingError(f"{path}: detected_schema={schema}; expected one of {sorted(allowed_schemas)}")


def read_source_file(path: Path, allowed_schemas: set[str]) -> tuple[SourceFile, pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ProcessingError(f"{path}: failed to read CSV: {exc}") from exc
    schema = detect_schema(list(df.columns))
    ensure_schema_allowed(path, schema, allowed_schemas)
    stat = path.stat()
    return (
        SourceFile(
            path=path,
            schema=schema,
            content_hash=file_sha256(path),
            size_bytes=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        ),
        df,
    )


def fail_bad_values(path: Path, schema: str, column: str, mask: pd.Series, values: pd.Series, expected: str) -> None:
    details = [f"row {csv_row(int(idx))} value={values.loc[idx]!r}" for idx in mask[mask].index[:5]]
    raise ProcessingError(
        f"{path}: detected_schema={schema}; column={column}; invalid value(s): "
        f"{'; '.join(details)}; expected {expected}"
    )


def numeric_series(path: Path, schema: str, df: pd.DataFrame, source_column: str, target_column: str) -> pd.Series:
    values = pd.to_numeric(df[source_column], errors="coerce")
    bad = values.isna() | df[source_column].isna()
    if bad.any():
        fail_bad_values(path, schema, source_column, bad, df[source_column], f"numeric {target_column}")
    return values.astype("float64")


def normalize_legacy(source: SourceFile, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(LEGACY_REQUIRED - set(df.columns))
    if missing:
        raise ProcessingError(f"{source.path}: detected_schema=legacy; missing required columns {missing}")
    sensor = pd.DataFrame()
    sensor["elapsed_seconds"] = numeric_series(source.path, source.schema, df, "uptime_ms", "elapsed_seconds") / 1000.0
    for column in MEASUREMENT_COLUMNS:
        sensor[column] = numeric_series(source.path, source.schema, df, column, column)
    label_text = df["label"].astype("string").str.strip().str.casefold()
    labels = label_text.map({"no smoking": 0, "smoking": 1})
    bad = labels.isna()
    if bad.any():
        fail_bad_values(source.path, source.schema, "label", bad, df["label"], "'no smoking' or 'smoking'")
    return sensor, labels.astype("int64")


def normalize_newer(source: SourceFile, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = NEWEST_REQUIRED if source.schema == "newer_with_predictions" else NEWER_REQUIRED
    missing = sorted(required - set(df.columns))
    if missing:
        raise ProcessingError(f"{source.path}: detected_schema={source.schema}; missing required columns {missing}")
    sensor = pd.DataFrame()
    mappings = {
        "ElapsedSeconds": "elapsed_seconds",
        "PM1_0": "pm1_0",
        "PM2_5": "pm2_5",
        "PM4_0": "pm4_0",
        "PM10": "pm10_0",
        "VOC": "voc_index",
        "NOx": "nox_index",
        "CO2": "co2_ppm",
    }
    for source_column, target_column in mappings.items():
        sensor[target_column] = numeric_series(source.path, source.schema, df, source_column, target_column)
    raw_label = numeric_series(source.path, source.schema, df, "ManualLabelRaw", "label")
    invalid = ~raw_label.isin([0, 1])
    if invalid.any():
        fail_bad_values(source.path, source.schema, "ManualLabelRaw", invalid, df["ManualLabelRaw"], "0 or 1")
    manual_text = df["ManualLabel"].astype("string").str.strip().str.casefold()
    expected_text = raw_label.map({0.0: "no smoking", 1.0: "smoking"})
    conflict = manual_text != expected_text
    if conflict.any():
        details = []
        for idx in conflict[conflict].index[:5]:
            details.append(
                f"row {csv_row(int(idx))} ManualLabelRaw={df['ManualLabelRaw'].loc[idx]!r} "
                f"ManualLabel={df['ManualLabel'].loc[idx]!r}"
            )
        raise ProcessingError(
            f"{source.path}: detected_schema={source.schema}; column=ManualLabelRaw/ManualLabel; "
            f"conflicting value(s): {'; '.join(details)}; expected 0=No Smoking and 1=Smoking"
        )
    return sensor, raw_label.astype("int64")


def normalize_source(source: SourceFile, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if source.schema == "legacy":
        return normalize_legacy(source, df)
    return normalize_newer(source, df)


def validate_sensor(path: Path, schema: str, sensor: pd.DataFrame, *, expect_zero_start: bool = False) -> None:
    if list(sensor.columns) != SENSOR_COLUMNS:
        raise ProcessingError(f"{path}: detected_schema={schema}; sensor columns {list(sensor.columns)}; expected {SENSOR_COLUMNS}")
    if sensor.empty:
        raise ProcessingError(f"{path}: detected_schema={schema}; no data rows")
    if sensor.columns.duplicated().any():
        raise ProcessingError(f"{path}: detected_schema={schema}; duplicate sensor columns")
    for column in SENSOR_COLUMNS:
        missing = sensor[column].isna()
        if missing.any():
            fail_bad_values(path, schema, column, missing, sensor[column], "non-missing required value")
        numeric = pd.to_numeric(sensor[column], errors="coerce")
        bad = numeric.isna()
        if bad.any():
            fail_bad_values(path, schema, column, bad, sensor[column], "numeric value")
        sensor[column] = numeric
    negative = sensor["elapsed_seconds"] < -1e-9
    if negative.any():
        fail_bad_values(path, schema, "elapsed_seconds", negative, sensor["elapsed_seconds"], "nonnegative time")
    decreasing = sensor["elapsed_seconds"].diff().fillna(0) < -1e-9
    if decreasing.any():
        fail_bad_values(path, schema, "elapsed_seconds", decreasing, sensor["elapsed_seconds"], "nondecreasing time")
    if expect_zero_start and abs(float(sensor["elapsed_seconds"].iloc[0])) > 1e-9:
        raise ProcessingError(f"{path}: first exported elapsed_seconds is {sensor['elapsed_seconds'].iloc[0]}, expected 0.0")


def validate_numeric_labels(path: Path, schema: str, labels: pd.Series) -> None:
    missing = labels.isna()
    if missing.any():
        fail_bad_values(path, schema, "label", missing, labels, "0 or 1")
    invalid = ~labels.isin([0, 1])
    if invalid.any():
        fail_bad_values(path, schema, "label", invalid, labels, "0 or 1")


def offset_timeline(sensor: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    original_first = float(sensor["elapsed_seconds"].iloc[0])
    original_final = float(sensor["elapsed_seconds"].iloc[-1])
    offset_sensor = sensor.copy()
    offset_sensor["elapsed_seconds"] = offset_sensor["elapsed_seconds"] - original_first
    offset_sensor.loc[offset_sensor.index[0], "elapsed_seconds"] = 0.0
    return offset_sensor, original_first, original_final, original_first


def typical_sample_interval(times: pd.Series) -> float:
    diffs = times.diff().dropna()
    positive = diffs[diffs > 0]
    if not positive.empty:
        return float(positive.median())
    return 1.0


def build_label_intervals(sensor: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, float]:
    times = sensor["elapsed_seconds"].reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    sample_interval = typical_sample_interval(times)
    rows: list[dict[str, object]] = []
    start_idx = 0
    for idx in range(1, len(labels)):
        if int(labels.iloc[idx]) != int(labels.iloc[start_idx]):
            start_time = float(times.iloc[start_idx])
            end_time = float(times.iloc[idx])
            rows.append(
                {
                    "Time": start_time,
                    "Length": end_time - start_time,
                    "Label": CLASS_NAMES[int(labels.iloc[start_idx])],
                    "Confidence": 1.0,
                    "Comment": "",
                }
            )
            start_idx = idx
    final_start = float(times.iloc[start_idx])
    final_end = float(times.iloc[-1]) + sample_interval
    rows.append(
        {
            "Time": final_start,
            "Length": final_end - final_start,
            "Label": CLASS_NAMES[int(labels.iloc[start_idx])],
            "Confidence": 1.0,
            "Comment": "",
        }
    )
    intervals = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    validate_intervals(intervals, float(times.iloc[0]), final_end)
    return intervals, sample_interval


def validate_intervals(intervals: pd.DataFrame, recording_start: float, recording_end: float) -> None:
    if list(intervals.columns) != LABEL_COLUMNS:
        raise ProcessingError(f"label columns {list(intervals.columns)}; expected {LABEL_COLUMNS}")
    if intervals.empty:
        raise ProcessingError("no label intervals generated")
    starts = pd.to_numeric(intervals["Time"], errors="coerce")
    lengths = pd.to_numeric(intervals["Length"], errors="coerce")
    if starts.isna().any() or lengths.isna().any():
        raise ProcessingError("label intervals contain non-numeric Time or Length")
    if (lengths <= 0).any():
        raise ProcessingError("label interval with non-positive duration")
    if abs(float(starts.iloc[0]) - recording_start) > 1e-9:
        raise ProcessingError("first label interval does not start at recording start")
    ends = starts + lengths
    for idx in range(1, len(intervals)):
        if starts.iloc[idx] < ends.iloc[idx - 1] - 1e-9:
            raise ProcessingError("label intervals overlap")
        if starts.iloc[idx] < starts.iloc[idx - 1] - 1e-9:
            raise ProcessingError("label intervals are not ordered")
    if abs(float(ends.iloc[-1]) - recording_end) > 1e-6:
        raise ProcessingError("final label interval does not match recording end")
    unexpected = set(intervals["Label"]) - set(CLASS_NAMES.values())
    if unexpected:
        raise ProcessingError(f"label intervals contain unexpected classes {sorted(unexpected)}")


def normalize_recording(path: Path, allowed_schemas: set[str]) -> NormalizedRecording:
    source, source_df = read_source_file(path, allowed_schemas)
    original_sensor, labels = normalize_source(source, source_df)
    validate_sensor(path, source.schema, original_sensor)
    validate_numeric_labels(path, source.schema, labels)
    # Raw timestamps must already be nondecreasing. The stable sort only preserves equal-time ordering.
    ordered = original_sensor.assign(__label=labels.to_numpy()).sort_values(
        by="elapsed_seconds", kind="mergesort", ignore_index=True
    )
    labels = ordered.pop("__label").astype("int64")
    original_sensor = ordered
    offset_sensor, original_first, original_final, offset = offset_timeline(original_sensor)
    validate_sensor(path, source.schema, offset_sensor, expect_zero_start=True)
    intervals, median_interval = build_label_intervals(offset_sensor, labels)
    output_end = float(intervals["Time"].iloc[-1] + intervals["Length"].iloc[-1])
    return NormalizedRecording(
        source=source,
        sensor=offset_sensor,
        label_intervals=intervals,
        source_row_count=len(source_df),
        interval_count=len(intervals),
        smoking_sample_count=int((labels == 1).sum()),
        no_smoking_sample_count=int((labels == 0).sum()),
        original_first_elapsed_time=original_first,
        original_final_elapsed_time=original_final,
        offset_applied=offset,
        offset_output_start_time=float(offset_sensor["elapsed_seconds"].iloc[0]),
        offset_output_end_time=output_end,
        median_positive_sample_interval=median_interval,
    )


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(manifest_path, dtype={"source_file_hash": "string", "source_hash": "string"})


def detect_old_manifest(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        return
    if "session_id" not in manifest.columns and {"generated_id", "dataset_output_filename", "label_output_filename"}.issubset(manifest.columns):
        print(
            "Detected old separate dataset/label manifest schema. "
            "It is not used for new session numbering; keep it only as historical traceability."
        )


def manifest_session_numbers(manifest: pd.DataFrame) -> set[int]:
    if manifest.empty or "session_number" not in manifest.columns:
        return set()
    return {int(value) for value in manifest["session_number"].dropna().tolist()}


def manifest_hashes(manifest: pd.DataFrame) -> set[str]:
    if manifest.empty or "source_file_hash" not in manifest.columns:
        return set()
    return set(manifest["source_file_hash"].dropna().astype("string").tolist())


def inspect_existing_sessions(data_dir: Path = DATA_DIR, manifest: pd.DataFrame | None = None) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    found: set[int] = set()
    errors: list[str] = []
    for item in data_dir.iterdir():
        if item.is_file():
            continue
        if not item.is_dir():
            continue
        match = SESSION_RE.match(item.name)
        if not match:
            errors.append(f"invalid session-folder name: {item}")
            continue
        number = int(match.group(1))
        if number in found:
            errors.append(f"duplicate session number: {item.name}")
        found.add(number)
        child_names = sorted(child.name for child in item.iterdir())
        required = {LABEL_FILENAME, SENSOR_FILENAME}
        optional_metadata = f"{item.name}.imsession"
        actual = set(child_names)
        missing = sorted(required - actual)
        unexpected = sorted(actual - required - {optional_metadata})
        if missing or unexpected:
            errors.append(
                f"malformed session {item}: missing={missing}, unexpected={unexpected}; "
                f"allowed={sorted(required | {optional_metadata})}"
            )

    if manifest is not None and not manifest.empty and "session_number" in manifest.columns:
        manifest_numbers = manifest_session_numbers(manifest)
        missing_dirs = sorted(manifest_numbers - found)
        missing_manifest = sorted(found - manifest_numbers)
        if missing_dirs:
            errors.append(f"manifest entry points to missing session folder(s): {missing_dirs}")
        if missing_manifest:
            errors.append(f"session folder(s) not represented in manifest: {missing_manifest}")
    if errors:
        raise ProcessingError("; ".join(errors))
    return max(found, default=0)


def already_processed(manifest: pd.DataFrame, source: SourceFile) -> bool:
    return source.content_hash in manifest_hashes(manifest)


def atomic_write_dataframe(df: pd.DataFrame, destination: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8", lineterminator="\n")
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_session(recording: NormalizedRecording, number: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_dir = session_path(number)
    if final_dir.exists():
        raise ProcessingError(f"refusing to overwrite existing session directory: {final_dir}")
    tmp_dir = DATA_DIR / f".{session_id(number)}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    try:
        atomic_write_dataframe(recording.sensor, tmp_dir / SENSOR_FILENAME)
        atomic_write_dataframe(recording.label_intervals, tmp_dir / LABEL_FILENAME)
        verify_session(recording, number, tmp_dir)
        tmp_dir.replace(final_dir)
        verify_session(recording, number, final_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return final_dir


def verify_session(recording: NormalizedRecording, number: int, directory: Path) -> None:
    if not directory.is_dir():
        raise ProcessingError(f"session path is not a folder: {directory}")
    if directory.name not in {session_id(number), f".{session_id(number)}.tmp"}:
        raise ProcessingError(f"folder name {directory.name} does not match {session_id(number)}")
    child_names = sorted(child.name for child in directory.iterdir())
    expected = [LABEL_FILENAME, SENSOR_FILENAME]
    if child_names != expected:
        raise ProcessingError(f"{directory}: contains {child_names}, expected {expected}")
    sensor_path = directory / SENSOR_FILENAME
    label_path = directory / LABEL_FILENAME
    sensor = pd.read_csv(sensor_path)
    labels = pd.read_csv(label_path)
    if list(sensor.columns) != SENSOR_COLUMNS:
        raise ProcessingError(f"{sensor_path}: columns {list(sensor.columns)}; expected {SENSOR_COLUMNS}")
    if len(sensor) != len(recording.sensor):
        raise ProcessingError(f"{sensor_path}: row count {len(sensor)}; expected {len(recording.sensor)}")
    for column in SENSOR_COLUMNS:
        if pd.to_numeric(sensor[column], errors="coerce").isna().any():
            raise ProcessingError(f"{sensor_path}: non-numeric value in {column}")
    if abs(float(sensor["elapsed_seconds"].iloc[0])) > 1e-9:
        raise ProcessingError(f"{sensor_path}: first elapsed_seconds is not zero")
    if (sensor["elapsed_seconds"] < -1e-9).any():
        raise ProcessingError(f"{sensor_path}: elapsed_seconds contains negative values")
    if (sensor["elapsed_seconds"].diff().fillna(0) < -1e-9).any():
        raise ProcessingError(f"{sensor_path}: elapsed_seconds is not nondecreasing")
    original_diffs = recording.sensor["elapsed_seconds"].diff().dropna().reset_index(drop=True)
    written_diffs = sensor["elapsed_seconds"].diff().dropna().reset_index(drop=True)
    if not original_diffs.sub(written_diffs).abs().le(1e-9).all():
        raise ProcessingError(f"{sensor_path}: sample spacing changed during export")
    validate_intervals(labels, 0.0, recording.offset_output_end_time)


def append_manifest_row(recording: NormalizedRecording, number: int, final_dir: Path) -> None:
    row = {
        "session_id": session_id(number),
        "session_number": number,
        "session_dir_path": str(final_dir),
        "sensor_file_path": str(final_dir / SENSOR_FILENAME),
        "label_file_path": str(final_dir / LABEL_FILENAME),
        "original_source_filename": recording.source.path.name,
        "original_full_source_path": str(recording.source.path),
        "source_format_version": recording.source.schema,
        "source_file_hash": recording.source.content_hash,
        "source_size_bytes": recording.source.size_bytes,
        "source_modified_time": recording.source.modified_time,
        "source_row_count": recording.source_row_count,
        "output_row_count": len(recording.sensor),
        "original_first_elapsed_time": recording.original_first_elapsed_time,
        "original_final_elapsed_time": recording.original_final_elapsed_time,
        "offset_applied": recording.offset_applied,
        "offset_output_start_time": recording.offset_output_start_time,
        "offset_output_end_time": recording.offset_output_end_time,
        "median_positive_sample_interval": recording.median_positive_sample_interval,
        "label_interval_count": recording.interval_count,
        "smoking_sample_count": recording.smoking_sample_count,
        "no_smoking_sample_count": recording.no_smoking_sample_count,
        "processing_datetime": datetime.now().isoformat(timespec="seconds"),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST_PATH.exists() and MANIFEST_PATH.stat().st_size > 0
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def process_batch(
    processor_name: str,
    source_dir: Path,
    allowed_schemas: set[str],
    *,
    dry_run: bool = False,
    force: bool = False,
    recursive: bool = False,
    continue_on_error: bool = False,
) -> list[ProcessResult]:
    source_dir = source_dir.expanduser().resolve()
    files = discover_csv_files(source_dir, recursive)
    manifest = load_manifest()
    detect_old_manifest(manifest)
    largest_session = inspect_existing_sessions(DATA_DIR, manifest if "session_number" in manifest.columns else None)
    next_number = max(largest_session, max(manifest_session_numbers(manifest), default=0)) + 1

    print(f"Processor: {processor_name}")
    print(f"Source folder: {source_dir}")
    print(f"CSV files discovered: {len(files)}")
    if not files:
        print_summary(processor_name, source_dir, [], largest_session)
        return []

    results: list[ProcessResult] = []
    for path in files:
        try:
            recording = normalize_recording(path, allowed_schemas)
            duplicate = already_processed(manifest, recording.source)
            if duplicate and not force:
                results.append(
                    ProcessResult(
                        source_path=path,
                        schema=recording.source.schema,
                        status="skipped",
                        message="source hash already present in session manifest",
                        original_range=(recording.original_first_elapsed_time, recording.original_final_elapsed_time),
                        offset_applied=recording.offset_applied,
                        exported_range=(recording.offset_output_start_time, recording.offset_output_end_time),
                    )
                )
                continue

            number = next_number
            next_number += 1
            sid = session_id(number)
            final_dir = session_path(number)
            result_kwargs = {
                "session_number": number,
                "session_id": sid,
                "session_path": final_dir,
                "source_rows": recording.source_row_count,
                "interval_count": recording.interval_count,
                "smoking_count": recording.smoking_sample_count,
                "no_smoking_count": recording.no_smoking_sample_count,
                "original_range": (recording.original_first_elapsed_time, recording.original_final_elapsed_time),
                "offset_applied": recording.offset_applied,
                "exported_range": (recording.offset_output_start_time, recording.offset_output_end_time),
            }
            if dry_run:
                results.append(
                    ProcessResult(
                        source_path=path,
                        schema=recording.source.schema,
                        status="planned",
                        message=f"dry run; duplicate_source={duplicate}; no files written",
                        **result_kwargs,
                    )
                )
                continue

            written_dir = write_session(recording, number)
            append_manifest_row(recording, number, written_dir)
            verify_manifest_entry(number, written_dir)
            results.append(
                ProcessResult(
                    source_path=path,
                    schema=recording.source.schema,
                    status="processed",
                    message="session written and verified",
                    **result_kwargs,
                )
            )
        except Exception as exc:
            results.append(ProcessResult(path, None, "failed", str(exc)))
            if not continue_on_error:
                break

    final_largest = inspect_existing_sessions(DATA_DIR, load_manifest() if MANIFEST_PATH.exists() else None)
    print_summary(processor_name, source_dir, results, final_largest)
    if any(result.status == "failed" for result in results):
        raise ProcessingError("one or more source files failed; see summary above")
    return results


def verify_manifest_entry(number: int, final_dir: Path) -> None:
    manifest = load_manifest()
    rows = manifest[manifest["session_number"].astype("int64") == number]
    if len(rows) != 1:
        raise ProcessingError(f"manifest does not contain exactly one entry for {session_id(number)}")
    row = rows.iloc[0]
    if Path(row["session_dir_path"]) != final_dir:
        raise ProcessingError(f"manifest session path mismatch for {session_id(number)}")
    if Path(row["sensor_file_path"]) != final_dir / SENSOR_FILENAME:
        raise ProcessingError(f"manifest sensor path mismatch for {session_id(number)}")
    if Path(row["label_file_path"]) != final_dir / LABEL_FILENAME:
        raise ProcessingError(f"manifest label path mismatch for {session_id(number)}")


def print_summary(processor_name: str, source_dir: Path, results: list[ProcessResult], final_largest: int) -> None:
    print("\nProcessing summary:")
    print(f"  processor used: {processor_name}")
    print(f"  source folder: {source_dir}")
    print(f"  already processed/skipped: {sum(r.status == 'skipped' for r in results)}")
    print(f"  newly processed: {sum(r.status == 'processed' for r in results)}")
    print(f"  planned only: {sum(r.status == 'planned' for r in results)}")
    print(f"  failed: {sum(r.status == 'failed' for r in results)}")
    schema_counts: dict[str, int] = {}
    for result in results:
        if result.schema:
            schema_counts[result.schema] = schema_counts.get(result.schema, 0) + 1
    for schema, count in sorted(schema_counts.items()):
        print(f"  detected {schema}: {count}")
    for result in results:
        print(f"  {result.status.upper()}: {result.source_path.name}")
        print(f"    schema: {result.schema or 'unknown'}")
        print(f"    session: {result.session_id or '-'}")
        if result.original_range:
            print(f"    original time range: {result.original_range[0]} to {result.original_range[1]}")
        if result.offset_applied is not None:
            print(f"    applied offset: {result.offset_applied}")
        if result.exported_range:
            print(f"    exported time range: {result.exported_range[0]} to {result.exported_range[1]}")
        print(f"    rows={result.source_rows or '-'} intervals={result.interval_count or '-'} "
              f"smoking={result.smoking_count if result.smoking_count is not None else '-'} "
              f"no_smoking={result.no_smoking_count if result.no_smoking_count is not None else '-'}")
        if result.session_path:
            print(f"    {result.source_path.name}")
            print(f"    -> {result.session_id}")
            print(f"       {SENSOR_FILENAME}")
            print(f"       {LABEL_FILENAME}")
            print(f"    session path: {result.session_path}")
        print(f"    message: {result.message}")
    print(f"  final largest session number: {final_largest}")
    print(f"  manifest path: {MANIFEST_PATH}")


def build_arg_parser(description: str, default_source_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source-dir", type=Path, default=default_source_dir, help=f"Source folder. Default: {default_source_dir}")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show planned session folders without writing outputs.")
    parser.add_argument("--force", action="store_true", help="Process a source even when its content hash is already in the manifest.")
    parser.add_argument("--recursive", action="store_true", help="Search the source folder recursively.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue independent files after a failure, then exit nonzero if any failed.",
    )
    return parser


def run_entry(processor_name: str, description: str, default_source_dir: Path, allowed_schemas: set[str]) -> int:
    parser = build_arg_parser(description, default_source_dir)
    args = parser.parse_args()
    try:
        process_batch(
            processor_name,
            args.source_dir,
            allowed_schemas,
            dry_run=args.dry_run,
            force=args.force,
            recursive=args.recursive,
            continue_on_error=args.continue_on_error,
        )
    except ProcessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
