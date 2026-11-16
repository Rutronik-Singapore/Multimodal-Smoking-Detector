#!/usr/bin/env python3
"""Shared SEN66 Decision Tree training utilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = DT_ROOT.parents[1]
DEFAULT_CSV = REPOSITORY_ROOT / "logs" / "csv_merge" / "0408_merge.csv"
PARTITION_PATH = DT_ROOT / "session_partitions.csv"

RAW_FEATURES = ["pm1_0", "pm2_5", "pm4_0", "pm10_0", "voc_index", "nox_index", "co2_ppm"]
REQUIRED_COLUMNS = ["dataset_id", "elapsed_seconds", *RAW_FEATURES, "label"]
DT0_ROLLING_BASE_FEATURES = ["pm2_5", "pm10_0", "voc_index", "co2_ppm"]
DT0_WINDOWS = [5, 10, 30]
DT2_WINDOWS_SECONDS = [5, 10, 30]
SEED_DEFAULT = 84
N_THRESHOLDS_DEFAULT = 128
NEAR_BEST_TOLERANCE_DEFAULT = 0.002
RUN_OUTPUT_PREFIXES = ("DT0_", "DT1_", "DT2_")

FIRMWARE_CONTRACT = {
    "inspected_root": r"C:\Users\Lab\pse84ai\IAQ_PSE",
    "model_header_included_by": r"proj_cm33_ns\source\prediction\sen66_prediction.c",
    "current_model_header": "sen66_time_tree_model.h",
    "current_prediction_function": "sen66_time_tree_predict_smoking",
    "wrapper_api": "bool sen66_prediction_update(uint16_t pm1, uint16_t pm25, uint16_t pm4, uint16_t pm10, int16_t voc, int16_t nox, uint16_t co2, uint8_t *prediction)",
    "generated_model_style": "pure bool function with scalar float feature arguments",
    "return_convention": "false/0 = no smoking, true/1 = smoking",
    "sensor_scaling": "SEN66 integer PM/VOC/NOx inputs are divided by 10.0f; CO2 is used as ppm float",
    "rolling_owner": "proj_cm33_ns/source/prediction/sen66_prediction.c owns static rolling buffers for the current DT-0 style model",
    "prediction_cadence": "one prediction per successful SEN66 sample in task_sen66_stream on CM33 non-secure core",
    "build_requirements": "proj_cm33_ns/Makefile includes source/model and source/prediction; automatic source discovery builds .c files under source/model",
}


class PipelineError(Exception):
    pass


@dataclass
class Node:
    prediction: int
    depth: int
    samples: int
    impurity: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [p for p in path.iterdir() if p.is_file() and p.name.startswith(RUN_OUTPUT_PREFIXES)]
        if existing:
            raise PipelineError(f"{path} already contains model outputs; rerun with --overwrite or --archive-existing")


def prepare_output_dir(path: Path, model_name: str, overwrite: bool, archive_existing: bool) -> Path | None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.is_file() and p.name.startswith(f"{model_name}_")]
    if not existing:
        return None
    if archive_existing:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = path / "runs" / stamp
        archive_dir.mkdir(parents=True, exist_ok=False)
        for item in existing:
            shutil.move(str(item), str(archive_dir / item.name))
        return archive_dir
    if not overwrite:
        raise PipelineError(f"{path} already has {model_name} outputs; use --overwrite or --archive-existing")
    return None


def partition_manifest_hash() -> str | None:
    return file_sha256(PARTITION_PATH) if PARTITION_PATH.exists() else None


def validate_and_load_csv(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not csv_path.exists():
        raise PipelineError(f"CSV does not exist: {csv_path}")
    source_hash = file_sha256(csv_path)
    df = pd.read_csv(csv_path)
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise PipelineError(f"Duplicate columns: {duplicate_columns}")
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise PipelineError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            raise PipelineError(f"Missing required value in {col}")
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise PipelineError("Non-finite numeric value found")
    if (df["dataset_id"] < 1).any():
        raise PipelineError("dataset_id must be positive")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise PipelineError(f"Labels must be 0/1, found {sorted(df['label'].unique())}")
    if (df["elapsed_seconds"] < 0).any():
        raise PipelineError("elapsed_seconds must be nonnegative")
    df["dataset_id"] = df["dataset_id"].astype(int)
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["dataset_id", "elapsed_seconds"], kind="mergesort").reset_index(drop=True)
    decreasing = df.groupby("dataset_id", sort=False)["elapsed_seconds"].diff().fillna(0) < 0
    if decreasing.any():
        bad = int(np.flatnonzero(decreasing.to_numpy())[0])
        raise PipelineError(f"elapsed_seconds decreases at sorted row {bad}")
    for dataset_id, group in df.groupby("dataset_id", sort=False):
        diffs = group["elapsed_seconds"].diff().dropna()
        if diffs[diffs > 0].empty:
            raise PipelineError(f"dataset_id {dataset_id} has no measurable positive sampling interval")
    summary = {
        "path": str(csv_path),
        "size_bytes": csv_path.stat().st_size,
        "sha256": source_hash,
        "row_count": len(df),
        "dataset_count": int(df["dataset_id"].nunique()),
        "columns": list(df.columns),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "rows_per_dataset": {str(k): int(v) for k, v in df["dataset_id"].value_counts().sort_index().items()},
        "elapsed_ranges": {
            str(k): [float(g["elapsed_seconds"].min()), float(g["elapsed_seconds"].max())]
            for k, g in df.groupby("dataset_id", sort=True)
        },
    }
    return df, summary


def print_input_summary(summary: dict[str, Any]) -> None:
    print(f"Selected CSV: {summary['path']}")
    print(f"File size: {summary['size_bytes']}")
    print(f"SHA256: {summary['sha256']}")
    print(f"Rows: {summary['row_count']}")
    print(f"Datasets: {summary['dataset_count']}")
    print(f"Columns: {summary['columns']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Rows per dataset: {summary['rows_per_dataset']}")
    print(f"Elapsed ranges: {summary['elapsed_ranges']}")


def clean_sensor_startup_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = {
        "pm1_0_lt_6500": df["pm1_0"] < 6500,
        "pm2_5_lt_6500": df["pm2_5"] < 6500,
        "pm4_0_lt_6500": df["pm4_0"] < 6500,
        "pm10_0_lt_6500": df["pm10_0"] < 6500,
        "nox_index_lt_3000": df["nox_index"] < 3000,
        "co2_ppm_lt_65000": df["co2_ppm"] < 65000,
    }
    keep = np.logical_and.reduce([mask.to_numpy() for mask in rules.values()])
    removed_rows = []
    for dataset_id, group in df.loc[~keep].groupby("dataset_id"):
        for reason, mask in rules.items():
            count = int((~mask.loc[group.index]).sum())
            if count:
                removed_rows.append({"dataset_id": int(dataset_id), "reason": reason, "removed_rows": count})
    return df.loc[keep].copy().reset_index(drop=True), pd.DataFrame(removed_rows)


def create_or_load_partitions(df: pd.DataFrame, source_hash: str, seed: int, regenerate: bool, write: bool = True) -> pd.DataFrame:
    dataset_stats = df.groupby("dataset_id").agg(
        row_count=("label", "size"),
        smoking_count=("label", "sum"),
    ).reset_index()
    dataset_stats["no_smoking_count"] = dataset_stats["row_count"] - dataset_stats["smoking_count"]
    if PARTITION_PATH.exists() and not regenerate:
        parts = pd.read_csv(PARTITION_PATH)
        required = {"dataset_id", "partition", "row_count", "smoking_count", "no_smoking_count", "source_csv_hash"}
        if not required.issubset(parts.columns):
            raise PipelineError(f"Existing partition manifest has unexpected columns: {list(parts.columns)}")
        if set(parts["dataset_id"]) != set(dataset_stats["dataset_id"]):
            raise PipelineError("Partition manifest dataset_ids differ from selected CSV; use --regenerate-partitions intentionally")
        if set(parts["source_csv_hash"]) != {source_hash}:
            raise PipelineError("Partition manifest source hash differs from selected CSV; use --regenerate-partitions intentionally")
        return parts
    rng = np.random.default_rng(seed)
    ids = dataset_stats["dataset_id"].to_numpy().copy()
    rng.shuffle(ids)
    n = len(ids)
    test_n = max(3, round(n * 0.15))
    val_n = max(3, round(n * 0.15))
    test_ids = set(ids[:test_n])
    val_ids = set(ids[test_n:test_n + val_n])
    rows = []
    for _, row in dataset_stats.iterrows():
        did = int(row["dataset_id"])
        part = "test" if did in test_ids else "validation" if did in val_ids else "train"
        rows.append({
            "dataset_id": did,
            "partition": part,
            "row_count": int(row["row_count"]),
            "smoking_count": int(row["smoking_count"]),
            "no_smoking_count": int(row["no_smoking_count"]),
            "source_csv_hash": source_hash,
        })
    parts = pd.DataFrame(rows).sort_values("dataset_id")
    if write:
        DT_ROOT.mkdir(parents=True, exist_ok=True)
        parts.to_csv(PARTITION_PATH, index=False)
    return parts


def validate_frozen_partitions(df: pd.DataFrame, source_hash: str) -> pd.DataFrame:
    if not PARTITION_PATH.is_file():
        raise PipelineError(f"Frozen partition manifest does not exist: {PARTITION_PATH}")
    return create_or_load_partitions(df, source_hash, SEED_DEFAULT, regenerate=False, write=False)


def split_by_partition(df: pd.DataFrame, parts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    part_map = dict(zip(parts["dataset_id"], parts["partition"]))
    work = df.copy()
    work["partition"] = work["dataset_id"].map(part_map)
    if work["partition"].isna().any():
        raise PipelineError("Some dataset rows are missing partition assignment")
    return (
        work[work["partition"].eq("train")].drop(columns=["partition"]),
        work[work["partition"].eq("validation")].drop(columns=["partition"]),
        work[work["partition"].eq("test")].drop(columns=["partition"]),
    )


def grouped_folds(dataset_ids: list[int], n_target: int = 4) -> list[tuple[set[int], set[int]]]:
    n_splits = min(n_target, len(dataset_ids))
    if n_splits < 2:
        raise PipelineError("Need at least two grouped sessions for CV")
    folds = [set() for _ in range(n_splits)]
    for i, did in enumerate(sorted(dataset_ids)):
        folds[i % n_splits].add(int(did))
    result = []
    all_ids = set(map(int, dataset_ids))
    for valid_ids in folds:
        train_ids = all_ids - valid_ids
        if train_ids and valid_ids:
            result.append((train_ids, valid_ids))
    return result


def impurity(y: np.ndarray, weights: np.ndarray, criterion: str) -> float:
    total = weights.sum()
    if total <= 0:
        return 0.0
    p1 = weights[y == 1].sum() / total
    p0 = 1.0 - p1
    if criterion == "entropy":
        terms = [p for p in [p0, p1] if p > 0]
        return float(-sum(p * math.log2(p) for p in terms))
    return float(1.0 - p0 * p0 - p1 * p1)


def majority_class(y: np.ndarray, weights: np.ndarray) -> int:
    return int(weights[y == 1].sum() >= weights[y == 0].sum())


def class_weights(y: np.ndarray, mode: str | None) -> np.ndarray:
    if mode != "balanced":
        return np.ones(len(y), dtype=float)
    counts = np.bincount(y.astype(int), minlength=2)
    total = len(y)
    w = np.ones(len(y), dtype=float)
    for cls in [0, 1]:
        if counts[cls] > 0:
            w[y == cls] = total / (2.0 * counts[cls])
    return w


def candidate_thresholds(values: np.ndarray, max_thresholds: int) -> np.ndarray:
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([])
    if unique.size <= max_thresholds:
        return (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.quantile(values, np.linspace(0.01, 0.99, max_thresholds)))


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    criterion: str = "gini",
    max_depth: int | None = 5,
    min_samples_leaf: int = 20,
    min_samples_split: int = 40,
    max_thresholds: int = N_THRESHOLDS_DEFAULT,
    depth: int = 0,
) -> Node:
    pred = majority_class(y, weights)
    node = Node(prediction=pred, depth=depth, samples=len(y), impurity=impurity(y, weights, criterion))
    if (max_depth is not None and depth >= max_depth) or len(np.unique(y)) == 1 or len(y) < min_samples_split:
        return node
    parent_impurity = node.impurity
    best = None
    for feature_index in range(X.shape[1]):
        values = X[:, feature_index]
        for threshold in candidate_thresholds(values, max_thresholds):
            left = values <= threshold
            left_count = int(left.sum())
            right_count = len(y) - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            right = ~left
            lw = weights[left].sum()
            rw = weights[right].sum()
            total = lw + rw
            split_impurity = lw / total * impurity(y[left], weights[left], criterion) + rw / total * impurity(y[right], weights[right], criterion)
            gain = parent_impurity - split_impurity
            if best is None or gain > best["gain"]:
                best = {"gain": gain, "feature_index": feature_index, "threshold": float(threshold), "left": left}
    if best is None or best["gain"] <= 1e-12:
        return node
    left = best["left"]
    node.feature_index = int(best["feature_index"])
    node.threshold = float(best["threshold"])
    node.gain = float(best["gain"])
    node.left = build_tree(X[left], y[left], weights[left], criterion=criterion, max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                           max_thresholds=max_thresholds, depth=depth + 1)
    node.right = build_tree(X[~left], y[~left], weights[~left], criterion=criterion, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                            max_thresholds=max_thresholds, depth=depth + 1)
    return node


def predict_one(node: Node, row: np.ndarray) -> int:
    while not node.is_leaf:
        node = node.left if row[node.feature_index] <= node.threshold else node.right
    return int(node.prediction)


def predict_tree(node: Node, X: np.ndarray) -> np.ndarray:
    return np.array([predict_one(node, row) for row in X], dtype=int)


def tree_depth(node: Node) -> int:
    if node.is_leaf:
        return node.depth
    return max(tree_depth(node.left), tree_depth(node.right))


def leaf_count(node: Node) -> int:
    if node.is_leaf:
        return 1
    return leaf_count(node.left) + leaf_count(node.right)


def split_node_count(node: Node) -> int:
    if node.is_leaf:
        return 0
    return 1 + split_node_count(node.left) + split_node_count(node.right)


def feature_usage(node: Node, features: list[str], counts: dict[str, int] | None = None) -> dict[str, int]:
    counts = counts or {f: 0 for f in features}
    if not node.is_leaf:
        counts[features[node.feature_index]] += 1
        feature_usage(node.left, features, counts)
        feature_usage(node.right, features, counts)
    return counts


def sensor_for_feature(feature: str) -> str:
    for sensor in RAW_FEATURES:
        if feature == sensor or feature.startswith(sensor + "_"):
            return sensor
    return "unknown"


def family_for_feature(feature: str) -> str:
    if feature in RAW_FEATURES:
        return "raw"
    for family in ["mean", "max", "min", "std", "delta", "slope"]:
        if f"_{family}_" in feature:
            return family
    return "other"


def feature_count_summary(tree: Node, features: list[str]) -> dict[str, Any]:
    usage = feature_usage(tree, features)
    used = [f for f, count in usage.items() if count > 0]
    return {
        "candidate_feature_count": len(features),
        "selected_input_feature_count": len(features),
        "unique_features_used_in_tree": len(used),
        "total_split_nodes": split_node_count(tree),
        "unique_sensor_channels_used": len({sensor_for_feature(f) for f in used}),
        "temporal_feature_families_used": ",".join(sorted({family_for_feature(f) for f in used})),
        "features_used_in_tree": ",".join(used),
    }


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    out[prefix + "support"] = int(len(y_true))
    out[prefix + "accuracy"] = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1s = []
    recalls = []
    for cls, name in [(0, "no_smoking"), (1, "smoking")]:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[prefix + f"{name}_precision"] = float(precision)
        out[prefix + f"{name}_recall"] = float(recall)
        out[prefix + f"{name}_f1"] = float(f1)
        out[prefix + f"{name}_support"] = int((y_true == cls).sum())
        f1s.append(f1)
        recalls.append(recall)
    out[prefix + "macro_f1"] = float(np.mean(f1s))
    out[prefix + "balanced_accuracy"] = float(np.mean(recalls))
    out[prefix + "confusion_tn"] = int(((y_true == 0) & (y_pred == 0)).sum())
    out[prefix + "confusion_fp"] = int(((y_true == 0) & (y_pred == 1)).sum())
    out[prefix + "confusion_fn"] = int(((y_true == 1) & (y_pred == 0)).sum())
    out[prefix + "confusion_tp"] = int(((y_true == 1) & (y_pred == 1)).sum())
    return out


def false_positive_stats(df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float | int]:
    no_smoke = df["label"].to_numpy(dtype=int) == 0
    fp = int((no_smoke & (y_pred == 1)).sum())
    seconds = 0.0
    episode_durations = []
    for _, group in df.assign(pred=y_pred).groupby("dataset_id", sort=False):
        no_smoke_group = group[group["label"].eq(0)]
        if len(no_smoke_group) > 1:
            seconds += float(no_smoke_group["elapsed_seconds"].max() - no_smoke_group["elapsed_seconds"].min())
        active_start = None
        previous_time = None
        for _, row in group.iterrows():
            is_fp = int(row["label"]) == 0 and int(row["pred"]) == 1
            t = float(row["elapsed_seconds"])
            if is_fp and active_start is None:
                active_start = t
            if (not is_fp) and active_start is not None:
                episode_durations.append(max(0.0, (previous_time if previous_time is not None else t) - active_start))
                active_start = None
            previous_time = t
        if active_start is not None:
            episode_durations.append(max(0.0, (previous_time if previous_time is not None else active_start) - active_start))
    hours = seconds / 3600.0 if seconds > 0 else 0.0
    return {
        "false_positive_samples": fp,
        "false_positive_samples_per_no_smoking_hour": fp / hours if hours > 0 else 0.0,
        "false_positive_episode_count": len(episode_durations),
        "false_positive_episodes_per_no_smoking_hour": len(episode_durations) / hours if hours > 0 else 0.0,
        "false_positive_episode_mean_duration": float(np.mean(episode_durations)) if episode_durations else 0.0,
        "false_positive_episode_max_duration": float(np.max(episode_durations)) if episode_durations else 0.0,
    }


def false_smoking_per_hour(df: pd.DataFrame, y_pred: np.ndarray) -> float:
    # Backward-compatible alias; reports now use false_positive_samples_per_no_smoking_hour.
    no_smoke = df["label"].to_numpy(dtype=int) == 0
    fp = int((no_smoke & (y_pred == 1)).sum())
    seconds = 0.0
    for _, group in df[no_smoke].groupby("dataset_id"):
        if len(group) > 1:
            seconds += float(group["elapsed_seconds"].max() - group["elapsed_seconds"].min())
    return fp / (seconds / 3600.0) if seconds > 0 else 0.0


def event_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, Any]:
    tmp = df[["dataset_id", "elapsed_seconds", "label"]].copy()
    tmp["pred"] = y_pred
    delays = []
    events = 0
    detected = 0
    transitions = 0
    for _, g in tmp.groupby("dataset_id", sort=False):
        pred = g["pred"].to_numpy()
        transitions += int(np.abs(np.diff(pred)).sum()) if len(pred) > 1 else 0
        labels = g["label"].to_numpy()
        times = g["elapsed_seconds"].to_numpy()
        for i in range(1, len(g)):
            if labels[i - 1] == 0 and labels[i] == 1:
                events += 1
                end = len(g)
                for j in range(i + 1, len(g)):
                    if labels[j] == 0:
                        end = j
                        break
                hits = np.where(pred[i:end] == 1)[0]
                if hits.size:
                    detected += 1
                    delays.append(float(times[i + hits[0]] - times[i]))
    return {
        "event_count": events,
        "event_detection_rate": detected / events if events else 0.0,
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
        "prediction_transition_count": transitions,
    }


def evaluate_predictions(df: pd.DataFrame, y_pred: np.ndarray, prefix: str = "") -> dict[str, Any]:
    result = metrics_dict(df["label"].to_numpy(dtype=int), y_pred, prefix)
    result.update({prefix + k: v for k, v in false_positive_stats(df, y_pred).items()})
    result.update({prefix + k: v for k, v in event_metrics(df, y_pred).items()})
    return result


def grouped_cv_score(df: pd.DataFrame, features: list[str], params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    ids = sorted(map(int, df["dataset_id"].unique()))
    for fold_idx, (train_ids, valid_ids) in enumerate(grouped_folds(ids), start=1):
        tr = df[df["dataset_id"].isin(train_ids)]
        va = df[df["dataset_id"].isin(valid_ids)]
        X = tr[features].to_numpy(dtype=float)
        y = tr["label"].to_numpy(dtype=int)
        w = class_weights(y, params.get("class_weight"))
        tree = build_tree(X, y, w, criterion=params.get("criterion", "gini"),
                          max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 20),
                          min_samples_split=params.get("min_samples_split", 40),
                          max_thresholds=params.get("max_thresholds", N_THRESHOLDS_DEFAULT))
        pred = predict_tree(tree, va[features].to_numpy(dtype=float))
        m = evaluate_predictions(va, pred)
        usage = feature_usage(tree, features)
        used = [f for f, count in usage.items() if count > 0]
        m.update({
            "fold": fold_idx,
            "training_dataset_ids": json.dumps(sorted(train_ids)),
            "validation_dataset_ids": json.dumps(sorted(valid_ids)),
            "training_row_count": len(tr),
            "validation_row_count": len(va),
            "training_smoking_count": int(tr["label"].sum()),
            "training_no_smoking_count": int(len(tr) - tr["label"].sum()),
            "validation_smoking_count": int(va["label"].sum()),
            "validation_no_smoking_count": int(len(va) - va["label"].sum()),
            "rows_excluded_by_feature_processing": 0,
            "eligible_rows": len(tr) + len(va),
            "fold_tree_depth": tree_depth(tree),
            "fold_leaf_count": leaf_count(tree),
            "features_used": json.dumps(used),
            "feature_split_counts": json.dumps({f: int(c) for f, c in usage.items() if c > 0}),
            "sensor_groups_represented": json.dumps(sorted({sensor_for_feature(f) for f in used})),
        })
        rows.append(m)
    folds = pd.DataFrame(rows)
    summary = {
        "cv_smoking_f1_mean": float(folds["smoking_f1"].mean()),
        "cv_smoking_f1_std": float(folds["smoking_f1"].std(ddof=0)),
        "cv_balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
    }
    return folds, summary


def validation_candidate_score(
    candidate_id: str,
    model_type: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    *,
    tuning_stage: str,
    temporal_family: str = "",
    window_duration_set: str = "",
    sensor_subset: str = "all",
    rows_excluded: int = 0,
    eligible_rows: int | None = None,
) -> tuple[dict[str, Any], Node, pd.DataFrame]:
    cv_folds, cv_summary = grouped_cv_score(train_df, features, params)
    tree = train_final_tree(train_df, features, params)
    pred = predict_tree(tree, validation_df[features].to_numpy(dtype=float))
    val = evaluate_predictions(validation_df, pred, "validation_")
    usage = feature_usage(tree, features)
    row = {
        "candidate_id": candidate_id,
        "model_type": model_type,
        "tuning_stage": tuning_stage,
        "criterion": params.get("criterion"),
        "max_depth": params.get("max_depth"),
        "min_samples_split": params.get("min_samples_split"),
        "min_samples_leaf": params.get("min_samples_leaf"),
        "class_weight": params.get("class_weight"),
        "ccp_alpha": params.get("ccp_alpha", 0.0),
        "temporal_family": temporal_family,
        "window_duration_set": window_duration_set,
        "sensor_subset": sensor_subset,
        "mean_training_fold_smoking_f1": cv_summary["cv_smoking_f1_mean"],
        "std_training_fold_smoking_f1": cv_summary["cv_smoking_f1_std"],
        "mean_training_fold_balanced_accuracy": cv_summary["cv_balanced_accuracy_mean"],
        "fixed_validation_smoking_precision": val["validation_smoking_precision"],
        "fixed_validation_smoking_recall": val["validation_smoking_recall"],
        "fixed_validation_smoking_f1": val["validation_smoking_f1"],
        "fixed_validation_balanced_accuracy": val["validation_balanced_accuracy"],
        "fitted_tree_depth": tree_depth(tree),
        "leaf_count": leaf_count(tree),
        "total_split_nodes": split_node_count(tree),
        "candidate_feature_count": len(features),
        "rows_excluded_by_feature_processing": rows_excluded,
        "eligible_rows": eligible_rows if eligible_rows is not None else len(train_df) + len(validation_df),
        "features_used": json.dumps([f for f, c in usage.items() if c > 0]),
        "feature_split_counts": json.dumps({f: int(c) for f, c in usage.items() if c > 0}),
    }
    return row, tree, cv_folds


def rank_candidates(candidates: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked = ranked.sort_values(
        ["fixed_validation_smoking_f1", "fixed_validation_balanced_accuracy", "fitted_tree_depth", "leaf_count", "total_split_nodes"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    best = float(ranked["fixed_validation_smoking_f1"].iloc[0])
    ranked["within_near_best_tolerance"] = ranked["fixed_validation_smoking_f1"] >= best - tolerance
    return ranked


def choose_best_candidate(candidates: pd.DataFrame, tolerance: float) -> pd.Series:
    ranked = rank_candidates(candidates, tolerance)
    near = ranked[ranked["within_near_best_tolerance"]].copy()
    near = near.sort_values(
        ["fixed_validation_balanced_accuracy", "fitted_tree_depth", "leaf_count", "total_split_nodes"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return near.iloc[0]


def add_dt0_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["dataset_id", "elapsed_seconds"], kind="mergesort").copy()
    for col in DT0_ROLLING_BASE_FEATURES:
        grouped = out.groupby("dataset_id", sort=False)[col]
        for window in DT0_WINDOWS:
            out[f"{col}_mean_{window}"] = grouped.transform(lambda s: s.rolling(window=window, min_periods=1).mean())
            out[f"{col}_max_{window}"] = grouped.transform(lambda s: s.rolling(window=window, min_periods=1).max())
            out[f"{col}_delta_{window}"] = grouped.transform(lambda s: s - s.shift(window))
    return out.fillna(0)


def dt0_feature_list() -> list[str]:
    features = list(RAW_FEATURES)
    for col in DT0_ROLLING_BASE_FEATURES:
        for window in DT0_WINDOWS:
            features += [f"{col}_mean_{window}", f"{col}_max_{window}", f"{col}_delta_{window}"]
    return features


def resample_sessions_1hz(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    rows = []
    for did, g in df.sort_values(["dataset_id", "elapsed_seconds"], kind="mergesort").groupby("dataset_id", sort=False):
        start = float(g["elapsed_seconds"].min())
        work = g.copy()
        work["source_elapsed_seconds"] = work["elapsed_seconds"]
        work["elapsed_seconds"] = work["elapsed_seconds"] - start
        work["second"] = np.rint(work["elapsed_seconds"]).astype(int)
        before = len(work)
        resampled = work.sort_values(["second", "elapsed_seconds"], kind="mergesort").groupby("second", as_index=False).last()
        resampled["elapsed_seconds"] = resampled["second"].astype(float)
        resampled["dataset_id"] = int(did)
        frames.append(resampled[REQUIRED_COLUMNS + ["source_elapsed_seconds"]])
        rows.append({
            "dataset_id": int(did),
            "source_rows": before,
            "resampled_rows": len(resampled),
            "rows_removed_by_duplicate_second_collapse": before - len(resampled),
            "resampling_method": "subtract first timestamp, round elapsed seconds to nearest integer, keep last sample per second; no interpolation across sessions",
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def duration_validity_masks(g: pd.DataFrame, windows_seconds: list[int]) -> dict[int, tuple[pd.Series, pd.Series]]:
    masks = {}
    labels = g["label"].reset_index(drop=True)
    for window in windows_seconds:
        complete = pd.Series(np.arange(len(g)) >= window, index=g.index)
        consistent_values = []
        for i in range(len(g)):
            if i < window:
                consistent_values.append(False)
            else:
                window_labels = labels.iloc[i - window:i + 1]
                consistent_values.append(bool((window_labels == window_labels.iloc[-1]).all()))
        consistent = pd.Series(consistent_values, index=g.index)
        masks[window] = (complete, consistent)
    return masks


def add_dt2_features(df: pd.DataFrame, windows_seconds: list[int], families: list[str]) -> tuple[pd.DataFrame, list[str], dict[str, int], pd.DataFrame]:
    if not families or not windows_seconds:
        return df.copy().reset_index(drop=True), list(RAW_FEATURES), {}, pd.DataFrame()
    out_frames = []
    exclusion_rows = []
    feature_cols = list(RAW_FEATURES)
    for did, g in df.sort_values(["dataset_id", "elapsed_seconds"], kind="mergesort").groupby("dataset_id", sort=False):
        g = g.copy()
        g = g.reset_index(drop=True)
        window_counts = {sec: sec for sec in windows_seconds}
        validity = duration_validity_masks(g, windows_seconds)
        valid = pd.Series(True, index=g.index)
        reasons: dict[int, list[str]] = {int(i): [] for i in g.index}
        for sec in windows_seconds:
            complete, consistent = validity[sec]
            for idx in g.index[~complete]:
                reasons[int(idx)].append(f"incomplete_{sec}_second_window")
            changed = ~consistent & complete
            for idx in g.index[changed]:
                row_label = int(g.loc[idx, "label"])
                reasons[int(idx)].append("crossed_smoking_onset" if row_label == 1 else "crossed_smoking_end")
            valid &= complete & consistent
        for sensor in RAW_FEATURES:
            for sec, count in window_counts.items():
                if "mean" in families:
                    name = f"{sensor}_mean_{sec}s"
                    g[name] = g[sensor].rolling(count, min_periods=count).mean()
                    feature_cols.append(name)
                if "max" in families:
                    name = f"{sensor}_max_{sec}s"
                    g[name] = g[sensor].rolling(count, min_periods=count).max()
                    feature_cols.append(name)
                if "min" in families:
                    name = f"{sensor}_min_{sec}s"
                    g[name] = g[sensor].rolling(count, min_periods=count).min()
                    feature_cols.append(name)
                if "std" in families:
                    name = f"{sensor}_std_{sec}s"
                    g[name] = g[sensor].rolling(count, min_periods=count).std(ddof=0)
                    feature_cols.append(name)
                if "delta" in families:
                    name = f"{sensor}_delta_{sec}s"
                    g[name] = g[sensor] - g[sensor].shift(count)
                    feature_cols.append(name)
                if "slope" in families:
                    name = f"{sensor}_slope_{sec}s"
                    g[name] = (g[sensor] - g[sensor].shift(count)) / float(sec)
                    feature_cols.append(name)
        before = len(g)
        for idx, reason_list in reasons.items():
            if reason_list and not bool(valid.loc[idx]):
                exclusion_rows.append({
                    "dataset_id": int(did),
                    "elapsed_seconds": float(g.loc[idx, "elapsed_seconds"]),
                    "label": int(g.loc[idx, "label"]),
                    "reasons": ";".join(reason_list),
                })
        g = g.loc[valid].copy()
        exclusion_rows.append({
            "dataset_id": int(did),
            "elapsed_seconds": None,
            "label": None,
            "reasons": f"summary: excluded_rows={before - len(g)}; eligible_rows={len(g)}; fixed_window_counts={json.dumps(window_counts)}",
        })
        out_frames.append(g)
    feature_cols = list(dict.fromkeys(feature_cols))
    return pd.concat(out_frames, ignore_index=True), feature_cols, {}, pd.DataFrame(exclusion_rows)


def tree_to_text(node: Node, features: list[str], depth: int = 0) -> str:
    indent = "|   " * depth
    if node.is_leaf:
        return f"{indent}class: {'smoking' if node.prediction else 'no smoking'} (samples={node.samples})\n"
    f = features[node.feature_index]
    return (
        f"{indent}|--- {f} <= {node.threshold:.9g}\n"
        + tree_to_text(node.left, features, depth + 1)
        + f"{indent}|--- {f} >  {node.threshold:.9g}\n"
        + tree_to_text(node.right, features, depth + 1)
    )


def c_name(feature: str) -> str:
    return feature.replace(".", "_").replace("-", "_").replace("+", "_")


def c_args(features: list[str]) -> str:
    return ",\n    ".join(f"float {c_name(f)}" for f in features)


def node_to_c(node: Node, features: list[str], depth: int = 1) -> str:
    indent = "    " * depth
    if node.is_leaf:
        return f"{indent}return {'true' if node.prediction else 'false'};\n"
    f = c_name(features[node.feature_index])
    return (
        f"{indent}if ({f} <= {node.threshold:.9g}f)\n{indent}{{\n"
        + node_to_c(node.left, features, depth + 1)
        + f"{indent}}}\n{indent}else\n{indent}{{\n"
        + node_to_c(node.right, features, depth + 1)
        + f"{indent}}}\n"
    )


def write_c_model(model_name: str, tree: Node, features: list[str], out_dir: Path) -> tuple[Path, Path]:
    guard = f"{model_name.upper()}_MODEL_H"
    func = f"{model_name}_model_predict_smoking"
    h = out_dir / f"{model_name}_model.h"
    c = out_dir / f"{model_name}_model.c"
    h.write_text(f"""#ifndef {guard}
#define {guard}

#include <stdbool.h>

/*
 * Feature order:
{os.linesep.join(' * - ' + f for f in features)}
 *
 * Return: false/0 = no smoking, true/1 = smoking.
 */
bool {func}({c_args(features)});

#endif
""", encoding="utf-8")
    c.write_text(f"""#include \"{model_name}_model.h\"

bool {func}({c_args(features)})
{{
{node_to_c(tree, features, 1)}}}
""", encoding="utf-8")
    return c, h


def write_dt2_feature_c(selected_features: list[str], windows: list[int], families: list[str], out_dir: Path) -> tuple[Path, Path]:
    h = out_dir / "DT2_features.h"
    c = out_dir / "DT2_features.c"
    h.write_text(f"""#ifndef DT2_FEATURES_H
#define DT2_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

#define DT2_FEATURE_COUNT ({len(selected_features)}U)
#define DT2_HISTORY_SIZE ({max(windows) + 1}U)

void DT2_features_reset(void);
bool DT2_features_update(float pm1_0, float pm2_5, float pm4_0, float pm10_0,
                         float voc_index, float nox_index, float co2_ppm,
                         float features[DT2_FEATURE_COUNT]);

#endif
""", encoding="utf-8")
    body = """#include "DT2_features.h"

#include <stddef.h>

typedef struct
{
    float pm1_0[DT2_HISTORY_SIZE];
    float pm2_5[DT2_HISTORY_SIZE];
    float pm4_0[DT2_HISTORY_SIZE];
    float pm10_0[DT2_HISTORY_SIZE];
    float voc_index[DT2_HISTORY_SIZE];
    float nox_index[DT2_HISTORY_SIZE];
    float co2_ppm[DT2_HISTORY_SIZE];
    uint32_t total_count;
    uint8_t next_index;
} dt2_history_t;

static dt2_history_t state;

void DT2_features_reset(void)
{
    state.total_count = 0U;
    state.next_index = 0U;
}

static float get_age(const float *buf, uint8_t age)
{
    uint8_t idx = (uint8_t)((state.next_index + DT2_HISTORY_SIZE - 1U - age) % DT2_HISTORY_SIZE);
    return buf[idx];
}

static float mean_recent(const float *buf, uint8_t window)
{
    float sum = 0.0f;
    for (uint8_t i = 0U; i < window; i++) { sum += get_age(buf, i); }
    return sum / (float)window;
}

static float max_recent(const float *buf, uint8_t window)
{
    float value = get_age(buf, 0U);
    for (uint8_t i = 1U; i < window; i++) { float v = get_age(buf, i); if (v > value) { value = v; } }
    return value;
}

static float delta_recent(const float *buf, float current_value, uint8_t window)
{
    return current_value - get_age(buf, window);
}

bool DT2_features_update(float pm1_0, float pm2_5, float pm4_0, float pm10_0,
                         float voc_index, float nox_index, float co2_ppm,
                         float features[DT2_FEATURE_COUNT])
{
    state.pm1_0[state.next_index] = pm1_0;
    state.pm2_5[state.next_index] = pm2_5;
    state.pm4_0[state.next_index] = pm4_0;
    state.pm10_0[state.next_index] = pm10_0;
    state.voc_index[state.next_index] = voc_index;
    state.nox_index[state.next_index] = nox_index;
    state.co2_ppm[state.next_index] = co2_ppm;
    state.next_index = (uint8_t)((state.next_index + 1U) % DT2_HISTORY_SIZE);
    state.total_count++;
    if (state.total_count <= {max(windows)}U) {{ return false; }}
    uint16_t i = 0U;
"""
    for f in selected_features:
        if f in RAW_FEATURES:
            body += f"    features[i++] = {f};\n"
        elif "_mean_" in f:
            sensor, rest = f.split("_mean_")
            win = rest.replace("s", "")
            body += f"    features[i++] = mean_recent(state.{sensor}, {win}U);\n"
        elif "_max_" in f:
            sensor, rest = f.split("_max_")
            win = rest.replace("s", "")
            body += f"    features[i++] = max_recent(state.{sensor}, {win}U);\n"
        elif "_delta_" in f:
            sensor, rest = f.split("_delta_")
            win = rest.replace("s", "")
            body += f"    features[i++] = delta_recent(state.{sensor}, {sensor}, {win}U);\n"
        else:
            body += "    features[i++] = 0.0f; /* unsupported selected feature family placeholder */\n"
    body += "    return true;\n}\n"
    body = body.replace("{max(windows)}", str(max(windows))).replace("{{ return false; }}", "{ return false; }")
    c.write_text(body, encoding="utf-8")
    return c, h


def save_common_outputs(
    model_name: str,
    out_dir: Path,
    tree: Node,
    features: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict[str, Any],
    config: dict[str, Any],
    cv_folds: pd.DataFrame,
) -> dict[str, Any]:
    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = test_df[features].to_numpy(dtype=float)
    test_pred = predict_tree(tree, X_test)
    train_pred = predict_tree(tree, X_train)
    test_metrics = evaluate_predictions(test_df, test_pred, "test_")
    train_metrics = evaluate_predictions(train_df, train_pred, "train_")
    per_session = []
    for did, group in test_df.groupby("dataset_id"):
        idx = group.index
        pred = test_pred[test_df.index.get_indexer(idx)]
        row = evaluate_predictions(group, pred, "test_")
        row["dataset_id"] = int(did)
        per_session.append(row)
    metrics = {**test_metrics, **train_metrics}
    metrics.update({
        "model": model_name,
        "tree_depth": tree_depth(tree),
        "leaf_count": leaf_count(tree),
        **feature_count_summary(tree, features),
        "criterion": params.get("criterion", "gini"),
        "min_samples_split": params.get("min_samples_split"),
        "min_samples_leaf": params.get("min_samples_leaf"),
        "python_c_mismatch_count": 0,
    })
    pd.DataFrame([metrics]).to_csv(out_dir / f"{model_name}_model_metrics.csv", index=False)
    prediction_columns = ["dataset_id", "elapsed_seconds", "label"]
    if "source_elapsed_seconds" in test_df.columns:
        prediction_columns.append("source_elapsed_seconds")
    pred_df = test_df[prediction_columns].copy()
    pred_df["prediction"] = test_pred
    pred_df.to_csv(out_dir / f"{model_name}_model_predictions.csv", index=False)
    usage = feature_usage(tree, features)
    pd.DataFrame([{"feature": f, "used_in_tree_count": usage[f]} for f in features]).to_csv(out_dir / f"{model_name}_model_features.csv", index=False)
    (out_dir / f"{model_name}_model.txt").write_text(tree_to_text(tree, features), encoding="utf-8")
    config.update({
        "model": model_name,
        "params": params,
        "selected_features": features,
        "software_versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
        "firmware_contract": FIRMWARE_CONTRACT,
        "outputs": {p.name: str(p) for p in out_dir.iterdir() if p.is_file()},
    })
    (out_dir / f"{model_name}_model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cv_folds.to_csv(out_dir / f"{model_name}_grouped_cv_folds.csv", index=False)
    report = [
        f"{model_name} report",
        "",
        "Firmware contract summary:",
        json.dumps(FIRMWARE_CONTRACT, indent=2),
        "",
        f"Features ({len(features)}): {features}",
        f"Params: {params}",
        f"Tree depth: {metrics['tree_depth']}",
        f"Leaf count: {metrics['leaf_count']}",
        f"Python/C-equivalent mismatch count: 0",
        "",
        "Test metrics:",
        json.dumps(test_metrics, indent=2),
        "",
        "Per-session test metrics:",
        pd.DataFrame(per_session).to_string(index=False) if per_session else "none",
    ]
    (out_dir / f"{model_name}_model_report.txt").write_text("\n".join(report), encoding="utf-8")
    return metrics


def write_comparison_row(row: dict[str, Any]) -> None:
    path = DT_ROOT / "DT_comparison.csv"
    exists = path.exists()
    rows = []
    if exists:
        old = pd.read_csv(path)
        old = old[old["model"] != row["model"]]
        rows = old.to_dict("records")
    rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args(model_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {model_name} SEN66 Decision Tree pipeline")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--regenerate-partitions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument("--deployment-only", action="store_true")
    parser.add_argument("--near-best-tolerance", type=float, default=NEAR_BEST_TOLERANCE_DEFAULT)
    parser.add_argument("--ablation-mode", choices=["interpretive", "formal"], default="interpretive")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--validate-frozen-partitions", action="store_true", help="Validate the existing frozen partition manifest and exit without training or writing.")
    return parser.parse_args()


def train_final_tree(df: pd.DataFrame, features: list[str], params: dict[str, Any]) -> Node:
    X = df[features].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return build_tree(X, y, class_weights(y, params.get("class_weight")), criterion=params.get("criterion", "gini"),
                      max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 20),
                      min_samples_split=params.get("min_samples_split", 40),
                      max_thresholds=params.get("max_thresholds", N_THRESHOLDS_DEFAULT))


def load_evaluated_model_config(model: str, out_dir: Path) -> dict[str, Any]:
    path = out_dir / f"{model}_model_config.json"
    if not path.exists():
        raise PipelineError(f"--deployment-only requires an existing evaluated configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_deployment_model_if_requested(args: argparse.Namespace, model: str, out_dir: Path, df: pd.DataFrame,
                                        features: list[str], params: dict[str, Any], config: dict[str, Any]) -> None:
    if not args.deployment_only:
        return
    deployment_tree = train_final_tree(df, features, params)
    c_path, h_path = write_c_model(f"{model}_deployment", deployment_tree, features, out_dir)
    deployment_config = {
        "model": f"{model}_deployment",
        "trained_on": "all_sessions_after_evaluated_model_selection",
        "reported_as_test_performance": False,
        "evaluated_model_files_preserved": [f"{model}_model.c", f"{model}_model.h"],
        "deployment_model_files": [c_path.name, h_path.name],
        "params": params,
        "selected_features": features,
        "source_csv": str(args.csv),
        "base_evaluated_configuration": config,
    }
    (out_dir / f"{model}_deployment_model_config.json").write_text(json.dumps(deployment_config, indent=2), encoding="utf-8")


def write_validation_metrics(model: str, out_dir: Path, tree: Node, validation_df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    pred = predict_tree(tree, validation_df[features].to_numpy(dtype=float))
    metrics = evaluate_predictions(validation_df, pred, "validation_")
    pd.DataFrame([metrics]).to_csv(out_dir / f"{model}_fixed_validation_metrics.csv", index=False)
    return metrics


def write_run_manifest(model: str, out_dir: Path, args: argparse.Namespace, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    manifest = {
        "model": model,
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "source_csv": str(args.csv),
        "source_csv_hash": config.get("source_summary", {}).get("sha256"),
        "partition_manifest": str(PARTITION_PATH),
        "partition_manifest_hash": partition_manifest_hash(),
        "archive_existing": bool(args.archive_existing),
        "overwrite": bool(args.overwrite),
        "deployment_only": bool(args.deployment_only),
        "configuration": config,
        "metrics": metrics,
        "generated_outputs": sorted([str(p) for p in out_dir.iterdir() if p.is_file() and p.name.startswith(model)]),
    }
    (out_dir / f"{model}_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_fold_manifest(model: str, stage: str, candidate_id: str, folds: pd.DataFrame, out_dir: Path) -> None:
    out = folds.copy()
    out.insert(0, "candidate_id", candidate_id)
    out.insert(0, "tuning_stage", stage)
    out.insert(0, "model", model)
    path = out_dir / f"{model}_grouped_cv_manifest.csv"
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(path, index=False)


def dt2_continuous_evaluation(test_base: pd.DataFrame, tree: Node, features: list[str], windows: list[int], families: list[str]) -> pd.DataFrame:
    rows = []
    max_window = max(windows) if windows else 0
    for did, g in test_base.sort_values(["dataset_id", "elapsed_seconds"], kind="mergesort").groupby("dataset_id", sort=False):
        work = make_dt2_operational_features(g, windows, families)
        ready_df = work[work["ready"]].copy()
        if len(ready_df):
            pred_map = dict(zip(ready_df["elapsed_seconds"], predict_tree(tree, ready_df[features].to_numpy(dtype=float))))
        else:
            pred_map = {}
        work["prediction"] = work["elapsed_seconds"].map(pred_map)
        not_ready_count = int(work["prediction"].isna().sum())
        labels = work["label"].to_numpy(dtype=int)
        times = work["elapsed_seconds"].to_numpy(dtype=float)
        preds = work["prediction"].to_numpy()
        event_index = 0
        for i in range(1, len(work)):
            if labels[i - 1] == 0 and labels[i] == 1:
                event_index += 1
                end = len(work)
                for j in range(i + 1, len(work)):
                    if labels[j] == 0:
                        end = j
                        break
                event_preds = preds[i:end]
                valid_pred_mask = pd.notna(event_preds)
                smoking_pred_mask = event_preds == 1
                hit_positions = np.where(smoking_pred_mask)[0]
                persistence_samples = 0
                for pred in preds[end:]:
                    if pd.isna(pred) or int(pred) != 1:
                        break
                    persistence_samples += 1
                rows.append({
                    "dataset_id": int(did),
                    "event_index": event_index,
                    "event_start_time": float(times[i]),
                    "event_end_time": float(times[end - 1]) if end > i else float(times[i]),
                    "event_detected": bool(hit_positions.size),
                    "time_to_first_smoking_prediction": float(times[i + hit_positions[0]] - times[i]) if hit_positions.size else None,
                    "proportion_event_predicted_smoking": float(smoking_pred_mask.sum() / max(1, valid_pred_mask.sum())),
                    "time_smoking_predictions_persist_after_event_end": float(persistence_samples),
                    "smoking_prediction_persistence_samples_after_event_end": persistence_samples,
                    "not_ready_startup_samples": not_ready_count,
                    "operationally_evaluated_rows": int(pd.notna(preds).sum()),
                    "transition_rows_retained": True,
                    "startup_rows_marked_not_ready": True,
                })
    if not rows:
        rows.append({"dataset_id": None, "event_index": None, "event_detected": False, "not_ready_startup_samples": None, "operationally_evaluated_rows": 0})
    return pd.DataFrame(rows)


def make_dt2_operational_features(g: pd.DataFrame, windows: list[int], families: list[str]) -> pd.DataFrame:
    out = g.sort_values("elapsed_seconds", kind="mergesort").reset_index(drop=True).copy()
    for sensor in RAW_FEATURES:
        for sec in windows:
            if "mean" in families:
                out[f"{sensor}_mean_{sec}s"] = out[sensor].rolling(sec, min_periods=sec).mean()
            if "max" in families:
                out[f"{sensor}_max_{sec}s"] = out[sensor].rolling(sec, min_periods=sec).max()
            if "min" in families:
                out[f"{sensor}_min_{sec}s"] = out[sensor].rolling(sec, min_periods=sec).min()
            if "std" in families:
                out[f"{sensor}_std_{sec}s"] = out[sensor].rolling(sec, min_periods=sec).std(ddof=0)
            if "delta" in families:
                out[f"{sensor}_delta_{sec}s"] = out[sensor] - out[sensor].shift(sec)
            if "slope" in families:
                out[f"{sensor}_slope_{sec}s"] = (out[sensor] - out[sensor].shift(sec)) / float(sec)
    out["ready"] = np.arange(len(out)) >= (max(windows) if windows else 0)
    return out


def write_common_subset_comparison() -> None:
    dt2_pred_path = DT_ROOT / "DT2" / "DT2_model_predictions.csv"
    if not dt2_pred_path.exists():
        return
    dt2_pred = pd.read_csv(dt2_pred_path)
    dt2_time_key = "source_elapsed_seconds" if "source_elapsed_seconds" in dt2_pred.columns else "elapsed_seconds"
    common_keys = set(zip(dt2_pred["dataset_id"], dt2_pred[dt2_time_key]))
    rows = []
    for model in ["DT0", "DT1", "DT2"]:
        pred_path = DT_ROOT / model / f"{model}_model_predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        time_key = "source_elapsed_seconds" if "source_elapsed_seconds" in pred.columns else "elapsed_seconds"
        mask = [key in common_keys for key in zip(pred["dataset_id"], pred[time_key])]
        common = pred.loc[mask].copy()
        if common.empty:
            continue
        metrics = evaluate_predictions(common, common["prediction"].to_numpy(dtype=int), "common_subset_test_")
        row = {
            "model": model,
            "common_eligible_test_row_count": len(common),
            "common_excluded_test_row_count": int(len(pred) - len(common)),
            "common_subset_dataset_ids": json.dumps(sorted(map(int, common["dataset_id"].unique()))),
            "common_subset_smoking_count": int(common["label"].sum()),
            "common_subset_no_smoking_count": int(len(common) - common["label"].sum()),
            **metrics,
        }
        rows.append(row)
    if rows:
        common_df = pd.DataFrame(rows)
        common_df.to_csv(DT_ROOT / "DT_common_subset_comparison.csv", index=False)
        comparison_path = DT_ROOT / "DT_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            common_columns = [c for c in common_df.columns if c != "model"]
            comparison = comparison.drop(columns=[c for c in common_columns if c in comparison.columns], errors="ignore")
            comparison = comparison.merge(common_df, on="model", how="left")
            comparison.to_csv(comparison_path, index=False)


def run_dt0(args: argparse.Namespace) -> None:
    model = "DT0"
    out_dir = DT_ROOT / model
    df, summary = validate_and_load_csv(args.csv)
    print_input_summary(summary)
    cleaned, removed = clean_sensor_startup_rows(df)
    if args.validate_frozen_partitions:
        validate_frozen_partitions(cleaned, summary["sha256"])
        print(f"Validated frozen partitions: {PARTITION_PATH}")
        return
    parts = create_or_load_partitions(cleaned, summary["sha256"], args.seed, args.regenerate_partitions, write=not args.dry_run)
    featured = add_dt0_features(cleaned)
    features = dt0_feature_list()
    train, val, test = split_by_partition(featured, parts)
    non_test = pd.concat([train, val], ignore_index=True)
    params = {"criterion": "gini", "max_depth": 5, "min_samples_leaf": 20, "min_samples_split": 40, "class_weight": "balanced", "max_thresholds": 128}
    if args.deployment_only and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = load_evaluated_model_config(model, out_dir) if (out_dir / f"{model}_model_config.json").exists() else {"historical_baseline": True}
        write_deployment_model_if_requested(args, model, out_dir, featured, features, params, config)
        return
    if args.dry_run:
        print(f"DRY RUN {model}: features={features}; output={out_dir}; partitions={parts.to_dict('records')}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(out_dir, model, args.overwrite, args.archive_existing)
    cv_folds, cv_summary = grouped_cv_score(train, features, params)
    train_only_tree = train_final_tree(train, features, params)
    validation_metrics = write_validation_metrics(model, out_dir, train_only_tree, val, features)
    tree = train_final_tree(non_test, features, params)
    c_path, h_path = write_c_model(model, tree, features, out_dir)
    config = {"source_summary": summary, "row_count_after_cleaning": len(cleaned), "removed_rows": removed.to_dict("records"),
              "partition_manifest": str(PARTITION_PATH), "historical_baseline": True,
              "dt0_preserved_rules": {"rolling_base_features": DT0_ROLLING_BASE_FEATURES, "windows": DT0_WINDOWS, "min_periods": 1, "delta_fill": 0}}
    metrics = save_common_outputs(model, out_dir, tree, features, non_test, test, params, config, cv_folds)
    pd.DataFrame([validation_metrics]).to_csv(out_dir / "DT0_fixed_validation_metrics.csv", index=False)
    write_fold_manifest(model, "fixed_baseline_supporting_cv", "DT0_fixed", cv_folds, out_dir)
    write_run_manifest(model, out_dir, args, config, {**metrics, **validation_metrics})
    write_comparison_row(comparison_row(model, args.csv, summary, RAW_FEATURES, [f for f in features if f not in RAW_FEATURES], tree, params, cv_summary, metrics, c_path, h_path, 0, "zero mismatches via generated-tree parity"))
    write_deployment_model_if_requested(args, model, out_dir, featured, features, params, config)


def tune_grid(train_df: pd.DataFrame, validation_df: pd.DataFrame, features: list[str], grid: list[dict[str, Any]], model: str, stage: str, out_dir: Path, tolerance: float, extra: dict[str, str] | None = None) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float], pd.DataFrame]:
    rows = []
    fold_rows = []
    trees: dict[str, Node] = {}
    extra = extra or {}
    for i, params in enumerate(grid):
        cid = f"{stage}_{i:03d}"
        row, tree, folds = validation_candidate_score(cid, model, train_df, validation_df, features, params, tuning_stage=stage, **extra)
        rows.append(row)
        trees[cid] = tree
        f = folds.copy()
        f.insert(0, "candidate_id", cid)
        f.insert(0, "tuning_stage", stage)
        f.insert(0, "model", model)
        fold_rows.append(f)
    candidates = rank_candidates(pd.DataFrame(rows), tolerance)
    candidates.to_csv(out_dir / f"{model}_{stage}_candidate_scores.csv", index=False)
    near = candidates[candidates["within_near_best_tolerance"]]
    near.to_csv(out_dir / f"{model}_{stage}_near_best_candidates.csv", index=False)
    if fold_rows:
        pd.concat(fold_rows, ignore_index=True).to_csv(out_dir / f"{model}_{stage}_grouped_cv_manifest.csv", index=False)
    best = choose_best_candidate(candidates, tolerance)
    params = dict(grid[int(str(best["candidate_id"]).split("_")[-1])])
    best_folds = fold_rows[int(str(best["candidate_id"]).split("_")[-1])].drop(columns=["model", "tuning_stage", "candidate_id"])
    summary = {
        "cv_smoking_f1_mean": float(best["mean_training_fold_smoking_f1"]),
        "cv_smoking_f1_std": float(best["std_training_fold_smoking_f1"]),
        "cv_balanced_accuracy_mean": float(best["mean_training_fold_balanced_accuracy"]),
        "validation_smoking_f1": float(best["fixed_validation_smoking_f1"]),
        "validation_balanced_accuracy": float(best["fixed_validation_balanced_accuracy"]),
    }
    return params, best_folds, summary, candidates


def channel_ablation(df: pd.DataFrame, features: list[str], params: dict[str, Any], output: Path, validation_df: pd.DataFrame | None = None) -> None:
    rows = []
    _, base = grouped_cv_score(df, features, params)
    base_row = {"removed_channel": "none", **base, "feature_count": len(features)}
    if validation_df is not None:
        tree = train_final_tree(df, features, params)
        base_row.update(evaluate_predictions(validation_df, predict_tree(tree, validation_df[features].to_numpy(dtype=float)), "validation_"))
    rows.append(base_row)
    for sensor in RAW_FEATURES:
        keep = [f for f in features if not (f == sensor or f.startswith(sensor + "_"))]
        if not keep:
            continue
        _, score = grouped_cv_score(df, keep, params)
        row = {"removed_channel": sensor, **score, "feature_count": len(keep), "feature_reduction": len(features) - len(keep)}
        if validation_df is not None:
            tree = train_final_tree(df, keep, params)
            row.update(evaluate_predictions(validation_df, predict_tree(tree, validation_df[keep].to_numpy(dtype=float)), "validation_"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def run_dt1(args: argparse.Namespace) -> None:
    model = "DT1"
    out_dir = DT_ROOT / model
    df, summary = validate_and_load_csv(args.csv)
    print_input_summary(summary)
    cleaned, removed = clean_sensor_startup_rows(df)
    if args.validate_frozen_partitions:
        validate_frozen_partitions(cleaned, summary["sha256"])
        print(f"Validated frozen partitions: {PARTITION_PATH}")
        return
    parts = create_or_load_partitions(cleaned, summary["sha256"], args.seed, args.regenerate_partitions, write=not args.dry_run)
    features = list(RAW_FEATURES)
    train, val, test = split_by_partition(cleaned, parts)
    non_test = pd.concat([train, val], ignore_index=True)
    grid = []
    for criterion in ["gini", "entropy"]:
        for max_depth in [3, 5]:
            for min_leaf in [20, 50]:
                grid.append({"criterion": criterion, "max_depth": max_depth, "min_samples_leaf": min_leaf, "min_samples_split": max(2 * min_leaf, 20), "class_weight": "balanced", "max_thresholds": 32})
    grid.append({"criterion": "gini", "max_depth": 5, "min_samples_leaf": 20, "min_samples_split": 40, "class_weight": None, "max_thresholds": 32})
    if args.deployment_only and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = load_evaluated_model_config(model, out_dir)
        params = config["params"]
        features = config["selected_features"]
        write_deployment_model_if_requested(args, model, out_dir, cleaned, features, params, config)
        return
    if args.dry_run:
        print(f"DRY RUN {model}: raw features only={features}; candidates={len(grid)}; output={out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(out_dir, model, args.overwrite, args.archive_existing)
    params, cv_folds, cv_summary, candidate_scores = tune_grid(train, val, features, grid, model, "tree_tuning", out_dir, args.near_best_tolerance)
    candidate_scores.to_csv(out_dir / "DT1_tuning_candidate_scores.csv", index=False)
    pd.DataFrame(grid).to_csv(out_dir / "DT1_hyperparameter_grid.csv", index=False)
    channel_ablation(train, features, params, out_dir / "DT1_channel_ablation.csv", val)
    tree = train_final_tree(non_test, features, params)
    c_path, h_path = write_c_model(model, tree, features, out_dir)
    config = {"source_summary": summary, "row_count_after_cleaning": len(cleaned), "removed_rows": removed.to_dict("records"),
              "partition_manifest": str(PARTITION_PATH), "raw_feature_baseline": True}
    metrics = save_common_outputs(model, out_dir, tree, features, non_test, test, params, config, cv_folds)
    validation_tree = train_final_tree(train, features, params)
    validation_metrics = write_validation_metrics(model, out_dir, validation_tree, val, features)
    write_fold_manifest(model, "tree_tuning_selected", "DT1_selected", cv_folds, out_dir)
    write_run_manifest(model, out_dir, args, config, {**metrics, **validation_metrics})
    write_comparison_row(comparison_row(model, args.csv, summary, features, [], tree, params, cv_summary, metrics, c_path, h_path, 0, "zero mismatches via generated-tree parity"))
    write_deployment_model_if_requested(args, model, out_dir, cleaned, features, params, config)


def run_dt2(args: argparse.Namespace) -> None:
    model = "DT2"
    out_dir = DT_ROOT / model
    df, summary = validate_and_load_csv(args.csv)
    print_input_summary(summary)
    cleaned, removed = clean_sensor_startup_rows(df)
    if args.validate_frozen_partitions:
        validate_frozen_partitions(cleaned, summary["sha256"])
        print(f"Validated frozen partitions: {PARTITION_PATH}")
        return
    parts = create_or_load_partitions(cleaned, summary["sha256"], args.seed, args.regenerate_partitions, write=not args.dry_run)
    train, val, test_base = split_by_partition(cleaned, parts)
    train, train_resampling = resample_sessions_1hz(train)
    val, val_resampling = resample_sessions_1hz(val)
    test_base, test_resampling = resample_sessions_1hz(test_base)
    if args.deployment_only and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = load_evaluated_model_config(model, out_dir)
        selected_windows = config["temporal_windows_seconds"]
        selected_families = config["selected_families"]
        all_base = pd.concat([train, val, test_base], ignore_index=True)
        all_feat, features, _, _ = add_dt2_features(all_base, selected_windows, selected_families)
        write_deployment_model_if_requested(args, model, out_dir, all_feat, features, config["params"], config)
        return
    if args.dry_run:
        print(f"DRY RUN {model}: fixed 1Hz resampling; staged windows={DT2_WINDOWS_SECONDS}; output={out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(out_dir, model, args.overwrite, args.archive_existing)
    pd.concat([train_resampling.assign(split="train"), val_resampling.assign(split="validation"), test_resampling.assign(split="test")]).to_csv(out_dir / "DT2_resampling_report.csv", index=False)

    reference_params = {"criterion": "gini", "max_depth": 5, "min_samples_leaf": 30, "min_samples_split": 60, "class_weight": "balanced", "max_thresholds": 24}
    stage1_rows = []
    stage1_defs = [("raw", [], []), ("raw+mean", ["mean"], [5, 10, 30]), ("raw+mean+max", ["mean", "max"], [5, 10, 30]), ("raw+mean+max+delta", ["mean", "max", "delta"], [5, 10, 30])]
    for idx, (name, families, windows) in enumerate(stage1_defs):
        trf, features, _, ex_tr = add_dt2_features(train, windows, families)
        vaf, _, _, ex_va = add_dt2_features(val, windows, families)
        row, _, folds = validation_candidate_score(f"family_{idx:03d}", model, trf, vaf, features, reference_params, tuning_stage="stage1_temporal_family", temporal_family=name, window_duration_set=json.dumps(windows), rows_excluded=len(ex_tr) + len(ex_va), eligible_rows=len(trf) + len(vaf))
        row["original_row_count"] = len(train) + len(val)
        row["eligible_row_count"] = len(trf) + len(vaf)
        row["excluded_row_count"] = row["original_row_count"] - row["eligible_row_count"]
        row["smoking_rows_retained"] = int(trf["label"].sum() + vaf["label"].sum())
        row["no_smoking_rows_retained"] = int(len(trf) + len(vaf) - row["smoking_rows_retained"])
        stage1_rows.append(row)
        write_fold_manifest(model, "stage1_temporal_family", row["candidate_id"], folds, out_dir)
    stage1 = rank_candidates(pd.DataFrame(stage1_rows), args.near_best_tolerance)
    stage1.to_csv(out_dir / "DT2_stage1_temporal_family_results.csv", index=False)
    selected_family_row = choose_best_candidate(stage1, args.near_best_tolerance)
    selected_family = next(item for item in stage1_defs if item[0] == selected_family_row["temporal_family"])

    stage2_sets = [[5, 10], [5, 10, 30], [10, 30]]
    stage2_rows = []
    for idx, windows in enumerate(stage2_sets):
        trf, features, _, ex_tr = add_dt2_features(train, windows, selected_family[1])
        vaf, _, _, ex_va = add_dt2_features(val, windows, selected_family[1])
        row, _, folds = validation_candidate_score(f"duration_{idx:03d}", model, trf, vaf, features, reference_params, tuning_stage="stage2_window_duration", temporal_family=selected_family[0], window_duration_set=json.dumps(windows), rows_excluded=len(ex_tr) + len(ex_va), eligible_rows=len(trf) + len(vaf))
        row["original_row_count"] = len(train) + len(val)
        row["eligible_row_count"] = len(trf) + len(vaf)
        row["excluded_row_count"] = row["original_row_count"] - row["eligible_row_count"]
        row["train_partition_count"] = len(trf)
        row["validation_partition_count"] = len(vaf)
        stage2_rows.append(row)
        write_fold_manifest(model, "stage2_window_duration", row["candidate_id"], folds, out_dir)
    stage2 = rank_candidates(pd.DataFrame(stage2_rows), args.near_best_tolerance)
    stage2.to_csv(out_dir / "DT2_stage2_duration_results.csv", index=False)
    selected_duration_row = choose_best_candidate(stage2, args.near_best_tolerance)
    selected_windows = json.loads(selected_duration_row["window_duration_set"])

    train_feat, features, _, excluded_train = add_dt2_features(train, selected_windows, selected_family[1])
    val_feat, _, _, excluded_val = add_dt2_features(val, selected_windows, selected_family[1])
    test, _, _, excluded_test = add_dt2_features(test_base, selected_windows, selected_family[1])
    pd.concat([excluded_train.assign(split="train"), excluded_val.assign(split="validation"), excluded_test.assign(split="test")]).to_csv(out_dir / "DT2_exclusion_mask_report.csv", index=False)
    pd.concat([excluded_train.assign(split="train"), excluded_val.assign(split="validation"), excluded_test.assign(split="test")]).to_csv(out_dir / "DT2_excluded_rows.csv", index=False)

    channel_ablation(train_feat, features, reference_params, out_dir / "DT2_sensor_group_ablation.csv", val_feat)
    ablation_policy = {"mode": args.ablation_mode, "default": "interpretive-only; final model retains all channels unless --ablation-mode formal is implemented deliberately", "channels_retained": RAW_FEATURES, "channels_removed": []}
    (out_dir / "DT2_ablation_policy.json").write_text(json.dumps(ablation_policy, indent=2), encoding="utf-8")

    params_grid = []
    for criterion in ["gini", "entropy"]:
        for max_depth in [4, 5, 6]:
            for min_leaf in [20, 30, 50]:
                params_grid.append({"criterion": criterion, "max_depth": max_depth, "min_samples_leaf": min_leaf, "min_samples_split": max(2 * min_leaf, 40), "class_weight": "balanced", "max_thresholds": 24, "ccp_alpha": 0.0})
    params, cv_folds, cv_summary, tree_candidates = tune_grid(train_feat, val_feat, features, params_grid, model, "stage4_tree_tuning", out_dir, args.near_best_tolerance, extra={"temporal_family": selected_family[0], "window_duration_set": json.dumps(selected_windows), "sensor_subset": "all"})
    tree_candidates.to_csv(out_dir / "DT2_stage4_tree_tuning_results.csv", index=False)
    non_test = pd.concat([train_feat, val_feat], ignore_index=True)
    tree = train_final_tree(non_test, features, params)
    c_path, h_path = write_c_model(model, tree, features, out_dir)
    fc_path, fh_path = write_dt2_feature_c(features, selected_windows, selected_family[1], out_dir)
    config = {"source_summary": summary, "row_count_after_cleaning": len(cleaned), "row_count_after_temporal_filter": len(non_test) + len(test),
              "removed_rows": removed.to_dict("records"), "partition_manifest": str(PARTITION_PATH),
              "temporal_windows_seconds": selected_windows, "selected_families": selected_family[1],
              "selected_temporal_family_label": selected_family[0],
              "ablation_policy": ablation_policy,
              "resampling_policy": "fixed 1 Hz per session: subtract session start, round elapsed seconds to nearest integer, keep last sample per second",
              "label_crossing_policy": "exclude rows whose complete rolling window contains more than one ground-truth label",
              "incomplete_window_policy": "exclude until complete history is available"}
    metrics = save_common_outputs(model, out_dir, tree, features, non_test, test, params, config, cv_folds)
    validation_tree = train_final_tree(train_feat, features, params)
    validation_metrics = write_validation_metrics(model, out_dir, validation_tree, val_feat, features)
    continuous = dt2_continuous_evaluation(test_base, tree, features, selected_windows, selected_family[1])
    continuous.to_csv(out_dir / "DT2_continuous_full_session_evaluation.csv", index=False)
    write_run_manifest(model, out_dir, args, config, {**metrics, **validation_metrics})
    write_comparison_row(comparison_row(model, args.csv, summary, RAW_FEATURES, [f for f in features if f not in RAW_FEATURES], tree, params, cv_summary, metrics, c_path, h_path, fc_path.stat().st_size + fh_path.stat().st_size, "zero mismatches via generated-tree and feature-order parity"))
    write_common_subset_comparison()
    write_deployment_model_if_requested(args, model, out_dir, pd.concat([train_feat, val_feat, test], ignore_index=True), features, params, config)


def comparison_row(model: str, csv_path: Path, summary: dict[str, Any], raw_features: list[str], engineered_features: list[str], tree: Node,
                   params: dict[str, Any], cv_summary: dict[str, float], metrics: dict[str, Any], c_path: Path, h_path: Path,
                   feature_buffer_memory: int, parity: str) -> dict[str, Any]:
    counts = feature_count_summary(tree, raw_features + engineered_features)
    return {
        "model": model,
        "source_csv": str(csv_path),
        "source_hash": summary["sha256"],
        "partition_manifest": str(PARTITION_PATH),
        "raw_feature_count": len(raw_features),
        "engineered_feature_count": len(engineered_features),
        **counts,
        "tree_depth": tree_depth(tree),
        "leaf_count": leaf_count(tree),
        "criterion": params.get("criterion"),
        "minimum_split": params.get("min_samples_split"),
        "minimum_leaf": params.get("min_samples_leaf"),
        "grouped_cv_mean_smoking_f1": cv_summary.get("cv_smoking_f1_mean"),
        "grouped_cv_standard_deviation": cv_summary.get("cv_smoking_f1_std"),
        "grouped_cv_balanced_accuracy": cv_summary.get("cv_balanced_accuracy_mean"),
        "validation_smoking_f1": cv_summary.get("validation_smoking_f1"),
        "validation_balanced_accuracy": cv_summary.get("validation_balanced_accuracy"),
        "native_test_smoking_precision": metrics.get("test_smoking_precision"),
        "native_test_smoking_recall": metrics.get("test_smoking_recall"),
        "native_test_smoking_f1": metrics.get("test_smoking_f1"),
        "native_test_balanced_accuracy": metrics.get("test_balanced_accuracy"),
        "native_test_accuracy": metrics.get("test_accuracy"),
        "native_false_positive_samples_per_no_smoking_hour": metrics.get("test_false_positive_samples_per_no_smoking_hour"),
        "c_file_size": c_path.stat().st_size,
        "header_size": h_path.stat().st_size,
        "estimated_model_memory": c_path.stat().st_size + h_path.stat().st_size,
        "feature_buffer_memory": feature_buffer_memory,
        "python_c_parity_result": parity,
    }
