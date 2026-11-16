from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVALUATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EVALUATION_ROOT.parents[1]
DECISIONTREE_ROOT = REPOSITORY_ROOT / "academic_study" / "DecisionTree"
DT_MODELS_ROOT = DECISIONTREE_ROOT
SOURCE_CSV = REPOSITORY_ROOT / "logs" / "csv_merge" / "0408_merge.csv"
MERGE_MANIFEST = REPOSITORY_ROOT / "logs" / "csv_merge" / "0408_merge_manifest.csv"
PARTITION_PATH = DECISIONTREE_ROOT / "session_partitions.csv"
DEEPCRAFT_STUDIO_ROOT = EVALUATION_ROOT / "DEEPCRAFT_Studio"

FEATURE_COLUMNS = [
    "pm1_0",
    "pm2_5",
    "pm4_0",
    "pm10_0",
    "voc_index",
    "nox_index",
    "co2_ppm",
]

TEST_PARTITION = "test"
ALIGNMENT_TOLERANCE_SECONDS = 0.55


@dataclass(frozen=True)
class DeepcraftModelSpec:
    model_id: str
    display_name: str
    project_root: Path
    model_folder: Path
    h5_path: Path
    prediction_run_name: str
    data_file_stem: str
    improj_path: Path
    header_path: Path
    source_path: Path


MODEL_SPECS = [
    DeepcraftModelSpec(
        model_id="DeepcraftInstant",
        display_name="Deepcraft Instant",
        project_root=DEEPCRAFT_STUDIO_ROOT / "InstantSmokingPredict",
        model_folder=DEEPCRAFT_STUDIO_ROOT / "InstantSmokingPredict" / "instant_model",
        h5_path=DEEPCRAFT_STUDIO_ROOT / "InstantSmokingPredict" / "conv1d-medium-balanced-0" / "conv1d-medium-balanced-0.h5",
        prediction_run_name="conv1d-medium-balanced-0",
        data_file_stem="conv1d-medium-balanced-00",
        improj_path=DEEPCRAFT_STUDIO_ROOT / "InstantSmokingPredict" / "InstantSmokingPredict.improj",
        header_path=DEEPCRAFT_STUDIO_ROOT / "InstantSmokingPredict" / "instant_model" / "instant_model.h",
        source_path=DEEPCRAFT_STUDIO_ROOT / "InstantSmokingPredict" / "instant_model" / "instant_model.c",
    ),
    DeepcraftModelSpec(
        model_id="DeepcraftTime",
        display_name="Deepcraft Time",
        project_root=DEEPCRAFT_STUDIO_ROOT / "TimeSmokingPredict",
        model_folder=DEEPCRAFT_STUDIO_ROOT / "TimeSmokingPredict" / "time_model",
        h5_path=DEEPCRAFT_STUDIO_ROOT / "TimeSmokingPredict" / "conv1d-medium-balanced-3" / "conv1d-medium-balanced-3.h5",
        prediction_run_name="conv1d-medium-balanced-3",
        data_file_stem="conv1d-medium-balanced-30",
        improj_path=DEEPCRAFT_STUDIO_ROOT / "TimeSmokingPredict" / "TimeSmokingPredict.improj",
        header_path=DEEPCRAFT_STUDIO_ROOT / "TimeSmokingPredict" / "time_model" / "time_model.h",
        source_path=DEEPCRAFT_STUDIO_ROOT / "TimeSmokingPredict" / "time_model" / "time_model.c",
    ),
]

RETAINED_NATIVE_COLUMNS = [
    "model",
    "display_name",
    "source_csv",
    "source_hash",
    "partition_manifest",
    "partition_hash",
    "evaluation_mode",
    "deepcraft_project",
    "deepcraft_model_folder",
    "h5_path",
    "h5_sha256",
    "header_sha256",
    "source_sha256",
    "input_feature_count",
    "input_features",
    "deepcraft_window_samples",
    "deepcraft_output_classes",
    "binary_mapping",
    "native_test_row_count",
    "native_test_source_row_count",
    "native_test_context_only_rows",
    "native_test_support",
    "native_test_accuracy",
    "native_test_balanced_accuracy",
    "native_test_no_smoking_precision",
    "native_test_no_smoking_recall",
    "native_test_no_smoking_f1",
    "native_test_no_smoking_support",
    "native_test_smoking_precision",
    "native_test_smoking_recall",
    "native_test_smoking_f1",
    "native_test_smoking_support",
    "native_test_macro_f1",
    "native_test_confusion_tn",
    "native_test_confusion_fp",
    "native_test_confusion_fn",
    "native_test_confusion_tp",
    "max_abs_alignment_error_seconds",
    "unlabeled_argmax_count",
]

RETAINED_COMMON_COLUMNS = [
    "model",
    "common_subset_dataset_ids",
    "common_eligible_test_row_count",
    "common_subset_smoking_count",
    "common_subset_no_smoking_count",
    "common_subset_test_support",
    "common_subset_test_accuracy",
    "common_subset_test_balanced_accuracy",
    "common_subset_test_no_smoking_precision",
    "common_subset_test_no_smoking_recall",
    "common_subset_test_no_smoking_f1",
    "common_subset_test_no_smoking_support",
    "common_subset_test_smoking_precision",
    "common_subset_test_smoking_recall",
    "common_subset_test_smoking_f1",
    "common_subset_test_smoking_support",
    "common_subset_test_macro_f1",
    "common_subset_test_confusion_tn",
    "common_subset_test_confusion_fp",
    "common_subset_test_confusion_fn",
    "common_subset_test_confusion_tp",
]

REPORT_NATIVE_COLUMNS = [
    "model",
    "native_test_row_count",
    "native_test_accuracy",
    "native_test_balanced_accuracy",
    "native_test_smoking_precision",
    "native_test_smoking_recall",
    "native_test_smoking_f1",
    "native_test_confusion_tn",
    "native_test_confusion_fp",
    "native_test_confusion_fn",
    "native_test_confusion_tp",
]

REPORT_COMMON_COLUMNS = [
    "model",
    "common_eligible_test_row_count",
    "common_subset_test_accuracy",
    "common_subset_test_balanced_accuracy",
    "common_subset_test_smoking_precision",
    "common_subset_test_smoking_recall",
    "common_subset_test_smoking_f1",
    "common_subset_test_confusion_tn",
    "common_subset_test_confusion_fp",
    "common_subset_test_confusion_fn",
    "common_subset_test_confusion_tp",
]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_dt_helpers_to_path() -> None:
    sys.path.insert(0, str(DT_MODELS_ROOT))


def load_dt_metrics():
    add_dt_helpers_to_path()
    from dt_pipeline_common import evaluate_predictions

    return evaluate_predictions


def parse_list_shape(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item) for item in re.findall(r"\d+", value)]


def parse_deepcraft_project(spec: DeepcraftModelSpec) -> dict[str, Any]:
    tree = ET.parse(spec.improj_path)
    root = tree.getroot()
    preprocessor = root.find(".//Preprocessor")
    sliding_unit = root.find(".//Preprocessor/Unit[@id='Core.Queues.SlidingWindow']")
    sliding_args = {}
    if sliding_unit is not None:
        sliding_args = {arg.get("param"): (arg.text or "") for arg in sliding_unit.findall("./Argument")}
    symbol_map = [
        {"name": node.get("name"), "id": int(node.get("id", "-1"))}
        for node in root.findall(".//SymbolMap/Symbol")
    ]
    data_files = []
    for data_file in root.findall(".//Database/DataFile"):
        data_files.append(
            {
                "name": data_file.get("name"),
                "path": data_file.get("path"),
                "usage": data_file.get("usage"),
                "samplerate": data_file.get("samplerate"),
                "shape": parse_list_shape(data_file.get("shape")),
            }
        )
    return {
        "preprocessor_inshape": parse_list_shape(preprocessor.get("inshape") if preprocessor is not None else None),
        "preprocessor_outshape": parse_list_shape(preprocessor.get("outshape") if preprocessor is not None else None),
        "preprocessor_infreq": preprocessor.get("infreq") if preprocessor is not None else None,
        "sliding_window_shape": parse_list_shape(sliding_args.get("window_shape")),
        "sliding_window_stride": sliding_args.get("stride"),
        "symbol_map": symbol_map,
        "data_files": data_files,
    }


def load_partitions() -> pd.DataFrame:
    partitions = pd.read_csv(PARTITION_PATH)
    required = {"dataset_id", "partition"}
    missing = required.difference(partitions.columns)
    if missing:
        raise ValueError(f"Partition file missing columns: {sorted(missing)}")
    return partitions[["dataset_id", "partition"]].copy()


def load_source_rows() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_CSV)
    source["dataset_id"] = source["dataset_id"].astype(int)
    source["label"] = source["label"].astype(int)
    partitions = load_partitions()
    manifest = pd.read_csv(MERGE_MANIFEST)
    manifest["dataset_id"] = manifest["dataset_id"].astype(int)
    manifest["relative_path"] = manifest.get("relative_path", manifest.get("source_filename", ""))
    manifest["schema_type"] = manifest.get("schema_type", manifest.get("detected_schema", ""))
    manifest["row_count"] = manifest.get("row_count", manifest.get("accepted_output_row_count", np.nan))
    manifest["smoking_count"] = manifest.get("smoking_count", np.nan)
    manifest["no_smoking_count"] = manifest.get("no_smoking_count", np.nan)
    source = source.merge(partitions, on="dataset_id", how="left", validate="many_to_one")
    source = source.merge(
        manifest[["dataset_id", "relative_path", "schema_type", "row_count", "smoking_count", "no_smoking_count"]],
        on="dataset_id",
        how="left",
        validate="many_to_one",
    )
    if source["partition"].isna().any():
        missing = sorted(source.loc[source["partition"].isna(), "dataset_id"].unique())
        raise ValueError(f"Missing frozen partitions for dataset IDs: {missing}")
    source = source.sort_values(["dataset_id", "elapsed_seconds"]).copy()
    source["row_index_in_session"] = source.groupby("dataset_id").cumcount()
    source["relative_elapsed_seconds"] = source["elapsed_seconds"] - source.groupby("dataset_id")[
        "elapsed_seconds"
    ].transform("min")
    return source


def prediction_data_path(spec: DeepcraftModelSpec, dataset_id: int) -> Path:
    return (
        spec.project_root
        / spec.prediction_run_name
        / "predictions"
        / "sessions"
        / f"session_{dataset_id:03d}"
        / f"predictions_{spec.prediction_run_name}"
        / f"{spec.data_file_stem}.data"
    )


def load_prediction_data(spec: DeepcraftModelSpec, dataset_id: int) -> pd.DataFrame:
    path = prediction_data_path(spec, dataset_id)
    if not path.exists():
        raise FileNotFoundError(path)
    predictions = pd.read_csv(path)
    predictions.columns = [column.strip() for column in predictions.columns]
    predictions = predictions.rename(
        columns={
            "Time (seconds)": "deepcraft_window_start_seconds",
            "Duration (seconds)": "deepcraft_window_duration_seconds",
            "pred_No Smoking": "score_no_smoking",
            "pred_Smoking": "score_smoking",
            "pred_unlabeled": "score_unlabeled",
        }
    )
    predictions["dataset_id"] = dataset_id
    predictions["prediction_data_path"] = str(path)
    predictions["deepcraft_window_end_seconds"] = (
        predictions["deepcraft_window_start_seconds"] + predictions["deepcraft_window_duration_seconds"]
    )
    return predictions


def align_model_predictions(spec: DeepcraftModelSpec, source_rows: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    window_shape = metadata["sliding_window_shape"]
    if len(window_shape) != 2:
        raise ValueError(f"{spec.model_id} has unsupported sliding window shape: {window_shape}")
    window_samples = int(window_shape[0])
    all_aligned = []

    for dataset_id, session_rows in source_rows.groupby("dataset_id", sort=True):
        predictions = load_prediction_data(spec, int(dataset_id))
        source_session = session_rows.sort_values("relative_elapsed_seconds").copy()
        match_source = source_session[
            [
                "dataset_id",
                "row_index_in_session",
                "elapsed_seconds",
                "relative_elapsed_seconds",
                "label",
                "partition",
                "relative_path",
                "schema_type",
            ]
        ].rename(
            columns={
                "elapsed_seconds": "source_elapsed_seconds",
                "relative_elapsed_seconds": "matched_relative_elapsed_seconds",
            }
        )

        aligned = pd.merge_asof(
            predictions.sort_values("deepcraft_window_end_seconds"),
            match_source.sort_values("matched_relative_elapsed_seconds"),
            by="dataset_id",
            left_on="deepcraft_window_end_seconds",
            right_on="matched_relative_elapsed_seconds",
            direction="nearest",
            tolerance=ALIGNMENT_TOLERANCE_SECONDS,
        )
        if aligned["source_elapsed_seconds"].isna().any():
            bad = aligned.loc[aligned["source_elapsed_seconds"].isna()].head()
            raise ValueError(f"{spec.model_id} session {dataset_id} has unaligned predictions:\n{bad}")

        aligned["alignment_error_seconds"] = (
            aligned["deepcraft_window_end_seconds"] - aligned["matched_relative_elapsed_seconds"]
        )
        aligned["window_length_samples"] = window_samples
        aligned["expected_first_source_row_index"] = window_samples - 1
        aligned["deepcraft_context_only_source_rows"] = len(source_session) - len(aligned)
        aligned["window_start_source_row_index"] = aligned["row_index_in_session"].astype(int) - window_samples + 1
        aligned["window_crosses_session_boundary"] = aligned["window_start_source_row_index"] < 0
        aligned["prediction_available"] = True
        score_cols = ["score_unlabeled", "score_no_smoking", "score_smoking"]
        score_array = aligned[score_cols].to_numpy()
        class_ids = score_array.argmax(axis=1)
        symbols = np.array(["unlabeled", "No Smoking", "Smoking"], dtype=object)
        aligned["predicted_deepcraft_class_id"] = class_ids
        aligned["predicted_deepcraft_symbol"] = symbols[class_ids]
        aligned["predicted_label"] = (aligned["predicted_deepcraft_symbol"] == "Smoking").astype(int)
        aligned["prediction_confidence"] = score_array.max(axis=1)
        aligned["model"] = spec.model_id
        all_aligned.append(aligned)

    result = pd.concat(all_aligned, ignore_index=True)
    if result["window_crosses_session_boundary"].any():
        raise ValueError(f"{spec.model_id} produced windows crossing session boundaries")
    return result


def native_metrics_for_model(
    spec: DeepcraftModelSpec,
    aligned: pd.DataFrame,
    source_rows: pd.DataFrame,
    metadata: dict[str, Any],
    evaluate_predictions,
) -> dict[str, Any]:
    window_samples = metadata["sliding_window_shape"][0]
    output: dict[str, Any] = {
        "model": spec.model_id,
        "display_name": spec.display_name,
        "source_csv": str(SOURCE_CSV),
        "source_hash": sha256_file(SOURCE_CSV),
        "partition_manifest": str(PARTITION_PATH),
        "partition_hash": sha256_file(PARTITION_PATH),
        "evaluation_mode": "saved_deepcraft_predictions_no_retraining",
        "deepcraft_project": str(spec.project_root),
        "deepcraft_model_folder": str(spec.model_folder),
        "h5_path": str(spec.h5_path),
        "h5_sha256": sha256_file(spec.h5_path),
        "header_sha256": sha256_file(spec.header_path),
        "source_sha256": sha256_file(spec.source_path),
        "input_feature_count": len(FEATURE_COLUMNS),
        "input_features": ",".join(FEATURE_COLUMNS),
        "deepcraft_window_samples": int(window_samples),
        "deepcraft_output_classes": "unlabeled,No Smoking,Smoking",
        "binary_mapping": "Smoking=1; No Smoking/unlabeled=0",
    }
    for partition_name, group in aligned.groupby("partition", sort=True):
        metrics_rows = group.rename(columns={"source_elapsed_seconds": "elapsed_seconds"})
        metrics = evaluate_predictions(metrics_rows, metrics_rows["predicted_label"].to_numpy(), f"{partition_name}_")
        output.update(metrics)
    test_rows = aligned[aligned["partition"] == TEST_PARTITION].copy()
    test_metrics_rows = test_rows.rename(columns={"source_elapsed_seconds": "elapsed_seconds"})
    output.update(evaluate_predictions(test_metrics_rows, test_metrics_rows["predicted_label"].to_numpy(), "native_test_"))
    source_test_count = int((source_rows["partition"] == TEST_PARTITION).sum())
    output["native_test_row_count"] = len(test_rows)
    output["native_test_source_row_count"] = source_test_count
    output["native_test_context_only_rows"] = int(source_test_count - len(test_rows))
    output["max_abs_alignment_error_seconds"] = float(test_rows["alignment_error_seconds"].abs().max())
    output["unlabeled_argmax_count"] = int((aligned["predicted_deepcraft_symbol"] == "unlabeled").sum())
    return {key: output.get(key, np.nan) for key in RETAINED_NATIVE_COLUMNS}


def load_dt_prediction_keys() -> dict[str, pd.DataFrame]:
    frames = {}
    partitions = load_partitions()
    for model_id in ["DT0", "DT1", "DT2"]:
        path = DT_MODELS_ROOT / model_id / f"{model_id}_model_predictions.csv"
        frame = pd.read_csv(path)
        frame["dataset_id"] = frame["dataset_id"].astype(int)
        if "partition" not in frame.columns:
            frame = frame.merge(partitions, on="dataset_id", how="left", validate="many_to_one")
        if "source_elapsed_seconds" not in frame.columns:
            frame["source_elapsed_seconds"] = frame["elapsed_seconds"]
        frame["source_elapsed_seconds"] = frame["source_elapsed_seconds"].astype(float)
        frame = frame[frame["partition"] == TEST_PARTITION].copy()
        pred_col = "predicted_label" if "predicted_label" in frame.columns else "prediction"
        if pred_col not in frame.columns:
            candidates = [col for col in frame.columns if "pred" in col.lower()]
            raise ValueError(f"Could not find prediction column for {model_id}; candidates={candidates}")
        frames[model_id] = frame[
            ["dataset_id", "source_elapsed_seconds", "label", pred_col]
        ].rename(columns={pred_col: f"{model_id}_predicted_label"})
    return frames


def evaluate_common_subset(
    deepcraft_frames: dict[str, pd.DataFrame],
    evaluate_predictions,
    out_dir: Path,
) -> pd.DataFrame:
    dt_frames = load_dt_prediction_keys()
    wide: pd.DataFrame | None = None
    for model_id, frame in dt_frames.items():
        next_frame = frame.copy()
        next_frame["source_elapsed_seconds_round"] = next_frame["source_elapsed_seconds"].round(6)
        keep = ["dataset_id", "source_elapsed_seconds_round", "label", f"{model_id}_predicted_label"]
        wide = next_frame[keep] if wide is None else wide.merge(next_frame[keep], on=["dataset_id", "source_elapsed_seconds_round", "label"])

    for model_id, frame in deepcraft_frames.items():
        next_frame = frame[frame["partition"] == TEST_PARTITION][
            ["dataset_id", "source_elapsed_seconds", "label", "predicted_label", "score_no_smoking", "score_smoking"]
        ].copy()
        next_frame["source_elapsed_seconds_round"] = next_frame["source_elapsed_seconds"].round(6)
        next_frame = next_frame.rename(
            columns={
                "predicted_label": f"{model_id}_predicted_label",
                "score_no_smoking": f"{model_id}_score_no_smoking",
                "score_smoking": f"{model_id}_score_smoking",
            }
        )
        wide = wide.merge(
            next_frame.drop(columns=["source_elapsed_seconds"]),
            on=["dataset_id", "source_elapsed_seconds_round", "label"],
        )

    if wide is None or wide.empty:
        raise ValueError("No common subset rows found")
    wide = wide.rename(columns={"source_elapsed_seconds_round": "source_elapsed_seconds"})
    wide.to_csv(out_dir / "five_model_common_subset_rows.csv", index=False)

    rows = []
    for model_id in ["DT0", "DT1", "DT2", *deepcraft_frames.keys()]:
        metrics_frame = wide[["dataset_id", "source_elapsed_seconds", "label", f"{model_id}_predicted_label"]].rename(
            columns={"source_elapsed_seconds": "elapsed_seconds"}
        )
        metrics = evaluate_predictions(metrics_frame, wide[f"{model_id}_predicted_label"].to_numpy(), "common_subset_test_")
        metrics["model"] = model_id
        metrics["common_subset_dataset_ids"] = json.dumps(sorted(wide["dataset_id"].unique().tolist()))
        metrics["common_eligible_test_row_count"] = len(wide)
        metrics["common_subset_smoking_count"] = int((wide["label"] == 1).sum())
        metrics["common_subset_no_smoking_count"] = int((wide["label"] == 0).sum())
        rows.append(metrics)
    result = pd.DataFrame(rows)
    result = result[[column for column in RETAINED_COMMON_COLUMNS if column in result.columns]]
    result.to_csv(out_dir / "five_model_common_subset_comparison.csv", index=False)
    return result


def write_text_report(out_dir: Path, native: pd.DataFrame, common: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        clean = frame.copy()
        for col in clean.columns:
            clean[col] = clean[col].map(
                lambda value: ""
                if pd.isna(value)
                else f"{value:.6g}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        header = "| " + " | ".join(clean.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in clean.to_numpy(dtype=str)]
        return "\n".join([header, separator, *body])

    report_path = out_dir / "five_model_report.md"
    lines = [
        "# SEN66 Five-Model Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "No Decision Tree model was retrained or regenerated. DEEPCRAFT metrics use saved trained-model prediction artifacts aligned to the frozen DecisionTree session partitions.",
        "",
        "This reduced package focuses on broad classification comparison: accuracy, balanced accuracy, Smoking precision/recall/F1, and confusion-matrix counts. Detailed continuous-session deployment metrics were removed from the submission package.",
        "",
        "The DEEPCRAFT windows were verified from project metadata as 6x7 for Instant and 30x7 for Time. Saved prediction timestamps are aligned to the source row at the trailing window end, and the alignment audit confirms windows remain within their source sessions.",
        "",
        "## Native Test Comparison",
        "",
        markdown_table(native[[column for column in REPORT_NATIVE_COLUMNS if column in native.columns]]),
        "",
        "## Common Test Subset Comparison",
        "",
        markdown_table(common[[column for column in REPORT_COMMON_COLUMNS if column in common.columns]]),
        "",
        "## Limitations",
        "",
        "The package uses exported DEEPCRAFT prediction artifacts because TensorFlow/Keras/h5py were not available in the evaluation runtime for direct `.h5` inference. The source data were collected under uncontrolled environmental conditions, so these results should support broad model comparison rather than fine-grained deployment claims.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_simple_png_report(out_dir: Path, native: pd.DataFrame, common: pd.DataFrame) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    width, height = 1600, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    draw.text((40, 30), "SEN66 Five-Model Evaluation", fill=(0, 0, 0), font=title_font)
    draw.text((40, 65), "Saved DEEPCRAFT predictions aligned to frozen DecisionTree test partitions", fill=(70, 70, 70), font=font)

    metrics = [
        ("Native Smoking F1", native.set_index("model")["native_test_smoking_f1"]),
        ("Native Balanced Accuracy", native.set_index("model")["native_test_balanced_accuracy"]),
        ("Common Smoking F1", common.set_index("model")["common_subset_test_smoking_f1"]),
        ("Common Balanced Accuracy", common.set_index("model")["common_subset_test_balanced_accuracy"]),
    ]
    colors = {
        "DT0": (62, 119, 180),
        "DT1": (74, 159, 96),
        "DT2": (201, 128, 48),
        "DeepcraftInstant": (165, 83, 155),
        "DeepcraftTime": (100, 100, 100),
    }
    y = 130
    for label, series in metrics:
        draw.text((40, y), label, fill=(0, 0, 0), font=font)
        x0 = 310
        bar_h = 24
        for idx, (model, value) in enumerate(series.items()):
            yy = y + idx * 34
            val = 0.0 if pd.isna(value) else float(value)
            draw.rectangle((x0, yy, x0 + int(800 * val), yy + bar_h), fill=colors.get(model, (90, 90, 90)))
            draw.text((x0 + 820, yy + 4), f"{model}: {val:.3f}", fill=(0, 0, 0), font=font)
        y += 205
    image.save(out_dir / "five_model_metric_summary.png")


def build_dt_native_rows() -> pd.DataFrame:
    comparison = pd.read_csv(DT_MODELS_ROOT / "DT_comparison.csv")
    keep = [
        "model",
        "source_csv",
        "source_hash",
        "partition_manifest",
        "native_test_accuracy",
        "native_test_no_smoking_precision",
        "native_test_no_smoking_recall",
        "native_test_no_smoking_f1",
        "native_test_no_smoking_support",
        "native_test_smoking_precision",
        "native_test_smoking_recall",
        "native_test_smoking_f1",
        "native_test_smoking_support",
        "native_test_macro_f1",
        "native_test_balanced_accuracy",
        "native_test_confusion_tn",
        "native_test_confusion_fp",
        "native_test_confusion_fn",
        "native_test_confusion_tp",
    ]
    existing = [col for col in keep if col in comparison.columns]
    dt = comparison[comparison["model"].isin(["DT0", "DT1", "DT2"])][existing].copy()
    dt["evaluation_mode"] = "existing_decision_tree_output_preserved"
    row_counts = {}
    native_counts: dict[str, dict[str, int]] = {}
    partitions = load_partitions()
    for model_id in ["DT0", "DT1", "DT2"]:
        frame = pd.read_csv(DT_MODELS_ROOT / model_id / f"{model_id}_model_predictions.csv")
        frame["dataset_id"] = frame["dataset_id"].astype(int)
        frame = frame.merge(partitions, on="dataset_id", how="left", validate="many_to_one")
        test = frame[frame["partition"] == TEST_PARTITION].copy()
        row_counts[model_id] = int(len(test))
        pred_col = "predicted_label" if "predicted_label" in test.columns else "prediction"
        labels = test["label"].astype(int)
        preds = test[pred_col].astype(int)
        native_counts[model_id] = {
            "native_test_support": int(len(test)),
            "native_test_no_smoking_support": int((labels == 0).sum()),
            "native_test_smoking_support": int((labels == 1).sum()),
            "native_test_confusion_tn": int(((labels == 0) & (preds == 0)).sum()),
            "native_test_confusion_fp": int(((labels == 0) & (preds == 1)).sum()),
            "native_test_confusion_fn": int(((labels == 1) & (preds == 0)).sum()),
            "native_test_confusion_tp": int(((labels == 1) & (preds == 1)).sum()),
        }
    dt["native_test_row_count"] = dt["model"].map(row_counts)
    dt["native_test_source_row_count"] = dt["native_test_row_count"]
    dt["native_test_context_only_rows"] = 0
    for metric_name in [
        "native_test_support",
        "native_test_no_smoking_support",
        "native_test_smoking_support",
        "native_test_confusion_tn",
        "native_test_confusion_fp",
        "native_test_confusion_fn",
        "native_test_confusion_tp",
    ]:
        dt[metric_name] = dt["model"].map(lambda model_id: native_counts[model_id][metric_name])
    dt["display_name"] = dt["model"]
    dt["evaluation_mode"] = "existing_decision_tree_output_preserved"
    dt = dt[[column for column in RETAINED_NATIVE_COLUMNS if column in dt.columns]]
    return dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved DEEPCRAFT predictions against frozen DecisionTree partitions.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty disposable directory for evaluation outputs.")
    return parser.parse_args()


def prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == EVALUATION_ROOT.resolve() or EVALUATION_ROOT.resolve() in resolved.parents:
        raise ValueError("output directory must not be inside the packaged DEEPCRAFT_training results")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    evaluate_predictions = load_dt_metrics()
    partition_hash_before = sha256_file(PARTITION_PATH)
    source_rows = load_source_rows()

    deepcraft_native_rows = []
    deepcraft_frames: dict[str, pd.DataFrame] = {}
    audit_rows = []

    for spec in MODEL_SPECS:
        metadata = parse_deepcraft_project(spec)
        model_dir = output_root / f"{spec.model_id}_predictions"
        model_dir.mkdir(parents=True, exist_ok=True)
        aligned = align_model_predictions(spec, source_rows, metadata)
        aligned.to_csv(model_dir / f"{spec.model_id}_aligned_predictions.csv", index=False)
        native = native_metrics_for_model(spec, aligned, source_rows, metadata, evaluate_predictions)
        deepcraft_native_rows.append(native)
        deepcraft_frames[spec.model_id] = aligned

        for dataset_id, group in aligned.groupby("dataset_id", sort=True):
            selected = pd.concat([group.head(2), group.iloc[[len(group) // 2]], group.tail(2)]).drop_duplicates()
            for row in selected.itertuples(index=False):
                audit_rows.append(
                    {
                        "model": spec.model_id,
                        "dataset_id": dataset_id,
                        "source_elapsed_seconds": row.source_elapsed_seconds,
                        "matched_relative_elapsed_seconds": row.matched_relative_elapsed_seconds,
                        "deepcraft_window_start_seconds": row.deepcraft_window_start_seconds,
                        "deepcraft_window_duration_seconds": row.deepcraft_window_duration_seconds,
                        "deepcraft_window_end_seconds": row.deepcraft_window_end_seconds,
                        "alignment_error_seconds": row.alignment_error_seconds,
                        "row_index_in_session": row.row_index_in_session,
                        "window_start_source_row_index": row.window_start_source_row_index,
                        "window_length_samples": row.window_length_samples,
                        "label": row.label,
                        "predicted_label": row.predicted_label,
                        "predicted_deepcraft_symbol": row.predicted_deepcraft_symbol,
                    }
                )

        manifest = {
            "model": spec.model_id,
            "display_name": spec.display_name,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(SOURCE_CSV),
            "source_hash": sha256_file(SOURCE_CSV),
            "partition_manifest": str(PARTITION_PATH),
            "partition_hash": partition_hash_before,
            "deepcraft_project": str(spec.project_root),
            "deepcraft_model_folder": str(spec.model_folder),
            "h5_path": str(spec.h5_path),
            "h5_sha256": sha256_file(spec.h5_path),
            "header_path": str(spec.header_path),
            "header_sha256": sha256_file(spec.header_path),
            "source_path": str(spec.source_path),
            "source_sha256": sha256_file(spec.source_path),
            "metadata": metadata,
            "prediction_alignment": {
                "prediction_time_interpretation": "saved .data Time + Duration is aligned to the source row at the trailing window end",
                "alignment_tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
                "max_abs_alignment_error_seconds": float(aligned["alignment_error_seconds"].abs().max()),
                "window_crosses_session_boundary_count": int(aligned["window_crosses_session_boundary"].sum()),
            },
            "label_mapping": {
                "source_labels": {"0": "no_smoking", "1": "smoking"},
                "deepcraft_classes": {"0": "unlabeled", "1": "No Smoking", "2": "Smoking"},
                "binary_decision": "Smoking maps to 1; No Smoking and unlabeled map to 0",
            },
            "dependency_note": "TensorFlow/Keras/h5py were unavailable in this runtime; evaluation uses saved DEEPCRAFT prediction artifacts rather than loading .h5 weights.",
        }
        (model_dir / f"{spec.model_id}_evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    deepcraft_native = pd.DataFrame(deepcraft_native_rows)
    deepcraft_native.to_csv(output_root / "deepcraft_native_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_root / "window_alignment_audit.csv", index=False)

    common = evaluate_common_subset(deepcraft_frames, evaluate_predictions, output_root)
    dt_native = build_dt_native_rows()
    native = pd.concat([dt_native, deepcraft_native], ignore_index=True, sort=False)
    native = native[[column for column in RETAINED_NATIVE_COLUMNS if column in native.columns]]
    native.to_csv(output_root / "five_model_native_comparison.csv", index=False)

    summary = native[["model", "native_test_accuracy", "native_test_balanced_accuracy", "native_test_smoking_precision", "native_test_smoking_recall", "native_test_smoking_f1"]].merge(
        common[
            [
                "model",
                "common_eligible_test_row_count",
                "common_subset_test_accuracy",
                "common_subset_test_balanced_accuracy",
                "common_subset_test_smoking_precision",
                "common_subset_test_smoking_recall",
                "common_subset_test_smoking_f1",
            ]
        ],
        on="model",
        how="left",
    )
    summary.to_csv(output_root / "five_model_summary.csv", index=False)
    write_text_report(output_root / "reports", native, common)
    write_simple_png_report(output_root / "reports", native, common)

    partition_hash_after = sha256_file(PARTITION_PATH)
    run_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(SOURCE_CSV),
        "source_hash": sha256_file(SOURCE_CSV),
        "partition_manifest": str(PARTITION_PATH),
        "partition_hash_before": partition_hash_before,
        "partition_hash_after": partition_hash_after,
        "partition_unchanged": partition_hash_before == partition_hash_after,
        "dt_outputs_modified": False,
        "models_retrained": False,
        "retained_metric_scope": "accuracy, balanced accuracy, Smoking precision/recall/F1, No-Smoking precision/recall/F1, confusion counts, native/common row counts, alignment audit metadata",
        "removed_metric_scope": "dedicated continuous-session deployment behavior, post-event persistence, operational readiness, extended transition stability, false-positive episode duration reporting",
        "outputs": {
            "deepcraft_native_metrics": str(output_root / "deepcraft_native_metrics.csv"),
            "five_model_native_comparison": str(output_root / "five_model_native_comparison.csv"),
            "five_model_common_subset_comparison": str(output_root / "five_model_common_subset_comparison.csv"),
            "five_model_common_subset_rows": str(output_root / "five_model_common_subset_rows.csv"),
            "five_model_summary": str(output_root / "five_model_summary.csv"),
            "window_alignment_audit": str(output_root / "window_alignment_audit.csv"),
            "report_markdown": str(output_root / "reports" / "five_model_report.md"),
            "report_png": str(output_root / "reports" / "five_model_metric_summary.png"),
        },
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(prepare_output_dir(args.output_dir))
