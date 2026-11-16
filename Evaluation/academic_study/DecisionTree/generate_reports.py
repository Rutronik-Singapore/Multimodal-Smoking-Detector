#!/usr/bin/env python3
"""Generate shareable visual summaries for the SEN66 DT model evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DT_ROOT = Path(__file__).resolve().parent
MODEL_DIRS = {"DT0": DT_ROOT / "DT0", "DT1": DT_ROOT / "DT1", "DT2": DT_ROOT / "DT2"}
MODEL_LABELS = {"DT0": "DT-0B", "DT1": "DT-1", "DT2": "DT-2"}
PARTITION_ORDER = ["train", "validation", "test"]
RAW_SENSORS = ["pm1_0", "pm2_5", "pm4_0", "pm10_0", "voc_index", "nox_index", "co2_ppm"]
COLORS = {
    "DT0": "#4E79A7",
    "DT1": "#59A14F",
    "DT2": "#F28E2B",
    "train": "#4E79A7",
    "validation": "#B07AA1",
    "test": "#E15759",
    "smoking": "#E15759",
    "no_smoking": "#59A14F",
    "selected": "#F28E2B",
    "muted": "#D6D6D6",
    "grid": "#E6E6E6",
    "text": "#222222",
}

REQUIRED_BY_MODEL = {
    "DT0": [
        "DT0_model_metrics.csv",
        "DT0_model_features.csv",
        "DT0_model_predictions.csv",
        "DT0_fixed_validation_metrics.csv",
        "DT0_grouped_cv_manifest.csv",
        "DT0_run_manifest.json",
    ],
    "DT1": [
        "DT1_model_metrics.csv",
        "DT1_model_features.csv",
        "DT1_model_predictions.csv",
        "DT1_fixed_validation_metrics.csv",
        "DT1_grouped_cv_manifest.csv",
        "DT1_tree_tuning_candidate_scores.csv",
        "DT1_channel_ablation.csv",
        "DT1_run_manifest.json",
    ],
    "DT2": [
        "DT2_model_metrics.csv",
        "DT2_model_features.csv",
        "DT2_model_predictions.csv",
        "DT2_fixed_validation_metrics.csv",
        "DT2_grouped_cv_manifest.csv",
        "DT2_stage1_temporal_family_results.csv",
        "DT2_stage2_duration_results.csv",
        "DT2_stage4_tree_tuning_candidate_scores.csv",
        "DT2_sensor_group_ablation.csv",
        "DT2_continuous_full_session_evaluation.csv",
        "DT2_resampling_report.csv",
        "DT2_run_manifest.json",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SEN66 DT model visual reports.")
    parser.add_argument("--regenerate-missing", action="store_true", help="Run existing DT scripts if required outputs are missing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace files already present in the report directory.")
    parser.add_argument("--report-dir", type=Path, default=DT_ROOT / "reports", help="Directory for report images and summary tables.")
    parser.add_argument("--models", nargs="+", choices=["DT0", "DT1", "DT2"], default=["DT0", "DT1", "DT2"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in font_names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(30, True)
FONT_SUBTITLE = load_font(20, True)
FONT = load_font(17)
FONT_SMALL = load_font(14)
FONT_SMALL_BOLD = load_font(14, True)
FONT_TINY = load_font(12)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def lighten(color: str, amount: float = 0.85) -> str:
    r, g, b = hex_to_rgb(color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def new_canvas(width: int, height: int, title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), title, fill=COLORS["text"], font=FONT_TITLE)
    if subtitle:
        draw.text((40, 68), subtitle, fill="#555555", font=FONT)
    return image, draw


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except TypeError:
        pass
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), str(text), font=font)
    return box[2] - box[0]


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    headers: list[str],
    rows: list[list[Any]],
    col_widths: list[int],
    row_height: int = 34,
    header_fill: str = "#F0F2F5",
    fills: list[str] | None = None,
    align: list[str] | None = None,
) -> int:
    align = align or ["left"] * len(headers)
    fills = fills or ["white"] * len(rows)
    cur_x = x
    for header, width in zip(headers, col_widths):
        draw.rectangle((cur_x, y, cur_x + width, y + row_height), fill=header_fill, outline=COLORS["grid"])
        draw.text((cur_x + 8, y + 8), str(header), fill=COLORS["text"], font=FONT_SMALL_BOLD)
        cur_x += width
    cur_y = y + row_height
    for row, fill in zip(rows, fills):
        cur_x = x
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            draw.rectangle((cur_x, cur_y, cur_x + width, cur_y + row_height), fill=fill, outline=COLORS["grid"])
            text = fmt(cell) if not isinstance(cell, str) else cell
            if align[i] == "right":
                tx = cur_x + width - 8 - text_width(draw, text, FONT_SMALL)
            elif align[i] == "center":
                tx = cur_x + (width - text_width(draw, text, FONT_SMALL)) // 2
            else:
                tx = cur_x + 8
            draw.text((tx, cur_y + 8), text, fill=COLORS["text"], font=FONT_SMALL)
            cur_x += width
        cur_y += row_height
    return cur_y


def draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    labels: list[str],
    values: list[float],
    title: str,
    color: str = "#4E79A7",
    x_max: float | None = None,
    value_digits: int = 3,
) -> None:
    draw.text((x, y), title, fill=COLORS["text"], font=FONT_SUBTITLE)
    if not labels:
        draw.text((x, y + 36), "No data", fill="#777777", font=FONT)
        return
    x_max = x_max if x_max is not None else max(max(values), 1.0)
    left_label_width = min(260, max(120, max(text_width(draw, label, FONT_SMALL) for label in labels) + 20))
    plot_x = x + left_label_width
    plot_w = width - left_label_width - 78
    bar_gap = 8
    bar_h = max(18, min(32, (height - 44 - bar_gap * (len(labels) - 1)) // max(1, len(labels))))
    cur_y = y + 42
    for label, value in zip(labels, values):
        draw.text((x, cur_y + 4), label, fill=COLORS["text"], font=FONT_SMALL)
        draw.rectangle((plot_x, cur_y, plot_x + plot_w, cur_y + bar_h), fill="#F4F4F4")
        filled = int(plot_w * (0 if x_max <= 0 else value / x_max))
        draw.rectangle((plot_x, cur_y, plot_x + filled, cur_y + bar_h), fill=color)
        draw.text((plot_x + plot_w + 8, cur_y + 3), fmt(float(value), value_digits), fill=COLORS["text"], font=FONT_SMALL)
        cur_y += bar_h + bar_gap


def draw_grouped_metric_table(draw: ImageDraw.ImageDraw, x: int, y: int, df: pd.DataFrame, metrics: list[tuple[str, str]]) -> int:
    headers = ["Metric"] + [MODEL_LABELS[m] for m in df["model"].tolist()]
    rows = []
    for key, label in metrics:
        row = [label]
        for model in df["model"].tolist():
            value = df.loc[df["model"] == model, key].iloc[0] if key in df.columns else None
            row.append(fmt(value, 3))
        rows.append(row)
    return draw_table(draw, x, y, headers, rows, [280, 150, 150, 150], align=["left", "right", "right", "right"])


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def missing_outputs(models: list[str]) -> list[Path]:
    missing: list[Path] = []
    for model in models:
        for name in REQUIRED_BY_MODEL[model]:
            path = MODEL_DIRS[model] / name
            if not path.exists():
                missing.append(path)
    for path in [DT_ROOT / "DT_comparison.csv", DT_ROOT / "DT_common_subset_comparison.csv", DT_ROOT / "session_partitions.csv"]:
        if not path.exists():
            missing.append(path)
    return missing


def regenerate_model(model: str, verbose: bool) -> None:
    script = MODEL_DIRS[model] / f"train_{model}.py"
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [sys.executable, str(script), "--archive-existing"]
    if verbose:
        print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(MODEL_DIRS[model]), check=True)


def ensure_outputs(models: list[str], regenerate_missing: bool, verbose: bool) -> tuple[bool, list[Path]]:
    missing = missing_outputs(models)
    regenerated = False
    if missing and not regenerate_missing:
        message = "Required evaluation outputs are missing. Re-run with --regenerate-missing or regenerate these files first:\n"
        message += "\n".join(str(path) for path in missing[:30])
        if len(missing) > 30:
            message += f"\n... and {len(missing) - 30} more"
        raise SystemExit(message)
    if missing:
        for model in models:
            if any(str(path).startswith(str(MODEL_DIRS[model])) for path in missing):
                regenerate_model(model, verbose)
                regenerated = True
        missing = missing_outputs(models)
        if missing:
            raise SystemExit("Outputs are still missing after regeneration:\n" + "\n".join(str(p) for p in missing))
    return regenerated, missing


def prepare_report_dir(report_dir: Path, overwrite: bool) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        "01_model_comparison.png",
        "02_confusion_matrices_native.png",
        "03_confusion_matrices_common_subset.png",
        "04_partition_summary.png",
        "05_feature_summary.png",
        "06_dt2_tuning_summary.png",
        "07_dt2_continuous_behavior.png",
        "08_metric_stage_comparison.png",
        "09_native_vs_common_subset.png",
        "model_comparison_summary.csv",
        "partition_summary.csv",
        "dt1_feature_summary.csv",
        "dt2_feature_summary.csv",
        "dt2_tuning_summary.csv",
        "dt2_continuous_summary.csv",
        "README_reports.md",
    ]
    existing = [report_dir / name for name in planned if (report_dir / name).exists()]
    if existing and not overwrite:
        raise SystemExit("Report outputs already exist. Re-run with --overwrite or choose a different --report-dir:\n" + "\n".join(str(p) for p in existing))


def model_summary(models: list[str]) -> pd.DataFrame:
    comparison = read_csv(DT_ROOT / "DT_comparison.csv")
    comparison = comparison[comparison["model"].isin(models)].copy()
    comparison["display_model"] = comparison["model"].map(MODEL_LABELS)
    cols = [
        "model",
        "display_model",
        "validation_smoking_f1",
        "validation_balanced_accuracy",
        "native_test_smoking_f1",
        "native_test_balanced_accuracy",
        "common_subset_test_smoking_f1",
        "common_subset_test_balanced_accuracy",
        "candidate_feature_count",
        "unique_features_used_in_tree",
        "total_split_nodes",
        "tree_depth",
        "leaf_count",
    ]
    return comparison[[c for c in cols if c in comparison.columns]]


def partition_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = read_csv(DT_ROOT / "session_partitions.csv")
    partition = parts.groupby("partition", as_index=False).agg(
        dataset_ids=("dataset_id", lambda s: ",".join(str(int(v)) for v in sorted(s))),
        dataset_count=("dataset_id", "count"),
        row_count=("row_count", "sum"),
        smoking_count=("smoking_count", "sum"),
        no_smoking_count=("no_smoking_count", "sum"),
    )
    partition["partition"] = pd.Categorical(partition["partition"], categories=PARTITION_ORDER, ordered=True)
    partition = partition.sort_values("partition").reset_index(drop=True)
    dataset = parts.sort_values("dataset_id").reset_index(drop=True)
    return partition, dataset


def load_model_metrics(models: list[str]) -> dict[str, pd.Series]:
    out = {}
    for model in models:
        out[model] = read_csv(MODEL_DIRS[model] / f"{model}_model_metrics.csv").iloc[0]
    return out


def make_model_comparison(report_dir: Path, summary: pd.DataFrame) -> Path:
    image, draw = new_canvas(1500, 980, "Overall DT Model Comparison", "Validation selection metrics, final test metrics, and model complexity")
    metric_rows = [
        ("validation_smoking_f1", "Validation Smoking F1"),
        ("validation_balanced_accuracy", "Validation Balanced Accuracy"),
        ("native_test_smoking_f1", "Native Test Smoking F1"),
        ("native_test_balanced_accuracy", "Native Test Balanced Accuracy"),
        ("common_subset_test_smoking_f1", "Common Subset Smoking F1"),
        ("common_subset_test_balanced_accuracy", "Common Subset Balanced Accuracy"),
    ]
    draw_grouped_metric_table(draw, 60, 125, summary, metric_rows)
    draw.text((870, 125), "Complexity", fill=COLORS["text"], font=FONT_SUBTITLE)
    complexity_rows = []
    for _, row in summary.iterrows():
        complexity_rows.append([
            row["display_model"],
            fmt(row.get("candidate_feature_count"), 0),
            fmt(row.get("unique_features_used_in_tree"), 0),
            fmt(row.get("tree_depth"), 0),
            fmt(row.get("leaf_count"), 0),
        ])
    draw_table(draw, 870, 165, ["Model", "Candidate", "Used", "Depth", "Leaves"], complexity_rows, [120, 110, 90, 80, 90], align=["left", "right", "right", "right", "right"])
    metrics = ["native_test_smoking_f1", "common_subset_test_smoking_f1", "native_test_balanced_accuracy", "common_subset_test_balanced_accuracy"]
    y = 430
    for metric in metrics:
        labels = summary["display_model"].tolist()
        values = [float(v) if not pd.isna(v) else 0.0 for v in summary[metric].tolist()]
        title = metric.replace("_", " ").replace("native test", "Native Test").replace("common subset test", "Common Subset").replace("smoking f1", "Smoking F1").replace("balanced accuracy", "Balanced Accuracy")
        draw_bar_chart(draw, 60, y, 650, 115, labels, values, title, "#4E79A7", x_max=1.0)
        y += 130
    draw_bar_chart(draw, 815, 430, 610, 170, summary["display_model"].tolist(), [float(v) for v in summary["candidate_feature_count"]], "Candidate Feature Count", "#9C755F", x_max=max(summary["candidate_feature_count"].max(), 1))
    draw_bar_chart(draw, 815, 635, 610, 170, summary["display_model"].tolist(), [float(v) for v in summary["leaf_count"]], "Leaf Count", "#76B7B2", x_max=max(summary["leaf_count"].max(), 1))
    path = report_dir / "01_model_comparison.png"
    save_image(image, path)
    return path


def confusion_rows(prefix: str, metrics: pd.Series | pd.DataFrame) -> tuple[int, int, int, int]:
    if isinstance(metrics, pd.DataFrame):
        row = metrics.iloc[0]
    else:
        row = metrics
    return (
        int(row[f"{prefix}confusion_tn"]),
        int(row[f"{prefix}confusion_fp"]),
        int(row[f"{prefix}confusion_fn"]),
        int(row[f"{prefix}confusion_tp"]),
    )


def draw_confusion_matrix(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, counts: tuple[int, int, int, int]) -> None:
    tn, fp, fn, tp = counts
    total = max(tn + fp + fn + tp, 1)
    cells = [[tn, fp], [fn, tp]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    draw.text((x, y), title, fill=COLORS["text"], font=FONT_SUBTITLE)
    draw.text((x + 128, y + 38), "Predicted", fill="#555555", font=FONT_SMALL_BOLD)
    draw.text((x + 92, y + 66), "No smoking", fill="#555555", font=FONT_SMALL)
    draw.text((x + 238, y + 66), "Smoking", fill="#555555", font=FONT_SMALL)
    draw.text((x, y + 146), "True", fill="#555555", font=FONT_SMALL_BOLD)
    draw.text((x + 10, y + 110), "No smoking", fill="#555555", font=FONT_SMALL)
    draw.text((x + 10, y + 220), "Smoking", fill="#555555", font=FONT_SMALL)
    max_cell = max(max(r) for r in cells) or 1
    cell_w, cell_h = 130, 90
    grid_x, grid_y = x + 100, y + 95
    for r in range(2):
        for c in range(2):
            value = cells[r][c]
            intensity = 1 - 0.55 * value / max_cell
            fill = lighten(COLORS["smoking"] if labels[r][c] in ["FP", "FN"] else COLORS["no_smoking"], intensity)
            x0 = grid_x + c * cell_w
            y0 = grid_y + r * cell_h
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=fill, outline="#BBBBBB")
            pct = value / total * 100
            draw.text((x0 + 12, y0 + 18), f"{labels[r][c]}: {value:,}", fill=COLORS["text"], font=FONT_SMALL_BOLD)
            draw.text((x0 + 12, y0 + 48), f"{pct:.1f}%", fill=COLORS["text"], font=FONT_SMALL)


def make_confusion_figure(report_dir: Path, models: list[str], common: bool = False) -> Path:
    title = "Common-Subset Test Confusion Matrices" if common else "Native Test Confusion Matrices"
    subtitle = "Same DT-2 eligible rows for all models" if common else "Each model evaluated on its native final test rows"
    image, draw = new_canvas(1500, 520, title, subtitle)
    source = read_csv(DT_ROOT / "DT_common_subset_comparison.csv") if common else None
    for i, model in enumerate(models):
        x = 60 + i * 480
        if common:
            row = source[source["model"] == model].iloc[0]
            counts = confusion_rows("common_subset_test_", row)
        else:
            row = read_csv(MODEL_DIRS[model] / f"{model}_model_metrics.csv").iloc[0]
            counts = confusion_rows("test_", row)
        draw_confusion_matrix(draw, x, 125, MODEL_LABELS[model], counts)
    filename = "03_confusion_matrices_common_subset.png" if common else "02_confusion_matrices_native.png"
    path = report_dir / filename
    save_image(image, path)
    return path


def make_partition_figure(report_dir: Path, partition: pd.DataFrame, dataset: pd.DataFrame) -> Path:
    image, draw = new_canvas(1550, 980, "Partition and Dataset Composition", "Session-level splits and class composition")
    rows = []
    fills = []
    for _, row in partition.iterrows():
        part = str(row["partition"])
        rows.append([part.title(), row["dataset_ids"], row["row_count"], row["smoking_count"], row["no_smoking_count"]])
        fills.append(lighten(COLORS.get(part, "#999999"), 0.86))
    draw_table(draw, 60, 125, ["Partition", "Dataset IDs", "Rows", "Smoking", "No smoking"], rows, [130, 540, 120, 120, 130], fills=fills, align=["left", "left", "right", "right", "right"])
    total_rows = max(float(partition["row_count"].sum()), 1.0)
    x, y = 60, 360
    draw.text((x, y), "Partition class counts", fill=COLORS["text"], font=FONT_SUBTITLE)
    bar_x, bar_y = x + 20, y + 55
    bar_w, bar_h = 760, 52
    for idx, (_, row) in enumerate(partition.iterrows()):
        py = bar_y + idx * 85
        part = str(row["partition"])
        smoking = float(row["smoking_count"])
        no_smoking = float(row["no_smoking_count"])
        total = max(smoking + no_smoking, 1.0)
        draw.text((x, py + 15), part.title(), fill=COLORS["text"], font=FONT_SMALL_BOLD)
        nx = bar_x + 145
        ns_w = int(bar_w * no_smoking / total)
        sm_w = int(bar_w * smoking / total)
        draw.rectangle((nx, py, nx + ns_w, py + bar_h), fill=COLORS["no_smoking"])
        draw.rectangle((nx + ns_w, py, nx + ns_w + sm_w, py + bar_h), fill=COLORS["smoking"])
        draw.rectangle((nx, py, nx + bar_w, py + bar_h), outline="#777777")
        draw.text((nx + bar_w + 12, py + 14), f"{int(total):,} rows", fill=COLORS["text"], font=FONT_SMALL)
    draw.rectangle((80, 650, 102, 672), fill=COLORS["no_smoking"])
    draw.text((110, 649), "No smoking", fill=COLORS["text"], font=FONT_SMALL)
    draw.rectangle((230, 650, 252, 672), fill=COLORS["smoking"])
    draw.text((260, 649), "Smoking", fill=COLORS["text"], font=FONT_SMALL)
    draw.text((900, 360), "Rows per dataset", fill=COLORS["text"], font=FONT_SUBTITLE)
    max_rows = max(dataset["row_count"].max(), 1)
    chart_x, chart_y = 900, 410
    for i, (_, row) in enumerate(dataset.iterrows()):
        dx = chart_x + i * 28
        h = int(430 * row["row_count"] / max_rows)
        color = COLORS.get(str(row["partition"]), "#999999")
        draw.rectangle((dx, chart_y + 430 - h, dx + 18, chart_y + 430), fill=color)
        if i % 2 == 0:
            draw.text((dx - 1, chart_y + 438), str(int(row["dataset_id"])), fill=COLORS["text"], font=FONT_TINY)
    draw.line((chart_x, chart_y + 430, chart_x + 625, chart_y + 430), fill="#777777")
    draw.text((chart_x, chart_y + 470), "Dataset ID", fill="#555555", font=FONT_SMALL)
    draw.text((chart_x, chart_y - 22), f"Max {int(max_rows):,} rows", fill="#555555", font=FONT_SMALL)
    path = report_dir / "04_partition_summary.png"
    save_image(image, path)
    return path


def sensor_for_feature(feature: str) -> str:
    for sensor in RAW_SENSORS:
        if feature == sensor or feature.startswith(sensor + "_"):
            return sensor
    return "other"


def family_for_feature(feature: str) -> str:
    if feature in RAW_SENSORS:
        return "raw"
    for family in ["mean", "max", "delta", "min", "std", "slope"]:
        if f"_{family}_" in feature:
            return family
    return "other"


def fold_feature_frequency(path: Path) -> Counter[str]:
    if not path.exists():
        return Counter()
    df = pd.read_csv(path)
    counts: Counter[str] = Counter()
    if "features_used" not in df.columns:
        return counts
    for raw in df["features_used"].dropna():
        try:
            values = json.loads(raw)
        except Exception:
            values = []
        for feature in set(values):
            counts[feature] += 1
    return counts


def make_feature_summaries(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    dt1_features = read_csv(MODEL_DIRS["DT1"] / "DT1_model_features.csv")
    dt1_freq = fold_feature_frequency(MODEL_DIRS["DT1"] / "DT1_grouped_cv_manifest.csv")
    dt1_rows = []
    for _, row in dt1_features.iterrows():
        feature = row["feature"]
        dt1_rows.append({
            "feature": feature,
            "sensor": sensor_for_feature(feature),
            "final_tree_split_count": int(row["used_in_tree_count"]),
            "fold_selection_count": int(dt1_freq.get(feature, 0)),
        })
    dt1_summary = pd.DataFrame(dt1_rows).sort_values(["final_tree_split_count", "fold_selection_count"], ascending=False)
    dt1_summary.to_csv(report_dir / "dt1_feature_summary.csv", index=False)

    dt2_features = read_csv(MODEL_DIRS["DT2"] / "DT2_model_features.csv")
    dt2_freq = fold_feature_frequency(MODEL_DIRS["DT2"] / "DT2_grouped_cv_manifest.csv")
    dt2_rows = []
    for _, row in dt2_features.iterrows():
        feature = row["feature"]
        dt2_rows.append({
            "feature": feature,
            "sensor": sensor_for_feature(feature),
            "feature_family": family_for_feature(feature),
            "final_tree_split_count": int(row["used_in_tree_count"]),
            "fold_selection_count": int(dt2_freq.get(feature, 0)),
        })
    dt2_summary = pd.DataFrame(dt2_rows).sort_values(["final_tree_split_count", "fold_selection_count"], ascending=False)
    dt2_summary.to_csv(report_dir / "dt2_feature_summary.csv", index=False)

    image, draw = new_canvas(1600, 1100, "DT-1 and DT-2 Feature / Sensor Summary", "Final tree usage, fold stability, and sensor ablation overview")
    top_dt1 = dt1_summary.head(7)
    draw_bar_chart(draw, 60, 125, 690, 285, top_dt1["feature"].tolist(), top_dt1["final_tree_split_count"].astype(float).tolist(), "DT-1 final tree feature split counts", COLORS["DT1"], x_max=max(top_dt1["final_tree_split_count"].max(), 1), value_digits=0)
    top_dt2 = dt2_summary.head(8)
    draw_bar_chart(draw, 840, 125, 690, 320, top_dt2["feature"].tolist(), top_dt2["final_tree_split_count"].astype(float).tolist(), "DT-2 final tree feature split counts", COLORS["DT2"], x_max=max(top_dt2["final_tree_split_count"].max(), 1), value_digits=0)

    dt1_ablation = read_csv(MODEL_DIRS["DT1"] / "DT1_channel_ablation.csv")
    dt1_ablation = dt1_ablation[dt1_ablation["removed_channel"] != "none"].copy()
    dt1_ablation = dt1_ablation.sort_values("validation_smoking_f1", ascending=True)
    draw_bar_chart(draw, 60, 500, 690, 285, dt1_ablation["removed_channel"].tolist(), dt1_ablation["validation_smoking_f1"].astype(float).tolist(), "DT-1 ablation validation Smoking F1 after removing channel", COLORS["DT1"], x_max=1.0)

    dt2_ablation = read_csv(MODEL_DIRS["DT2"] / "DT2_sensor_group_ablation.csv")
    dt2_ablation = dt2_ablation[dt2_ablation["removed_channel"] != "none"].copy()
    dt2_ablation = dt2_ablation.sort_values("validation_smoking_f1", ascending=True)
    draw_bar_chart(draw, 840, 500, 690, 285, dt2_ablation["removed_channel"].tolist(), dt2_ablation["validation_smoking_f1"].astype(float).tolist(), "DT-2 ablation validation Smoking F1 after removing channel", COLORS["DT2"], x_max=1.0)

    sensor_counts = dt2_summary.groupby("sensor", as_index=False)["final_tree_split_count"].sum()
    sensor_counts = sensor_counts.sort_values("final_tree_split_count", ascending=False)
    family_counts = dt2_summary.groupby("feature_family", as_index=False)["final_tree_split_count"].sum()
    family_counts = family_counts.sort_values("final_tree_split_count", ascending=False)
    draw_bar_chart(draw, 60, 830, 690, 210, sensor_counts["sensor"].tolist(), sensor_counts["final_tree_split_count"].astype(float).tolist(), "DT-2 final tree usage by sensor group", "#9C755F", x_max=max(sensor_counts["final_tree_split_count"].max(), 1), value_digits=0)
    draw_bar_chart(draw, 840, 830, 690, 210, family_counts["feature_family"].tolist(), family_counts["final_tree_split_count"].astype(float).tolist(), "DT-2 final tree usage by feature family", "#76B7B2", x_max=max(family_counts["final_tree_split_count"].max(), 1), value_digits=0)
    path = report_dir / "05_feature_summary.png"
    save_image(image, path)
    return dt1_summary, dt2_summary, path


def selected_candidate(df: pd.DataFrame) -> pd.Series:
    if "candidate_rank" in df.columns:
        return df.sort_values("candidate_rank").iloc[0]
    return df.sort_values(["fixed_validation_smoking_f1", "fixed_validation_balanced_accuracy"], ascending=False).iloc[0]


def make_dt2_tuning_summary(report_dir: Path) -> tuple[pd.DataFrame, Path]:
    stage1 = read_csv(MODEL_DIRS["DT2"] / "DT2_stage1_temporal_family_results.csv")
    stage2 = read_csv(MODEL_DIRS["DT2"] / "DT2_stage2_duration_results.csv")
    stage4 = read_csv(MODEL_DIRS["DT2"] / "DT2_stage4_tree_tuning_candidate_scores.csv")
    ablation = read_csv(MODEL_DIRS["DT2"] / "DT2_sensor_group_ablation.csv")
    rows = []
    for stage_name, df, label_col in [
        ("Temporal family", stage1, "temporal_family"),
        ("Window duration", stage2, "window_duration_set"),
        ("Tree tuning", stage4, "candidate_id"),
    ]:
        best_id = selected_candidate(df)["candidate_id"]
        for _, row in df.iterrows():
            rows.append({
                "stage": stage_name,
                "candidate_id": row["candidate_id"],
                "candidate_name": row.get(label_col, row["candidate_id"]),
                "validation_smoking_f1": row.get("fixed_validation_smoking_f1"),
                "validation_balanced_accuracy": row.get("fixed_validation_balanced_accuracy"),
                "eligible_row_count": row.get("eligible_row_count"),
                "selected": row["candidate_id"] == best_id,
            })
    tuning_summary = pd.DataFrame(rows)
    tuning_summary.to_csv(report_dir / "dt2_tuning_summary.csv", index=False)

    image, draw = new_canvas(1600, 1120, "DT-2 Staged Tuning Summary", "Validation-driven selection without using the test partition")
    stage1_labels = stage1["temporal_family"].fillna(stage1["candidate_id"]).astype(str).tolist()
    draw_bar_chart(draw, 60, 125, 710, 260, stage1_labels, stage1["fixed_validation_smoking_f1"].astype(float).tolist(), "Stage 1: temporal family validation Smoking F1", COLORS["DT2"], x_max=1.0)
    best1 = selected_candidate(stage1)
    draw.text((90, 390), f"Selected: {best1.get('temporal_family', best1['candidate_id'])}", fill=COLORS["selected"], font=FONT_SMALL_BOLD)

    stage2_labels = stage2["window_duration_set"].astype(str).tolist()
    draw_bar_chart(draw, 860, 125, 650, 260, stage2_labels, stage2["fixed_validation_smoking_f1"].astype(float).tolist(), "Stage 2: duration set validation Smoking F1", COLORS["DT2"], x_max=1.0)
    best2 = selected_candidate(stage2)
    draw.text((890, 390), f"Selected: {best2.get('window_duration_set', best2['candidate_id'])}", fill=COLORS["selected"], font=FONT_SMALL_BOLD)

    ablation_plot = ablation.copy()
    ablation_plot["candidate"] = ablation_plot["removed_channel"].apply(lambda x: "retain all" if x == "none" else f"remove {x}")
    ablation_plot = ablation_plot.sort_values("validation_smoking_f1", ascending=True)
    draw_bar_chart(draw, 60, 480, 710, 315, ablation_plot["candidate"].tolist(), ablation_plot["validation_smoking_f1"].astype(float).tolist(), "Stage 3: sensor ablation validation Smoking F1", "#9C755F", x_max=1.0)
    draw.text((90, 800), "Default policy: interpretive-only; final DT-2 retains all channels.", fill="#555555", font=FONT_SMALL)

    top_stage4 = stage4.sort_values("candidate_rank").head(8).copy()
    labels = [
        f"{row['candidate_id']} d{int(row['max_depth'])} leaf{int(row['min_samples_leaf'])} {row['criterion']}"
        for _, row in top_stage4.iterrows()
    ]
    draw_bar_chart(draw, 860, 480, 650, 360, labels, top_stage4["fixed_validation_smoking_f1"].astype(float).tolist(), "Stage 4: top tree-tuning candidates", "#4E79A7", x_max=1.0)
    best4 = selected_candidate(stage4)
    draw.text((890, 850), f"Selected: {best4['candidate_id']} by F1/complexity tie-break", fill=COLORS["selected"], font=FONT_SMALL_BOLD)

    sample_rows = []
    for _, row in stage2.iterrows():
        sample_rows.append([row["window_duration_set"], row.get("eligible_row_count"), row.get("excluded_row_count"), row.get("fixed_validation_balanced_accuracy")])
    draw_table(draw, 60, 900, ["Duration set", "Eligible", "Excluded", "Val Balanced Acc."], sample_rows, [180, 110, 110, 150], align=["left", "right", "right", "right"])
    path = report_dir / "06_dt2_tuning_summary.png"
    save_image(image, path)
    return tuning_summary, path


def make_dt2_continuous(report_dir: Path) -> tuple[pd.DataFrame, Path]:
    continuous = read_csv(MODEL_DIRS["DT2"] / "DT2_continuous_full_session_evaluation.csv")
    continuous.to_csv(report_dir / "dt2_continuous_summary.csv", index=False)
    image, draw = new_canvas(1500, 880, "DT-2 Continuous Operational Behavior", "Full test sessions retain startup and transition periods")
    rows = []
    fills = []
    for _, row in continuous.iterrows():
        detected = bool(row["event_detected"])
        rows.append([
            f"Dataset {int(row['dataset_id'])}",
            "Detected" if detected else "Not detected",
            row["time_to_first_smoking_prediction"],
            row["proportion_event_predicted_smoking"],
            row["time_smoking_predictions_persist_after_event_end"],
            row["not_ready_startup_samples"],
        ])
        fills.append("#EAF5EA" if detected else "#FBECEC")
    draw_table(
        draw,
        60,
        125,
        ["Test session", "Event", "Delay (s)", "Coverage", "Persistence (s)", "Not ready"],
        rows,
        [160, 130, 120, 120, 150, 120],
        fills=fills,
        align=["left", "left", "right", "right", "right", "right"],
    )
    labels = [f"Dataset {int(v)}" for v in continuous["dataset_id"]]
    draw_bar_chart(draw, 60, 360, 640, 180, labels, continuous["time_to_first_smoking_prediction"].fillna(0).astype(float).tolist(), "First smoking prediction delay (seconds)", COLORS["DT2"], x_max=max(continuous["time_to_first_smoking_prediction"].fillna(0).max(), 1), value_digits=1)
    draw_bar_chart(draw, 800, 360, 610, 180, labels, continuous["proportion_event_predicted_smoking"].astype(float).tolist(), "Smoking-event coverage", COLORS["smoking"], x_max=1.0)
    draw_bar_chart(draw, 60, 600, 640, 180, labels, continuous["time_smoking_predictions_persist_after_event_end"].fillna(0).astype(float).tolist(), "Prediction persistence after event end (seconds)", "#9C755F", x_max=max(continuous["time_smoking_predictions_persist_after_event_end"].fillna(0).max(), 1), value_digits=1)
    draw_bar_chart(draw, 800, 600, 610, 180, labels, continuous["not_ready_startup_samples"].astype(float).tolist(), "Startup samples marked not ready", "#76B7B2", x_max=max(continuous["not_ready_startup_samples"].max(), 1), value_digits=0)
    path = report_dir / "07_dt2_continuous_behavior.png"
    save_image(image, path)
    return continuous, path


def make_metric_stage_comparison(report_dir: Path, models: list[str]) -> Path:
    metrics = load_model_metrics(models)
    image, draw = new_canvas(1500, 930, "Training / Validation / Test Performance Overview", "Smoking F1 by evaluation stage")
    headers = ["Model", "Train", "Grouped CV", "Validation", "Native Test", "Common Subset"]
    common = read_csv(DT_ROOT / "DT_common_subset_comparison.csv")
    rows = []
    for model in models:
        m = metrics[model]
        validation = read_csv(MODEL_DIRS[model] / f"{model}_fixed_validation_metrics.csv").iloc[0]
        cv = read_csv(MODEL_DIRS[model] / f"{model}_grouped_cv_manifest.csv")
        common_row = common[common["model"] == model].iloc[0]
        rows.append([
            MODEL_LABELS[model],
            m.get("train_smoking_f1"),
            cv["smoking_f1"].mean() if "smoking_f1" in cv.columns else None,
            validation.get("validation_smoking_f1"),
            m.get("test_smoking_f1"),
            common_row.get("common_subset_test_smoking_f1"),
        ])
    draw_table(draw, 60, 125, headers, rows, [150, 130, 130, 130, 130, 150], align=["left", "right", "right", "right", "right", "right"])
    stages = headers[1:]
    x0, y0 = 90, 390
    group_w = 420
    for idx, row in enumerate(rows):
        label = row[0]
        values = [0 if pd.isna(v) else float(v) for v in row[1:]]
        draw_bar_chart(draw, x0 + idx * 470, y0, group_w, 380, stages, values, label, COLORS[models[idx]], x_max=1.0)
    path = report_dir / "08_metric_stage_comparison.png"
    save_image(image, path)
    return path


def make_native_vs_common(report_dir: Path, summary: pd.DataFrame) -> Path:
    image, draw = new_canvas(1450, 760, "Native vs Common-Subset Test Comparison", "Shows how DT-2 eligibility filtering affects like-for-like comparison")
    for idx, metric_pair in enumerate([
        ("native_test_smoking_f1", "common_subset_test_smoking_f1", "Smoking F1"),
        ("native_test_balanced_accuracy", "common_subset_test_balanced_accuracy", "Balanced Accuracy"),
    ]):
        x = 70 + idx * 690
        native_col, common_col, title = metric_pair
        draw.text((x, 125), title, fill=COLORS["text"], font=FONT_SUBTITLE)
        labels = summary["display_model"].tolist()
        max_v = 1.0
        for i, (_, row) in enumerate(summary.iterrows()):
            y = 185 + i * 130
            draw.text((x, y + 16), labels[i], fill=COLORS["text"], font=FONT_SMALL_BOLD)
            native = 0.0 if pd.isna(row[native_col]) else float(row[native_col])
            common = 0.0 if pd.isna(row[common_col]) else float(row[common_col])
            bx = x + 145
            bw = 360
            draw.rectangle((bx, y, bx + int(bw * native / max_v), y + 28), fill="#4E79A7")
            draw.text((bx + bw + 12, y + 3), f"Native {native:.3f}", fill=COLORS["text"], font=FONT_SMALL)
            draw.rectangle((bx, y + 42, bx + int(bw * common / max_v), y + 70), fill="#F28E2B")
            draw.text((bx + bw + 12, y + 45), f"Common {common:.3f}", fill=COLORS["text"], font=FONT_SMALL)
    draw.rectangle((90, 640, 112, 662), fill="#4E79A7")
    draw.text((120, 638), "Native test", fill=COLORS["text"], font=FONT_SMALL)
    draw.rectangle((250, 640, 272, 662), fill="#F28E2B")
    draw.text((280, 638), "Common subset", fill=COLORS["text"], font=FONT_SMALL)
    path = report_dir / "09_native_vs_common_subset.png"
    save_image(image, path)
    return path


def write_readme(report_dir: Path, image_paths: list[Path], table_paths: list[Path], regenerated: bool) -> Path:
    lines = [
        "# SEN66 DT Model Evaluation Reports",
        "",
        "Generated by `generate_reports.py` from the current DT-0B, DT-1 and DT-2 evaluation outputs.",
        "",
        f"Missing model outputs regenerated during this run: {'yes' if regenerated else 'no'}",
        "",
        "## Images",
    ]
    descriptions = {
        "01_model_comparison.png": "Overall model metrics and complexity summary.",
        "02_confusion_matrices_native.png": "Native final test confusion matrices for each model.",
        "03_confusion_matrices_common_subset.png": "Common-subset confusion matrices using the same DT-2 eligible test rows.",
        "04_partition_summary.png": "Train, validation and test dataset composition.",
        "05_feature_summary.png": "DT-1 and DT-2 feature usage, fold stability and sensor ablation summary.",
        "06_dt2_tuning_summary.png": "DT-2 staged temporal, duration, sensor and tree-selection summary.",
        "07_dt2_continuous_behavior.png": "DT-2 continuous operational event behavior on test datasets.",
        "08_metric_stage_comparison.png": "Train, grouped-CV, validation and test Smoking F1 overview.",
        "09_native_vs_common_subset.png": "Native versus common-subset test metric comparison.",
    }
    for path in image_paths:
        lines.append(f"- `{path.name}`: {descriptions.get(path.name, 'Generated visual report.')}")
    lines += ["", "## Summary Tables"]
    for path in table_paths:
        lines.append(f"- `{path.name}`")
    lines += [
        "",
        "Detailed CSVs, prediction outputs, manifests and generated model files remain in the model folders.",
        "",
        "## Rerun",
        "",
        "```powershell",
        "python academic_study/DecisionTree/generate_reports.py --report-dir generated_reports",
        "```",
    ]
    path = report_dir / "README_reports.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def verify_reports(report_dir: Path, image_paths: list[Path], summary: pd.DataFrame) -> None:
    for path in image_paths:
        if not path.exists() or path.stat().st_size < 2000:
            raise RuntimeError(f"Report image was not generated correctly: {path}")
        with Image.open(path) as image:
            if image.width < 1000 or image.height < 500:
                raise RuntimeError(f"Report image is unexpectedly small: {path} {image.size}")
    common = read_csv(DT_ROOT / "DT_common_subset_comparison.csv")
    counts = set(common["common_eligible_test_row_count"].astype(int).tolist())
    if len(counts) != 1:
        raise RuntimeError(f"Common-subset row counts differ across models: {sorted(counts)}")
    report_summary = pd.read_csv(report_dir / "model_comparison_summary.csv")
    for _, row in summary.iterrows():
        check = report_summary[report_summary["model"] == row["model"]].iloc[0]
        for col in ["native_test_smoking_f1", "common_subset_test_smoking_f1"]:
            a = row.get(col)
            b = check.get(col)
            if not (pd.isna(a) and pd.isna(b)) and abs(float(a) - float(b)) > 1e-12:
                raise RuntimeError(f"Summary value mismatch for {row['model']} {col}: {a} != {b}")


def main() -> None:
    args = parse_args()
    regenerated, _ = ensure_outputs(args.models, args.regenerate_missing, args.verbose)
    prepare_report_dir(args.report_dir, args.overwrite)

    summary = model_summary(args.models)
    summary.to_csv(args.report_dir / "model_comparison_summary.csv", index=False)
    partition, dataset = partition_summaries()
    partition.to_csv(args.report_dir / "partition_summary.csv", index=False)

    image_paths: list[Path] = []
    image_paths.append(make_model_comparison(args.report_dir, summary))
    image_paths.append(make_confusion_figure(args.report_dir, args.models, common=False))
    image_paths.append(make_confusion_figure(args.report_dir, args.models, common=True))
    image_paths.append(make_partition_figure(args.report_dir, partition, dataset))
    _, _, feature_path = make_feature_summaries(args.report_dir)
    image_paths.append(feature_path)
    _, tuning_path = make_dt2_tuning_summary(args.report_dir)
    image_paths.append(tuning_path)
    _, continuous_path = make_dt2_continuous(args.report_dir)
    image_paths.append(continuous_path)
    image_paths.append(make_metric_stage_comparison(args.report_dir, args.models))
    image_paths.append(make_native_vs_common(args.report_dir, summary))

    table_paths = [
        args.report_dir / "model_comparison_summary.csv",
        args.report_dir / "partition_summary.csv",
        args.report_dir / "dt1_feature_summary.csv",
        args.report_dir / "dt2_feature_summary.csv",
        args.report_dir / "dt2_tuning_summary.csv",
        args.report_dir / "dt2_continuous_summary.csv",
    ]
    readme_path = write_readme(args.report_dir, image_paths, table_paths, regenerated)
    verify_reports(args.report_dir, image_paths, summary)
    print(f"Reports written to: {args.report_dir}")
    print(f"README: {readme_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Summary tables: {len(table_paths)}")
    print(f"Missing outputs regenerated: {'yes' if regenerated else 'no'}")


if __name__ == "__main__":
    main()
