import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEPLOYED_DIR = Path(__file__).resolve().parent
CSV_PATH = DEPLOYED_DIR / "merged_log.CSV"
OUTPUT_DIR = DEPLOYED_DIR / "reruns" / "DecisionTree"

MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 20
N_THRESHOLDS = 128

RAW_FEATURES = [
    "pm1_0",
    "pm2_5",
    "pm4_0",
    "pm10_0",
    "voc_index",
    "nox_index",
    "co2_ppm",
]

ROLLING_BASE_FEATURES = [
    "pm2_5",
    "pm10_0",
    "voc_index",
    "co2_ppm",
]

WINDOWS = [5, 10, 30]


@dataclass
class Node:
    prediction: int
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self):
        return self.feature_index is None


def clean_sensor_startup_rows(df):
    return df[
        (df["pm1_0"] < 6500)
        & (df["pm2_5"] < 6500)
        & (df["pm4_0"] < 6500)
        & (df["pm10_0"] < 6500)
        & (df["nox_index"] < 3000)
        & (df["co2_ppm"] < 65000)
    ].copy()


def add_rolling_features(df):
    df = df.sort_values(["source_file", "uptime_ms"]).copy()

    for col in ROLLING_BASE_FEATURES:
        grouped = df.groupby("source_file", sort=False)[col]

        for window in WINDOWS:
            df[f"{col}_mean_{window}"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_max_{window}"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).max()
            )
            df[f"{col}_delta_{window}"] = grouped.transform(lambda s: s - s.shift(window))

    return df.fillna(0)


def build_feature_list():
    rolling_features = []
    for col in ROLLING_BASE_FEATURES:
        for window in WINDOWS:
            rolling_features.extend(
                [
                    f"{col}_mean_{window}",
                    f"{col}_max_{window}",
                    f"{col}_delta_{window}",
                ]
            )

    return RAW_FEATURES + rolling_features


def weighted_gini(y, weights):
    total = weights.sum()
    if total <= 0:
        return 0.0

    p1 = weights[y == 1].sum() / total
    p0 = 1.0 - p1
    return 1.0 - p0 * p0 - p1 * p1


def majority_class(y, weights):
    smoking_weight = weights[y == 1].sum()
    no_smoking_weight = weights[y == 0].sum()
    return 1 if smoking_weight >= no_smoking_weight else 0


def candidate_thresholds(values):
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([])

    if unique.size <= N_THRESHOLDS:
        thresholds = (unique[:-1] + unique[1:]) / 2.0
    else:
        quantiles = np.linspace(0.01, 0.99, N_THRESHOLDS)
        thresholds = np.unique(np.quantile(values, quantiles))

    return thresholds


def best_split(X, y, weights):
    parent_gini = weighted_gini(y, weights)
    best = None

    for feature_index in range(X.shape[1]):
        values = X[:, feature_index]

        for threshold in candidate_thresholds(values):
            left_mask = values <= threshold
            left_count = int(left_mask.sum())
            right_count = len(y) - left_count

            if left_count < MIN_SAMPLES_LEAF or right_count < MIN_SAMPLES_LEAF:
                continue

            right_mask = ~left_mask
            left_weight = weights[left_mask].sum()
            right_weight = weights[right_mask].sum()
            total_weight = left_weight + right_weight

            split_gini = (
                left_weight / total_weight * weighted_gini(y[left_mask], weights[left_mask])
                + right_weight / total_weight * weighted_gini(y[right_mask], weights[right_mask])
            )
            gain = parent_gini - split_gini

            if best is None or gain > best["gain"]:
                best = {
                    "gain": gain,
                    "feature_index": feature_index,
                    "threshold": float(threshold),
                    "left_mask": left_mask,
                }

    return best


def build_tree(X, y, weights, depth=0):
    prediction = majority_class(y, weights)
    node = Node(prediction=prediction, depth=depth)

    if depth >= MAX_DEPTH or len(np.unique(y)) == 1 or len(y) < MIN_SAMPLES_LEAF * 2:
        return node

    split = best_split(X, y, weights)
    if split is None or split["gain"] <= 1e-9:
        return node

    left_mask = split["left_mask"]
    right_mask = ~left_mask

    node.feature_index = split["feature_index"]
    node.threshold = split["threshold"]
    node.left = build_tree(X[left_mask], y[left_mask], weights[left_mask], depth + 1)
    node.right = build_tree(X[right_mask], y[right_mask], weights[right_mask], depth + 1)

    return node


def predict_one(node, row):
    while not node.is_leaf:
        if row[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.prediction


def predict_tree(node, X):
    return np.array([predict_one(node, row) for row in X], dtype=int)


def balanced_weights(y):
    counts = np.bincount(y, minlength=2)
    total = len(y)
    weights = np.zeros(len(y), dtype=float)

    for cls in [0, 1]:
        if counts[cls] > 0:
            weights[y == cls] = total / (2.0 * counts[cls])

    return weights


def print_metrics(y_true, y_pred):
    labels = ["no smoking", "smoking"]
    print("Confusion matrix, rows=true, columns=predicted")
    print("              pred no smoking  pred smoking")

    for cls, label in enumerate(labels):
        row = [int(((y_true == cls) & (y_pred == pred)).sum()) for pred in [0, 1]]
        print(f"true {label:10s} {row[0]:16d} {row[1]:13d}")

    print("\nClassification report")
    print("label         precision    recall        f1   support")

    for cls, label in enumerate(labels):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int((y_true == cls).sum())
        print(f"{label:12s} {precision:9.3f} {recall:9.3f} {f1:9.3f} {support:9d}")

    accuracy = float((y_true == y_pred).mean())
    print(f"\naccuracy: {accuracy:.3f}")


def tree_to_text(node, features, depth=0):
    indent = "|   " * depth

    if node.is_leaf:
        return f"{indent}class: {'smoking' if node.prediction else 'no smoking'}\n"

    feature = features[node.feature_index]
    text = f"{indent}|--- {feature} <= {node.threshold:.6f}\n"
    text += tree_to_text(node.left, features, depth + 1)
    text += f"{indent}|--- {feature} >  {node.threshold:.6f}\n"
    text += tree_to_text(node.right, features, depth + 1)
    return text


def c_safe_name(feature_name):
    return feature_name.replace(".", "_").replace("-", "_")


def node_to_c(node, features, depth=1):
    indent = "    " * depth

    if node.is_leaf:
        return f"{indent}return {'true' if node.prediction else 'false'};\n"

    feature = c_safe_name(features[node.feature_index])

    code = ""
    code += f"{indent}if ({feature} <= {node.threshold:.6f}f)\n"
    code += f"{indent}{{\n"
    code += node_to_c(node.left, features, depth + 1)
    code += f"{indent}}}\n"
    code += f"{indent}else\n"
    code += f"{indent}{{\n"
    code += node_to_c(node.right, features, depth + 1)
    code += f"{indent}}}\n"

    return code


def function_args(features):
    return ",\n                                ".join(
        f"float {c_safe_name(feature)}" for feature in features
    )


def write_c_model(tree, features, tree_text):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h_path = OUTPUT_DIR / "sen66_time_tree_model.h"
    c_path = OUTPUT_DIR / "sen66_time_tree_model.c"
    txt_path = OUTPUT_DIR / "sen66_time_tree_model.txt"
    features_path = OUTPUT_DIR / "sen66_time_tree_features.txt"

    args = function_args(features)

    h_code = f"""#ifndef SEN66_TIME_TREE_MODEL_H
#define SEN66_TIME_TREE_MODEL_H

#include <stdbool.h>

bool sen66_time_tree_predict_smoking({args});

#endif
"""

    c_code = f"""#include "sen66_time_tree_model.h"

bool sen66_time_tree_predict_smoking({args})
{{
"""
    c_code += node_to_c(tree, features)
    c_code += "}\n"

    h_path.write_text(h_code)
    c_path.write_text(c_code)
    txt_path.write_text(tree_text)
    features_path.write_text("\n".join(features) + "\n")

    print(f"\nWrote: {h_path}")
    print(f"Wrote: {c_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {features_path}")


def prepare_data():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df.drop(columns=["elapsed_time", "predicted_label", "hoi"], errors="ignore")

    required = set(RAW_FEATURES + ["label", "source_file", "uptime_ms"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded rows: {len(df):,}")
    print("\nRows by source file:")
    print(df["source_file"].value_counts().to_string())
    print("\nRows by label:")
    print(df["label"].value_counts().to_string())

    df = clean_sensor_startup_rows(df)
    df = add_rolling_features(df)
    features = build_feature_list()
    df["target"] = df["label"].str.lower().eq("smoking").astype(int)

    print(f"\nRows after startup/sentinel cleanup: {len(df):,}")
    print(f"Feature count: {len(features)}")

    return df, features


def holdout_evaluation(df, features):
    print("\n=== File holdout evaluation ===")

    for holdout_file in sorted(df["source_file"].unique()):
        train_df = df[df["source_file"] != holdout_file]
        test_df = df[df["source_file"] == holdout_file]

        print(f"\nHoldout: {holdout_file}")
        if test_df["target"].nunique() < 2:
            print("Skipped metrics: holdout file has only one label.")
            print(test_df["label"].value_counts().to_string())
            continue

        X_train = train_df[features].to_numpy(dtype=float)
        y_train = train_df["target"].to_numpy(dtype=int)
        X_test = test_df[features].to_numpy(dtype=float)
        y_test = test_df["target"].to_numpy(dtype=int)

        tree = build_tree(X_train, y_train, balanced_weights(y_train))
        y_pred = predict_tree(tree, X_test)
        print_metrics(y_test, y_pred)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the historical deployed SEN66 decision tree.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Training CSV. Default: deployed/merged_log.CSV")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Rerun output directory; packaged exports are never the default target.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty rerun output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate resolved input/output paths without training or writing files.")
    return parser.parse_args()


def validate_output_dir(output_dir, overwrite):
    resolved = output_dir.expanduser().resolve()
    if resolved == DEPLOYED_DIR.resolve():
        raise ValueError("refusing to write packaged deployed artifacts; choose a rerun output directory")
    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise ValueError(f"rerun output directory is not empty: {resolved}; use --overwrite only for rerun outputs")
    return resolved


def main():
    global CSV_PATH, OUTPUT_DIR
    args = parse_args()
    CSV_PATH = args.csv.expanduser().resolve()
    OUTPUT_DIR = validate_output_dir(args.output_dir, args.overwrite) if not args.dry_run else validate_output_dir(args.output_dir, overwrite=True)
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)
    if args.dry_run:
        print(f"DRY RUN input CSV: {CSV_PATH}")
        print(f"DRY RUN rerun output: {OUTPUT_DIR}")
        return
    df, features = prepare_data()
    holdout_evaluation(df, features)

    X = df[features].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    print("\n=== Training final tree on all cleaned rows ===")
    tree = build_tree(X, y, balanced_weights(y))
    tree_text = tree_to_text(tree, features)
    print(tree_text)
    write_c_model(tree, features, tree_text)


if __name__ == "__main__":
    main()
