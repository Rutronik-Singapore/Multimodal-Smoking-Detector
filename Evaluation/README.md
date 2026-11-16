# Environmental Machine Learning — Evaluation

This archive contains the data, scripts, model exports, and saved results for the environmental branch of the PSoC Edge E84 smoking-detection study. It supports two distinct tasks: tracing the decision tree used in the firmware and comparing three academic Decision Tree variants with two DEEPCRAFT Studio models. The camera model and the BLE dashboard are maintained separately.

## Start with the saved results

Open the [five-model report](academic_study/DEEPCRAFT_training/reports/five_model_report.md) or the [summary CSV](academic_study/DEEPCRAFT_training/five_model_summary.csv). Neither requires Python or Studio.

| Model | Native-test smoking F1 | Common-subset smoking F1 |
| --- | ---: | ---: |
| DT0 | 0.807 | 0.828 |
| DT1 | 0.805 | 0.826 |
| DT2 | 0.798 | 0.798 |
| DEEPCRAFT Instant | 0.649 | 0.619 |
| DEEPCRAFT Time | 0.783 | 0.804 |

The native figure uses each model's eligible test rows. The common-subset figure uses the same **3,140 matched test rows** for all five models. These results describe 22 sessions collected under uncontrolled conditions. They are useful for comparing the implemented workflows and observed behavior, not as a claim of field accuracy or a substitute for independent validation.

## Layout

| Path | Contents |
| --- | --- |
| [`deployed/`](deployed/) | Historical firmware trainer, its own `merged_log.CSV`, and deployed C/H model exports. |
| [`logs/csv_log/`](logs/csv_log/) | Original recordings in `legacy/` and `pse84/`. |
| [`logs/csv_merge/`](logs/csv_merge/) | Log merger, canonical `0408_merge.csv`, and its source-recording manifest. |
| [`academic_study/DecisionTree/`](academic_study/DecisionTree/) | DT0 (sample-count rolling features), DT1 (instantaneous features), DT2 (1 Hz temporal features), frozen session partitions, and reports. |
| [`academic_study/DEEPCRAFT_data_prep/`](academic_study/DEEPCRAFT_data_prep/) | CSV preparation scripts, prepared sessions, and the processing manifest. |
| [`academic_study/DEEPCRAFT_training/`](academic_study/DEEPCRAFT_training/) | Studio projects/exports, saved predictions, Python evaluator, and five-model results. |

**Do not interchange the two training datasets.** `deployed/train_DecisionTree.py` is the historical firmware trainer and reads `deployed/merged_log.CSV`. The academic DT scripts and DEEPCRAFT evaluator use `logs/csv_merge/0408_merge.csv` and [`session_partitions.csv`](academic_study/DecisionTree/session_partitions.csv).

## Windows setup

Open PowerShell in the extracted repository root. Use Python 3.12 and install NumPy, pandas, and Pillow in a virtual environment. Pillow is used for report images.

```powershell
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install numpy pandas Pillow
```

Keep the checkout path short. If Git for Windows reports a path-length error from the nested Studio exports, clone with `git clone -c core.longpaths=true <repository-url>` or move to a shorter parent directory.

## Use the scripts

Run these commands from the repository root in PowerShell after the setup above. The bundled data and predictions are already prepared. To reproduce the published comparison, skip the merge, preparation, and training steps and go straight to [Reproduce the comparison without retraining](#reproduce-the-comparison-without-retraining).

### Merge raw logs into a new dataset

Run the merger when you need one normalized CSV from multiple recordings. If you omit the input folders, it reads `logs/csv_log/legacy/` and `logs/csv_log/pse84/`. If you omit `--output-dir`, it writes to `logs/csv_merge/`. The example below keeps new outputs in a separate folder next to the project:

```powershell
& $py .\logs\csv_merge\merge_sen66_csv_logs.py --output-dir ..\sen66-merge-rerun
```

The script uses your computer's local date in `DDMM` format when you omit `--date-prefix`. A run on 16 November writes `1611_merge.csv` and `1611_merge_manifest.csv`. You can supply `--date-prefix` for a specific prefix or pass one or more raw-log folders as positional arguments to merge a different set. To keep multiple runs from the same day, use different output folders. The script refuses to replace an existing output unless you pass `--overwrite`.

Keep the merged CSV and its matching manifest together. The manifest maps each `dataset_id` to its source recording. The bundled `0408_merge.csv` and `0408_merge_manifest.csv` are the historical study inputs. Leave them in place when creating a new dataset.

### Check the input filename before running the next script

**A new merge does not automatically become the input to the next script.** Copy the exact output path printed by the merger. For a new run, check the trainer's `--csv` argument or the script's documented input setting. Don't assume the script will find the newest date-prefixed file.

| Next step | Which input to use |
| --- | --- |
| Repeat the academic Decision Tree training on the study data | Omit `--csv` to use the bundled `logs/csv_merge/0408_merge.csv` and its frozen partition manifest. |
| Train an academic Decision Tree on a different merge | Pass `--csv` with the exact path to your dated `*_merge.csv`, not the `*_merge_manifest.csv`. Use a separate experiment copy with a partition manifest created for that dataset. |
| Rebuild the historical firmware tree | Use `deployed/merged_log.CSV` by default. If overriding it with `--csv`, provide a file with the deployed trainer's schema, including `source_file` and `uptime_ms`. The academic merge is not a drop-in replacement. |
| Prepare DEEPCRAFT training sessions | Point `--source-dir` to a folder of original recordings, not to the merged CSV. |
| Reproduce the five-model comparison | Keep the evaluator on the bundled historical CSV, its matching merge manifest, frozen partitions, and saved predictions. In `evaluate_deepcraft_models.py`, `SOURCE_CSV` and `MERGE_MANIFEST` point to the historical pair. `--output-dir` only changes where results are written. A different study dataset requires matching partitions, Decision Tree results, and Studio predictions. |

For example, in a **separate experiment copy**, move that copy's `academic_study/DecisionTree/session_partitions.csv` out of the Decision Tree folder before the first training run on different data. The first trainer creates a new split. Use the same merged CSV and split for subsequent academic trainers in that experiment. Replace `1611_merge.csv` below with the actual filename printed by the merger and adjust the path if the experiment copy is in a different location.

```powershell
& $py .\academic_study\DecisionTree\DT0\train_DT0.py --csv "..\sen66-merge-rerun\1611_merge.csv" --archive-existing
```

### Prepare recordings for DEEPCRAFT Studio

Use the wrapper that matches the recording format. By default, the legacy and PSE84 wrappers read their respective folders under `logs/csv_log/`. Use `--source-dir` to point to a folder containing new recordings of that format. For example:

```powershell
& $py academic_study\DEEPCRAFT_data_prep\prepare_deepcraft_legacy.py --source-dir ..\new-legacy-logs
& $py academic_study\DEEPCRAFT_data_prep\prepare_deepcraft_pse84.py --source-dir ..\new-pse84-logs
```

Run the command for the format you have. For each new recording, the script creates a numbered session folder with `sensor.csv` and `label.label` under `academic_study/DEEPCRAFT_data_prep/data/` and updates `processing_manifest.csv`. The script skips recordings whose raw-file hashes are already in that manifest. Use a working copy if you want to keep the original prepared sessions and manifest as a baseline. These steps prepare data for import into Studio but do not train a model.

The optional `academic_study/DEEPCRAFT_data_prep/optional/process_new_deepcraft_csv.py` wrapper is for split CSVs in the newer format that include prediction fields. Supply their folder with `--source-dir` or place them under `academic_study/DEEPCRAFT_data_prep/split/`.

### Rebuild the historical firmware decision tree

```powershell
& $py deployed\train_DecisionTree.py --output-dir ..\sen66-deployed-tree-rerun
```

This trains on `deployed/merged_log.CSV` and writes fresh C/H exports, a tree description, and a feature list to the named rerun directory. Without `--output-dir`, it writes to `deployed/reruns/DecisionTree/`. Use a new or empty directory. The trainer does not flash the board or copy the output into the firmware repository.

### Retrain the academic Decision Tree models

Use a **separate copy of the full project** for academic retraining. These trainers write to their own `DT0/`, `DT1/`, or `DT2/` folders and can update shared comparison CSVs under `academic_study/DecisionTree/`. They do not accept an output-directory override. From that copy, with `$py` pointing to your Python environment, run the models you want to retrain:

```powershell
& $py academic_study\DecisionTree\DT0\train_DT0.py --archive-existing
& $py academic_study\DecisionTree\DT1\train_DT1.py --archive-existing
& $py academic_study\DecisionTree\DT2\train_DT2.py --archive-existing
```

Each trainer uses the canonical merged CSV and the frozen partitions by default. `--archive-existing` moves existing per-model outputs to a timestamped `runs/` folder before writing new ones. It does not isolate the shared comparison tables, which is why the separate copy matters. The new outputs include the trained model, predictions, metrics, and run information. Leave `--regenerate-partitions` off when reproducing the study.

## Reproduce the comparison without retraining

Run the evaluator against the bundled Studio `.data` prediction exports. Use a new or empty output directory outside the packaged `DEEPCRAFT_training/` results folder.

```powershell
& $py academic_study\DEEPCRAFT_training\evaluate_deepcraft_models.py --output-dir ..\sen66-deepcraft-rerun
```

The evaluator uses the canonical CSV and frozen partitions for ground truth, aligns the saved Studio outputs, and combines them with the bundled DT results. It does **not** load the H5 weights for inference, retrain the models, or require a Studio install. A baseline rerun produces 27,248 aligned Instant rows and 26,720 aligned Time rows across the exported sessions. For fresh Decision Tree figures and tables from saved outputs, use a separate report folder:

```powershell
& $py academic_study\DecisionTree\generate_reports.py --report-dir ..\sen66-dt-reports
```

Do not use `--regenerate-missing` for this workflow. It can invoke a trainer.

## Preserve the baseline

Keep [.gitattributes](.gitattributes) with the repository. Git must preserve the original bytes of the canonical CSV and raw logs. The DT tooling checks the canonical CSV against the frozen partition hash, and DEEPCRAFT data preparation uses raw-log hashes to recognize existing recordings. Do not regenerate the canonical merge or the partitions to clear a hash mismatch. Keep new outputs separate from bundled models, predictions, manifests, and reports.

The bundled Studio `.improj` files still contain session links from the original project layout. Reopening the projects with their training data in Studio requires relinking to `academic_study/DEEPCRAFT_data_prep/data/`. This is separate from rerunning the saved-prediction evaluator.

## Companion repositories

- [PSoC Edge E84 Air Quality firmware](https://github.com/Jifsopeon/kit_pse84_ai_iaq/tree/master) — sensor acquisition, deployed environmental inference, the separate camera model, and BLE publishing.
- [PSE84 Air Quality Monitor](https://github.com/Jifsopeon/BLE_APP/tree/master) — live monitoring, manual labels, and CSV recording.
