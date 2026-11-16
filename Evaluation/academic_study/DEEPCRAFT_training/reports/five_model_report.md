# SEN66 Five-Model Evaluation Report

Generated: 2026-09-11T08:16:20.439796+00:00

No Decision Tree model was retrained or regenerated. DEEPCRAFT metrics use saved trained-model prediction artifacts aligned to the frozen DecisionTree session partitions.

This reduced package focuses on broad classification comparison: accuracy, balanced accuracy, Smoking precision/recall/F1, and confusion-matrix counts. Detailed continuous-session deployment metrics were removed from the submission package.

The DEEPCRAFT windows were verified from project metadata as 6x7 for Instant and 30x7 for Time. Saved prediction timestamps are aligned to the source row at the trailing window end, and the alignment audit confirms windows remain within their source sessions.

## Native Test Comparison

| model | native_test_row_count | native_test_accuracy | native_test_balanced_accuracy | native_test_smoking_precision | native_test_smoking_recall | native_test_smoking_f1 | native_test_confusion_tn | native_test_confusion_fp | native_test_confusion_fn | native_test_confusion_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DT0 | 3419 | 0.814858 | 0.835027 | 0.703664 | 0.946429 | 0.807189 | 1461 | 558 | 75 | 1325 |
| DT1 | 3419 | 0.815736 | 0.833471 | 0.709467 | 0.931429 | 0.805435 | 1485 | 534 | 96 | 1304 |
| DT2 | 3140 | 0.816561 | 0.824449 | 0.736093 | 0.871363 | 0.798036 | 1426 | 408 | 168 | 1138 |
| DeepcraftInstant | 3404 | 0.76322 | 0.728289 | 0.832215 | 0.531429 | 0.648649 | 1854 | 150 | 656 | 744 |
| DeepcraftTime | 3332 | 0.77611 | 0.801625 | 0.660451 | 0.961429 | 0.783013 | 1240 | 692 | 54 | 1346 |

## Common Test Subset Comparison

| model | common_eligible_test_row_count | common_subset_test_accuracy | common_subset_test_balanced_accuracy | common_subset_test_smoking_precision | common_subset_test_smoking_recall | common_subset_test_smoking_f1 | common_subset_test_confusion_tn | common_subset_test_confusion_fp | common_subset_test_confusion_fn | common_subset_test_confusion_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DT0 | 3140 | 0.833439 | 0.852456 | 0.725129 | 0.965544 | 0.828243 | 1356 | 478 | 45 | 1261 |
| DT1 | 3140 | 0.833121 | 0.849868 | 0.730271 | 0.949464 | 0.825566 | 1376 | 458 | 66 | 1240 |
| DT2 | 3140 | 0.816561 | 0.824449 | 0.736093 | 0.871363 | 0.798036 | 1426 | 408 | 168 | 1138 |
| DeepcraftInstant | 3140 | 0.743949 | 0.708723 | 0.812968 | 0.499234 | 0.618596 | 1684 | 150 | 654 | 652 |
| DeepcraftTime | 3140 | 0.801274 | 0.827235 | 0.68119 | 0.981623 | 0.804266 | 1234 | 600 | 24 | 1282 |

## Limitations

The package uses exported DEEPCRAFT prediction artifacts because TensorFlow/Keras/h5py were not available in the evaluation runtime for direct `.h5` inference. The source data were collected under uncontrolled environmental conditions, so these results should support broad model comparison rather than fine-grained deployment claims.
