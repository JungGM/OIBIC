# Improved standalone XGBoost

Final submission CSVs: composition.csv (253 rows), forecast.csv (17,520 rows). Original CSV contents preserved.
Prediction: 2025-06-01 01:00 through 2026-06-01 00:00 KST, 8,760 hours per group.
G1: 134 sites / 20,465.655 kW; G2: 119 sites / 22,189.715 kW. Original reference assignment retained.

## Model and training
Observed training labels: 2021-01-19 13:00 through 2025-05-31 17:00 KST. 2,189,472 nonnegative daytime rows.
Standalone XGBoost: 52 features, round17 forecasts only, 300 trees, depth8, learning_rate0.03, min_child_weight100, reg_lambda30, subsample0.8, colsample_bytree0.8, seed42, hist.
Normalized SMP*k*capacity sample weights; target clip(energy/capacity,0,1.2); nonnegative predictions; nighttime zero.
No LightGBM blending, no group calibration, no V1 prediction mixing. No observed weather or post-cutoff generation used. Arrival timestamps are unavailable; cutoff is enforced on label ts.

## Historical validation
Same model structure trained through 2024-12-31 17:00: Jan-May2025 complete-fleet 3,624-hour penalty 76,648,860 KRW versus V1 82,998,020 KRW; reduction 7.64977%.
These are historical validation results, not the unknown competition-period score of the final retrained model.

## Reproduce
Python3.12: install requirements.txt, set environment variable OIBC_DATA to the absolute competition data directory, then run python train_predict.py.
Existing forecast.csv is never overwritten. To rerun, copy code and feature_sets.json into a fresh directory first.
train_predict.py is the final entry point. run_experiment.py and baseline_reference.py provide the original feature functions; do not invoke their other experimental stages. LightGBM is an import compatibility dependency and is not fitted or blended.
The trained UBJ model remains preserved locally; this folder provides CSVs and retraining code.

## Checks
Official README checks passed: columns, row counts, timezone +09:00, timestamps, group capacity, no duplicates/missing/negative forecasts. Reloaded model prediction difference:0.
See training_manifest.json and validation_report.json for settings and original CSV SHA256 hashes.
Only GitHub upload was requested. No competition submission was performed.
