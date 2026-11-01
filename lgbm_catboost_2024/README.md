# LightGBM + CatBoost: common 2024 experiment

Status: full 2022–2023 refit, sealed 2024 inference and original common scoring completed locally. This folder contains the reproducible code, frozen settings and aggregate results.

Train: 2022-01-01 01:00 through 2023-12-31 17:00 KST.
Forecast: 2024-01-01 01:00 through 2025-01-01 00:00 KST (17,568 group rows).
Use reference A→G1/B→G2 for the initial model comparison.


## 모델과 결과

2022~2023년 제공 자료로 학습한 LightGBM + CatBoost의 50:50 혼합이다. 각 모델은 160개 특징을 사용하고 설비용량당 발전량을 예측한다. 신규 발전소를 포함해 253개소의 2024년 8,784시간을 예측했다.

공통 로컬 채점 정산금은 **107,603,178.80원**, 벌금은 **2,189,990.90원**이다. 실적이 모두 있는 **178시간만 채점한 값**이며 연간 전체 또는 공식 서버 점수가 아니다. [집계 결과](results/2024/README.md), [특징·수식](docs/features_and_formulas.md), [선택·최종 확인](docs/selection_and_holdout.md)을 참고한다.

원본 데이터·정답·모델 바이너리·예측 CSV는 로컬에 보관한다. 아래 명령의 `<...>`는 실제 로컬 경로로 바꿔 실행한다. 전체 재현은 제공 자료를 준비한 뒤 환경 설치 → prepare_training → 개발 검증/선택 또는 고정 설정 사용 → 전체 재학습 → predict_2024 → 공통 채점 순서다.

## Environment

Python 3.12. Create a virtual environment and install `requirements.txt` (pinned versions).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Activate the environment or use its Python executable for all commands below. On Linux/macOS use `.venv/bin/python`.
Data stays outside this repository. Provide its directory via `--data` to the scorer.
The source data layout is sites.csv, train/generation.parquet, train/smp.csv and weather parquet files.

## Common scorer

The original scorer and team rules are in scoring/. source.json records source commit and SHA256.

```powershell
python scoring/score_2024.py --data <DATA_ROOT> --composition <LOCAL_COMPOSITION> --forecast <LOCAL_FORECAST> --out <LOCAL_SCORE_JSON>
```

Create the output parent directory before scoring. The recorded run has already produced and sealed its predictions locally; this command documents how to reproduce scoring.
Only complete fleet hours are scored (178 in the audited data), not all annual hours.

## Validation contract

Train the development model through 2022-12-31 17:00. Select features and blend weights using 2023 development labels available through 2023-09-30 17:00. Freeze choices before the October holdout. Holdout labels end at the common 2023-12-31 17:00 cutoff. Refit on the common full training interval only after settings are frozen. Add whole-site exclusion validation for unseen sites. Forecast issues must be available by the previous day 17:00; ending-hour dates define target days.

The final selection retains 160 input columns for each model. The 104-column LightGBM alternative was evaluated during development. Missing truth is never zero-filled. Actual 2024 generation/weather and target-day actual SMP are excluded from feature/parameter selection. Historical SMP is a training loss weight only.

## Publication

Follow the common team rules: source data, ground truth, fitted models and forecast CSVs stay local. Publish code, configuration, aggregate results and file hashes. No competition data is sent to external AI services.

## Feature generation

```powershell
python scripts/build_features.py --data <DATA_ROOT> --out <LOCAL_CACHE> --start "2022-01-01 01:00" --end "2023-12-31 17:00"
```

Reads only sites and train forecast files; no generation, observed weather or SMP. Writes monthly per-site shards and a local manifest. Use a fresh output directory per version/date range (same-named shards are replaced). Optional --site-limit is for smoke runs. Full physical target days are processed before cropping requested hours. Six solar midpoint samples per hour; SPA sunrise/sunset. Round issue availability follows the supplied data contract, since separate issue timestamps are absent. Invalid negative precipitation/wind becomes NaN; humidity/cloud clipped to 0–100. Incomplete temporal windows remain NaN. Model training must separately filter labels to configured cutoffs. No global test statistics are fitted.

## Development training (stage 4)

```powershell
python scripts/prepare_training.py --data <DATA_ROOT> --out <LOCAL_TRAINING_CACHE>
python scripts/train_validate.py --cache <LOCAL_TRAINING_CACHE> --out <LOCAL_DEVELOPMENT_RESULTS>
```

The materializer stores only observed nonnegative labels in the common 2022–2023 interval; no missing labels are fabricated. It expands forecast context to full issued days independently of label cutoffs. Source hashes and feature-engine hash are recorded. Historical SMP weights are joined only to their training/validation rows. Exactly-zero generation is preserved. Model training ends at 2022-12-31 17:00 KST. Development validation ends at 2023-09-30 17:00 KST; October–December rows are filtered at parquet read time.

Both models use up to 600 trees, early stopping on development loss, seed 20261101 and four threads. The first blend is fixed at 50:50 for diagnosis; stage 5 selects weights/features. Metrics include seen/unseen sites, daylight, peak hours, and month. These are site-level diagnostics and not the official full-fleet group score.

A second run hides 20 percent of training sites using a deterministic hash. Early stopping uses validation from retained training sites; excluded-site labels are evaluated only after fitting. Models and per-site predictions stay local. Public results should contain aggregate metrics only.

## Stage 5: frozen selection and final confirmation

```powershell
python scripts/select_and_holdout.py --cache <LOCAL_TRAINING_CACHE> --development <LOCAL_DEVELOPMENT_RESULTS> --data <DATA_ROOT> --out <FRESH_LOCAL_SELECTION_RESULTS>
```

The stage-5 run has already evaluated the held-out October–December period once. Do not use a repeat run to tune to this period. The selected public configuration is configs/final_model.json: both models 160 columns, blend 50:50, LightGBM 484 trees and CatBoost 600 trees for the full refit. Development ablation tested LightGBM104 versus LightGBM160 with the existing CatBoost160. It did not independently ablate all six added groups or train CatBoost104. Blend weights were selected on complete fixed-cohort group loss with overall/unseen site-loss guards; near-ties within 0.1 percent favor 50:50. This is a partial-cohort diagnostic, not the official full-fleet score. Frozen model hashes and settings are written before reading the final holdout. The confirmation uses models trained on 2022 only; the completed full refit subsequently used the full 2022–2023 interval.

## Full refit and 2024 inference (stage 6)

Run sequentially on memory-limited machines; do not fit both models concurrently.

```powershell
python scripts/refit_final.py --cache <LOCAL_TRAINING_CACHE> --out <FRESH_LOCAL_FINAL_MODELS> --model lightgbm
python scripts/refit_final.py --cache <LOCAL_TRAINING_CACHE> --out <FRESH_LOCAL_FINAL_MODELS> --model catboost --prepare-cat-pool
python scripts/refit_final.py --cache <LOCAL_TRAINING_CACHE> --out <FRESH_LOCAL_FINAL_MODELS> --model catboost
python scripts/predict_2024.py --data <DATA_ROOT> --models <LOCAL_FINAL_MODELS> --out <FRESH_LOCAL_2024_RESULTS>
```

The full refit uses all 1,444,607 observed nonnegative 2022–2023 labels in the common interval, with no early stopping or parameter changes. Each model records configuration/cache hashes, training bounds, features and model hash. CatBoost's pool is prequantized at the CPU default 254 borders in a separate process, saved to disk, then loaded into a fresh training process to release raw-matrix memory. The --prepare-cat-pool command is required before the CatBoost refit command. Inference reads only site metadata and the 2024 forecast subset from the provided train weather files, not generation, observed weather or SMP. It creates each site's features independently, clips each component prediction below at zero, blends at the frozen 50:50 ratio and sums to reference groups. All 253 sites are predicted, including 105 sites without 2022–2023 training labels.

Output composition.csv and forecast.csv are local only. The script reloads and checks exact columns, all site/group-hour keys, nonnegative finite values, explicit KST and both group capacities. prediction_manifest.json seals file/model/config hashes. The common scorer is a separate next step and must use the already sealed prediction, without tuning to its result. No final model or forecast file should be added to public GitHub.

Provenance note: `model_sha256` inside configs/final_model.json identifies the selection-stage source models. Full-refit hashes are separately recorded in local lightgbm_run.json/catboost_run.json and prediction_manifest.json. The frozen configuration file is not changed after the full refit.

## Common 2024 score (stage 7)

Settlement 107,603,178.80 KRW; penalty 2,189,990.90 KRW. Only 178 complete-fleet hours were scored, out of 8,784 forecast hours. This is a local common score, not an annual or official server score. See results/2024 for aggregate results and hashes. Predictions remain local and unchanged after scoring.
