"""Paper-informed, local-only OIBC production training and submission builder.

All decisions precede the next forecast's 17:00 issue deadline. No network code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import features as ft
import grouping
from validate_submission import validate

TZ = "Asia/Seoul"
SEED = 20261029
THREADS = 6
HERE = Path(__file__).resolve().parent
T0 = time.time()
# Finite, declared search; this is not an exhaustive or global optimum claim.
CONFIGS = [
    dict(name="official_lgb", kind="baseline", history_months=24),
    dict(name="lgb_l2", kind="lgb", history_months=24, cap_weight=.5,
         objective="regression", n_estimators=1000, num_leaves=63, min_child_samples=160),
    dict(name="lgb_l2_wide", kind="lgb", history_months=24, cap_weight=.5,
         objective="regression", n_estimators=1000, num_leaves=127, min_child_samples=200),
    dict(name="lgb_l1", kind="lgb", history_months=24, cap_weight=1.,
         objective="regression_l1", n_estimators=1000, num_leaves=63, min_child_samples=160),
    dict(name="lgb_recent", kind="lgb", history_months=12, cap_weight=.5,
         objective="regression", n_estimators=850, num_leaves=63, min_child_samples=160),
    dict(name="xgb_regular", kind="xgb", history_months=24, cap_weight=.5,
         n_estimators=800, max_depth=6, min_child_weight=20, reg_lambda=10),
    dict(name="xgb_deep", kind="xgb", history_months=24, cap_weight=.5,
         n_estimators=650, max_depth=8, min_child_weight=30, reg_lambda=15),
]


def log(message):
    print(f"[{time.time()-T0:8.1f}s] {message}", flush=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=str, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(value):
    return pd.Timestamp(value, tz=TZ)


def issue_deadline(target):
    ts = pd.Timestamp(target)
    if ts.tz is None:
        raise ValueError("Timezone required")
    return (ts.tz_convert(TZ)-pd.Timedelta(hours=1)).normalize()-pd.Timedelta(hours=7)


def history_start(origin, months=24):
    return origin.normalize()-pd.DateOffset(months=months)+pd.Timedelta(hours=1)


def source_manifest(data):
    names = ["README.md", "baseline.py", "sites.csv", "smp.csv", "train/smp.csv",
             "train/generation.parquet", "train/forecast_wa.parquet", "train/forecast_wb.parquet",
             "test/forecast_wa.parquet", "test/forecast_wb.parquet"]
    return {name: sha256(data/name) for name in names}


def design(d, names):
    return d[names].astype("float32")


def fit_one(d, schema, config, origin, directory):
    """Only observed daylight targets before the issue deadline reach fitting."""
    name = config["name"]
    directory.mkdir(parents=True, exist_ok=True)
    d = d.loc[(d.ts >= history_start(origin, config["history_months"])) &
              (d.ts <= issue_deadline(origin)) & d.energy.notna() & (d.elev > 0)]
    if d.empty or not np.isfinite(d.energy).all():
        raise ValueError("No finite training targets")
    y = (d.energy/d.cap).clip(0, 1.2).astype("float32")
    log(f"Fitting {name}: {len(d):,} daylight rows; cutoff {issue_deadline(origin)}")
    if config["kind"] == "baseline":
        fitted = {}
        for r in [10, 17]:
            model = lgb.LGBMRegressor(n_estimators=600, learning_rate=.06, num_leaves=127,
                min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                verbosity=-1, n_jobs=THREADS, random_state=SEED, deterministic=True,
                force_col_wise=True)
            model.fit(design(d, schema[f"base{r}"]), y)
            model.booster_.save_model(str(directory/f"{name}_r{r}.txt"))
            fitted[r] = model
    else:
        excluded = {"name", "kind", "history_months", "cap_weight"}
        params = {k: v for k, v in config.items() if k not in excluded}
        weights = (d.cap.to_numpy()/d.cap.median())**config["cap_weight"]
        weights = weights/weights.mean()
        if config["kind"] == "lgb":
            fitted = lgb.LGBMRegressor(**params, learning_rate=.035, subsample=.85,
                subsample_freq=1, colsample_bytree=.85, reg_lambda=8, reg_alpha=.1,
                max_bin=127, verbosity=-1, n_jobs=THREADS, random_state=SEED,
                deterministic=True, force_col_wise=True)
            cats = [c for c in schema.get("categorical", []) if c in schema["enhanced"]]
            fitted.fit(design(d, schema["enhanced"]), y, sample_weight=weights,
                       categorical_feature=cats)
            fitted.booster_.save_model(str(directory/f"{name}.txt"))
        else:
            fitted = xgb.XGBRegressor(**params, learning_rate=.035, tree_method="hist",
                max_bin=128, subsample=.85, colsample_bytree=.85, reg_alpha=.1,
                n_jobs=THREADS, random_state=SEED)
            fitted.fit(design(d, schema["enhanced"]), y, sample_weight=weights)
            fitted.save_model(directory/f"{name}.ubj")
    save_json(directory/f"{name}.json", {"config": config, "training_rows": len(d),
        "origin": origin, "cutoff": issue_deadline(origin), "first_target": d.ts.min(),
        "last_target": d.ts.max(), "schema": schema,
        "target": "capacity-normalized utilization clipped [0,1.2]; no target imputation"})
    return fitted


def predict_one(model, d, schema, config):
    p = np.zeros(len(d), dtype="float64")
    day = d.elev.to_numpy() > 0
    if day.any():
        if config["kind"] == "baseline":
            u = sum(model[r].predict(design(d.loc[day], schema[f"base{r}"]))
                    for r in [10, 17])/2
        else:
            u = model.predict(design(d.loc[day], schema["enhanced"]))
        # Same nonnegative postprocessing for all candidates; no tuned clipping.
        p[day] = np.maximum(u, 0)*d.loc[day, "cap"].to_numpy()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prediction")
    return p


def panel(d, columns, site_ids, prices):
    """Require full truth for every retained hour. Never manufacture missing truth."""
    if d.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate panel keys")
    times = pd.DatetimeIndex(d.ts.unique()).sort_values()
    index = pd.MultiIndex.from_product([times, site_ids], names=["ts", "site_id"])
    z = d.set_index(["ts", "site_id"]).reindex(index)
    if z.energy.isna().any() or z[list(columns)].isna().any().any():
        raise ValueError("Complete fleet truth and predictions required for group scoring")
    y = z.energy.to_numpy(dtype=float).reshape(len(times), len(site_ids))
    pp = {c: z[c].to_numpy(dtype=float).reshape(y.shape) for c in columns}
    pr = prices.set_index("ts").reindex(times)
    w = (pr.smp*pr.k).to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(w).all() or (w < 0).any() or any(
            not np.isfinite(v).all() for v in pp.values()):
        raise ValueError("Invalid truth/price")
    return times, y, pp, w, pr.smp.to_numpy(dtype=float)


def group_sum(a, assignment):
    return np.column_stack([a[:, assignment == 1].sum(axis=1),
                            a[:, assignment == 0].sum(axis=1)])


def cost(actual, prediction, w, assignment):
    return float(np.sum(w[:, None]*np.abs(group_sum(prediction-actual, assignment))))


def select_ensemble(folds, names):
    """0.1 simplex grid, <=3 components, baseline-normalized chronological loss."""
    history = []
    bases = [max(float(f["losses"][0]), 1e-12) for f in folds]
    importance = np.asarray([f.get("importance", 1.) for f in folds], dtype=float)
    importance /= importance.sum()
    best = None
    for count in [1, 2, 3]:
        for components in itertools.combinations(range(len(names)), count):
            units = [(10,)] if count == 1 else [u for u in itertools.product(range(1, 10), repeat=count)
                                                if sum(u) == 10]
            for allocation in units:
                weights = np.zeros(len(names))
                weights[list(components)] = np.array(allocation)/10
                losses = []
                for f in folds:
                    # Group aggregation is linear, so ensemble search needs no site-level loop.
                    p = np.tensordot(weights, f["group_predictions"], axes=1)
                    losses.append(float(np.sum(f["w"][:, None]*np.abs(p-f["group_actual"]))))
                ratios = np.array(losses)/bases
                key = (float(ratios @ importance), float(ratios.max()), count)
                item = {"weights": dict(zip(names, weights.tolist())),
                        "mean_monthly_relative_loss": key[0], "monthly_relative_loss": ratios.tolist()}
                history.append(item)
                if best is None or key < best[0]:
                    best = (key, weights.copy(), item)
    return best[1], best[2], sorted(history, key=lambda x: x["mean_monthly_relative_loss"])


def metrics(times, actual, prediction, w, smp, assignment):
    error = prediction-actual
    gerror = group_sum(error, assignment)
    weighted_error = float(np.sum(w[:, None]*np.abs(gerror)))
    revenue = float(np.sum(smp*actual.sum(axis=1)))
    return {"hours": len(times), "site_mae_kwh": float(np.abs(error).mean()),
        "site_rmse_kwh": float(np.sqrt(np.mean(error**2))),
        "group_mae_kwh": float(np.abs(gerror).mean()), "weighted_error": weighted_error,
        "penalty_krw": weighted_error*.1, "revenue_krw": revenue,
        "settlement_krw": revenue-.1*weighted_error,
        "fleet_error_lower_bound": float(np.sum(w*np.abs(error.sum(axis=1))))}


def daily_losses(times, actual, prediction, w, assignment):
    loss = w*np.abs(group_sum(prediction-actual, assignment)).sum(axis=1)
    # Date denotes the right-closed operating day, including its final 00:00.
    day = (times-pd.Timedelta(hours=1)).normalize()
    return pd.Series(loss, index=day).groupby(level=0).sum()


def ensure_train_cache(args, start="2022-07", end="2025-05"):
    return ft.ensure_cache(args.data, args.cache, start, end, split="train")


def select_models(args, schema, sites, prices):
    folder = args.run/"selection"
    folder.mkdir()
    names = [c["name"] for c in CONFIGS]
    ref = reference(sites)
    folds, reports = [], []
    coverage_reports = []
    blocks = [("seasonal_proxy", stamp("2024-07-01 01:00"), stamp("2024-12-31 17:00"), list(range(7, 13))),
              ("complete_fleet", stamp("2025-01-01 01:00"), stamp("2025-02-28 17:00"), [1, 2])]
    for block, origin, end, months in blocks:
        train = ft.load_rows(args.cache, history_start(origin), issue_deadline(origin), labeled=True)
        valid = ft.load_rows(args.cache, origin, end)
        if block == "seasonal_proxy":
            coverage = valid.groupby("site_id").energy.count()/valid.ts.nunique()
            ids = sites.loc[sites.site_id.isin(coverage[coverage >= .95].index), "site_id"].tolist()
            if len(ids) < 50:
                raise ValueError("Insufficient stable cohort for seasonal proxy")
        else:
            ids = sites.site_id.tolist()
        for c in CONFIGS:
            model = fit_one(train, schema, c, origin, folder/block/"models")
            valid[c["name"]] = predict_one(model, valid, schema, c)
            del model
            gc.collect()
        valid[["site_id", "ts", "energy"]+names].to_parquet(
            folder/block/"out_of_time_predictions.parquet", index=False, compression="zstd")
        del train
        gc.collect()
        assignment = ref[sites.site_id.isin(ids).to_numpy()]
        for month in months:
            d = valid.loc[((valid.ts-pd.Timedelta(hours=1)).dt.month == month) & valid.site_id.isin(ids)]
            counts = d.groupby("ts").energy.count()
            complete = counts[counts == len(ids)].index
            d = d.loc[d.ts.isin(complete)]
            times, y, pp, w, smp = panel(d, names, ids, prices)
            if len(times) < 168 or np.sum(y.sum(axis=1) > 0) < 50:
                raise ValueError("Insufficient seasonal scoring hours")
            coverage_reports.append({"block": block, "month": month, "cohort_sites": len(ids),
                "hours": len(times), "capacity_kw": float(sites.loc[sites.site_id.isin(ids), "capacity_kw"].sum()),
                "scope": "partial fixed-cohort proxy" if block == "seasonal_proxy" else "all 253 sites"})
            losses = [cost(y, pp[n], w, assignment) for n in names]
            folds.append({"losses": losses, "group_predictions": np.array([group_sum(pp[n], assignment) for n in names]),
                "group_actual": group_sum(y, assignment), "w": w, "importance": .5/len(months)})
            for n in names:
                reports.append({"block": block, "month": month, "model": n,
                                **metrics(times, y, pp[n], w, smp, assignment)})
        del valid
        gc.collect()
    weights, selected, history = select_ensemble(folds, names)
    save_json(folder/"selection.json", {"selected": selected, "search_count": len(history),
        "top_30": history[:30], "monthly_candidates": reports, "coverage": coverage_reports,
        "block_weights": {"seasonal_partial_cohort": .5, "winter_complete_fleet": .5},
        "selection_end": "2025-02-28 17:00+09:00"})
    log(f"Selected ensemble: {selected}")
    return weights


def reference(sites):
    if set(sites.current_group) != {"A", "B"}:
        raise ValueError("Unexpected reference grouping")
    return (sites.current_group == "A").to_numpy(dtype=np.int8)


def development(args, schema, sites, prices, weights):
    folder = args.run/"development"
    folder.mkdir()
    origin = stamp("2025-03-01 01:00")
    train = ft.load_rows(args.cache, history_start(origin), issue_deadline(origin), labeled=True)
    active = [(i, c) for i, c in enumerate(CONFIGS) if weights[i] > 0 or i == 0]
    models = {i: fit_one(train, schema, c, origin, folder/"models") for i, c in active}
    del train
    gc.collect()
    frames = []
    for month in [3, 4, 5]:
        start = stamp(f"2025-{month:02d}-01 01:00")
        end = stamp(f"2025-{month+1:02d}-01 00:00")-pd.Timedelta(hours=7)
        d = ft.load_rows(args.cache, start, end)
        p = np.zeros(len(d))
        base = None
        for i, c in active:
            pred = predict_one(models[i], d, schema, c)
            p += weights[i]*pred
            if i == 0:
                base = pred
        z = d[["site_id", "ts", "energy"]].copy()
        z["ensemble"] = p
        z["official_lgb"] = base
        frames.append(z)
        log(f"Development predictions ready for {month:02d}")
    del models
    gc.collect()
    z = pd.concat(frames, ignore_index=True)
    z.to_parquet(folder/"site_predictions.parquet", index=False, compression="zstd")
    ref = reference(sites)
    march = z.loc[(z.ts-pd.Timedelta(hours=1)).dt.month == 3]
    times, y, pp, w, smp = panel(march, ["ensemble"], sites.site_id.tolist(), prices)
    candidate, search = grouping.optimize(pp["ensemble"]-y, w, sites.capacity_kw.to_numpy(), ref,
                                         seed=SEED, steps=args.group_steps, restarts=5)
    save_json(folder/"group_search.json", search)
    april = z.loc[(z.ts-pd.Timedelta(hours=1)).dt.month == 4]
    at, ay, ap, aw, asm = panel(april, ["ensemble"], sites.site_id.tolist(), prices)
    before = daily_losses(at, ay, ap["ensemble"], aw, ref)
    after = daily_losses(at, ay, ap["ensemble"], aw, candidate)
    gate = grouping.daily_gate(before.to_numpy(), after.to_numpy(), min_improvement=.01, bootstrap=2000)
    # The exact API key is deliberately explicit: no truthy dict as acceptance.
    accepted = bool(gate["accepted"])
    selected = candidate if accepted else ref
    save_json(folder/"group_gate.json", {"gate": gate, "accepted": accepted,
        "reference": metrics(at, ay, ap["ensemble"], aw, asm, ref),
        "candidate": metrics(at, ay, ap["ensemble"], aw, asm, candidate),
        "rule": ">=1% April loss improvement AND paired-day bootstrap lower90 >0; >=14days",
        "candidate_frozen_by": "2025-03-31 17:00+09:00",
        "gate_end": "2025-04-30 17:00+09:00"})
    # Freeze before looking at May metrics. Both earlier years were seen previously.
    save_json(args.run/"frozen_choice_before_may_audit.json", {
        "weights": dict(zip([c["name"] for c in CONFIGS], weights.tolist())),
        "groups": selected.tolist(), "group_candidate_accepted": accepted,
        "warning": "2024/2025 were inspected in prior work. May is frozen only within this run, not an untouched external holdout."})
    reports = []
    for month in [3, 4, 5]:
        d = z.loc[(z.ts-pd.Timedelta(hours=1)).dt.month == month]
        ts, y, pp, w, smp = panel(d, ["official_lgb", "ensemble"], sites.site_id.tolist(), prices)
        for label, pname, assignment in [("official_reference", "official_lgb", ref),
                ("ensemble_reference", "ensemble", ref), ("selected_system", "ensemble", selected)]:
            reports.append({"month": month, "model": label, **metrics(ts, y, pp[pname], w, smp, assignment)})
    save_json(folder/"monthly_metrics.json", reports)
    log(f"Group candidate accepted: {accepted}. May development audit saved.")
    return selected, reports


def production(args, schema, sites, weights, assignments):
    folder = args.run/"production"
    folder.mkdir()
    origin = stamp("2025-06-01 01:00")
    train = ft.load_rows(args.cache, history_start(origin), issue_deadline(origin), labeled=True)
    models = {}
    for i, c in enumerate(CONFIGS):
        if weights[i] > 0:
            models[i] = fit_one(train, schema, c, origin, folder/"models")
    del train
    gc.collect()
    ft.ensure_cache(args.data, args.cache, "2025-06", "2026-05", split="test")
    frames = []
    for month in pd.period_range("2025-06", "2026-05", freq="M"):
        start = month.start_time.tz_localize(TZ)+pd.Timedelta(hours=1)
        end = (month+1).start_time.tz_localize(TZ)
        d = ft.load_rows(args.cache, start, end, split="test")
        p = np.zeros(len(d))
        for i, model in models.items():
            p += weights[i]*predict_one(model, d, schema, CONFIGS[i])
        z = d[["site_id", "ts"]].copy()
        z["prediction_kwh"] = p
        frames.append(z)
        log(f"Production forecast ready {month}")
    p = pd.concat(frames, ignore_index=True)
    expected = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(args.data/"smp.csv").ts)).tz_convert(TZ)
    if len(p) != 253*len(expected) or set(p.ts) != set(expected):
        raise ValueError("Official target grid mismatch")
    p.to_parquet(folder/"site_predictions.parquet", index=False, compression="zstd")
    composition = pd.DataFrame({"site_id": sites.site_id, "group": np.where(assignments == 1, "G1", "G2")})
    p["group"] = p.site_id.map(composition.set_index("site_id").group)
    forecast = p.groupby(["group", "ts"], as_index=False).prediction_kwh.sum().rename(
        columns={"prediction_kwh": "forecast_kwh"}).sort_values(["group", "ts"])
    args.submission.mkdir(parents=True, exist_ok=True)
    for name in ["composition.csv", "forecast.csv"]:
        if (args.submission/name).exists():
            raise ValueError("Refusing to overwrite an existing submission; choose another --submission")
    composition.to_csv(args.submission/"composition.csv", index=False)
    forecast.to_csv(args.submission/"forecast.csv", index=False)
    errors = validate(args.submission/"composition.csv", args.submission/"forecast.csv", args.data/"sites.csv")
    if errors:
        raise ValueError(str(errors))
    cap = sites.assign(group=composition.group).groupby("group").capacity_kw.sum().to_dict()
    receipt = {"format_validation": "PASS", "composition_rows": len(composition),
        "forecast_rows": len(forecast), "site_prediction_rows": len(p),
        "target_start": expected.min(), "target_end": expected.max(),
        "fit_cutoff": issue_deadline(origin), "group_capacity_kw": cap,
        "files": {n: sha256(args.submission/n) for n in ["composition.csv", "forecast.csv"]},
        "model_source_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")},
        "run": str(args.run), "no_server_submission": True,
        "warning": "Local structural validation only; official target-year accuracy is unknown."}
    save_json(args.submission/"검증결과.json", receipt)
    save_json(args.run/"completion.json", receipt)
    log(f"PASS: official-period submission generated at {args.submission}")
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=HERE/"_local"/"cache")
    ap.add_argument("--run", type=Path, default=HERE/"_local"/"paper_v2_seasonal")
    ap.add_argument("--submission", type=Path, default=HERE/"제출")
    ap.add_argument("--group-steps", type=int, default=20000)
    args = ap.parse_args()
    args.data, args.cache, args.run, args.submission = [p.resolve() for p in
        [args.data, args.cache, args.run, args.submission]]
    if args.run.exists():
        raise ValueError("Use a new --run directory to preserve audit records")
    if args.data == args.run or args.data in args.run.parents:
        raise ValueError("Do not write into the supplied data directory")
    args.run.mkdir(parents=True)
    sites = pd.read_csv(args.data/"sites.csv").sort_values("site_id").reset_index(drop=True)
    if len(sites) != 253 or sites.site_id.duplicated().any():
        raise ValueError("Expected 253 unique sites")
    prices = pd.read_csv(args.data/"train/smp.csv")
    prices["ts"] = pd.DatetimeIndex(pd.to_datetime(prices.ts)).tz_convert(TZ)
    prices["k"] = np.where(prices.ts.dt.hour.between(11, 15), 3, 1)
    manifest = source_manifest(args.data)
    save_json(args.run/"protocol_before_training.json", {"configs": CONFIGS,
        "seed": SEED, "threads": THREADS, "source_sha256": manifest,
        "model_selection": "two chronological origins: train through Jun30 2024 17 for Jul-Dec2024 stable >=95% availability cohort proxy; train through Dec31 2024 17 for Jan-Feb2025 full253; equal 0.5 weight per block, equal monthly relative baseline loss within block; <=3-component 0.1 simplex ensemble",
        "group_selection": "models refit through Feb28 17; search March through31 17; gate April through30 17; then freeze",
        "audit": "May through31 17, previously seen historical data, never claim an untouched test",
        "final_fit_cutoff": "2025-05-31 17:00+09:00", "no_external_data": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__, "xgboost": xgb.__version__},
        "code_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")}})
    schema = ensure_train_cache(args)
    weights = select_models(args, schema, sites, prices)
    assignments, reports = development(args, schema, sites, prices, weights)
    receipt = production(args, schema, sites, weights, assignments)
    if source_manifest(args.data) != manifest:
        raise ValueError("Source files changed during this run")
    save_json(args.run/"source_integrity.json", {"unchanged": True, "source_sha256": manifest})
    log(f"Completed in {(time.time()-T0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
