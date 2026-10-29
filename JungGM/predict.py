"""Recreate an OIBC submission from a completed local run, without fitting.

Example:
  py predict.py --data DATA --run _local/paper_v2_seasonal --out REPRODUCTION

Only supplied local files and saved model weights are read. No network calls.
The output must be new or empty; existing submissions are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import features as ft
import pipeline as p
from validate_submission import validate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def overlaps(a, b):
    """Paths are already resolved, including any symlink/junction target."""
    return a == b or a in b.parents or b in a.parents


def check_paths(data, run, cache, out):
    if not data.is_dir() or not run.is_dir():
        raise ValueError("--data and --run must be existing directories")
    for protected in (data, run, cache):
        if overlaps(out, protected):
            raise ValueError("--out must not overlap the supplied dataset, saved run, or cache")
    if overlaps(cache, data) or overlaps(cache, run):
        raise ValueError("--cache must be separate from the supplied dataset and saved run")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("--out must be a new or empty directory; refusing to overwrite files")


def verify_sources(data, run):
    protocol = read_json(run/"protocol_before_training.json")
    complete = read_json(run/"completion.json")
    integrity = read_json(run/"source_integrity.json")
    if complete.get("format_validation") != "PASS" or integrity.get("unchanged") is not True:
        raise ValueError("A successfully completed run with source-integrity verification is required")
    expected = protocol.get("source_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Run protocol has no original source hashes")
    actual = p.source_manifest(data)
    different = sorted(set(expected) ^ set(actual) | {name for name in set(expected) & set(actual)
                                                   if expected[name] != actual[name]})
    if different:
        raise ValueError("Original source SHA256 mismatch: " + ", ".join(different))
    if integrity.get("source_sha256") != expected:
        raise ValueError("Run source-integrity record disagrees with its initial protocol")
    # These files define inference semantics; changes require an explicit new run.
    for name in ("pipeline.py", "features.py", "validate_submission.py"):
        expected_code = protocol.get("code_sha256", {}).get(name)
        if expected_code is None or p.sha256(p.HERE/name) != expected_code:
            raise ValueError("Inference source changed since model training: " + name)
    return protocol, actual


def verify_schema(current, saved):
    # A byte-identical dataset copied to another folder may have new mtimes;
    # original bytes are checked separately against the run protocol.
    keys = ("version", "base10", "base17", "enhanced", "categorical", "units", "code_sha256")
    if any(current.get(key) != saved.get(key) for key in keys):
        raise ValueError("Cached feature schema differs from saved model schema")


def load_models(run, schema, weights):
    """Load active serialized estimators and verify their feature order.

    Returns ([(weight, fitted_model, config), ...], {relative_file: sha256}).
    LightGBM Boosters have the same predict interface used by predict_one.
    """
    directory = Path(run)/"production"/"models"
    configs = {config["name"]: config for config in p.CONFIGS}
    if set(weights) != set(configs):
        raise ValueError("Frozen ensemble names differ from the declared pipeline configurations")
    active, hashes = [], {}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        meta_path = directory/f"{name}.json"
        metadata = read_json(meta_path)
        config = configs[name]
        if metadata.get("config") != config:
            raise ValueError("Saved model configuration differs from pipeline: " + name)
        verify_schema(schema, metadata.get("schema", {}))
        origin = pd.Timestamp(metadata["origin"])
        cutoff = pd.Timestamp(metadata["cutoff"])
        if origin != p.stamp("2025-06-01 01:00") or cutoff != p.issue_deadline(origin):
            raise ValueError("Saved production model has an unexpected issue deadline")
        if pd.Timestamp(metadata["last_target"]) > cutoff or metadata["training_rows"] <= 0:
            raise ValueError("Saved training metadata violates the prediction deadline")
        paths = [meta_path]
        if config["kind"] == "baseline":
            fitted = {}
            for r in (10, 17):
                path = directory/f"{name}_r{r}.txt"
                model = lgb.Booster(model_file=str(path))
                if model.feature_name() != schema[f"base{r}"]:
                    raise ValueError("Saved baseline feature order mismatch")
                fitted[r] = model
                paths.append(path)
        elif config["kind"] == "lgb":
            path = directory/f"{name}.txt"
            fitted = lgb.Booster(model_file=str(path))
            if fitted.feature_name() != schema["enhanced"]:
                raise ValueError("Saved LightGBM feature order mismatch")
            paths.append(path)
        elif config["kind"] == "xgb":
            path = directory/f"{name}.ubj"
            fitted = xgb.XGBRegressor(n_jobs=p.THREADS)
            fitted.load_model(path)
            if fitted.get_booster().feature_names != schema["enhanced"]:
                raise ValueError("Saved XGBoost feature order mismatch")
            paths.append(path)
        else:
            raise ValueError("Unsupported saved model kind")
        for path in paths:
            hashes[str(path.relative_to(run))] = p.sha256(path)
        active.append((float(weight), fitted, config))
    if not active:
        raise ValueError("Frozen ensemble has no active estimators")
    return active, hashes


def reproduce(data, run, cache, out):
    data, run, cache, out = [Path(path).resolve() for path in (data, run, cache, out)]
    check_paths(data, run, cache, out)
    protocol, sources = verify_sources(data, run)
    frozen_path = run/"frozen_choice_before_may_audit.json"
    frozen = read_json(frozen_path)
    weights = frozen.get("weights", {})
    values = np.asarray(list(weights.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Frozen ensemble weights must be finite, nonnegative and sum to one")
    sites = pd.read_csv(data/"sites.csv").sort_values("site_id").reset_index(drop=True)
    assignments = np.asarray(frozen.get("groups", []))
    if len(sites) != 253 or sites.site_id.duplicated().any() or assignments.shape != (253,) or not np.isin(assignments, [0, 1]).all():
        raise ValueError("Invalid frozen 253-site grouping")
    expected = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(data/"smp.csv").ts)).tz_convert(p.TZ)
    if len(expected) != 8760 or expected.has_duplicates or not expected.equals(pd.date_range(
            p.stamp("2025-06-01 01:00"), p.stamp("2026-06-01 00:00"), freq="h")):
        raise ValueError("Official target timestamps differ from the submission contract")
    schema = ft.ensure_cache(data, cache, "2025-06", "2026-05", split="test")
    models, model_hashes = load_models(run, schema, weights)
    composition = pd.DataFrame({"site_id": sites.site_id,
        "group": np.where(assignments == 1, "G1", "G2")})
    capacities = sites.assign(group=composition.group).groupby("group").capacity_kw.sum()
    if len(capacities) != 2 or capacities.min() < 20000:
        raise ValueError("Frozen grouping violates the 20MW capacity constraint")
    group_map = composition.set_index("site_id").group
    frames = []
    site_rows = 0
    for month in pd.period_range("2025-06", "2026-05", freq="M"):
        start, end = ft.month_bounds(month)
        d = ft.load_rows(cache, start, end, split="test")
        times = expected[(expected >= start) & (expected <= end)]
        counts = d.groupby("site_id").ts.nunique()
        if len(d) != 253*len(times) or d.duplicated(["site_id", "ts"]).any() or set(d.site_id) != set(sites.site_id) or not counts.eq(len(times)).all() or set(d.ts) != set(times):
            raise ValueError("Incomplete or duplicate target site/hour grid")
        prediction = np.zeros(len(d), dtype=float)
        for weight, model, config in models:
            prediction += weight*p.predict_one(model, d, schema, config)
        z = d[["site_id", "ts"]].copy()
        z["group"] = z.site_id.map(group_map)
        z["forecast_kwh"] = prediction
        frames.append(z.groupby(["group", "ts"], as_index=False).forecast_kwh.sum())
        site_rows += len(d)
        del d, z
        gc.collect()
        p.log(f"Reproduced saved models for {month}")
    forecast = pd.concat(frames, ignore_index=True).sort_values(["group", "ts"])
    if len(forecast) != 17520 or not np.isfinite(forecast.forecast_kwh).all() or (forecast.forecast_kwh < 0).any():
        raise ValueError("Invalid final grouped forecast")
    # Check again immediately before writing, including an initially empty dir.
    check_paths(data, run, cache, out)
    out.mkdir(parents=True, exist_ok=True)
    composition.to_csv(out/"composition.csv", index=False)
    forecast.to_csv(out/"forecast.csv", index=False)
    errors = validate(out/"composition.csv", out/"forecast.csv", data/"sites.csv")
    if errors:
        raise ValueError("Reproduced submission failed validation: " + "; ".join(errors))
    receipt = {"format_validation": "PASS", "inference_only": True, "refitted": False,
        "run": str(run), "source_sha256_verified": sources,
        "frozen_choice_sha256": p.sha256(frozen_path), "model_files_sha256": model_hashes,
        "weights": weights, "group_capacity_kw": capacities.to_dict(),
        "site_prediction_rows": site_rows, "composition_rows": len(composition), "forecast_rows": len(forecast),
        "target_start": expected.min(), "target_end": expected.max(),
        "fit_cutoff": protocol["final_fit_cutoff"],
        "files": {name: p.sha256(out/name) for name in ("composition.csv", "forecast.csv")},
        "inference_code_sha256": {name: p.sha256(p.HERE/name) for name in ("predict.py", "pipeline.py", "features.py", "validate_submission.py")},
        "no_server_submission": True,
        "warning": "Local format and saved-model reproduction only; official target-year accuracy is unknown."}
    p.save_json(out/"receipt.json", receipt)
    p.log(f"PASS: saved-model reproduction written to {out}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=p.HERE/"_local"/"cache")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.data, args.run, args.cache, args.out)


if __name__ == "__main__":
    main()
