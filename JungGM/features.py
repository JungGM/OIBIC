"""Forecast-only, local monthly feature cache for the OIBC 2026 dataset.

The solar approximation is adapted from the organizer's supplied baseline.py.
Only supplied anonymized metadata and forecasts issued by the prior-day 17:00
deadline enter X. Generation is a separate optional label, never a feature.
All timestamps label the end of the preceding one-hour operating interval.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
VERSION = 1
WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press",
      "wa_wind_speed", "wa_wind_dir", "wa_rain", "wa_snow"]
WB = ["wb_temp", "wb_feel_temp", "wb_wet_bulb_temp", "wb_dew_point",
      "wb_wind_speed", "wb_wind_dir", "wb_wind_gust_spd", "wb_rel_hum",
      "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_rain_prob",
      "wb_snow_prob", "wb_ice_prob", "wb_total_liq", "wb_rain", "wb_snow",
      "wb_ice", "wb_cloud"]
BASE_WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain",
           "wa_wind_speed", "wa_ground_press"]
BASE_WB = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",
           "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]
GEOMETRY = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon"]
CATEGORICAL = ["site_index", "wa_station_index", "wb_station_index"]
REVISION = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed",
            "wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_total_liq",
            "wb_precip_prob", "wb_vis"]
SMOOTH = ["wa_cloud", "wb_cloud", "wb_uv_idx"]


def _timestamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)


def month_bounds(month):
    """Return inclusive bounds for a right-closed operating month."""
    m = pd.Period(month, freq="M")
    return (_timestamp(m.start_time) + pd.Timedelta(hours=1),
            _timestamp((m + 1).start_time))


def _months(start, end):
    def period(value):
        if isinstance(value, str) and len(value) == 7:
            return pd.Period(value, freq="M")
        return (_timestamp(value) - pd.Timedelta(hours=1)).tz_localize(None).to_period("M")
    a, b = period(start), period(end)
    if a > b:
        raise ValueError("start must precede end")
    return pd.period_range(a, b, freq="M")


def solar(ts, lat, lon):
    """Organizer's NOAA approximation at the hour midpoint; no external data."""
    ts = pd.DatetimeIndex(ts)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                       - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
            - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
            - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    tst = (hour - 0.5) * 60 + eqtime + 4 * (np.asarray(lon) - 135.0)
    ha, la = np.radians(tst / 4.0 - 180.0), np.radians(lat)
    cosz = np.clip(np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha), -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha),
                      np.tan(decl)*np.cos(la) - np.sin(la)*np.cos(ha))) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")


def issued_day_mean(values, window):
    """Trailing forecast-trajectory mean, resetting after each 24h target day.

    Input must be site-major, then operating-day-major, with exactly 24 hours
    per day starting at 01:00. No observed weather or future labels are read.
    """
    x = np.asarray(values, dtype=np.float32)
    if window < 1 or window > 24 or x.size % 24:
        raise ValueError("Expected full 24h forecast days and window in [1,24]")
    a = x.reshape(-1, 24)
    finite = np.isfinite(a)
    sums = np.pad(np.cumsum(np.where(finite, a, 0), axis=1, dtype=np.float64), ((0, 0), (1, 0)))
    counts = np.pad(np.cumsum(finite, axis=1), ((0, 0), (1, 0)))
    end = np.arange(1, 25)
    start = np.maximum(0, end-window)
    n = counts[:, end] - counts[:, start]
    return np.divide(sums[:, end]-sums[:, start], n,
             out=np.full(a.shape, np.nan), where=n > 0).astype("float32").ravel()


def feature_schema():
    raw = [f"r{r}_{c}" for r in (10, 17) for c in WA+WB]
    extra = CATEGORICAL + ["wa_dist", "wb_dist", "hour_sin", "hour_cos", "cosz_positive"]
    extra += [f"revision_{c}" for c in REVISION]
    for r in (10, 17):
        extra += [f"r{r}_provider_{c}_difference" for c in ("cloud", "temp", "humidity", "wind")]
        extra += [f"r{r}_{c}_mean{w}" for c in SMOOTH for w in (3, 7)]
        extra += [f"r{r}_{c}_daily_mean" for c in SMOOTH]
        extra += [f"r{r}_{provider}_wind_{trig}" for provider in ("wa", "wb") for trig in ("sin", "cos")]
        extra += [f"r{r}_{provider}_cloud_solar" for provider in ("wa", "wb")]
        extra += [f"r{r}_weather_missing"]
    schema = {"version": VERSION,
              "base10": [f"r10_{c}" for c in BASE_WA+BASE_WB] + GEOMETRY,
              "base17": [f"r17_{c}" for c in BASE_WA+BASE_WB] + GEOMETRY,
              "enhanced": raw + GEOMETRY + extra,
              "categorical": CATEGORICAL,
              "units": {"wind": "m/s for both providers and rounds", "snow": "mm for both providers and rounds"},
              "labels": "energy is optional, never in X; absent original rows stay NaN",
              "temporal_features": "supplied issued forecast trajectory only; reset within each 01:00..24:00 operating day",
              "solar_source": "adapted from supplied official baseline.py solar()"}
    if len(schema["enhanced"]) != len(set(schema["enhanced"])) or "energy" in schema["enhanced"]:
        raise RuntimeError("Invalid feature schema")
    return schema


def build_month(sites, fa, fb, index, generation=None):
    """Build a complete site/hour grid. Public for synthetic contract tests."""
    index = pd.DatetimeIndex(index).tz_convert(TZ)
    if not len(index) or len(index) % 24 or index[0].hour != 1:
        raise ValueError("Expected complete operating days beginning 01:00 KST")
    if not index.equals(pd.date_range(index[0], periods=len(index), freq="h")):
        raise ValueError("Timestamp index must be unique and consecutive")
    if sites.site_id.duplicated().any() or not np.isfinite(sites.capacity_kw).all() or (sites.capacity_kw <= 0).any():
        raise ValueError("Invalid site metadata")
    meta = sites.copy()
    for field in CATEGORICAL:
        source = "site_id" if field == "site_index" else field.replace("_index", "")
        mapping = {value: i for i, value in enumerate(sorted(meta[source].unique()))}
        meta[field] = meta[source].map(mapping).astype("int16")
    base = pd.MultiIndex.from_product([meta.site_id, index], names=["site_id", "ts"]).to_frame(index=False)
    d = base.merge(meta, on="site_id", how="left", validate="many_to_one", sort=False)
    for frame, station, columns in ((fa, "wa_station", WA), (fb, "wb_station", WB)):
        if frame.duplicated([station, "round", "ts"]).any():
            raise ValueError("Duplicate forecast station/round/ts key")
        if not frame["round"].isin([10, 17]).all():
            raise ValueError("Unrecognized forecast issue round")
        for r in (10, 17):
            f = frame.loc[frame["round"] == r, [station, "ts"]+columns].copy()
            for c in columns:
                f[c] = pd.to_numeric(f[c], errors="raise").astype("float32")
            if station == "wb_station":
                f["wb_wind_speed"] /= 3.6
                f["wb_wind_gust_spd"] /= 3.6
                f["wb_snow"] *= 10.0
            f = f.rename(columns={c: f"r{r}_{c}" for c in columns})
            d = d.merge(f, on=[station, "ts"], how="left", validate="many_to_one", sort=False)
    # Explicit order makes reshape-based forecast-day statistics unambiguous.
    d = d.sort_values(["site_index", "ts"], kind="stable").reset_index(drop=True)
    ts = pd.DatetimeIndex(d.ts)
    if not (ts.hour[::24] == 1).all():
        raise ValueError("Forecast-day alignment failed")
    elev, azim, cosz = solar(ts, d.lat.to_numpy(), d.lon.to_numpy())
    doy, hour = ts.dayofyear.to_numpy(), ts.hour.to_numpy()
    derived = {"elev": elev, "azim": azim, "cosz": cosz,
               "doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25),
               "hour": hour, "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
               "cap": d.capacity_kw.to_numpy(), "cosz_positive": np.maximum(cosz, 0)}
    for c in REVISION:
        derived[f"revision_{c}"] = d[f"r17_{c}"].to_numpy()-d[f"r10_{c}"].to_numpy()
    for r in (10, 17):
        prefix = f"r{r}_"
        for feature, a, b in (("cloud", "wa_cloud", "wb_cloud"), ("temp", "wa_temp", "wb_temp"),
                               ("humidity", "wa_humidity", "wb_rel_hum"), ("wind", "wa_wind_speed", "wb_wind_speed")):
            derived[f"{prefix}provider_{feature}_difference"] = d[prefix+a].to_numpy()-d[prefix+b].to_numpy()
        for c in SMOOTH:
            values = d[prefix+c].to_numpy()
            for w in (3, 7):
                derived[f"{prefix}{c}_mean{w}"] = issued_day_mean(values, w)
            a = values.reshape(-1, 24)
            finite = np.isfinite(a)
            denom = finite.sum(axis=1)
            means = np.divide(np.where(finite, a, 0).sum(axis=1), denom,
                              out=np.full(len(a), np.nan), where=denom > 0)
            derived[f"{prefix}{c}_daily_mean"] = np.repeat(means, 24)
        for provider in ("wa", "wb"):
            radians = np.radians(d[f"{prefix}{provider}_wind_dir"].to_numpy())
            derived[f"{prefix}{provider}_wind_sin"] = np.sin(radians)
            derived[f"{prefix}{provider}_wind_cos"] = np.cos(radians)
            cloud = np.clip(d[f"{prefix}{provider}_cloud"].to_numpy()/100, 0, 1)
            derived[f"{prefix}{provider}_cloud_solar"] = derived["cosz_positive"]*(1-cloud)**2
        derived[f"{prefix}weather_missing"] = d[[prefix+c for c in WA+WB]].isna().sum(axis=1).to_numpy()
    features = feature_schema()["enhanced"]
    # Compact numeric storage; the source site identifier remains separate.
    d = pd.concat([d, pd.DataFrame(derived, index=d.index)], axis=1)
    d = d[["site_id", "ts"]+features].astype({c: "int16" if c in CATEGORICAL else "float32" for c in features}).copy()
    if np.isinf(d[features].to_numpy()).any():
        raise ValueError("Infinite forecast-derived feature")
    if generation is None:
        d["energy"] = np.nan
    else:
        g = generation[["site_id", "ts", "energy"]].copy()
        if g.duplicated(["site_id", "ts"]).any() or not np.isfinite(g.energy).all():
            raise ValueError("Duplicate keys or nonfinite original generation")
        if set(g.site_id)-set(sites.site_id):
            raise ValueError("Unknown generation site")
        d = d.merge(g, on=["site_id", "ts"], how="left", validate="one_to_one", sort=False)
        # Preserve original label precision for exact group-score accounting.
        d["energy"] = d.energy.astype("float64")
    return d


def _source_signature(data):
    paths = [data/"sites.csv", data/"train/generation.parquet"]
    paths += [data/split/f"forecast_{provider}.parquet" for split in ("train", "test") for provider in ("wa", "wb")]
    return {str(p.relative_to(data)): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths}


def get_schema(cache):
    schema = json.loads((Path(cache)/"schema.json").read_text(encoding="utf-8"))
    if schema.get("version") != VERSION:
        raise ValueError("Feature cache version mismatch; use a fresh cache")
    return schema


def ensure_cache(data, cache, start, end, split="train"):
    """Build missing monthly full-grid shards, using filtered Parquet reads.

    start/end accept YYYY-MM strings or timezone-aware timestamps. Timestamp
    boundaries select whole operating months; load_rows applies exact cutoffs.
    Existing shards are reused only if source metadata and feature schema match.
    """
    data, cache = Path(data).resolve(), Path(cache).resolve()
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    cache.mkdir(parents=True, exist_ok=True)
    schema = feature_schema()
    schema["source_signature"] = _source_signature(data)
    schema["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    schema_path = cache/"schema.json"
    if schema_path.exists():
        previous = get_schema(cache)
        if previous != schema:
            raise ValueError("Source/features changed; select a fresh cache directory")
    else:
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    folder = cache/split
    folder.mkdir(exist_ok=True)
    sites = pd.read_csv(data/"sites.csv")
    for month in _months(start, end):
        path = folder/f"{month}.parquet"
        if path.exists():
            continue
        left, right = month_bounds(month)
        filters = [("ts", ">=", left.to_pydatetime()), ("ts", "<=", right.to_pydatetime())]
        fa = pd.read_parquet(data/split/"forecast_wa.parquet", columns=["wa_station", "round", "ts"]+WA, filters=filters)
        fb = pd.read_parquet(data/split/"forecast_wb.parquet", columns=["wb_station", "round", "ts"]+WB, filters=filters)
        gen = pd.read_parquet(data/"train/generation.parquet", columns=["site_id", "ts", "energy"], filters=filters) if split == "train" else None
        for frame in (fa, fb, gen):
            if frame is not None:
                frame["ts"] = pd.to_datetime(frame.ts).dt.tz_convert(TZ)
        d = build_month(sites, fa, fb, pd.date_range(left, right, freq="h"), gen)
        temporary = path.with_suffix(".partial.parquet")
        d.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
        print(f"Features {split}/{month}: {len(d):,} rows, {d.energy.notna().sum():,} original labels", flush=True)
    return schema


def load_rows(cache, start, end, labeled=False, split="train"):
    """Read exact inclusive timestamp bounds; optionally observed daylight only.

    root pipeline supplies each training cutoff. This function never expands it
    past `end`; unlabeled validation/inference retains every site/hour row.
    """
    cache = Path(cache)
    get_schema(cache)
    start, end = _timestamp(start), _timestamp(end)
    if start > end:
        raise ValueError("start must precede end")
    frames = []
    for month in _months(start, end):
        path = cache/split/f"{month}.parquet"
        filters = [("ts", ">=", start.to_pydatetime()), ("ts", "<=", end.to_pydatetime())]
        if labeled:
            filters += [("elev", ">", 0)]
        # Arrow filtering of null values differs across versions; apply the
        # label mask in pandas so missing targets can never become training y.
        d = pd.read_parquet(path, filters=filters)
        if labeled:
            d = d.loc[d.energy.notna() & (d.elev > 0)].copy()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
