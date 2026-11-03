"""Frozen chronological improvement experiment. All competition data stay local.

Stages: train (new cost-aligned estimator), evaluate (one selected candidate),
produce (only when April gate and May historical audit both pass).
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import QuantileRegressor
import features as ft
import pipeline as p
from improvement_features import add_features
from validate_submission import validate

HERE = Path(__file__).resolve().parent
RUN = HERE/'_local/improvement_v3'
OLD = HERE/'_local/paper_v2_seasonal'
CACHE = HERE/'_local/cache'
SEED = 20261102
PARAMS = dict(objective='regression', n_estimators=900, learning_rate=.035,
              num_leaves=63, min_child_samples=200, reg_lambda=12,
              colsample_bytree=.85, subsample=.85, subsample_freq=1,
              max_bin=127, n_jobs=4, random_state=SEED, verbosity=-1,
              deterministic=True, force_col_wise=True)
RAW = ['cosz_positive', 'r17_wa_cloud', 'r17_wb_cloud', 'r17_wb_uv_idx',
       'r17_wb_temp', 'r17_wb_rel_hum', 'revision_wa_cloud', 'revision_wb_cloud',
       'r17_weather_missing']
CAL = ['util', 'solar', 'cloud', 'uv', 'temp', 'humidity', 'disagreement',
       'revision', 'hour_sin', 'hour_cos', 'peak', 'group_code', 'missing']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prices(data):
    x = pd.read_csv(data/'train/smp.csv')
    x.ts = pd.to_datetime(x.ts)
    x['weight'] = x.smp*np.where(x.ts.dt.hour.between(11, 15), 3, 1)
    return x.set_index('ts')


def protocol():
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN/'protocol_before_results.json'
    obj = dict(seed=SEED, parameters=PARAMS,
        estimator_fit='24 months before 2025-03-01, labels <= 2025-02-28 17:00 KST',
        estimator_weight='capacity * historical SMP * k; not a feature',
        calibration_fit='2024-07 through 2025-02 time-external original predictions; July-Dec fixed 225-site proxy',
        selection='March: blends 0/.25/.5/.75/1; calibrators none, scale, hour-scale, LAD alpha .001/.005; strengths .5/1',
        gate='One March-selected candidate; April >=1% penalty improvement and 3-day block bootstrap lower90 >0',
        audit='May previously inspected historical data; require nonnegative improvement; never claim untouched test',
        production='Refit new estimator through May31 17; freeze selected recipe; original submission never overwritten',
        max_correction_fraction=.12, reference_groups_fixed=True,
        external_data=False, official_score_used_for_parameter_search=False,
        teammate_source_commit='f33a5296f09f20e1dad4c66ab179973bbcbd8261')
    if path.exists() and read(path) != obj:
        raise ValueError('Protocol mismatch; use a new run instead of changing the search')
    if not path.exists():
        p.save_json(path, obj)


def train(data, production=False, origin_override=None, dest_override=None):
    origin = p.stamp(origin_override or ('2025-06-01 01:00' if production else '2025-03-01 01:00'))
    dest = Path(dest_override) if dest_override else RUN/('production' if production else 'development')
    dest.mkdir(exist_ok=True)
    model_path = dest/'cost_lgb.txt'
    schema = ft.get_schema(CACHE)
    if model_path.exists() and (dest/'metadata.json').exists():
        metadata = read(dest/'metadata.json')
        if metadata['params'] != PARAMS or metadata['origin'] != str(origin) or metadata['cutoff'] != str(p.issue_deadline(origin)):
            raise ValueError('Saved estimator does not match requested protocol')
        return lgb.Booster(model_file=str(model_path)), metadata['features']
    pr = prices(data)
    frames = []
    start = p.history_start(origin, 24)
    cutoff = p.issue_deadline(origin)
    for month in ft._months(start, cutoff):
        d, extra = add_features(pd.read_parquet(CACHE/'train'/f'{month}.parquet'))
        d = d.loc[(d.ts >= start) & (d.ts <= cutoff) & d.energy.notna() & (d.solar_hour_integral > 0)]
        names = schema['enhanced']+extra
        frames.append(d[['ts', 'energy', 'cap']+[n for n in names if n != 'cap']])
        del d
    d = pd.concat(frames, ignore_index=True); del frames; gc.collect()
    weights = d.cap.to_numpy()*pr.weight.reindex(pd.DatetimeIndex(d.ts)).to_numpy()
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid historical training weights')
    y = (d.energy/d.cap).clip(0, 1.2).to_numpy(dtype='float32')
    weights /= weights.mean()
    metadata = dict(features=names, cutoff=str(cutoff), origin=str(origin),
                    rows=len(d), params=PARAMS, first_target=str(d.ts.min()),
                    last_target=str(d.ts.max()))
    p.log(f"Cost-aligned LightGBM: {len(d):,} rows, {len(names)} features, cutoff {cutoff}")
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(d[names], y, sample_weight=weights,
              categorical_feature=[c for c in schema['categorical'] if c in names])
    model.booster_.save_model(str(model_path))
    p.save_json(dest/'metadata.json', metadata)
    del d, y, weights; gc.collect()
    return model.booster_, names


def predict_new(model, names, split, months, out):
    if out.exists():
        return
    frames = []
    for month in months:
        d, _ = add_features(pd.read_parquet(CACHE/split/f'{month}.parquet'))
        mask = d.solar_hour_integral.to_numpy() > 0
        pred = np.zeros(len(d))
        pred[mask] = np.maximum(model.predict(d.loc[mask, names], num_threads=4), 0)*d.loc[mask, 'cap'].to_numpy()
        z = d[['site_id', 'ts']].copy(); z['new'] = pred
        frames.append(z)
        p.log(f'New model predicted {month}')
        del d; gc.collect()
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)


def old_oof():
    frames = []
    for part in ('seasonal_proxy', 'complete_fleet'):
        d = pd.read_parquet(OLD/'selection'/part/'out_of_time_predictions.parquet')
        coverage = d.groupby('site_id').energy.count()/d.ts.nunique()
        ids = coverage[coverage >= .95].index if part == 'seasonal_proxy' else coverage.index
        d = d[d.site_id.isin(ids)].copy()
        complete = d.groupby('ts').energy.count()
        d = d[d.ts.isin(complete[complete == len(ids)].index)].copy()
        d['base'] = .2*d.official_lgb+.3*d.lgb_l1+.5*d.xgb_deep
        frames.append(d[['site_id', 'ts', 'energy', 'base']])
    return pd.concat(frames, ignore_index=True)


def group_frame(site, data, split='train'):
    """Only complete panels in site input are accepted, including fixed proxy cohorts."""
    meta = pd.read_csv(data/'sites.csv').set_index('site_id')
    comp = pd.read_csv(HERE/'제출/composition.csv').set_index('site_id').group
    pr = prices(data) if split == 'train' else None
    result = []
    month_keys = (site.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    for month, sub in site.groupby(month_keys, sort=True):
        sub = sub.copy()
        n = sub.site_id.nunique()
        if sub.duplicated(['site_id', 'ts']).any() or not sub.groupby('ts').size().eq(n).all():
            raise ValueError('Incomplete panel; never zero-fill missing targets')
        cols = ['site_id', 'ts']+RAW
        weather = pd.read_parquet(CACHE/split/f'{month}.parquet', columns=cols)
        sub = sub.merge(weather, on=['site_id', 'ts'], validate='one_to_one', how='left')
        sub['group'] = sub.site_id.map(comp)
        sub['capacity'] = sub.site_id.map(meta.capacity_kw)
        for g, z in sub.groupby('group'):
            sums = z.groupby('ts')[['base', 'capacity']+(['energy'] if 'energy' in z else [])+(['new'] if 'new' in z else [])].sum()
            if 'energy' in z and not np.isfinite(z.energy).all():
                raise ValueError('Missing original truth')
            feat = {}
            for c in RAW:
                v = z[c].to_numpy(dtype=float); cap = z.capacity.to_numpy()
                top = pd.Series(np.where(np.isfinite(v), v*cap, 0), index=z.ts).groupby(level=0).sum()
                bot = pd.Series(np.where(np.isfinite(v), cap, 0), index=z.ts).groupby(level=0).sum()
                feat[c] = top/bot.replace(0, np.nan)
            f = pd.DataFrame(feat).reindex(sums.index)
            sums['solar'] = f.cosz_positive
            sums['cloud'] = (f.r17_wa_cloud+f.r17_wb_cloud)/200
            sums['uv'] = f.r17_wb_uv_idx
            sums['temp'] = f.r17_wb_temp
            sums['humidity'] = f.r17_wb_rel_hum/100
            sums['disagreement'] = (f.r17_wa_cloud-f.r17_wb_cloud).abs()/100
            sums['revision'] = (f.revision_wa_cloud.abs()+f.revision_wb_cloud.abs())/200
            sums['missing'] = f.r17_weather_missing
            sums['group'] = g; sums['group_code'] = float(g == 'G2')
            h = sums.index.hour.to_numpy()
            sums['hour_sin'] = np.sin(2*np.pi*(h-.5)/24)
            sums['hour_cos'] = np.cos(2*np.pi*(h-.5)/24)
            sums['peak'] = ((h >= 11) & (h <= 15)).astype(float)
            sums['util'] = sums.base/sums.capacity
            sums['sites'] = z.groupby('ts').site_id.nunique()
            if pr is not None:
                sums['weight'] = pr.weight.reindex(sums.index)
                sums['smp'] = pr.smp.reindex(sums.index)
            result.append(sums.reset_index())
    return pd.concat(result, ignore_index=True).sort_values(['ts', 'group']).reset_index(drop=True)


def weighted_median(x, w):
    x, w = np.asarray(x), np.asarray(w)
    ix = np.argsort(x)
    return float(x[ix[np.searchsorted(np.cumsum(w[ix]), w.sum()/2)]])


def calibrators(d):
    d = d[(d.base > 1) & (d.weight > 0)].copy()
    recipes = [dict(kind='none', strength=0)]
    for kind in ('scale', 'hour_scale'):
        factors = {}
        for g, z in d.groupby('group'):
            factors[g] = float(np.clip(weighted_median(z.energy/z.base, z.weight*z.base), .88, 1.12))
            if kind == 'hour_scale':
                for h, x in z.groupby(z.ts.dt.hour):
                    raw = weighted_median(x.energy/x.base, x.weight*x.base)
                    # Shrink hourly estimates toward the group estimate.
                    factors[f'{g}_{h}'] = float(np.clip(.5*raw+.5*factors[g], .88, 1.12))
        for strength in (.5, 1.):
            recipes.append(dict(kind=kind, strength=strength, factors=factors))
    x = d[CAL].to_numpy(dtype=float)
    median = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in x.T])
    x = np.where(np.isfinite(x), x, median)
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-8] = 1
    x = (x-mean)/scale
    for alpha in (.001, .005):
        model = QuantileRegressor(quantile=.5, alpha=alpha, solver='highs')
        model.fit(x, (d.energy-d.base)/d.capacity, sample_weight=d.weight*d.capacity)
        for strength in (.5, 1.):
            recipes.append(dict(kind='lad', alpha=alpha, strength=strength,
                median=median.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                coef=model.coef_.tolist(), intercept=float(model.intercept_)))
    return recipes


def apply(d, recipe):
    b = d.base.to_numpy(dtype=float)
    kind = recipe['kind']
    if kind == 'none':
        return b.copy()
    if kind in ('scale', 'hour_scale'):
        factors = recipe['factors']
        f = np.array([factors.get(f'{g}_{h}', factors[g]) if kind == 'hour_scale' else factors[g]
                      for g, h in zip(d.group, d.ts.dt.hour)])
        delta = b*(f-1)
    else:
        x = d[CAL].to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, recipe['median'])
        x = (x-np.array(recipe['mean']))/np.array(recipe['scale'])
        delta = (x@np.array(recipe['coef'])+recipe['intercept'])*d.capacity.to_numpy()
    return np.maximum(0, b+recipe['strength']*np.clip(delta, -.12*b, .12*b))


def blend(d, weight):
    z = d.copy()
    z['base'] = (1-weight)*d.base+weight*d['new']
    z['util'] = z.base/z.capacity
    return z


def loss(d, prediction):
    return float(.1*np.sum(d.weight.to_numpy()*np.abs(prediction-d.energy.to_numpy())))


def block_gate(d, candidate):
    day = (d.ts-pd.Timedelta(hours=1)).dt.normalize()
    b = .1*d.weight.to_numpy()*np.abs(d.base.to_numpy()-d.energy.to_numpy())
    c = .1*d.weight.to_numpy()*np.abs(candidate-d.energy.to_numpy())
    daily = pd.DataFrame(dict(day=day, base=b, candidate=c)).groupby('day').sum()
    savings = (daily.base-daily.candidate).to_numpy()
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(savings), size=(2000, int(np.ceil(len(savings)/3))))
    ix = ((starts[:, :, None]+np.arange(3)) % len(savings)).reshape(2000, -1)[:, :len(savings)]
    lower = float(np.quantile(savings[ix].mean(1), .05))
    gain = float(1-daily.candidate.sum()/daily.base.sum())
    return dict(accepted=bool(gain >= .01 and lower > 0), improvement_fraction=gain,
                lower90_daily_saving_krw=lower, days=len(daily), block_days=3)


def evaluate(data):
    protocol()
    fpath = RUN/'calibration_frame.parquet'
    if not fpath.exists():
        group_frame(old_oof(), data).to_parquet(fpath, index=False)
    fit = pd.read_parquet(fpath)
    ep = RUN/'evaluation_frame.parquet'
    if not ep.exists():
        site = pd.read_parquet(OLD/'development/site_predictions.parquet').rename(columns={'ensemble': 'base'})
        new = pd.read_parquet(RUN/'development/predictions.parquet')
        site = site.merge(new, on=['site_id', 'ts'], how='left', validate='one_to_one')
        if not np.isfinite(site['new']).all():
            raise ValueError('Missing new prediction')
        group_frame(site, data).to_parquet(ep, index=False)
    ev = pd.read_parquet(ep)
    recipes = calibrators(fit)
    month = (ev.ts-pd.Timedelta(hours=1)).dt.month
    march = ev[month == 3]
    results = []
    for nw in (0., .25, .5, .75, 1.):
        z = blend(march, nw)
        for i, recipe in enumerate(recipes):
            results.append(dict(new_weight=nw, recipe_index=i,
                                penalty_krw=loss(z, apply(z, recipe))))
    best = min(results, key=lambda x: x['penalty_krw'])
    selected = dict(new_weight=best['new_weight'], recipe=recipes[best['recipe_index']])
    p.save_json(RUN/'frozen_before_april.json', dict(selected=selected, march_search=results))
    p.log(f'March froze: new weight {selected["new_weight"]}, correction {selected["recipe"]["kind"]}')
    rows = []
    for m in (3, 4, 5):
        d = ev[month == m]
        z = blend(d, selected['new_weight'])
        pred = apply(z, selected['recipe'])
        base_loss, cand = loss(d, d.base), loss(d, pred)
        row = dict(month=m, hours=d.ts.nunique(), sites=253, baseline_penalty_krw=base_loss,
                   new_model_only_penalty_krw=loss(d, d['new']), candidate_penalty_krw=cand,
                   gain_pct=100*(1-cand/base_loss))
        rows.append(row)
        if m == 4:
            gate = block_gate(d, pred)
        p.log(str(row))
    accepted = bool(gate['accepted'] and rows[-1]['gain_pct'] >= 0)
    p.save_json(RUN/'evaluation.json', dict(months=rows, april_gate=gate,
                    accepted=accepted, selected=selected, candidate_count=len(results),
                    caveat='All historical periods have been inspected previously; no untouched test claim.'))
    pd.DataFrame(rows).to_csv(RUN/'monthly_comparison.csv', index=False)
    p.log(f'Adoption gate: {accepted}; April: {gate}')


def produce(data):
    result = read(RUN/'evaluation.json')
    if not result['accepted']:
        p.log('Candidate rejected; original official submission preserved.'); return
    model, names = train(data, production=True)
    outpred = RUN/'production/predictions.parquet'
    predict_new(model, names, 'test', pd.period_range('2025-06','2026-05',freq='M'), outpred)
    old = pd.read_parquet(OLD/'production/site_predictions.parquet').rename(columns={'prediction_kwh':'base'})
    site = old.merge(pd.read_parquet(outpred), on=['site_id','ts'], validate='one_to_one')
    if len(site) != 253*8760:
        raise ValueError('Incomplete target grid')
    # Use the frozen calibration coefficients, avoiding a production-only refit change.
    frame = group_frame(site, data, split='test')
    selected = result['selected']; z = blend(frame, selected['new_weight'])
    z['forecast_kwh'] = apply(z, selected['recipe'])
    out = HERE/'제출/improved_v3'; out.mkdir(exist_ok=True)
    original = HERE/'제출/composition.csv'
    (out/'composition.csv').write_bytes(original.read_bytes())
    z[['group','ts','forecast_kwh']].sort_values(['group','ts']).to_csv(out/'forecast.csv', index=False)
    errors = validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors: raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS', official_submitted=False,
        files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']},
        selection=selected, evaluation=result, model_sha256=p.sha256(RUN/'production/cost_lgb.txt')))
    p.log(f'Validated new submission at {out}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['train','evaluate','produce'],required=True)
    args=ap.parse_args();protocol()
    if args.stage=='train':
        model,names=train(args.data)
        predict_new(model,names,'train',pd.period_range('2025-03','2025-05',freq='M'),RUN/'development/predictions.parquet')
    elif args.stage=='evaluate':evaluate(args.data)
    else:produce(args.data)


if __name__=='__main__':main()
