"""Fixed-group comparison of public and locally selected JungGM weights; no fitting."""
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/suwan_projectV2'))
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import pipeline as p
import features as ft

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'work/suwan_v2_reproduction'
DATA = ROOT / 'work/oibc_data'
CACHE = ROOT / 'work/suwan_v2_cache'
PUBLIC = {'official_lgb': .2, 'lgb_l1': .3, 'xgb_deep': .5}

def write(name, value):
    p.save_json(OUT / name, value)

def score(d, sites, prices, names, block):
    assert not d.duplicated(['site_id', 'ts']).any()
    counts = d.groupby('ts').energy.count()
    d = d[d.ts.isin(counts[counts == len(sites)].index)].copy()
    assert len(d) and np.isfinite(d[['energy'] + names].to_numpy()).all()
    d['group'] = d.site_id.map(sites.set_index('site_id').current_group)
    assert d.group.notna().all()
    # Independent pandas aggregation; do not use pipeline metrics or panel.
    agg = d.groupby(['ts', 'group'])[['energy'] + names].sum()
    ts = agg.index.get_level_values('ts')
    price = prices.set_index('ts').smp.reindex(ts).to_numpy()
    assert np.isfinite(price).all() and (price >= 0).all()
    weights = price * np.where(ts.hour.isin(range(11, 16)), 3, 1)
    month = (ts - pd.Timedelta(hours=1)).strftime('%Y-%m')
    rows = []
    for m in sorted(set(month)):
        mask = month == m
        for name in names:
            err = (agg[name] - agg.energy).to_numpy()[mask]
            rows.append(dict(block=block, month=m, model=name,
                sites=len(sites), hours=int(mask.sum() // 2),
                penalty_krw=float(.1 * np.sum(weights[mask] * np.abs(err))),
                group_bias_kwh=float(err.mean()),
                group_mae_kwh=float(np.abs(err).mean())))
    return rows

def main():
    selection = json.loads((RUN / 'selection/selection.json').read_text())
    local = selection['selected']['weights']
    sites = pd.read_csv(DATA / 'sites.csv').sort_values('site_id').reset_index(drop=True)
    prices = pd.read_csv(DATA / 'train/smp.csv')
    prices['ts'] = pd.to_datetime(prices.ts).dt.tz_convert(p.TZ)
    protocol = json.loads((RUN / 'protocol_before_training.json').read_text())
    assert p.source_manifest(DATA) == protocol['source_sha256'], 'Raw source changed'
    for name, sha in protocol['code_sha256'].items():
        assert p.sha256(p.HERE / name) == sha, name
    verified = 0
    for path in RUN.glob('**/*_checkpoint.json'):
        ck = json.loads(path.read_text())
        for name, sha in ck['model_sha256'].items():
            assert p.sha256(path.parent / name) == sha, name
        meta = json.loads(path.with_name(path.name.replace('_checkpoint', '')).read_text())
        assert pd.Timestamp(meta['last_target']) <= pd.Timestamp(meta['cutoff'])
        assert 'energy' not in meta['schema']['enhanced']
        verified += 1
    rows = []
    for block in ['seasonal_proxy', 'complete_fleet']:
        d = pd.read_parquet(RUN / f'selection/{block}/out_of_time_predictions.parquet')
        ids = sites
        if block == 'seasonal_proxy':
            coverage = d.groupby('site_id').energy.count() / d.ts.nunique()
            ids = sites[sites.site_id.isin(coverage[coverage >= .95].index)]
        d = d[d.site_id.isin(ids.site_id)].copy()
        d['public_fixed'] = sum(d[n] * w for n, w in PUBLIC.items())
        d['local_selected'] = sum(d[n] * w for n, w in local.items() if w)
        rows += score(d, ids, prices,
                      ['official_lgb', 'public_fixed', 'local_selected'], block)
        del d
        gc.collect()
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index=['block','month'], columns='model', values='penalty_krw')
    ratios = pivot.div(pivot.official_lgb, axis=0)
    aggregate = ratios.groupby(level='block').mean().mean().to_dict()
    assert np.isclose(aggregate['local_selected'],
                      selection['selected']['mean_monthly_relative_loss'], rtol=1e-10)
    frame.to_csv(OUT / 'selection_comparison.csv', index=False)
    write('selection_check.json', dict(status='PASS', checkpoint_count=verified,
          independent_scoring_matches=True, source_and_code_hashes_match=True,
          weights={'public_fixed': PUBLIC, 'local_selected': local},
          selection_objective=aggregate, versions=protocol['versions']))
    print('Selection independently verified:', aggregate, flush=True)
    devpath = RUN / 'development/site_predictions.parquet'
    if not (RUN / 'development_done.json').exists():
        write('status.json', {'stage':'waiting_for_development', 'complete':False})
        return
    dev = pd.read_parquet(devpath)
    raw = pd.read_parquet(DATA / 'train/generation.parquet')
    check = dev[['site_id','ts','energy']].merge(raw, on=['site_id','ts'],
             suffixes=('_saved','_raw'), validate='one_to_one')
    assert len(check) == len(dev)
    np.testing.assert_allclose(check.energy_saved, check.energy_raw, rtol=0, atol=0)
    del raw, check
    gc.collect()
    schema = json.loads((CACHE / 'schema.json').read_text())
    configs = {c['name']: c for c in p.CONFIGS}
    directory = RUN / 'development/models'
    devrows = []
    max_diff = 0.
    for month in [3, 4, 5]:
        start = p.stamp(f'2025-{month:02d}-01 01:00')
        end = p.stamp(f'2025-{month+1:02d}-01 00:00') - pd.Timedelta(hours=7)
        d = ft.load_rows(CACHE, start, end)
        z = dev[(dev.ts >= start) & (dev.ts <= end)].copy()
        for name in ['lgb_l1','xgb_deep']:
            if name == 'lgb_l1':
                model = lgb.Booster(model_file=str(directory / (name + '.txt')))
            else:
                model = xgb.XGBRegressor(n_jobs=2)
                model.load_model(directory / (name + '.ubj'))
            pred = p.predict_one(model, d, schema, configs[name])
            part = d[['site_id','ts']].copy()
            part[name] = pred
            z = z.merge(part, on=['site_id','ts'], validate='one_to_one')
            del model, part, pred
            gc.collect()
        z['local_selected'] = sum(z[n] * w for n, w in local.items() if w)
        np.testing.assert_allclose(z.local_selected, z.ensemble, rtol=1e-7, atol=1e-6)
        max_diff = max(max_diff, float(np.max(np.abs(z.local_selected-z.ensemble))))
        z['public_fixed'] = sum(z[n] * w for n, w in PUBLIC.items())
        devrows += score(z, sites, prices,
                         ['official_lgb','public_fixed','local_selected'], 'development')
        print('Compared development month', month, flush=True)
        del d,z
        gc.collect()
    result = pd.DataFrame(devrows)
    result.to_csv(OUT / 'development_comparison.csv', index=False)
    original = pd.DataFrame(json.loads((RUN/'development/monthly_metrics.json').read_text()))
    for row in devrows:
        if row['model'] == 'public_fixed':
            continue
        label = 'official_reference' if row['model']=='official_lgb' else 'ensemble_reference'
        expected = original[(original.month==int(row['month'][-2:])) & (original.model==label)].iloc[0]
        assert np.isclose(row['penalty_krw'], expected.penalty_krw, rtol=1e-10)
    write('status.json', {'stage':'complete','complete':True,
          'scope':'Fixed-reference-group baseline comparison; no new model fitting or calibration',
          'raw_truth_matches':True,'independent_development_scores_match':True,
          'saved_model_prediction_max_abs_difference':max_diff,
          'official_period_production':'Separate pre-existing reproduction, not needed for this comparison'})
    print(result.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
