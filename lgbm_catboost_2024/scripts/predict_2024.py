"""Predict all 2024 group-hours, with frozen final models and no truth access."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from features import build_site, weather


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(composition, forecast, sites, ix):
    if list(composition.columns) != ['site_id', 'group']:
        raise ValueError('Composition schema')
    if len(composition) != 253 or not composition.site_id.is_unique or set(composition.site_id) != set(sites.site_id):
        raise ValueError('Composition coverage')
    if set(composition.group) != {'G1', 'G2'}:
        raise ValueError('Composition groups')
    capacity = composition.merge(sites[['site_id', 'capacity_kw']], on='site_id', validate='one_to_one').groupby('group').capacity_kw.sum()
    if (capacity < 20000).any():
        raise ValueError('Group capacity below 20MW')
    if list(forecast.columns) != ['group', 'ts', 'forecast_kwh']:
        raise ValueError('Forecast schema')
    expected = pd.MultiIndex.from_product([['G1', 'G2'], ix])
    actual = pd.MultiIndex.from_frame(forecast[['group', 'ts']])
    if len(forecast) != len(expected) or not actual.is_unique or len(expected.difference(actual)):
        raise ValueError('Forecast grid')
    if not np.isfinite(forecast.forecast_kwh).all() or (forecast.forecast_kwh < 0).any():
        raise ValueError('Invalid forecast values')
    return capacity.to_dict()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'forecast.csv').exists():
        raise ValueError('Sealed forecast exists; do not overwrite')
    cfg_path = ROOT/'configs/final_model.json'
    config = json.loads(cfg_path.read_text(encoding='utf-8'))
    exp = json.loads((ROOT/'configs/experiment_2024.json').read_text(encoding='utf-8'))
    runs = {name: json.loads((a.models/(name+'_run.json')).read_text(encoding='utf-8')) for name in ['lightgbm', 'catboost']}
    for name, run in runs.items():
        path = a.models/('lightgbm.txt' if name == 'lightgbm' else 'catboost.cbm')
        if run['frozen_config_sha256'] != sha(cfg_path) or sha(path) != run['model_sha256']:
            raise ValueError('Model/config hash mismatch')
        if pd.Timestamp(run['train_end']) > pd.Timestamp(exp['train_end']):
            raise ValueError('Model trained past allowed cutoff')
    sites = pd.read_csv(a.data/'sites.csv')
    if len(sites) != 253 or not sites.site_id.is_unique or (sites.capacity_kw <= 0).any():
        raise ValueError('Invalid site metadata')
    ix = pd.date_range(exp['forecast_start'], exp['forecast_end'], freq='h')
    assert len(ix) == 8784
    started = time.time()
    print(json.dumps({'phase': 'forecast_weather', 'hours': len(ix), 'sites': len(sites)}), flush=True)
    wa = weather(a.data/'train/forecast_wa.parquet', 'a', ix, sites.wa_station.unique())
    wb = weather(a.data/'train/forecast_wb.parquet', 'b', ix, sites.wb_station.unique())
    models = {'lightgbm': lgb.Booster(model_file=str(a.models/'lightgbm.txt')),
              'catboost': CatBoostRegressor().load_model(str(a.models/'catboost.cbm'))}
    totals = {name: {g: np.zeros(len(ix), dtype='float64') for g in ['G1', 'G2']}
              for name in models}
    categories = runs['lightgbm']['categories']
    seen = set(runs['lightgbm']['observed_site_ids'])
    groups = sites.current_group.map(exp['reference_groups'])
    if groups.isna().any():
        raise ValueError('Unknown reference group')
    composition = pd.DataFrame({'site_id': sites.site_id, 'group': groups})
    for j, site in enumerate(sites.itertuples(index=False)):
        d = build_site(site, ix, wa[site.wa_station], wb[site.wb_station])
        group = exp['reference_groups'][site.current_group]
        for name, model in models.items():
            x = d[config['features'][name]].copy()
            for c in categories:
                if name == 'lightgbm':
                    known = x[c].where(x[c].isin(categories[c]))
                    x[c] = pd.Categorical(known, categories=categories[c])
                else:
                    x[c] = x[c].astype('object').fillna('__missing__')
            if name == 'lightgbm':
                pred = model.predict(x, num_threads=config['threads'])
            else:
                pred = model.predict(x, thread_count=config['threads'])
            pred = np.maximum(pred, 0)*site.capacity_kw
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite site prediction')
            totals[name][group] += pred
            del x
        del d
        if (j+1) % 20 == 0 or j+1 == len(sites):
            print(json.dumps({'phase': 'predict', 'sites_done': j+1, 'sites_total': len(sites),
                              'seconds': time.time()-started}), flush=True)
            gc.collect()
    weight = config['selection']['lightgbm_weight']
    forecast = pd.concat([pd.DataFrame({'group': group, 'ts': ix,
                            'forecast_kwh': weight*totals['lightgbm'][group]+(1-weight)*totals['catboost'][group]})
                          for group in ['G1', 'G2']], ignore_index=True)
    capacities = validate(composition, forecast, sites, ix)
    composition.to_csv(a.out/'composition.csv', index=False)
    forecast.to_csv(a.out/'forecast.csv', index=False)
    # Reload exported timestamps and numbers, not just in-memory arrays.
    c = pd.read_csv(a.out/'composition.csv')
    f = pd.read_csv(a.out/'forecast.csv')
    if not f.ts.str.endswith('+09:00').all():
        raise ValueError('Export lost KST timezone')
    f['ts'] = pd.to_datetime(f.ts)
    validate(c, f, sites, ix)
    report = {'status': 'sealed_and_format_validated', 'requested_hours': len(ix),
              'forecast_rows': len(f), 'composition_rows': len(c), 'capacity_kw': capacities,
              'train_start': runs['lightgbm']['train_start'], 'train_end': runs['lightgbm']['train_end'],
              'training_rows': runs['lightgbm']['rows'], 'training_sites': len(seen),
              'unseen_forecast_sites': int((~sites.site_id.isin(seen)).sum()),
              'forecast_start': str(ix[0]), 'forecast_end': str(ix[-1]),
              'blend_lightgbm_weight': weight, 'features': {k: len(v) for k, v in config['features'].items()},
              'iterations': config['iterations'], 'seconds': time.time()-started,
              'truth_or_smp_read_during_prediction': False, 'official_score_computed': False, 'common_score_computed': False,
              'frozen_config_sha256': sha(cfg_path), 'feature_engine_sha256': sha(ROOT/'src/features.py'),
              'script_sha256': sha(__file__),
              'model_sha256': {name: run['model_sha256'] for name, run in runs.items()},
              'forecast_sha256': sha(a.out/'forecast.csv'), 'composition_sha256': sha(a.out/'composition.csv')}
    (a.out/'prediction_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
