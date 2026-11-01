"""Materialize observed 2022/2023 training rows locally, respecting common cutoff."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import build_site, weather, NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/experiment_2024.json').read_text(encoding='utf-8'))
    code_hash = hashlib.sha256((root/'src/features.py').read_bytes()).hexdigest()
    sites = pd.read_csv(a.data/'sites.csv')
    gen = pd.read_parquet(a.data/'train/generation.parquet', filters=[('ts', '>=', pd.Timestamp(config['train_start'])), ('ts', '<=', pd.Timestamp(config['train_end']))])
    gen['ts'] = pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
    gen = gen[gen.ts.between(pd.Timestamp(config['train_start']), pd.Timestamp(config['train_end']))].copy()
    if gen.duplicated(['site_id', 'ts']).any():
        raise ValueError('Duplicate truth')
    negative = int((gen.energy < 0).sum())
    gen = gen[np.isfinite(gen.energy) & (gen.energy >= 0)]
    prices = pd.read_csv(a.data/'train/smp.csv')
    prices['ts'] = pd.to_datetime(prices.ts).dt.tz_convert('Asia/Seoul')
    if prices.ts.duplicated().any():
        raise ValueError('Duplicate historical price')
    signatures = {str(f.relative_to(a.data)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in [a.data/'sites.csv', a.data/'train/generation.parquet',
                            a.data/'train/smp.csv', a.data/'train/forecast_wa.parquet',
                            a.data/'train/forecast_wb.parquet']}
    manifest = {'feature_engine_sha256': code_hash, 'inputs': signatures, 'features': NAMES,
                'negative_labels_excluded': negative, 'cutoff': config['train_end'], 'years': {}}
    for year in [2022, 2023]:
        begin = pd.Timestamp(f'{year}-01-01 01:00', tz='Asia/Seoul')
        finish = pd.Timestamp(f'{year+1}-01-01 00:00', tz='Asia/Seoul')
        # Features use the whole issued target day; labels are cut independently.
        ix = pd.date_range(begin, finish, freq='h')
        labels = gen[gen.ts.between(begin, finish)]
        scope = sites[sites.site_id.isin(labels.site_id.unique())]
        dest = a.out/str(year)
        dest.mkdir(exist_ok=True)
        print(json.dumps({'year': year, 'observed_sites': len(scope), 'rows': len(labels), 'phase': 'weather'}), flush=True)
        wa = weather(a.data/'train/forecast_wa.parquet', 'a', ix, scope.wa_station.unique())
        wb = weather(a.data/'train/forecast_wb.parquet', 'b', ix, scope.wb_station.unique())
        count = 0
        for j, site in enumerate(scope.itertuples(index=False)):
            d = build_site(site, ix, wa[site.wa_station], wb[site.wb_station])
            y = labels[labels.site_id == site.site_id][['site_id', 'ts', 'energy']]
            d = d.merge(y, on=['site_id', 'ts'], how='inner', validate='one_to_one')
            d = d.merge(prices, on='ts', how='left', validate='many_to_one')
            if d.smp.isna().any() or (d.smp < 0).any():
                raise ValueError('Historical SMP missing or negative')
            d['target_cf'] = (d.energy/d.capacity_kw).astype('float32')
            d['sample_weight'] = (d.capacity_kw*d.smp*np.where(d.ts.dt.hour.between(11, 15), 3, 1)).astype('float32')
            d.to_parquet(dest/f'{site.site_id}.parquet', index=False, compression='zstd')
            count += len(d)
            if (j+1) % 10 == 0 or j+1 == len(scope):
                print(json.dumps({'year': year, 'sites_done': j+1, 'sites_total': len(scope), 'rows': count}), flush=True)
        if count != len(labels):
            raise ValueError('Training row loss')
        manifest['years'][str(year)] = {'rows': count, 'sites': len(scope)}
        (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Training cache complete.', flush=True)


if __name__ == '__main__':
    main()
