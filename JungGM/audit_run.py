"""Independently recompute saved development scores and submission aggregation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from validate_submission import validate


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(data, run, submission):
    sites = pd.read_csv(data/'sites.csv')
    composition = pd.read_csv(submission/'composition.csv')
    errors = validate(submission/'composition.csv', submission/'forecast.csv', data/'sites.csv')
    if errors:
        raise ValueError(errors)
    protocol = json.loads((run/'protocol_before_training.json').read_text(encoding='utf-8'))
    for name, value in protocol['source_sha256'].items():
        if digest(data/name) != value:
            raise ValueError('Original file changed: '+name)
    frozen = json.loads((run/'frozen_choice_before_may_audit.json').read_text(encoding='utf-8'))
    weights = np.array(list(frozen['weights'].values()))
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid frozen ensemble')
    ordered = composition.set_index('site_id').reindex(sites.sort_values('site_id').site_id)
    if not np.array_equal((ordered.group=='G1').to_numpy(dtype=int), frozen['groups']):
        raise ValueError('Submission groups differ from frozen grouping')
    model_meta = list((run/'production'/'models').glob('*.json'))
    expected_models = {name for name,weight in frozen['weights'].items() if weight > 0}
    if {p.stem for p in model_meta} != expected_models:
        raise ValueError('Saved production models differ from active frozen ensemble')
    for path in model_meta:
        meta = json.loads(path.read_text(encoding='utf-8'))
        if pd.Timestamp(meta['last_target']) > pd.Timestamp('2025-05-31 17:00', tz='Asia/Seoul'):
            raise ValueError('Final fitting crosses first issue deadline')
        if 'energy' in meta['schema']['enhanced']:
            raise ValueError('Label included in features')
    raw = pd.read_parquet(data/'train/generation.parquet')
    dev = pd.read_parquet(run/'development'/'site_predictions.parquet')
    truth = dev[['site_id','ts','energy']].merge(raw, on=['site_id','ts'], how='left',
        validate='one_to_one', suffixes=('_saved','_original'))
    if truth.energy_original.isna().any() or not np.array_equal(truth.energy_saved, truth.energy_original):
        raise ValueError('Saved development truth differs from original precision or missing labels')
    prices = pd.read_csv(data/'train/smp.csv')
    prices['ts'] = pd.to_datetime(prices.ts).dt.tz_convert('Asia/Seoul')
    prices['k'] = np.where(prices.ts.dt.hour.isin([11,12,13,14,15]), 3, 1)
    ref = sites.set_index('site_id').current_group.map({'A':'G1','B':'G2'})
    selected = composition.set_index('site_id').group
    reports = []
    for model, col, groups in [('official_reference','official_lgb',ref),
            ('ensemble_reference','ensemble',ref),('selected_system','ensemble',selected)]:
        z = dev.assign(group=dev.site_id.map(groups))
        sums = z.groupby(['group','ts'], as_index=False)[[col,'energy']].sum()
        sums = sums.merge(prices, on='ts', validate='many_to_one')
        sums['month'] = (sums.ts-pd.Timedelta(hours=1)).dt.month
        for month, frame in sums.groupby('month'):
            penalty = .1*float((frame.smp*frame.k*(frame[col]-frame.energy).abs()).sum())
            reports.append({'month':int(month),'model':model,'penalty_krw':penalty})
    recorded = json.loads((run/'development'/'monthly_metrics.json').read_text(encoding='utf-8'))
    for calculated in reports:
        expected = next(r for r in recorded if r['month']==calculated['month'] and r['model']==calculated['model'])
        if not np.isclose(calculated['penalty_krw'], expected['penalty_krw'], rtol=1e-12, atol=1e-5):
            raise ValueError('Independent score differs from saved score')
    site_pred = pd.read_parquet(run/'production'/'site_predictions.parquet')
    if site_pred.duplicated(['site_id','ts']).any() or len(site_pred) != 253*8760:
        raise ValueError('Invalid production site grid')
    target_times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(data/'smp.csv').ts)).tz_convert('Asia/Seoul').sort_values()
    if set(site_pred.site_id) != set(sites.site_id):
        raise ValueError('Production site identifiers differ from official metadata')
    for _, frame in site_pred.groupby('site_id'):
        if not pd.DatetimeIndex(frame.ts).sort_values().equals(target_times):
            raise ValueError('Production per-site target timestamps differ from official grid')
    sums = site_pred.assign(group=site_pred.site_id.map(selected)).groupby(['group','ts']).prediction_kwh.sum().sort_index()
    forecast = pd.read_csv(submission/'forecast.csv')
    forecast['ts'] = pd.to_datetime(forecast.ts).dt.tz_convert('Asia/Seoul')
    saved = forecast.set_index(['group','ts']).forecast_kwh.sort_index()
    if not sums.index.equals(saved.index) or not np.allclose(sums.to_numpy(),saved.to_numpy(),rtol=1e-12,atol=1e-7):
        raise ValueError('Submitted forecast differs from site predictions and fixed grouping')
    receipt = {'passed':True,'raw_source_hashes_unchanged':True,'original_truth_precision':True,
        'independent_group_scores':reports,'production_aggregation_equal':True,
        'production_site_rows':len(site_pred),'forecast_rows':len(forecast),
        'submission_sha256':{n:digest(submission/n) for n in ['composition.csv','forecast.csv']},
        'scope':'Independent local arithmetic/structure checks, not official server acceptance or future accuracy'}
    (run/'independent_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','run','submission']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    result=audit(a.data,a.run,a.submission)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
