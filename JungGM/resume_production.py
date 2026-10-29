"""Resume interrupted final fitting with frozen choices and unchanged inputs."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

import features as ft
import pipeline as p
import predict
from validate_submission import validate


def resume(data, run, cache, submission):
    data, run, cache, submission = [Path(v).resolve() for v in (data, run, cache, submission)]
    for output in (run, cache, submission):
        if predict.overlaps(data, output):
            raise ValueError('Output must be separate from original data')
    if predict.overlaps(run, cache) or predict.overlaps(submission, run) or predict.overlaps(submission, cache):
        raise ValueError('Run, cache, and submission must be separate')
    if (run/'completion.json').exists() or any((submission/n).exists() for n in ('composition.csv', 'forecast.csv')):
        raise ValueError('Refusing to overwrite completed output')
    protocol = predict.read_json(run/'protocol_before_training.json')
    if p.source_manifest(data) != protocol['source_sha256']:
        raise ValueError('Original sources changed')
    for name, digest in protocol['code_sha256'].items():
        if name in ('pipeline.py', 'features.py', 'grouping.py', 'validate_submission.py') and p.sha256(p.HERE/name) != digest:
            raise ValueError('Training source changed: '+name)
    frozen = predict.read_json(run/'frozen_choice_before_may_audit.json')
    weights = frozen['weights']
    values = np.array(list(weights.values()))
    if set(weights) != {c['name'] for c in p.CONFIGS} or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError('Invalid frozen weights')
    sites = pd.read_csv(data/'sites.csv').sort_values('site_id').reset_index(drop=True)
    assignments = np.asarray(frozen['groups'])
    if len(sites) != 253 or assignments.shape != (253,) or not np.isin(assignments, [0, 1]).all():
        raise ValueError('Invalid frozen groups')
    schema = ft.get_schema(cache)
    origin = p.stamp('2025-06-01 01:00')
    directory = run/'production'/'models'
    directory.mkdir(parents=True, exist_ok=True)
    reused, missing = [], []
    models = {}
    for config in p.CONFIGS:
        name = config['name']
        if weights[name] <= 0:
            continue
        if (directory/f'{name}.json').exists():
            one = {key: value if key == name else 0. for key, value in weights.items()}
            loaded, hashes = predict.load_models(run, schema, one)
            models[name] = loaded[0][1]
            reused.append({'name': name, 'files_sha256': hashes})
            p.log('Reusing verified completed model: '+name)
        else:
            missing.append(config)
    p.save_json(run/'resume_record.json', {'frozen_choice_sha256': p.sha256(run/'frozen_choice_before_may_audit.json'),
        'reused': reused, 'fit_missing': [c['name'] for c in missing], 'selection_repeated': False})
    if missing:
        train = ft.load_rows(cache, p.history_start(origin), p.issue_deadline(origin), labeled=True)
        for config in missing:
            models[config['name']] = p.fit_one(train, schema, config, origin, directory)
        del train
        gc.collect()
    ft.ensure_cache(data, cache, '2025-06', '2026-05', split='test')
    frames = []
    for month in pd.period_range('2025-06', '2026-05', freq='M'):
        start, end = ft.month_bounds(month)
        d = ft.load_rows(cache, start, end, split='test')
        pred = np.zeros(len(d))
        for config in p.CONFIGS:
            name = config['name']
            if weights[name] > 0:
                pred += weights[name]*p.predict_one(models[name], d, schema, config)
        z = d[['site_id', 'ts']].copy()
        z['prediction_kwh'] = pred
        frames.append(z)
        p.log(f'Production forecast ready {month}')
    predictions = pd.concat(frames, ignore_index=True)
    expected = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(data/'smp.csv').ts)).tz_convert(p.TZ)
    if len(predictions) != 253*len(expected) or predictions.duplicated(['site_id','ts']).any() or set(predictions.ts) != set(expected) or set(predictions.site_id) != set(sites.site_id):
        raise ValueError('Invalid production grid')
    predictions.to_parquet(run/'production'/'site_predictions.parquet', index=False, compression='zstd')
    composition = pd.DataFrame({'site_id': sites.site_id, 'group': np.where(assignments == 1, 'G1', 'G2')})
    predictions['group'] = predictions.site_id.map(composition.set_index('site_id').group)
    forecast = predictions.groupby(['group','ts'], as_index=False).prediction_kwh.sum().rename(columns={'prediction_kwh':'forecast_kwh'}).sort_values(['group','ts'])
    submission.mkdir(parents=True, exist_ok=True)
    composition.to_csv(submission/'composition.csv', index=False)
    forecast.to_csv(submission/'forecast.csv', index=False)
    errors = validate(submission/'composition.csv', submission/'forecast.csv', data/'sites.csv')
    if errors:
        raise ValueError(errors)
    if p.source_manifest(data) != protocol['source_sha256']:
        raise ValueError('Original sources changed during resume')
    receipt = {'format_validation': 'PASS', 'composition_rows': len(composition), 'forecast_rows': len(forecast),
        'site_prediction_rows': len(predictions), 'target_start': expected.min(), 'target_end': expected.max(),
        'fit_cutoff': p.issue_deadline(origin), 'group_capacity_kw': sites.assign(group=composition.group).groupby('group').capacity_kw.sum().to_dict(),
        'files': {n: p.sha256(submission/n) for n in ('composition.csv','forecast.csv')},
        'model_source_sha256': {v.name: p.sha256(v) for v in p.HERE.glob('*.py')}, 'run': str(run),
        'no_server_submission': True, 'resumed': True,
        'warning': 'Local structural validation only; official target-year accuracy is unknown.'}
    p.save_json(submission/'검증결과.json', receipt)
    p.save_json(run/'completion.json', receipt)
    p.save_json(run/'source_integrity.json', {'unchanged': True, 'source_sha256': protocol['source_sha256']})
    p.log('PASS: resumed final fitting and submission generation complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=p.HERE/'_local'/'cache')
    parser.add_argument('--submission', type=Path, default=p.HERE/'제출')
    args = parser.parse_args()
    resume(args.data, args.run, args.cache, args.submission)
