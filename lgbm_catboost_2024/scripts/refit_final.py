"""Refit one frozen model on the full common 2022–2023 training interval."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['lightgbm', 'catboost'], required=True)
    p.add_argument('--prepare-cat-pool', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model_path = a.out/('lightgbm.txt' if a.model == 'lightgbm' else 'catboost.cbm')
    if model_path.exists() or (a.out/(a.model+'_run.json')).exists():
        raise ValueError('Final model exists; use a fresh run directory instead of overwriting')
    config_path = ROOT/'configs/final_model.json'
    experiment = json.loads((ROOT/'configs/experiment_2024.json').read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    cache_manifest = json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'))
    if cache_manifest['feature_engine_sha256'] != sha(ROOT/'src/features.py'):
        raise ValueError('Stale feature cache')
    features = config['features'][a.model]
    if cache_manifest['features'] != config['features']['catboost']:
        raise ValueError('Feature order mismatch')
    if a.model == 'catboost' and not a.prepare_cat_pool:
        pool_path = a.out/'catboost_training.quantized'
        pool_meta_path = a.out/'catboost_pool_run.json'
        meta = json.loads(pool_meta_path.read_text(encoding='utf-8'))
        if meta['frozen_config_sha256'] != sha(config_path) or meta['cache_manifest_sha256'] != sha(a.cache/'manifest.json'):
            raise ValueError('Stale quantized pool metadata')
        if meta['quantized_pool_sha256'] != sha(pool_path):
            raise ValueError('Quantized pool hash mismatch')
        print(json.dumps({'phase': 'load_quantized_pool', 'rows': meta['rows']}), flush=True)
        pool = Pool('quantized://'+pool_path.resolve().as_posix(), thread_count=config['threads'])
        if pool.num_row() != meta['rows'] or pool.num_col() != len(features):
            raise ValueError('Quantized pool shape mismatch')
        params = dict(config['training_parameters']['catboost'])
        params.update({'random_seed': config['seed'], 'thread_count': config['threads'],
                       'allow_writing_files': False, 'verbose': 100})
        model = CatBoostRegressor(**params)
        began = time.time()
        model.fit(pool)
        model.save_model(str(model_path))
        if model.tree_count_ != config['iterations']['catboost']:
            raise ValueError('Unexpected final CatBoost iteration count')
        meta.update({'status': 'complete', 'seconds': time.time()-began,
                     'trained_iterations': model.tree_count_, 'model_sha256': sha(model_path),
                     'script_sha256': sha(__file__), 'training_process_loaded_quantized_only': True})
        (a.out/'catboost_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(json.dumps({k: meta[k] for k in ['model', 'status', 'rows', 'observed_sites', 'trained_iterations', 'seconds', 'model_sha256']}), flush=True)
        return
    start = pd.Timestamp(experiment['train_start'])
    end = pd.Timestamp(experiment['train_end'])
    cols = list(dict.fromkeys(['site_id', 'ts']+features+['target_cf', 'sample_weight']))
    files = sorted((a.cache/'2022').glob('*.parquet'))+sorted((a.cache/'2023').glob('*.parquet'))
    print(json.dumps({'phase': 'load', 'model': a.model, 'files': len(files)}), flush=True)
    d = pd.concat([pd.read_parquet(f, columns=cols, filters=[('ts', '>=', start), ('ts', '<=', end)])
                   for f in files], ignore_index=True)
    expected = sum(v['rows'] for v in cache_manifest['years'].values())
    if len(d) != expected or d.duplicated(['site_id', 'ts']).any():
        raise ValueError('Incomplete or duplicated final training data')
    if d.ts.min() < start or d.ts.max() > end or not np.isfinite(d.target_cf).all():
        raise ValueError('Invalid final training bounds/labels')
    if (d.target_cf < 0).any() or not np.isfinite(d.sample_weight).all() or d.sample_weight.sum() <= 0:
        raise ValueError('Invalid training values')
    observed = sorted(d.site_id.unique())
    categories = {c: sorted(d[c].dropna().unique()) for c in ['wa_station', 'wb_station']}
    meta = {'model': a.model, 'rows': len(d), 'observed_sites': len(observed),
            'train_start': str(d.ts.min()), 'train_end': str(d.ts.max()),
            'features': features, 'observed_site_ids': observed, 'categories': categories,
            'frozen_config_sha256': sha(config_path), 'cache_manifest_sha256': sha(a.cache/'manifest.json'),
            'feature_engine_sha256': sha(ROOT/'src/features.py'), 'script_sha256': sha(__file__),
            'truth_2024_used': False, 'early_stopping': False,
            'parameters': config['training_parameters'][a.model]}
    y = d.pop('target_cf').to_numpy(copy=True)
    weights = d.pop('sample_weight').to_numpy(copy=True)
    weights /= weights.mean()
    d.drop(columns=['site_id', 'ts'], inplace=True)
    # Column selection is checked without duplicating the large numeric matrix.
    if list(d.columns) != features:
        d = d[features]
    began = time.time()
    print(json.dumps({'phase': 'refit', 'model': a.model, 'rows': meta['rows'],
                      'sites': len(observed), 'features': len(features)}), flush=True)
    if a.model == 'lightgbm':
        for c in categories:
            d[c] = pd.Categorical(d[c], categories=categories[c])
        params = dict(config['training_parameters']['lightgbm'])
        rounds = params.pop('n_estimators')
        params.update({'seed': config['seed'], 'num_threads': config['threads'], 'verbosity': -1})
        dataset = lgb.Dataset(d, label=y, weight=weights, categorical_feature=list(categories), free_raw_data=True)
        dataset.construct()
        del d, y, weights
        gc.collect()
        def progress(env):
            if (env.iteration+1) % 100 == 0 or env.iteration+1 == rounds:
                print(json.dumps({'model': 'lightgbm', 'iteration': env.iteration+1,
                                  'iterations': rounds, 'seconds': time.time()-began}), flush=True)
        model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[progress])
        model.save_model(str(model_path))
        meta['trained_iterations'] = model.current_iteration()
    else:
        for c in categories:
            d[c] = d[c].astype('object').fillna('__missing__')
        pool = Pool(d, label=y, weight=weights, cat_features=list(categories), thread_count=config['threads'])
        del d, y, weights
        gc.collect()
        # CPU default 254 borders, unchanged model settings; quantization reduces RAM.
        pool.quantize(border_count=254, random_seed=config['seed'])
        gc.collect()
        print(json.dumps({'phase': 'pool_quantized', 'rows': pool.num_row(), 'features': pool.num_col()}), flush=True)
        pool_path = a.out/'catboost_training.quantized'
        pool.save(str(pool_path))
        meta.update({'status': 'quantized_pool_ready', 'quantized_pool_sha256': sha(pool_path),
                     'prequantization': {'border_count': 254, 'random_seed': config['seed']}})
        (a.out/'catboost_pool_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(json.dumps({'phase': 'quantized_pool_saved', 'rows': meta['rows'],
                          'bytes': pool_path.stat().st_size}), flush=True)
        return
    meta.update({'status': 'complete', 'seconds': time.time()-began, 'model_sha256': sha(model_path)})
    if meta['trained_iterations'] != config['iterations'][a.model]:
        raise ValueError('Iteration count differs from frozen selection')
    (a.out/(a.model+'_run.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ['model', 'status', 'rows', 'observed_sites', 'trained_iterations', 'seconds', 'model_sha256']}), flush=True)


if __name__ == '__main__':
    main()
