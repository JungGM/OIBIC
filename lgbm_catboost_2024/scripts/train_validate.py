"""First development validation; no October-December holdout or 2024 truth read."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from features import NAMES


def metrics(d, prediction):
    e = np.asarray(prediction)-d.target_cf.to_numpy()
    cap = d.capacity_kw.to_numpy()
    w = d.sample_weight.to_numpy()
    return {'rows': len(d), 'sites': int(d.site_id.nunique()),
            'mae_cf': float(np.mean(abs(e))), 'mae_kwh': float(np.mean(abs(e)*cap)),
            'weighted_mae_cf': float(np.sum(w*abs(e))/np.sum(w)),
            'bias_kwh': float(np.mean(e*cap)),
            'site_penalty_proxy_krw': float(.1*np.sum(w*abs(e)))}


def matrix(d, model, categories):
    x = d[NAMES].copy()
    for col in ['wa_station', 'wb_station']:
        if model == 'lightgbm':
            x[col] = pd.Categorical(x[col], categories=categories[col])
        else:
            x[col] = x[col].astype('object').fillna('__missing__')
    return x


def fit_pair(train, valid, out, categories, seed, threads, label):
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    report = {}
    weights = train.sample_weight.to_numpy()
    weights = weights/weights.mean()
    eval_weight = valid.sample_weight.to_numpy()
    eval_weight = eval_weight/eval_weight.mean()
    for name in ['lightgbm', 'catboost']:
        print(json.dumps({'phase': 'fit', 'experiment': label, 'model': name,
                          'train_rows': len(train), 'validation_rows': len(valid)}), flush=True)
        begin = time.time()
        x = matrix(train, name, categories)
        v = matrix(valid, name, categories)
        if name == 'lightgbm':
            model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=600,
                                     learning_rate=.04, num_leaves=31, min_child_samples=200,
                                     reg_lambda=5., colsample_bytree=.85,
                                     n_jobs=threads, random_state=seed, verbosity=-1,
                                     deterministic=True, force_col_wise=True)
            model.fit(x, train.target_cf, sample_weight=weights,
                      eval_set=[(v, valid.target_cf)], eval_sample_weight=[eval_weight],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
            prediction = model.predict(v)
            model.booster_.save_model(str(out/'lightgbm.txt'))
            iterations = model.best_iteration_
        else:
            model = CatBoostRegressor(iterations=600, learning_rate=.04, depth=6,
                                      loss_function='MAE', l2_leaf_reg=5.,
                                      random_seed=seed, thread_count=threads,
                                      allow_writing_files=False, verbose=False)
            pool = Pool(x, train.target_cf, weight=weights, cat_features=['wa_station', 'wb_station'])
            eval_pool = Pool(v, valid.target_cf, weight=eval_weight, cat_features=['wa_station', 'wb_station'])
            model.fit(pool, eval_set=eval_pool, early_stopping_rounds=60, use_best_model=True)
            prediction = model.predict(v)
            model.save_model(str(out/'catboost.cbm'))
            iterations = model.tree_count_
        preds[name] = np.maximum(prediction, 0)
        report[name] = {**metrics(valid, preds[name]), 'iterations': int(iterations), 'seconds': time.time()-begin}
        print(json.dumps({'phase': 'fitted', 'experiment': label, 'model': name, **report[name]}), flush=True)
        del x, v, model
    # Fixed blend for first diagnostic only. Weight optimization belongs to stage 5.
    preds['blend_50_50'] = .5*(preds['lightgbm']+preds['catboost'])
    report['blend_50_50'] = metrics(valid, preds['blend_50_50'])
    known = set(train.site_id)
    for name, prediction in preds.items():
        breakdown = {}
        for label2, mask in [('seen', valid.site_id.isin(known)), ('unseen', ~valid.site_id.isin(known)),
                             ('peak_11_15', valid.ts.dt.hour.between(11, 15)),
                             ('daylight', valid.solar_q > 0)]:
            if mask.any():
                breakdown[label2] = metrics(valid.loc[mask], prediction[mask.to_numpy()])
        report[name]['breakdown'] = breakdown
        report[name]['monthly'] = {str(month): metrics(valid.loc[mask], prediction[mask.to_numpy()]) for month in sorted(valid.ts.dt.month.unique()) if (mask := valid.ts.dt.month.eq(month)).any()}
    local = valid[['site_id', 'ts', 'energy', 'capacity_kw']].copy()
    for name, prediction in preds.items():
        local[name+'_kwh'] = prediction*valid.capacity_kw.to_numpy()
    local.to_parquet(out/'validation_predictions.parquet', index=False)
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    config = json.loads((ROOT/'configs/experiment_2024.json').read_text(encoding='utf-8'))
    a = pd.concat([pd.read_parquet(f) for f in sorted((args.cache/'2022').glob('*.parquet'))], ignore_index=True)
    # Read only development rows, not the final holdout, from disk.
    b = pd.concat([pd.read_parquet(f, filters=[('ts', '<=', pd.Timestamp(config['development_end']))])
                   for f in sorted((args.cache/'2023').glob('*.parquet'))], ignore_index=True)
    a = a[a.ts <= pd.Timestamp(config['development_train_end'])].reset_index(drop=True)
    b = b[b.ts >= pd.Timestamp(config['development_start'])].reset_index(drop=True)
    assert a.ts.max() < b.ts.min() and b.ts.max() <= pd.Timestamp(config['development_end'])
    categories = {c: sorted(set(a[c].dropna())) for c in ['wa_station', 'wb_station']}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'config_sha256': hashlib.sha256((ROOT/'configs/experiment_2024.json').read_bytes()).hexdigest(), 'status': 'development_only', 'train_rows': len(a), 'development_rows': len(b),
              'train_end': str(a.ts.max()), 'validation_end': str(b.ts.max()), 'features': len(NAMES),
              'oct_dec_holdout_evaluated': False, 'truth_2024_used': False,
              'note': 'Site-weighted loss is a proxy, not the official group score. Blend fixed at 50:50.'}
    report['main'] = fit_pair(a, b, args.out/'main', categories, config['seed'], config['threads'], 'main')
    candidates = sorted(a.site_id.unique(), key=lambda s: hashlib.sha256((str(config['seed'])+s).encode()).hexdigest())
    hidden = set(candidates[:max(1, len(candidates)//5)])
    cold_train = a[~a.site_id.isin(hidden)].reset_index(drop=True)
    # Early stopping uses retained sites. Entirely hidden sites are evaluated after model fitting.
    retained_validation = b[b.site_id.isin(set(cold_train.site_id))].reset_index(drop=True)
    cold_categories = {c: sorted(set(cold_train[c].dropna())) for c in ['wa_station', 'wb_station']}
    report['cold_development'] = fit_pair(cold_train, retained_validation, args.out/'cold', cold_categories,
                                         config['seed'], config['threads'], 'whole_site_exclusion')
    hidden_valid = b[b.site_id.isin(hidden)].reset_index(drop=True)
    if len(hidden_valid):
        cold_results = {}
        for name in ['lightgbm', 'catboost']:
            if name == 'lightgbm':
                model = lgb.Booster(model_file=str(args.out/'cold/lightgbm.txt'))
            else:
                model = CatBoostRegressor().load_model(str(args.out/'cold/catboost.cbm'))
            prediction = np.maximum(model.predict(matrix(hidden_valid, name, cold_categories)), 0)
            cold_results[name] = metrics(hidden_valid, prediction)
        report['whole_site_holdout'] = {'excluded_site_count': len(hidden), 'metrics': cold_results}
    (args.out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
