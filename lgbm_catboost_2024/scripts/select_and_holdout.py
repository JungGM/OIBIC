"""Stage 5: one feature ablation, guarded group blend selection, sealed holdout."""
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from train_validate import metrics


def load_period(cache, year, start, end, columns=None):
    return pd.concat([pd.read_parquet(f, columns=columns,
                     filters=[('ts', '>=', start), ('ts', '<=', end)])
                     for f in sorted((cache/str(year)).glob('*.parquet'))], ignore_index=True)


def matrix(d, names, categories=None):
    x = d[names].copy()
    for col in ['wa_station', 'wb_station']:
        if col not in x:
            continue
        if categories is not None:
            known = x[col].where(x[col].isin(categories[col]))
            x[col] = pd.Categorical(known, categories=categories[col])
        else:
            x[col] = x[col].astype('object').fillna('__missing__')
    return x


def aggregation(d, known, site_groups):
    """Only fixed training cohort, and hours complete for that cohort."""
    cohort = d[d.site_id.isin(known)]
    counts = cohort.groupby('ts').site_id.nunique()
    complete = counts[counts == len(known)].index
    mask = d.site_id.isin(known) & d.ts.isin(complete)
    sub = d.loc[mask, ['site_id', 'ts', 'energy', 'smp']].copy()
    sub['group'] = sub.site_id.map(site_groups)
    if sub.group.isna().any() or not len(sub):
        raise ValueError('No valid fixed-cohort aggregate')
    keys = pd.MultiIndex.from_frame(sub[['ts', 'group']])
    codes, unique = pd.factorize(keys)
    actual = np.bincount(codes, weights=sub.energy)
    price = sub.groupby(['ts', 'group'], sort=False).smp.first().reindex(unique).to_numpy()
    hours = pd.DatetimeIndex(unique.get_level_values(0)).hour
    w = price*np.where(np.isin(hours, [11, 12, 13, 14, 15]), 3., 1.)
    def loss(prediction_kwh):
        pred = np.bincount(codes, weights=np.asarray(prediction_kwh)[mask.to_numpy()])
        return float(np.sum(w*np.abs(pred-actual)))
    return loss, {'cohort_sites': len(known), 'complete_hours': len(complete),
                  'group_hours': len(unique), 'official_full_fleet_score': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if (a.out/'frozen_selection.json').exists():
        raise ValueError('Selection already frozen here; do not repeat holdout selection')
    a.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/experiment_2024.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'configs/features_160.json').read_text(encoding='utf-8'))
    all_names = [x['name'] for x in manifest['features']]
    base_names = [x['name'] for x in manifest['features'] if x['status'] == 'base']
    assert len(all_names) == 160 and len(base_names) == 104
    train = load_period(a.cache, 2022, pd.Timestamp(config['train_start']), pd.Timestamp(config['development_train_end']))
    valid = load_period(a.cache, 2023, pd.Timestamp(config['development_start']), pd.Timestamp(config['development_end']))
    known = set(train.site_id)
    categories = {c: sorted(train[c].dropna().unique()) for c in ['wa_station', 'wb_station']}
    groups = pd.read_csv(a.data/'sites.csv').set_index('site_id').current_group.map(config['reference_groups'])
    group_loss, cohort = aggregation(valid, known, groups)
    saved = pd.read_parquet(a.development/'main/validation_predictions.parquet')
    saved = valid[['site_id', 'ts']].merge(saved, on=['site_id', 'ts'], how='left', validate='one_to_one')
    if saved[['lightgbm_kwh', 'catboost_kwh']].isna().any().any():
        raise ValueError('Development predictions incomplete')
    pred_full = saved.lightgbm_kwh.to_numpy()/valid.capacity_kw.to_numpy()
    pred_cat = saved.catboost_kwh.to_numpy()/valid.capacity_kw.to_numpy()
    print(json.dumps({'phase': 'feature_ablation', 'train_rows': len(train), 'validation_rows': len(valid),
                      'features': len(base_names), 'cohort': cohort}), flush=True)
    started = time.time()
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=600,
                             learning_rate=.04, num_leaves=31, min_child_samples=200,
                             reg_lambda=5., colsample_bytree=.85, n_jobs=config['threads'],
                             random_state=config['seed'], verbosity=-1, deterministic=True,
                             force_col_wise=True)
    x = matrix(train, base_names, categories)
    v = matrix(valid, base_names, categories)
    weights = train.sample_weight.to_numpy()
    ew = valid.sample_weight.to_numpy()
    model.fit(x, train.target_cf, sample_weight=weights/weights.mean(),
              eval_X=v, eval_y=valid.target_cf, eval_sample_weight=[ew/ew.mean()],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)])
    reduced_path = a.out/'lightgbm_104.txt'
    model.booster_.save_model(str(reduced_path))
    pred_reduced = np.maximum(model.predict(v), 0)
    reduced_report = {**metrics(valid, pred_reduced), 'iterations': int(model.best_iteration_),
                      'seconds': time.time()-started}
    del x, v, model, train
    gc.collect()
    print(json.dumps({'phase': 'feature_ablation_complete', **reduced_report}), flush=True)

    candidates = []
    unseen = ~valid.site_id.isin(known)
    for variant, pl in [('lgb160_cat160', pred_full), ('lgb104_cat160', pred_reduced)]:
        for alpha in np.linspace(0, 1, 21):
            cf = alpha*pl+(1-alpha)*pred_cat
            row = {'variant': variant, 'lightgbm_weight': float(alpha),
                   'weighted_mae_cf': metrics(valid, cf)['weighted_mae_cf'],
                   'unseen_weighted_mae_cf': metrics(valid.loc[unseen], cf[unseen.to_numpy()])['weighted_mae_cf'],
                   'partial_group_weighted_absolute_error': group_loss(cf*valid.capacity_kw.to_numpy())}
            candidates.append(row)
    # Prespecified guards: neither overall nor unseen site-weighted loss may
    # exceed the corresponding best candidate by more than 1%.
    best_site = min(x['weighted_mae_cf'] for x in candidates)
    best_unseen = min(x['unseen_weighted_mae_cf'] for x in candidates)
    eligible = [x for x in candidates if x['weighted_mae_cf'] <= 1.01*best_site
                and x['unseen_weighted_mae_cf'] <= 1.01*best_unseen]
    if not eligible:
        raise ValueError('No candidate satisfies the preregistered guards')
    best_group = min(x['partial_group_weighted_absolute_error'] for x in eligible)
    near_best = [x for x in eligible if x['partial_group_weighted_absolute_error'] <= 1.001*best_group]
    chosen = min(near_best, key=lambda x: (abs(x['lightgbm_weight']-.5), x['variant']))
    selected_names = base_names if chosen['variant'] == 'lgb104_cat160' else all_names
    src_lgb = reduced_path if chosen['variant'] == 'lgb104_cat160' else a.development/'main/lightgbm.txt'
    selected_dir = a.out/'selected_models'
    selected_dir.mkdir(exist_ok=True)
    shutil.copy2(src_lgb, selected_dir/'lightgbm.txt')
    shutil.copy2(a.development/'main/catboost.cbm', selected_dir/'catboost.cbm')
    main_report = json.loads((a.development/'main/metrics.json').read_text(encoding='utf-8'))
    freeze = {'status': 'frozen_before_holdout_read', 'selection': chosen,
              'selection_period': [config['development_start'], config['development_end']],
              'holdout_period': [config['holdout_start'], config['holdout_end']],
              'holdout_models_trained_on': '2022 only; no refit on 2023 labels before this holdout',
              'features': {'lightgbm': selected_names, 'catboost': all_names},
              'iterations': {'lightgbm': reduced_report['iterations'] if len(selected_names) == 104 else main_report['lightgbm']['iterations'],
                             'catboost': main_report['catboost']['iterations']},
              'categories': categories, 'seed': config['seed'], 'threads': config['threads'],
              'guards': {'maximum_overall_and_unseen_loss_above_best': .01, 'group_near_tie_tolerance': .001},
              'cohort': cohort,
              'model_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in selected_dir.iterdir()},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'truth_2024_used': False}
    (a.out/'feature_ablation_and_grid.json').write_text(json.dumps({'lightgbm104': reduced_report, 'candidates': candidates}, indent=2), encoding='utf-8')
    frozen_path = a.out/'frozen_selection.json'
    frozen_path.write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    freeze_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    print(json.dumps({'phase': 'selection_frozen', 'selection': chosen, 'features_lightgbm': len(selected_names),
                      'features_catboost': len(all_names), 'frozen_sha256': freeze_hash}), flush=True)
    del valid, saved, pred_full, pred_reduced, pred_cat
    gc.collect()

    # Only now read the final October–December holdout, once, with settings fixed.
    holdout = load_period(a.cache, 2023, pd.Timestamp(config['holdout_start']), pd.Timestamp(config['holdout_end']))
    outputs = {}
    for name in ['lightgbm', 'catboost']:
        if name == 'lightgbm':
            m = lgb.Booster(model_file=str(selected_dir/'lightgbm.txt'))
            x = matrix(holdout, selected_names, categories)
            prediction = m.predict(x)
        else:
            m = CatBoostRegressor().load_model(str(selected_dir/'catboost.cbm'))
            x = matrix(holdout, all_names)
            prediction = m.predict(x, thread_count=config['threads'])
        outputs[name] = np.maximum(prediction, 0)
        if not np.isfinite(outputs[name]).all():
            raise ValueError('Invalid holdout prediction')
        del m, x
        gc.collect()
    alpha = chosen['lightgbm_weight']
    outputs['selected_blend'] = alpha*outputs['lightgbm']+(1-alpha)*outputs['catboost']
    report = {'status': 'holdout_evaluated_once_after_freeze', 'frozen_selection_sha256': freeze_hash,
              'selection': chosen, 'rows': len(holdout), 'sites': int(holdout.site_id.nunique()),
              'holdout_start': str(holdout.ts.min()), 'holdout_end': str(holdout.ts.max()),
              'official_score': False, 'truth_2024_used': False, 'metrics': {}}
    hloss, hcohort = aggregation(holdout, known, groups)
    report['cohort'] = hcohort
    for name, prediction in outputs.items():
        diag = metrics(holdout, prediction)
        diag['partial_group_weighted_absolute_error'] = hloss(prediction*holdout.capacity_kw.to_numpy())
        diag['breakdown'] = {}
        for label, mask in [('seen', holdout.site_id.isin(known)), ('unseen', ~holdout.site_id.isin(known)),
                            ('daylight', holdout.solar_q > 0), ('peak_11_15', holdout.ts.dt.hour.between(11, 15))]:
            if mask.any():
                diag['breakdown'][label] = metrics(holdout.loc[mask], prediction[mask.to_numpy()])
        report['metrics'][name] = diag
    local = holdout[['site_id', 'ts', 'energy', 'capacity_kw']].copy()
    for name, prediction in outputs.items():
        local[name+'_kwh'] = prediction*holdout.capacity_kw.to_numpy()
    local.to_parquet(a.out/'holdout_predictions.parquet', index=False)
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != freeze_hash:
        raise ValueError('Frozen configuration changed during holdout')
    (a.out/'holdout_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'phase': 'holdout_complete', 'rows': len(holdout), 'sites': report['sites'],
                      'selected_blend': report['metrics']['selected_blend']}), flush=True)


if __name__ == '__main__':
    main()
