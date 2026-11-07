"""Reproduce the pinned 11.03 compact XGBoost on JungGM's exact OOF rows.

Uses only local participant data. Historical months have already been inspected;
these are exploratory chronological checks, never an untouched final test.
"""
import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pipeline as p
import features as ft
import improvement_experiment as ex
from validate_submission import validate

RUN = ex.HERE/'_local/improvement_v6_compact_xgb'
SOURCE = ex.HERE.parent/'11.03'
SOURCE_HASHES = {
    'baseline_reference.py':'31dbe968f13148b88f1a85849b4a009e732300539ec9a7b32d5773e08af2a8dd',
    'run_experiment.py':'9ba626324081def68e67b657a98f7e8383d2f2bd2fdfabbd80a1a2c0c6fa0f35',
    'feature_sets.json':'d03abc393ce22cdd01c718a1b8128dd005cdd98a9ab52d260726cb80b6e38ca9'}
PARAMS = dict(n_estimators=300, max_depth=8, learning_rate=.03,
    min_child_weight=100, reg_lambda=30, subsample=.8, colsample_bytree=.8,
    objective='reg:squarederror', tree_method='hist', n_jobs=4, random_state=42)


def protocol():
    RUN.mkdir(parents=True, exist_ok=True)
    obj = dict(version=6, source_hashes=SOURCE_HASHES, params=PARAMS,
        training='all provided observed labels from 2021, up to origin previous day 17:00',
        features='52 exact 11.03 features; provided round17 only; no labels or SMP in X',
        weight='capacity * historical SMP * k, normalized mean 1',
        origins=['2024-07-01 01:00','2025-01-01 01:00','2025-03-01 01:00'],
        selection='July-Dec relative monthly loss 50%, Jan-Feb 50%, original identical cohorts/hours',
        blends=[0,.25,.5,.75,1.], reference_groups_fixed=True,
        gate='April >=1% and paired 3-day block lower90 >0; May >=0%; March >=-1%',
        disclosure='Repeated historical research; not an untouched holdout. No official score tuning.')
    path=RUN/'protocol_before_results.json'
    if path.exists() and ex.read(path)!=obj:raise ValueError('Protocol changed')
    if not path.exists():p.save_json(path,obj)


def source(data):
    for name,h in SOURCE_HASHES.items():
        if p.sha256(SOURCE/name)!=h:raise ValueError(f'Pinned source changed: {name}')
    os.environ['OIBC_DATA']=str(data.resolve())
    sys.dont_write_bytecode=True
    # Reviewed module top levels only define functions/constants; never run its CLI.
    for name in ['baseline_reference','run_experiment']:
        spec=importlib.util.spec_from_file_location(name,SOURCE/f'{name}.py')
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
        spec.loader.exec_module(mod)
    return mod


def weather(data, split):
    a=pd.read_parquet(data/split/'forecast_wa.parquet')
    b=pd.read_parquet(data/split/'forecast_wb.parquet')
    for d,station in [(a,'wa_station'),(b,'wb_station')]:
        if d.duplicated([station,'round','ts']).any():raise ValueError('Duplicate weather keys')
    return a,b


def build(src, g, sites, a, b, ix=None):
    d=src.v1.b.features(g,sites,a,b,ix,17)
    names=src.enhanced(d)
    if names!=ex.read(SOURCE/'feature_sets.json')['enhanced'] or len(names)!=52:
        raise ValueError('Feature contract mismatch')
    if g is not None and not d[['site_id','ts']].equals(g[['site_id','ts']]):
        raise ValueError('Row order mismatch')
    if d.duplicated(['site_id','ts']).any():raise ValueError('Duplicate features')
    return d,names


def train_one(data, origin, dest):
    dest.mkdir(parents=True,exist_ok=True)
    origin=p.stamp(origin);cutoff=p.issue_deadline(origin)
    src=source(data);model_path=dest/'compact_xgb.ubj'
    if model_path.exists() and (dest/'metadata.json').exists():
        record=ex.read(dest/'metadata.json')
        if record['protocol']!=ex.read(RUN/'protocol_before_results.json') or record['cutoff']!=str(cutoff):
            raise ValueError('Saved model provenance mismatch')
        model=XGBRegressor(n_jobs=4);model.load_model(model_path)
        return model,record['features']
    sites=src.inputs()
    g=pd.read_parquet(data/'train/generation.parquet',filters=[('ts','>=',p.stamp('2021-01-01')),('ts','<=',cutoff)])
    a,b=weather(data,'train')
    a=a[a.ts.isin(g.ts.unique())];b=b[b.ts.isin(g.ts.unique())]
    d,names=build(src,g,sites,a,b)
    mask=(d.elev>0)&np.isfinite(d.energy)&(d.energy>=0)&(d.ts<=cutoff)
    y=(d.loc[mask,'energy']/d.loc[mask,'capacity_kw']).clip(0,1.2).astype('float64')
    weight=d.loc[mask,'ts'].map(ex.prices(data).weight)*d.loc[mask,'capacity_kw']
    if not np.isfinite(weight).all() or (weight<0).any() or weight.mean()<=0:
        raise ValueError('Invalid historical weights')
    weight/=weight.mean()
    record=dict(origin=str(origin),cutoff=str(cutoff),first_target=str(d.loc[mask,'ts'].min()),
        last_target=str(d.loc[mask,'ts'].max()),rows=int(mask.sum()),features=names,
        protocol=ex.read(RUN/'protocol_before_results.json'))
    X=d.loc[mask,names].astype('float32')
    del d,g,a,b;gc.collect()
    p.log(f'Compact XGB {origin}: {len(X):,} rows, {len(names)} features')
    model=XGBRegressor(**PARAMS).fit(X,y,sample_weight=weight)
    model.save_model(model_path);record['model_sha256']=p.sha256(model_path)
    p.save_json(dest/'metadata.json',record)
    del X,y,weight;gc.collect()
    return model,names


def predict(data,model,names,split,months,out):
    src=source(data);sites=src.inputs();a,b=weather(data,split);parts=[]
    loaded=XGBRegressor(n_jobs=4);loaded.load_model(out.parent/'compact_xgb.ubj')
    maximum=0.
    for month in months:
        start,end=ft.month_bounds(month);ix=pd.date_range(start,end,freq='h')
        d,actual_names=build(src,None,sites,a[a.ts.isin(ix)],b[b.ts.isin(ix)],ix)
        if actual_names!=names:raise ValueError('Inference features differ')
        X=d[names].astype('float32');raw=model.predict(X)
        diff=float(np.max(abs(raw-loaded.predict(X))));maximum=max(maximum,diff)
        pred=np.maximum(raw,0)*d.capacity_kw.to_numpy();pred[d.elev.to_numpy()<=0]=0
        z=d[['site_id','ts']].copy();z['new']=pred
        if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid prediction')
        parts.append(z);del d,X;gc.collect()
    if maximum!=0:raise ValueError('Saved model inference differs')
    pd.concat(parts,ignore_index=True).to_parquet(out,index=False)
    p.save_json(out.parent/'reproduction.json',dict(max_abs_saved_model_difference=maximum,
        rows=sum(len(d) for d in parts),model_sha256=p.sha256(out.parent/'compact_xgb.ubj')))
    p.log(f'Predictions and exact saved-model reproduction: {out.parent.name}')


def train(data):
    for part,origin,start,end in [
        ('seasonal_proxy','2024-07-01 01:00','2024-07','2024-12'),
        ('complete_fleet','2025-01-01 01:00','2025-01','2025-02'),
        ('development','2025-03-01 01:00','2025-03','2025-05')]:
        dest=RUN/part;model,names=train_one(data,origin,dest)
        predict(data,model,names,'train',pd.period_range(start,end,freq='M'),dest/'predictions.parquet')
        del model;gc.collect()


def evaluate(data):
    # Identical selection, fixed groups and spring adoption gate as v5.
    import group_loss_experiment as group_experiment
    group_experiment.evaluate(data,run=RUN)
    old=pd.read_parquet(ex.OLD/'development/site_predictions.parquet').rename(columns={'ensemble':'base'})
    site=old.merge(pd.read_parquet(RUN/'development/predictions.parquet'),on=['site_id','ts'],validate='one_to_one')
    g=ex.group_frame(site,data);diagnostics=[]
    nw=ex.read(RUN/'evaluation.json')['selected_new_weight']
    for month,z in g.groupby((g.ts-pd.Timedelta(hours=1)).dt.month):
        daylight=(z.base>1)|(z['new']>1)
        a=(z.base-z.energy)[daylight].to_numpy();b=(z['new']-z.energy)[daylight].to_numpy()
        subset=site[(site.ts-pd.Timedelta(hours=1)).dt.month==month]
        peak=z.ts.dt.hour.between(11,15)
        diagnostics.append(dict(month=int(month),daylight_group_residual_correlation=float(np.corrcoef(a,b)[0,1]),
            site_mae_base_kwh=float(abs(subset.base-subset.energy).mean()),
            site_mae_new_kwh=float(abs(subset['new']-subset.energy).mean()),
            peak_baseline_penalty=ex.loss(z[peak],z.loc[peak,'base']),
            peak_candidate_penalty=ex.loss(z[peak],(1-nw)*z.loc[peak,'base']+nw*z.loc[peak,'new']),
            new_only_bias_group_kwh=float((z['new']-z.energy).mean()),
            baseline_bias_group_kwh=float((z.base-z.energy).mean())))
    p.save_json(RUN/'diagnostics.json',diagnostics)


def produce(data):
    result=ex.read(RUN/'evaluation.json')
    if not result['accepted']:
        p.log('Not adopted; official submission preserved.');return
    dest=RUN/'production';model,names=train_one(data,'2025-06-01 01:00',dest)
    predict(data,model,names,'test',pd.period_range('2025-06','2026-05',freq='M'),dest/'predictions.parquet')
    old=pd.read_parquet(ex.OLD/'production/site_predictions.parquet')
    new=pd.read_parquet(dest/'predictions.parquet')
    d=old.merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
    if len(d)!=253*8760 or not np.isfinite(d['new']).all():raise ValueError('Incomplete production')
    nw=result['selected_new_weight'];d['forecast_kwh']=(1-nw)*d.prediction_kwh+nw*d['new']
    comp=pd.read_csv(ex.HERE/'제출/composition.csv');d['group']=d.site_id.map(comp.set_index('site_id').group)
    out=ex.HERE/'제출/improved_v6';out.mkdir(exist_ok=True)
    comp.to_csv(out/'composition.csv',index=False)
    d.groupby(['group','ts'],as_index=False).forecast_kwh.sum().to_csv(out/'forecast.csv',index=False)
    errors=validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS',official_submitted=False,result=result,
        files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']}))
    p.log(f'Validated candidate: {out}')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--stage',required=True,choices=['train','evaluate','produce'])
    args=ap.parse_args();protocol();globals()[args.stage](args.data)
