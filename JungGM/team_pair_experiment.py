"""Pinned teammate L1 LightGBM/CatBoost, with common chronological evaluation.

Feature formulas and model settings come from lgbm_catboost_2024. Adaptation:
rolling 24-month training windows at our three origins, no early stopping or
retuning. All feature caches, fitted models and predictions remain local.
"""
import argparse
import gc
import importlib.util
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import pipeline as p
import improvement_experiment as ex
import group_loss_experiment as common
from validate_submission import validate

RUN=ex.HERE/'_local/improvement_v7_team_pair'
SOURCE=ex.HERE.parent/'lgbm_catboost_2024'
PINS={'src/features.py':'30168cc877a64231ceee4967ac6c1bdc27cf47adf9cb820a4e5671785760bbf2',
      'configs/features_160.json':'8d561f03cf22117f1fa820583536f80eb1c9d54936e6b4788b9ebc9b0d1965d3',
      'configs/final_model.json':'1c69bca3fc752fcc2e4d0aa3926df91b637853fea6f048f899c2fbce6f77ddfc'}
PARTS={'seasonal_proxy':('2024-07-01 01:00','2024-07-01 01:00','2025-01-01 00:00'),
       'complete_fleet':('2025-01-01 01:00','2025-01-01 01:00','2025-03-01 00:00'),
       'development':('2025-03-01 01:00','2025-03-01 01:00','2025-06-01 00:00'),
       'production':('2025-06-01 01:00','2025-06-01 01:00','2026-06-01 00:00')}


def source():
    for name,h in PINS.items():
        if p.sha256(SOURCE/name)!=h:raise ValueError(f'Teammate source changed: {name}')
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('teammate_features',SOURCE/'src/features.py')
    src=importlib.util.module_from_spec(spec);spec.loader.exec_module(src)
    return src


def protocol():
    RUN.mkdir(parents=True,exist_ok=True);source()
    obj=dict(version=7,source_hashes=PINS,
        model=ex.read(SOURCE/'configs/final_model.json')['training_parameters'],
        feature_count=160,seed=20261101,threads=4,teammate_lgb_weight=.5,
        train='previous 24 months, original observed nonnegative labels, cutoff prior day17',
        target='unclipped capacity factor; daytime and nighttime, same as teammate',
        training_weight='capacity * historical SMP * k, float32 normalized mean1',
        selection='original July-Dec 225-site and Jan-Feb 253-site OOF; block equal monthly relative loss',
        new_weights=[0,.25,.5,.75,1.],reference_groups_fixed=True,
        gate='April >=1%, 3-day block lower90>0; May>=0%; March>=-1%',
        disclosure='Repeated exploratory historical comparisons; no untouched test; not exact official teammate binary')
    path=RUN/'protocol_before_results.json'
    if path.exists() and ex.read(path)!=obj:raise ValueError('Protocol changed')
    if not path.exists():p.save_json(path,obj)


def prepare(data,split='train'):
    src=source();dest=RUN/'cache'/split;dest.mkdir(parents=True,exist_ok=True)
    start,end=('2022-07-01 01:00','2025-06-01 00:00') if split=='train' else PARTS['production'][1:]
    ix=pd.date_range(p.stamp(start),p.stamp(end),freq='h')
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id')
    paths=[dest/f'{sid}.parquet' for sid in sites.site_id]
    if (dest/'manifest.json').exists():
        manifest=ex.read(dest/'manifest.json')
        if manifest['source_hashes']!=PINS or not all(f.exists() for f in paths):raise ValueError('Incomplete/stale cache')
        return
    p.log(f'Team pair {split} feature cache: {len(ix)} hours x253, forecast-only full target days')
    wa=src.weather(data/split/'forecast_wa.parquet','a',ix,sites.wa_station.unique())
    wb=src.weather(data/split/'forecast_wb.parquet','b',ix,sites.wb_station.unique())
    for i,site in enumerate(sites.itertuples(index=False)):
        path=dest/f'{site.site_id}.parquet'
        if not path.exists():
            d=src.build_site(site,ix,wa[site.wa_station],wb[site.wb_station])
            if len(d)!=len(ix) or not pd.DatetimeIndex(d.ts).equals(ix):raise ValueError('Feature grid mismatch')
            d.to_parquet(path,index=False,compression='zstd')
        if (i+1)%25==0 or i==len(sites)-1:p.log(f'{split} feature cache {i+1}/{len(sites)} sites')
    p.save_json(dest/'manifest.json',dict(source_hashes=PINS,features=src.NAMES,
        start=str(ix.min()),end=str(ix.max()),hours=len(ix),sites=len(sites),labels_read=False))


def skip_unselected_development(data,part):
    if part!='development':return False
    path=RUN/'frozen_before_spring.json'
    if path.exists():
        record=ex.read(path)
        for name,h in record['prediction_sha256'].items():
            if p.sha256(RUN/name/'predictions.parquet')!=h:raise ValueError('Selection inputs changed')
        selected=record['selected']
    else:selected=common.select_model(data,run=RUN)
    if selected['new_weight']!=0:return False
    p.save_json(RUN/'early_stop.json',dict(selected_new_weight=0,
        reason='Original-only selection has exactly zero gain and cannot pass the unchanged >=1% gate.',
        development_new_model_fitted=False,development_new_model_score=None))
    p.log('Seasonal selection retained original only; no unused spring estimator is fitted.')
    return True


def fit(data,part,kind,skip_selection=False,cache_root=None):
    if not skip_selection and skip_unselected_development(data,part):return
    origin=p.stamp(PARTS[part][0]);cutoff=p.issue_deadline(origin);start=p.history_start(origin,24)
    dest=RUN/part;dest.mkdir(exist_ok=True)
    model_path=dest/('lightgbm.txt' if kind=='lightgbm' else 'catboost.cbm')
    record_path=dest/f'{kind}_metadata.json'
    cfg=ex.read(SOURCE/'configs/final_model.json');names=cfg['features'][kind]
    if model_path.exists() and record_path.exists():
        record=ex.read(record_path)
        if (record['protocol']!=ex.read(RUN/'protocol_before_results.json') or
            record['origin']!=str(origin) or record['cutoff']!=str(cutoff) or
            record['model_sha256']!=p.sha256(model_path)):
            raise ValueError('Saved model provenance mismatch')
        return
    # Read training labels independently after full-day feature calculation.
    truth=pd.read_parquet(data/'train/generation.parquet',filters=[('ts','>=',start),('ts','<=',cutoff)])
    truth=truth[np.isfinite(truth.energy)&(truth.energy>=0)]
    if truth.duplicated(['site_id','ts']).any():raise ValueError('Duplicate truth')
    frames=[]
    for sid,y in truth.groupby('site_id',sort=True):
        cache=RUN/'cache/train' if cache_root is None else Path(cache_root)
        d=pd.read_parquet(cache/f'{sid}.parquet',filters=[('ts','>=',start),('ts','<=',cutoff)])
        z=d.merge(y,on=['site_id','ts'],how='inner',validate='one_to_one')
        frames.append(z)
    d=pd.concat(frames,ignore_index=True);del frames
    if len(d)!=len(truth):raise ValueError('Training row loss')
    if d.ts.max()>cutoff:raise ValueError('Future label')
    prices=ex.prices(data)
    y=(d.energy/d.capacity_kw).to_numpy(dtype='float32')
    # Preserve teammate float32 weight construction and row order.
    weights=(d.capacity_kw*d.ts.map(prices.smp)*np.where(d.ts.dt.hour.between(11,15),3,1)).to_numpy(dtype='float32')
    if not np.isfinite(weights).all() or (weights<0).any() or weights.mean()<=0:raise ValueError('Invalid weights')
    weights/=weights.mean()
    categories={c:sorted(d[c].dropna().unique()) for c in ['wa_station','wb_station']}
    record=dict(origin=str(origin),cutoff=str(cutoff),first_target=str(d.ts.min()),last_target=str(d.ts.max()),
        rows=len(d),features=names,categories=categories,protocol=ex.read(RUN/'protocol_before_results.json'))
    d=d[names];del truth;gc.collect()
    params=dict(cfg['training_parameters'][kind])
    p.log(f'Team {kind} {part}: {len(d):,} observed rows, {len(names)} features')
    if kind=='lightgbm':
        for c,values in categories.items():d[c]=pd.Categorical(d[c],categories=values)
        probe=d.iloc[np.linspace(0,len(d)-1,min(1024,len(d)),dtype=int)].copy()
        rounds=params.pop('n_estimators');params.update(seed=cfg['seed'],num_threads=4,verbosity=-1)
        dataset=lgb.Dataset(d,label=y,weight=weights,categorical_feature=list(categories),free_raw_data=True)
        dataset.construct();del d,y,weights;gc.collect()
        model=lgb.train(params,dataset,num_boost_round=rounds,callbacks=[lambda env:p.log(f'LGB {part} {env.iteration+1}/{rounds}') if (env.iteration+1)%100==0 else None])
        model.save_model(str(model_path))
    else:
        for c in categories:d[c]=d[c].astype('object').fillna('__missing__')
        probe=d.iloc[np.linspace(0,len(d)-1,min(1024,len(d)),dtype=int)].copy()
        pool=Pool(d,label=y,weight=weights,cat_features=list(categories),thread_count=4)
        del d,y,weights;gc.collect()
        pool.quantize(border_count=254,random_seed=cfg['seed'])
        p.log(f'CatBoost {part}: quantized training pool ready')
        params.update(random_seed=cfg['seed'],thread_count=4,allow_writing_files=False,verbose=100)
        model=CatBoostRegressor(**params).fit(pool);model.save_model(str(model_path))
    loaded=lgb.Booster(model_file=str(model_path)) if kind=='lightgbm' else CatBoostRegressor().load_model(str(model_path))
    options={'num_threads':4} if kind=='lightgbm' else {'thread_count':4}
    difference=float(np.max(abs(model.predict(probe,**options)-loaded.predict(probe,**options))))
    if difference!=0:raise ValueError('Persisted model prediction differs')
    record['saved_model_probe_rows']=len(probe);record['saved_model_max_abs_difference']=difference
    record['model_sha256']=p.sha256(model_path);p.save_json(record_path,record)


def predict(data,part):
    if skip_unselected_development(data,part):return
    dest=RUN/part;start,end=map(p.stamp,PARTS[part][1:]);split='test' if part=='production' else 'train'
    ix=pd.date_range(start,end,freq='h');cfg=ex.read(SOURCE/'configs/final_model.json')
    lgb_model=lgb.Booster(model_file=str(dest/'lightgbm.txt'))
    cat=CatBoostRegressor().load_model(str(dest/'catboost.cbm'))
    meta=ex.read(dest/'lightgbm_metadata.json');frames=[]
    capacities=pd.read_csv(data/'sites.csv').set_index('site_id').capacity_kw
    for sid in sorted(capacities.index):
        d=pd.read_parquet(RUN/'cache'/split/f'{sid}.parquet',filters=[('ts','>=',start),('ts','<=',end)])
        if len(d)!=len(ix) or not pd.DatetimeIndex(d.ts).equals(ix):raise ValueError('Inference grid mismatch')
        x=d[cfg['features']['lightgbm']].copy()
        for c,values in meta['categories'].items():x[c]=pd.Categorical(x[c].where(x[c].isin(values)),categories=values)
        a=np.maximum(lgb_model.predict(x,num_threads=4),0)*float(capacities.loc[sid])
        for c in meta['categories']:x[c]=d[c].astype('object').fillna('__missing__')
        b=np.maximum(cat.predict(x[cfg['features']['catboost']],thread_count=4),0)*float(capacities.loc[sid])
        z=d[['site_id','ts']].copy();z['new']=.5*a+.5*b
        z['team_lgb']=a;z['team_cat']=b
        if not np.isfinite(z['new']).all():raise ValueError('Nonfinite predictions')
        frames.append(z)
    full=pd.concat(frames,ignore_index=True)
    if len(full)!=253*len(ix) or full.duplicated(['site_id','ts']).any():raise ValueError('Incomplete predictions')
    full.to_parquet(dest/'components.parquet',index=False)
    full[['site_id','ts','new']].to_parquet(dest/'predictions.parquet',index=False)
    p.log(f'Team pair {part} predictions complete: {len(full):,} site-hours')


def subprocess_stage(data,stage,part,kind=None):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--data',str(data),'--stage',stage,'--part',part]
    if kind:cmd+=['--model',kind]
    subprocess.run(cmd,check=True)


def train(data,parts=None):
    for part in parts or ['seasonal_proxy','complete_fleet','development']:
        for kind in ['lightgbm','catboost']:subprocess_stage(data,'fit',part,kind)
        subprocess_stage(data,'predict',part)


def evaluate(data):
    common.evaluate(data,run=RUN)
    if not (RUN/'development/components.parquet').exists():return
    old=pd.read_parquet(ex.OLD/'development/site_predictions.parquet').rename(columns={'ensemble':'base'})
    components=pd.read_parquet(RUN/'development/components.parquet')
    d=old[['site_id','ts','energy','base']].merge(components,on=['site_id','ts'],validate='one_to_one')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    d['group']=d.site_id.map(comp)
    g=d.groupby(['ts','group'])[['energy','base','new','team_lgb','team_cat']].sum().reset_index()
    g['weight']=g.ts.map(ex.prices(data).weight)
    rows=[]
    for month,z in g.groupby((g.ts-pd.Timedelta(hours=1)).dt.month):
        daylight=(z.base>1)|(z['new']>1)
        rows.append(dict(month=int(month),hours=int(z.ts.nunique()),
            component_penalties={name:ex.loss(z,z[name]) for name in ['base','new','team_lgb','team_cat']},
            daylight_residual_correlation=float(np.corrcoef((z.base-z.energy)[daylight],(z['new']-z.energy)[daylight])[0,1])))
    p.save_json(RUN/'diagnostics.json',rows)


def produce(data):
    r=ex.read(RUN/'evaluation.json')
    if not r['accepted']:
        p.log('Not adopted; original submission preserved.');return
    prepare(data,'test');train(data,['production'])
    old=pd.read_parquet(ex.OLD/'production/site_predictions.parquet')
    new=pd.read_parquet(RUN/'production/predictions.parquet')
    d=old.merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
    if len(d)!=253*8760 or not np.isfinite(d['new']).all():raise ValueError('Incomplete production')
    nw=r['selected_new_weight'];d['forecast_kwh']=(1-nw)*d.prediction_kwh+nw*d['new']
    comp=pd.read_csv(ex.HERE/'제출/composition.csv');d['group']=d.site_id.map(comp.set_index('site_id').group)
    out=ex.HERE/'제출/improved_v7';out.mkdir(exist_ok=True)
    comp.to_csv(out/'composition.csv',index=False)
    d.groupby(['group','ts'],as_index=False).forecast_kwh.sum().to_csv(out/'forecast.csv',index=False)
    errors=validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS',official_submitted=False,result=r,
        files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']}))


if __name__=='__main__':
    sys.stdout.reconfigure(line_buffering=True)
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['prepare','train','fit','predict','evaluate','produce'],required=True)
    ap.add_argument('--part',choices=list(PARTS));ap.add_argument('--model',choices=['lightgbm','catboost'])
    args=ap.parse_args();protocol()
    if args.stage=='fit':fit(args.data,args.part,args.model)
    elif args.stage=='predict':predict(args.data,args.part)
    elif args.stage=='evaluate':evaluate(args.data)
    else:globals()[args.stage](args.data)
