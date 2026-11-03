"""Separate, predeclared three-expert selection using already fitted team models.

Never change the v7 50:50 method after seeing results. This experiment selects
15 convex combinations of original/LGB/Cat on July-Feb only, then applies the
same spring gate. Unused components need no spring/production refit.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import pipeline as p
import improvement_experiment as ex
import group_loss_experiment as common
import team_pair_experiment as team
from validate_submission import validate

RUN=ex.HERE/'_local/improvement_v8_components'
TEAM=team.RUN


def protocol():
    RUN.mkdir(parents=True,exist_ok=True);team.source()
    obj=dict(version=8,source_hashes=team.PINS,
        experts=['original_ensemble','team_lightgbm','team_catboost'],
        choices=[dict(base=i/4,lightgbm=j/4,catboost=(4-i-j)/4) for i in range(4,-1,-1) for j in range(4-i+1)],
        selection='Same July-Dec 225-site and Jan-Feb 253-site rows; monthly relative cost, block weights .5/.5',
        gate='Unchanged April >=1% and lower90>0, May>=0%, March>=-1%; fixed reference groups',
        model_settings=ex.read(team.SOURCE/'configs/final_model.json')['training_parameters'],
        training='Same 24-month origin/previous-day17 cutoff as v7; reuse fitted models when available',
        disclosure='Additional exploratory experiment declared after v7 summer pair results, before winter pair results; no untouched holdout')
    path=RUN/'protocol_before_results.json'
    if path.exists() and ex.read(path)!=obj:raise ValueError('Component protocol changed')
    if not path.exists():p.save_json(path,obj)


def select(data):
    hashes={part:p.sha256(TEAM/part/'components.parquet') for part in ['seasonal_proxy','complete_fleet']}
    frozen=RUN/'frozen_before_spring.json'
    if frozen.exists():
        record=ex.read(frozen)
        if record['component_prediction_sha256']!=hashes:raise ValueError('Component selection inputs changed')
        return record['selected']
    old=ex.old_oof()
    new=pd.concat([pd.read_parquet(TEAM/part/'components.parquet',columns=['site_id','ts','team_lgb','team_cat'])
                   for part in ['seasonal_proxy','complete_fleet']])
    d=old.merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
    if not np.isfinite(d[['energy','base','team_lgb','team_cat']]).all().all():raise ValueError('Incomplete selection panel')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    d['group']=d.site_id.map(comp)
    g=d.groupby(['ts','group'])[['energy','base','team_lgb','team_cat']].sum().reset_index()
    g['weight']=g.ts.map(ex.prices(data).weight)
    choices=[]
    for weights in ex.read(RUN/'protocol_before_results.json')['choices']:
        rows=[]
        for month,z in g.groupby((g.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')):
            pred=weights['base']*z.base+weights['lightgbm']*z.team_lgb+weights['catboost']*z.team_cat
            base=ex.loss(z,z.base);cost=ex.loss(z,pred)
            rows.append(dict(month=month,base=base,candidate=cost,ratio=cost/base,hours=int(z.ts.nunique())))
        score=.5*np.mean([r['ratio'] for r in rows[:6]])+.5*np.mean([r['ratio'] for r in rows[6:]])
        choices.append(dict(weights=weights,new_weight=1-weights['base'],score=float(score),months=rows))
    selected=min(choices,key=lambda x:x['score'])
    p.save_json(frozen,dict(selected=selected,choices=choices,component_prediction_sha256=hashes))
    p.log(f'Frozen component weights: {selected["weights"]}; selection score {selected["score"]:.6f}')
    return selected


def model_paths(data,part,weights):
    result={}
    for kind in ['lightgbm','catboost']:
        if weights[kind]==0:continue
        filename='lightgbm.txt' if kind=='lightgbm' else 'catboost.cbm'
        existing=TEAM/part/filename
        if existing.exists():
            record=ex.read(existing.parent/f'{kind}_metadata.json')
            origin=p.stamp(team.PARTS[part][0])
            if (record['origin']!=str(origin) or record['cutoff']!=str(p.issue_deadline(origin)) or
                record['model_sha256']!=p.sha256(existing) or record['protocol']['source_hashes']!=team.PINS):
                raise ValueError('Reused component provenance mismatch')
            result[kind]=existing
        else:
            previous=team.RUN
            try:
                team.RUN=RUN
                team.fit(data,part,kind,skip_selection=True,cache_root=TEAM/'cache/train')
            finally:team.RUN=previous
            result[kind]=RUN/part/filename
    return result


def predict(data,part,selected):
    weights=selected['weights'];nw=selected['new_weight']
    if nw==0:
        p.log('Original-only selection; no unused component is fitted or scored.');return
    dest=RUN/part;dest.mkdir(exist_ok=True)
    paths=model_paths(data,part,weights)
    cfg=ex.read(team.SOURCE/'configs/final_model.json')
    start,end=map(p.stamp,team.PARTS[part][1:]);ix=pd.date_range(start,end,freq='h')
    split='test' if part=='production' else 'train'
    models={kind:(lgb.Booster(model_file=str(path)) if kind=='lightgbm' else CatBoostRegressor().load_model(str(path)))
            for kind,path in paths.items()}
    metas={kind:ex.read(path.parent/f'{kind}_metadata.json') for kind,path in paths.items()}
    capacities=pd.read_csv(data/'sites.csv').set_index('site_id').capacity_kw
    parts=[]
    for sid,capacity in capacities.sort_index().items():
        d=pd.read_parquet(TEAM/'cache'/split/f'{sid}.parquet',filters=[('ts','>=',start),('ts','<=',end)])
        if len(d)!=len(ix) or not pd.DatetimeIndex(d.ts).equals(ix):raise ValueError('Component prediction grid mismatch')
        pred=np.zeros(len(d))
        for kind,model in models.items():
            x=d[cfg['features'][kind]].copy()
            for c,values in metas[kind]['categories'].items():
                x[c]=pd.Categorical(x[c].where(x[c].isin(values)),categories=values) if kind=='lightgbm' else x[c].astype('object').fillna('__missing__')
            options={'num_threads':4} if kind=='lightgbm' else {'thread_count':4}
            pred+=weights[kind]/nw*np.maximum(model.predict(x,**options),0)*float(capacity)
        if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid component predictions')
        z=d[['site_id','ts']].copy();z['new']=pred;parts.append(z)
    full=pd.concat(parts,ignore_index=True)
    if len(full)!=253*len(ix) or full.duplicated(['site_id','ts']).any():raise ValueError('Incomplete components')
    full.to_parquet(dest/'predictions.parquet',index=False)
    p.save_json(dest/'prediction_provenance.json',dict(weights=weights,rows=len(full),
        component_model_sha256={k:p.sha256(v) for k,v in paths.items()},
        source_model_paths={k:str(v.relative_to(ex.HERE)).replace('\\','/') for k,v in paths.items()}))


def evaluate(data):
    selected=select(data)
    predict(data,'development',selected)
    common.evaluate(data,run=RUN,selection=selected)
    if selected['new_weight']==0:return
    old=pd.read_parquet(ex.OLD/'development/site_predictions.parquet').rename(columns={'ensemble':'base'})
    d=old.merge(pd.read_parquet(RUN/'development/predictions.parquet'),on=['site_id','ts'],validate='one_to_one')
    g=ex.group_frame(d,data);nw=selected['new_weight'];rows=[]
    for month,z in g.groupby((g.ts-pd.Timedelta(hours=1)).dt.month):
        daylight=(z.base>1)|(z['new']>1);peak=z.ts.dt.hour.between(11,15)
        rows.append(dict(month=int(month),
            daylight_residual_correlation=float(np.corrcoef((z.base-z.energy)[daylight],(z['new']-z.energy)[daylight])[0,1]),
            peak_baseline_penalty=ex.loss(z[peak],z.loc[peak,'base']),
            peak_candidate_penalty=ex.loss(z[peak],(1-nw)*z.loc[peak,'base']+nw*z.loc[peak,'new'])))
    p.save_json(RUN/'diagnostics.json',rows)


def produce(data):
    result=ex.read(RUN/'evaluation.json')
    if not result['accepted']:
        p.log('Component candidate not adopted; official submission preserved.');return
    selected=ex.read(RUN/'frozen_before_spring.json')['selected']
    team.prepare(data,'test');predict(data,'production',selected)
    d=pd.read_parquet(ex.OLD/'production/site_predictions.parquet')
    d=d.merge(pd.read_parquet(RUN/'production/predictions.parquet'),on=['site_id','ts'],how='left',validate='one_to_one')
    if len(d)!=253*8760 or not np.isfinite(d['new']).all():raise ValueError('Incomplete production grid')
    nw=selected['new_weight'];d['forecast_kwh']=(1-nw)*d.prediction_kwh+nw*d['new']
    comp=pd.read_csv(ex.HERE/'제출/composition.csv');d['group']=d.site_id.map(comp.set_index('site_id').group)
    out=ex.HERE/'제출/improved_v8';out.mkdir(exist_ok=True)
    comp.to_csv(out/'composition.csv',index=False)
    d.groupby(['group','ts'],as_index=False).forecast_kwh.sum().to_csv(out/'forecast.csv',index=False)
    errors=validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS',official_submitted=False,result=result,
        files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['declare','select','evaluate','produce'],required=True)
    args=ap.parse_args();protocol()
    if args.stage!='declare':globals()[args.stage](args.data)
