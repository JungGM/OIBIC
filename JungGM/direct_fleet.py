"""Direct aggregate generation forecaster. Private data never leave the machine.

Missing historical production is never filled: each training aggregate uses
exactly its observed sites for both labels and forecast features. Inference
uses all sites. Chronological validation uses the same cohorts as the V8 run.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pipeline as p
import features as ft
import improvement_experiment as ex
import fleet_residual as fr
from validate_submission import validate
from submission_guard import ensure_writable

RUN=ex.HERE/'_local/direct_fleet_v10'
FEATURES=[n for n in ft.feature_schema()['enhanced'] if n not in ft.CATEGORICAL and n not in ['cap','wa_dist','wb_dist']]
SPREAD=['r17_wa_cloud','r17_wb_cloud','r17_wb_uv_idx','r17_wb_rel_hum','cosz_positive','lat','lon']
CONFIGS=[dict(objective=o,solar_normalized=s,leaves=l) for o in ['regression_l1','regression'] for s in [False,True] for l in [7,15]]


def aggregate(z,meta,observed):
    if observed:z=z[z.energy.notna()&(z.energy>=0)].copy()
    else:z=z.copy()
    z['capacity']=z.site_id.map(meta.capacity_kw);z['group']=z.site_id.map(meta.current_group.map({'A':'G1','B':'G2'}))
    pieces=[]
    for group,x in z.groupby('group'):
        key=x.ts;w=x.capacity
        cap=w.groupby(key).sum();count=x.site_id.groupby(key).nunique()
        columns={'capacity':cap,'sites':count}
        if observed:columns['energy']=x.energy.groupby(key).sum()
        for c in FEATURES:
            good=np.isfinite(x[c]);weight=w.where(good,0)
            denom=weight.groupby(key).sum().replace(0,np.nan)
            mean=(x[c].fillna(0)*weight).groupby(key).sum()/denom
            columns['avg_'+c]=mean
            if c in SPREAD:
                sq=(x[c].fillna(0)**2*weight).groupby(key).sum()/denom
                columns['spread_'+c]=np.sqrt(np.maximum(sq-mean**2,0))
        g=pd.DataFrame(columns);g['group']=group;g['group_code']=float(group=='G2')
        if observed:g=g[g.sites>=20]
        pieces.append(g.reset_index())
    return pd.concat(pieces,ignore_index=True)


def build(data,kind):
    RUN.mkdir(parents=True,exist_ok=True)
    target=RUN/f'{kind}.parquet'
    if target.exists():return pd.read_parquet(target)
    meta=pd.read_csv(data/'sites.csv').set_index('site_id');frames=[]
    if kind=='training':months=pd.period_range('2022-07','2025-05',freq='M');keys=None
    elif kind=='production':months=pd.period_range('2025-06','2026-05',freq='M');keys=None
    else:
        keys=fr.source_panel()[['site_id','ts']]
        months=pd.period_range('2024-07','2025-05',freq='M')
    for month in months:
        split='test' if kind=='production' else 'train'
        columns=['site_id','ts']+FEATURES+(['energy'] if kind=='training' else [])
        z=pd.read_parquet(ex.CACHE/split/f'{month}.parquet',columns=columns)
        if keys is not None:z=z.merge(keys,on=['site_id','ts'],how='inner',validate='one_to_one')
        frames.append(aggregate(z,meta,kind=='training'))
        p.log(f'Direct fleet {kind} {month}')
    d=pd.concat(frames,ignore_index=True).sort_values(['ts','group']).reset_index(drop=True)
    if kind=='training':d['weight']=d.ts.map(ex.prices(data).weight)
    elif kind=='validation':
        b=fr.build(data)[['ts','group','v8','energy','weight','capacity']].rename(columns={'capacity':'reference_capacity'})
        d=d.merge(b,on=['ts','group'],validate='one_to_one')
        if not np.allclose(d.capacity,d.reference_capacity):raise ValueError('Validation population differs')
    d.to_parquet(target,index=False);return d


def names(d):return [c for c in d if c.startswith(('avg_','spread_')) or c=='group_code']
def normalizer(d,cfg):return np.maximum(d.avg_cosz_positive.to_numpy(),.15) if cfg['solar_normalized'] else np.ones(len(d))


def fit(train,origin,cfg):
    cutoff=p.issue_deadline(p.stamp(origin))
    z=train[(train.ts<=cutoff)&(train.avg_cosz_positive>0)&(train.sites>=20)]
    model=lgb.LGBMRegressor(objective=cfg['objective'],n_estimators=600,learning_rate=.025,
        num_leaves=cfg['leaves'],min_child_samples=60,reg_lambda=10,
        colsample_bytree=.8,max_bin=127,n_jobs=4,verbosity=-1,random_state=20261103,
        deterministic=True,force_col_wise=True)
    norm=normalizer(z,cfg);y=z.energy/z.capacity/norm
    w=z.weight*z.capacity*norm;w/=w.mean()
    model.fit(z[names(z)],y,sample_weight=w)
    return model,dict(cutoff=str(cutoff),last_label=str(z.ts.max()),rows=len(z),features=names(z),config=cfg)


def predict(model,d,cfg):
    return np.where(d.avg_cosz_positive>0,np.maximum(0,model.predict(d[names(d)],num_threads=4))*normalizer(d,cfg)*d.capacity,0)


def evaluate(data):
    train=build(data,'training');valid=build(data,'validation')
    choices=[]
    for i,cfg in enumerate(CONFIGS):
        chunks=[]
        for start,end in [('2024-07-01 01:00','2025-01-01 00:00'),('2025-01-01 01:00','2025-03-01 00:00')]:
            model,_=fit(train,start,cfg)
            z=valid[valid.ts.between(p.stamp(start),p.stamp(end))].copy()
            z['direct']=predict(model,z,cfg);chunks.append(z)
        z=pd.concat(chunks,ignore_index=True)
        z.to_parquet(RUN/f'selection_{i}.parquet',index=False)
        for strength in [.25,.5,.75,1.]:
            rows=fr.summary(z,(1-strength)*z.v8+strength*z.direct)
            score=.5*np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows[:6]])+.5*np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows[6:]])
            choices.append(dict(config=cfg,strength=strength,score=float(score),months=rows))
        p.log(f'Direct fleet selection {i+1}/{len(CONFIGS)}')
    selected=min(choices,key=lambda v:v['score']);p.save_json(RUN/'selected.json',dict(selected=selected,choices=choices))
    model,record=fit(train,'2025-03-01 01:00',selected['config'])
    z=valid[valid.ts.between(p.stamp('2025-03-01 01:00'),p.stamp('2025-06-01 00:00'))].copy()
    z['direct']=predict(model,z,selected['config']);a=selected['strength'];z['candidate']=(1-a)*z.v8+a*z.direct
    rows=fr.summary(z,z.candidate);gain=100*(1-fr.penalty(z,z.candidate)/fr.penalty(z,z.v8))
    accepted=selected['score']<1 and gain>=1 and min(r['gain_pct'] for r in rows)>-2
    result=dict(accepted=bool(accepted),selected=selected,months=rows,total_gain_pct=gain,fit=record)
    p.save_json(RUN/'evaluation.json',result);z.to_parquet(RUN/'development_predictions.parquet',index=False)
    model.booster_.save_model(str(RUN/'development_model.txt'))
    p.log(str({k:v for k,v in result.items() if k not in ['fit','selected']}))


def produce(data):
    ensure_writable(ex.HERE/'제출/improved_v10')
    r=ex.read(RUN/'evaluation.json')
    if not r['accepted']:raise ValueError('Rejected by chronological validation')
    cfg=r['selected']['config'];a=r['selected']['strength']
    model,record=fit(build(data,'training'),'2025-06-01 01:00',cfg)
    model.booster_.save_model(str(RUN/'production_model.txt'));record['model_sha256']=p.sha256(RUN/'production_model.txt')
    d=build(data,'production');d['direct']=predict(model,d,cfg)
    v8=pd.read_csv(ex.HERE/'제출/improved_v8/forecast.csv');v8.ts=pd.to_datetime(v8.ts).dt.tz_convert('Asia/Seoul')
    d=d.merge(v8,on=['ts','group'],validate='one_to_one');d.forecast_kwh=(1-a)*d.forecast_kwh+a*d.direct
    out=ex.HERE/'제출/improved_v10';out.mkdir(parents=True,exist_ok=True)
    pd.read_csv(ex.HERE/'제출/composition.csv').to_csv(out/'composition.csv',index=False)
    d[['group','ts','forecast_kwh']].to_csv(out/'forecast.csv',index=False)
    errors=validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS',official_submitted=False,fit=record,result=r,
        files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']}))
    p.log('V10 production complete and validated')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate');a=ap.parse_args()
    globals()[a.stage](a.data)
