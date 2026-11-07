"""Physical target normalization + site-aware regression, local-only experiment."""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pipeline as p
import improvement_experiment as ex
import team_pair_experiment as team
import fleet_residual as fr
from validate_submission import validate
from submission_guard import ensure_writable

RUN=ex.HERE/'_local/solar_normalized_v11'
NORMALIZE=True
KIND='lightgbm'
CANDIDATE='improved_v11'
PARAMS=dict(objective='regression',n_estimators=900,learning_rate=.035,num_leaves=63,
    min_child_samples=200,reg_lambda=20,colsample_bytree=.85,max_bin=127,n_jobs=4,
    verbosity=-1,random_state=20261103,deterministic=True,force_col_wise=True)


class CatPredictor:
    def __init__(self,model):self.model=model
    def predict(self,x,num_threads=4):
        x=x.copy()
        for c in ['wa_station','wb_station','site_code']:x[c]=x[c].astype(str)
        return self.model.predict(x,thread_count=num_threads)


def inputs(data):
    meta=pd.read_csv(data/'sites.csv').set_index('site_id').sort_index()
    cfg=ex.read(team.SOURCE/'configs/final_model.json')
    names=cfg['features']['lightgbm']+['site_code','site_lat','site_lon']
    categories={c:sorted(meta[c].unique()) for c in ['wa_station','wb_station']}
    return meta,names,categories


def design(d,sid,meta,categories):
    d=d.copy();d['site_code']=int(sid.replace('PV',''))
    d['site_lat']=float(meta.loc[sid,'lat']);d['site_lon']=float(meta.loc[sid,'lon'])
    for c,values in categories.items():d[c]=pd.Categorical(d[c],categories=values)
    d['site_code']=pd.Categorical(d.site_code,categories=range(1,254))
    return d


def train(data,part):
    dest=RUN/part;dest.mkdir(parents=True,exist_ok=True)
    meta,names,cats=inputs(data);origin=p.stamp(team.PARTS[part][0]);cutoff=p.issue_deadline(origin)
    path=dest/('model.cbm' if KIND=='catboost' else 'model.txt')
    if path.exists():
        record=ex.read(dest/'metadata.json')
        if (record['cutoff']!=str(cutoff) or record['params']!=PARAMS or record['model_sha256']!=p.sha256(path)
                or record['normalization']!=('max(solar_q,.15)' if NORMALIZE else 'none')):raise ValueError('Stale model')
        if KIND=='catboost':
            from catboost import CatBoostRegressor
            return CatPredictor(CatBoostRegressor().load_model(str(path)))
        return lgb.Booster(model_file=str(path))
    start=p.history_start(origin,24)
    truth=pd.read_parquet(data/'train/generation.parquet',filters=[('ts','>=',start),('ts','<=',cutoff)])
    truth=truth[np.isfinite(truth.energy)&(truth.energy>=0)]
    frames=[]
    for i,(sid,y) in enumerate(truth.groupby('site_id',sort=True)):
        d=pd.read_parquet(team.RUN/'cache/train'/f'{sid}.parquet',filters=[('ts','>=',start),('ts','<=',cutoff)])
        d=d.merge(y,on=['site_id','ts'],validate='one_to_one');d=d[d.solar_q>0]
        frames.append(design(d,sid,meta,cats))
        if (i+1)%50==0:p.log(f'{part} design {i+1} sites')
    d=pd.concat(frames,ignore_index=True);del frames,truth;gc.collect()
    norm=np.maximum(d.solar_q.to_numpy(),.15) if NORMALIZE else np.ones(len(d))
    y=(d.energy/d.capacity_kw).to_numpy(dtype=float)/norm
    # Preserve squared error in original capacity-factor units after changing target scale.
    weight=(d.capacity_kw*d.ts.map(ex.prices(data).weight)).to_numpy()*norm**2;weight/=weight.mean()
    record=dict(cutoff=str(cutoff),last_label=str(d.ts.max()),first_label=str(d.ts.min()),rows=len(d),
                features=names,params=PARAMS,kind=KIND,normalization='max(solar_q,.15)' if NORMALIZE else 'none',categories=cats)
    if d.ts.max()>cutoff:raise ValueError('Future target')
    p.log(f'Solar normalized {part}: {len(d):,} rows, {len(names)} features')
    if KIND=='catboost':
        from catboost import CatBoostRegressor,Pool
        for c in ['wa_station','wb_station','site_code']:d[c]=d[c].astype(str)
        pool=Pool(d[names],label=y,weight=weight,cat_features=['wa_station','wb_station','site_code'],thread_count=2)
        del d,y,weight;gc.collect()
        model=CatBoostRegressor(**PARAMS).fit(pool)
        model.save_model(str(path));record['model_sha256']=p.sha256(path);p.save_json(dest/'metadata.json',record)
        return CatPredictor(model)
    params=PARAMS.copy();rounds=params.pop('n_estimators')
    dataset=lgb.Dataset(d[names],label=y,weight=weight,params={'max_bin':PARAMS['max_bin'],'num_threads':4},categorical_feature=['wa_station','wb_station','site_code'],free_raw_data=True)
    dataset.construct();del d,y,weight;gc.collect()
    model=lgb.train(params,dataset,num_boost_round=rounds,
        callbacks=[lambda env:p.log(f'{part} {env.iteration+1}/{rounds}') if (env.iteration+1)%100==0 else None])
    model.save_model(str(path));record['model_sha256']=p.sha256(path);p.save_json(dest/'metadata.json',record)
    return model


def infer(data,part):
    out=RUN/part/'predictions.parquet'
    if out.exists():return pd.read_parquet(out)
    model=train(data,part);meta,names,cats=inputs(data)
    start,end=map(p.stamp,team.PARTS[part][1:]);split='test' if part=='production' else 'train'
    frames=[]
    for sid in meta.index:
        d=pd.read_parquet(team.RUN/'cache'/split/f'{sid}.parquet',filters=[('ts','>=',start),('ts','<=',end)])
        d=design(d,sid,meta,cats);norm=np.maximum(d.solar_q.to_numpy(),.15) if NORMALIZE else np.ones(len(d))
        pred=np.where(d.solar_q>0,np.maximum(model.predict(d[names],num_threads=4),0)*norm*float(meta.loc[sid,'capacity_kw']),0)
        z=d[['site_id','ts']].copy();z['new']=pred;frames.append(z)
    result=pd.concat(frames,ignore_index=True);result.to_parquet(out,index=False);return result


def comparison(data,parts):
    pred=pd.concat([infer(data,s) for s in parts],ignore_index=True)
    old=fr.source_panel();d=old.merge(pred,on=['site_id','ts'],validate='one_to_one')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    d['group']=d.site_id.map(comp)
    g=d.groupby(['ts','group'])[['energy','v8','new']].sum().reset_index()
    g['weight']=g.ts.map(ex.prices(data).weight)
    return g


def evaluate(data):
    RUN.mkdir(parents=True,exist_ok=True)
    g=comparison(data,['seasonal_proxy','complete_fleet'])
    choices=[]
    for strength in [.25,.5,.75,1.]:
        rows=fr.summary(g,(1-strength)*g.v8+strength*g['new'])
        score=.5*np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows[:6]])+.5*np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows[6:]])
        choices.append(dict(strength=strength,score=float(score),months=rows))
    selected=min(choices,key=lambda v:v['score']);p.save_json(RUN/'selected.json',dict(selected=selected,choices=choices))
    g=comparison(data,['development']);a=selected['strength'];g['candidate']=(1-a)*g.v8+a*g['new']
    rows=fr.summary(g,g.candidate);gain=100*(1-fr.penalty(g,g.candidate)/fr.penalty(g,g.v8))
    accepted=selected['score']<1 and gain>=1 and min(r['gain_pct'] for r in rows)>-2
    result=dict(accepted=bool(accepted),selected=selected,months=rows,total_gain_pct=gain)
    p.save_json(RUN/'evaluation.json',result);p.log(str(result))


def produce(data):
    ensure_writable(ex.HERE/'제출'/CANDIDATE)
    r=ex.read(RUN/'evaluation.json')
    if not r['accepted']:raise ValueError('Candidate rejected')
    a=r['selected']['strength'];new=infer(data,'production')
    base=fr.source_panel(True).merge(new,on=['site_id','ts'],validate='one_to_one')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv');base['group']=base.site_id.map(comp.set_index('site_id').group)
    base['forecast_kwh']=(1-a)*base.v8+a*base['new']
    dest=ex.HERE/'제출'/CANDIDATE;dest.mkdir(parents=True,exist_ok=True)
    comp.to_csv(dest/'composition.csv',index=False)
    base.groupby(['group','ts'],as_index=False).forecast_kwh.sum().to_csv(dest/'forecast.csv',index=False)
    errors=validate(dest/'composition.csv',dest/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(dest/'validation.json',dict(format_validation='PASS',official_submitted=False,result=r,
        fit=ex.read(RUN/'production/metadata.json'),files={n:p.sha256(dest/n) for n in ['composition.csv','forecast.csv']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate');a=ap.parse_args()
    globals()[a.stage](a.data)
