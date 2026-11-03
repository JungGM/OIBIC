"""Forecast-only fleet residual stacking with chronological development checks.

All base predictions used for training are out of time. Each calibration fold
also stops labels at previous-day 17:00. No test prices or actuals are read.
Private matrices/models live in _local; only aggregates are printed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pipeline as p
import improvement_experiment as ex
import team_pair_experiment as team
from validate_submission import validate
from submission_guard import ensure_writable

RUN = ex.HERE/'_local/fleet_residual_v9'
RAW = ['cosz_positive','elev','azim','doy_sin','doy_cos','hour_sin','hour_cos']
RAW += [f'r{r}_{c}' for r in (10,17) for c in
        ['wa_cloud','wa_temp','wa_humidity','wa_ground_press','wa_wind_speed','wa_rain','wa_snow',
         'wb_cloud','wb_uv_idx','wb_rel_hum','wb_temp','wb_vis','wb_ceiling','wb_total_liq','wb_wind_speed']]
RAW += ['revision_'+c for c in ['wa_cloud','wb_cloud','wb_uv_idx','wb_temp','wb_rel_hum']]
CONFIGS = [dict(objective=o,leaves=n,rounds=r) for o in ['regression_l1','regression']
           for n,r in [(7,120),(7,300),(15,180)]]
STRENGTHS = [.25,.5,.75,1.]


def source_panel(production=False):
    if production:
        d=pd.read_parquet(ex.OLD/'production/site_predictions.parquet').rename(columns={'prediction_kwh':'base'})
        c=pd.read_parquet(team.RUN/'production/components.parquet')
    else:
        d=pd.concat([ex.old_oof(),pd.read_parquet(ex.OLD/'development/site_predictions.parquet').rename(columns={'ensemble':'base'})[['site_id','ts','energy','base']]],ignore_index=True)
        c=pd.concat([pd.read_parquet(team.RUN/s/'components.parquet') for s in ['seasonal_proxy','complete_fleet','development']],ignore_index=True)
    d=d.merge(c[['site_id','ts','team_cat','team_lgb']],on=['site_id','ts'],how='left',validate='one_to_one')
    if not np.isfinite(d[['base','team_cat','team_lgb']]).all().all(): raise ValueError('Missing base predictions')
    d['v8']=.75*d.base+.25*d.team_cat
    return d


def build(data,production=False):
    RUN.mkdir(parents=True,exist_ok=True)
    path=RUN/('production.parquet' if production else 'historical.parquet')
    if path.exists(): return pd.read_parquet(path)
    d=source_panel(production)
    meta=pd.read_csv(data/'sites.csv').set_index('site_id')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    frames=[]
    for month,z in d.groupby((d.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')):
        f=pd.read_parquet(ex.CACHE/('test' if production else 'train')/f'{month}.parquet',columns=['site_id','ts']+RAW)
        z=z.merge(f,on=['site_id','ts'],how='left',validate='one_to_one')
        z['capacity']=z.site_id.map(meta.capacity_kw);z['group']=z.site_id.map(comp)
        for group,x in z.groupby('group'):
            key=x.ts
            cap=x.capacity.groupby(key).sum()
            n=x.site_id.groupby(key).nunique()
            if not n.eq(n.iloc[0]).all(): raise ValueError('Variable cohort within month')
            g=pd.DataFrame({'capacity':cap,'sites':n})
            for c in ['base','v8','team_cat','team_lgb']+([] if production else ['energy']):
                if not np.isfinite(x[c]).all(): raise ValueError('Missing target or prediction')
                g[c]=x[c].groupby(key).sum()
            for c in RAW:
                finite=np.isfinite(x[c]);w=x.capacity.where(finite,0)
                den=w.groupby(key).sum().replace(0,np.nan)
                mean=(x[c].fillna(0)*w).groupby(key).sum()/den
                sq=(x[c].fillna(0)**2*w).groupby(key).sum()/den
                g['mean_'+c]=mean;g['std_'+c]=np.sqrt(np.maximum(sq-mean**2,0))
            g['base_util']=g.base/cap;g['v8_util']=g.v8/cap
            g['cat_delta']=(g.team_cat-g.base)/cap;g['lgb_delta']=(g.team_lgb-g.base)/cap
            g['group_code']=float(group=='G2');g['group']=group
            g['hour']=g.index.hour
            g['peak']=g.index.hour.isin(range(11,16)).astype(float)
            # No daily reduction here: the evaluation panel excludes missing
            # labels, so day means must be constructed upstream on full forecasts.
            frames.append(g.reset_index())
        p.log(f'Fleet matrix {month}')
    out=pd.concat(frames,ignore_index=True).sort_values(['ts','group']).reset_index(drop=True)
    if not production:
        out['weight']=out.ts.map(ex.prices(data).weight)
        if not np.isfinite(out.weight).all():raise ValueError('Missing historical price')
    out.to_parquet(path,index=False)
    return out


def names(d):
    return [c for c in d if c.startswith(('mean_','std_','day_')) or c in
            ['base_util','v8_util','cat_delta','lgb_delta','hour','peak','group_code']]


def fit(d,origin,cfg):
    cutoff=p.issue_deadline(p.stamp(origin))
    z=d[(d.ts<=cutoff)&(d.v8>1)].copy()
    if z.empty or z.ts.max()>cutoff:raise ValueError('Invalid training cutoff')
    features=names(d)
    model=lgb.LGBMRegressor(objective=cfg['objective'],n_estimators=cfg['rounds'],
        num_leaves=cfg['leaves'],learning_rate=.025,min_child_samples=100,
        reg_lambda=25,colsample_bytree=.8,max_bin=63,n_jobs=4,
        deterministic=True,force_col_wise=True,random_state=20261103,verbosity=-1)
    weights=z.weight*z.capacity;weights/=weights.mean()
    model.fit(z[features],(z.energy-z.v8)/z.capacity,sample_weight=weights)
    return model,dict(cutoff=str(cutoff),last_label=str(z.ts.max()),rows=len(z),features=features,config=cfg)


def predict(model,d,strength):
    delta=model.predict(d[names(d)],num_threads=4)*d.capacity.to_numpy()
    # Absolute correction cannot create nighttime generation or dominate a weak baseline.
    bound=.35*d.v8.to_numpy()
    return np.where(d.v8.to_numpy()>1,np.maximum(0,d.v8.to_numpy()+strength*np.clip(delta,-bound,bound)),d.v8)


def penalty(d,pred):return float(.1*np.sum(d.weight*np.abs(np.asarray(pred)-d.energy)))


def summary(d,pred):
    d=d.copy();d['candidate']=pred
    rows=[]
    for month,z in d.groupby((d.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')):
        base=penalty(z,z.v8);cost=penalty(z,z.candidate)
        rows.append(dict(month=month,hours=int(z.ts.nunique()),base_penalty=base,
                         candidate_penalty=cost,gain_pct=100*(1-cost/base)))
    return rows


def evaluate(data):
    d=build(data)
    frozen=RUN/'selected.json'
    if not frozen.exists():
        # Broad shortlist on two forward periods only; spring is not used to select.
        folds=[('2024-10-01 01:00','2025-01-01 00:00'),('2025-01-01 01:00','2025-03-01 00:00')]
        candidates=[]
        for cfg in CONFIGS:
            predictions=[]
            for start,end in folds:
                model,record=fit(d,start,cfg)
                z=d[d.ts.between(p.stamp(start),p.stamp(end))].copy()
                for strength in STRENGTHS:z[f'pred_{strength}']=predict(model,z,strength)
                predictions.append(z)
            z=pd.concat(predictions)
            for strength in STRENGTHS:
                rows=summary(z,z[f'pred_{strength}'])
                score=np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows])
                candidates.append(dict(config=cfg,strength=strength,score=float(score),months=rows))
            p.log(f'Calibration selection {cfg}')
        best=min(candidates,key=lambda v:v['score'])
        p.save_json(frozen,dict(selected=best,choices=candidates,history_sha256=p.sha256(RUN/'historical.parquet')))
    selected=ex.read(frozen)['selected']
    model,record=fit(d,'2025-03-01 01:00',selected['config'])
    z=d[d.ts.between(p.stamp('2025-03-01 01:00'),p.stamp('2025-06-01 00:00'))].copy()
    z['candidate']=predict(model,z,selected['strength'])
    rows=summary(z,z.candidate)
    total_gain=100*(1-penalty(z,z.candidate)/penalty(z,z.v8))
    # Paired daily uncertainty for the complete three-month comparison.
    z['saving']=.1*z.weight*(abs(z.v8-z.energy)-abs(z.candidate-z.energy))
    daily=z.groupby((z.ts-pd.Timedelta(hours=1)).dt.date).saving.sum().to_numpy()
    rng=np.random.default_rng(20261103);n=len(daily)
    samples=np.stack([daily[(rng.integers(n,size=(int(np.ceil(n/3)),1))+np.arange(3))%n].ravel()[:n].mean() for _ in range(2000)])
    lower=float(np.quantile(samples,.05))
    accepted=selected['score']<1 and total_gain>=1 and lower>0 and min(r['gain_pct'] for r in rows)>-2
    result=dict(accepted=bool(accepted),selected=selected,months=rows,total_gain_pct=total_gain,
                lower90_daily_saving=lower,fit=record,disclosure='Repeated historical development; not an untouched holdout')
    p.save_json(RUN/'evaluation.json',result);z.to_parquet(RUN/'development_predictions.parquet',index=False)
    model.booster_.save_model(str(RUN/'development_model.txt'))
    p.log(str(result))


def produce(data):
    ensure_writable(ex.HERE/'제출/improved_v9')
    result=ex.read(RUN/'evaluation.json')
    if not result['accepted']:raise ValueError('Chronological validation rejected candidate')
    cfg=result['selected'];d=build(data)
    model,record=fit(d,'2025-06-01 01:00',cfg['config'])
    model.booster_.save_model(str(RUN/'production_model.txt'))
    record['model_sha256']=p.sha256(RUN/'production_model.txt')
    out=build(data,True);out['forecast_kwh']=predict(model,out,cfg['strength'])
    dest=ex.HERE/'제출/improved_v9';dest.mkdir(parents=True,exist_ok=True)
    comp=pd.read_csv(ex.HERE/'제출/composition.csv');comp.to_csv(dest/'composition.csv',index=False)
    out[['group','ts','forecast_kwh']].to_csv(dest/'forecast.csv',index=False)
    errors=validate(dest/'composition.csv',dest/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(dest/'validation.json',dict(format_validation='PASS',official_submitted=False,fit=record,
        result=result,files={n:p.sha256(dest/n) for n in ['composition.csv','forecast.csv']}))
    p.log('V9 production complete and validated')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate')
    a=ap.parse_args();globals()[a.stage](a.data)
