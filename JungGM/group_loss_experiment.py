"""Exploratory group-aware boosting after the recorded v3/v4 rejections.

Warm starts from the same 155-feature cost-weighted estimator. Adds a bounded
number of trees with a coupled group pseudo-Huber + individual squared loss.
No missing label is filled; the group term requires >=80% observed group capacity.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pipeline as p
import features as ft
import improvement_experiment as ex
from improvement_features import add_features
from validate_submission import validate

RUN=ex.HERE/'_local/improvement_v5_group_loss'
V4=ex.HERE/'_local/improvement_v4_seasonal'
LAMBDA=.75
DELTA=.05
ROUNDS=250


class GroupObjective:
    """Gradient of a coupled convex loss and a diagonal Hessian upper bound."""
    def __init__(self, target, capacity, weight, codes, active, mix=LAMBDA, delta=DELTA):
        self.y=np.asarray(target,dtype=float)
        self.c=np.asarray(capacity,dtype=float)
        self.w=np.asarray(weight,dtype=float)
        self.codes=np.asarray(codes,dtype=int)
        self.mass=np.bincount(self.codes,weights=self.c)
        self.mix=mix*np.asarray(active,dtype=float)
        self.delta=delta
        if not all(np.isfinite(x).all() for x in [self.y,self.c,self.w,self.mix]):
            raise ValueError('Nonfinite objective inputs')
        if (self.c<=0).any() or (self.w<0).any() or (self.mass<=0).any():
            raise ValueError('Invalid objective weights')

    def group_error(self, error):
        return np.bincount(self.codes,weights=self.c*error)/self.mass

    def value(self, prediction):
        e=np.asarray(prediction)-self.y
        ge=self.group_error(e)[self.codes]
        phi=self.delta**2*(np.sqrt(1+(ge/self.delta)**2)-1)
        return float(np.sum(self.w*((1-self.mix)*.5*e**2+self.mix*phi)))

    def __call__(self, prediction, dataset=None):
        e=np.asarray(prediction)-self.y
        ge=self.group_error(e)[self.codes]
        scale=np.sqrt(1+(ge/self.delta)**2)
        grad=self.w*((1-self.mix)*e+self.mix*ge/scale)
        hess=self.w*((1-self.mix)+self.mix/(scale**3))
        return grad,np.maximum(hess,1e-8)


def protocol():
    RUN.mkdir(parents=True,exist_ok=True)
    obj=dict(mix=LAMBDA,delta=DELTA,extra_trees=ROUNDS,learning_rate=.025,
        num_leaves=31,min_child_samples=200,reg_lambda=20,
        min_observed_group_capacity=.8,
        selection='Same July-Dec 225-site proxy 50%, Jan-Feb 253-site 50%; blend grid 0,.25,.5,.75,1',
        gate='April gain>=1% and 3-day block lower90>0; May nonnegative; March degradation<=1%',
        groups='Reference fixed; no new group search',
        caveat='Exploratory third experiment; all historical evaluation periods already inspected',
        official_score_search=False)
    path=RUN/'protocol_before_results.json'
    if path.exists() and ex.read(path)!=obj:raise ValueError('Protocol changed')
    if not path.exists():p.save_json(path,obj)


def train_one(data,origin,base_dir,dest):
    origin=p.stamp(origin);cutoff=p.issue_deadline(origin)
    dest.mkdir(exist_ok=True)
    meta=ex.read(base_dir/'metadata.json');names=meta['features']
    if pd.Timestamp(meta['cutoff'])!=cutoff:raise ValueError('Warm start cutoff mismatch')
    path=dest/'group_lgb.txt'
    if path.exists() and (dest/'metadata.json').exists():
        record=ex.read(dest/'metadata.json')
        if (record['protocol']!=ex.read(RUN/'protocol_before_results.json') or
            pd.Timestamp(record['cutoff'])!=cutoff or
            record['warm_start_sha256']!=p.sha256(base_dir/'cost_lgb.txt')):
            raise ValueError('Group estimator provenance mismatch')
        return lgb.Booster(model_file=str(path)),names
    site_meta=pd.read_csv(data/'sites.csv').set_index('site_id')
    group_capacity=site_meta.groupby('current_group').capacity_kw.sum()
    pr=ex.prices(data)
    frames=[]
    start=p.history_start(origin,24)
    for month in ft._months(start,cutoff):
        d,_=add_features(pd.read_parquet(ex.CACHE/'train'/f'{month}.parquet'))
        mask=(d.ts>=start)&(d.ts<=cutoff)&d.energy.notna()&(d.solar_hour_integral>0)
        frames.append(d.loc[mask,['site_id','ts','energy']+names])
        del d
    d=pd.concat(frames,ignore_index=True);del frames;gc.collect()
    capacity=d.cap.to_numpy(dtype=float)
    weight=capacity*pr.weight.reindex(pd.DatetimeIndex(d.ts)).to_numpy()
    weight/=weight.mean()
    group=d.site_id.map(site_meta.current_group)
    codes,unique=pd.factorize(pd.MultiIndex.from_arrays([d.ts,group]),sort=False)
    observed=np.bincount(codes,weights=capacity)
    total=np.array([group_capacity[g] for _,g in unique])
    active=(observed/total>=.8)[codes]
    y=(d.energy/d.cap).clip(0,1.2).to_numpy(dtype='float32')
    objective=GroupObjective(y,capacity,weight,codes,active)
    params=dict(objective=objective,metric='None',learning_rate=.025,num_leaves=31,
                min_data_in_leaf=200,lambda_l2=20,feature_fraction=.85,max_bin=127,
                num_threads=4,seed=ex.SEED,verbosity=-1,deterministic=True,
                force_col_wise=True,feature_pre_filter=False)
    ds=lgb.Dataset(d[names],label=y,categorical_feature=ft.get_schema(ex.CACHE)['categorical'],free_raw_data=True)
    record=dict(features=names,origin=str(origin),cutoff=str(cutoff),rows=len(d),
                first_target=str(d.ts.min()),last_target=str(d.ts.max()),
                group_active_row_fraction=float(active.mean()),warm_start_sha256=p.sha256(base_dir/'cost_lgb.txt'),
                protocol=ex.read(RUN/'protocol_before_results.json'))
    p.log(f'Group-aware continuation {origin}: {len(d):,} rows; active group term {active.mean():.1%}')
    model=lgb.train(params,ds,num_boost_round=ROUNDS,init_model=str(base_dir/'cost_lgb.txt'))
    model.save_model(str(path));p.save_json(dest/'metadata.json',record)
    del d,ds,objective;gc.collect()
    return model,names


def train(data):
    runs=[('seasonal_proxy','2024-07-01 01:00','2024-07','2024-12',V4/'seasonal_proxy'),
          ('complete_fleet','2025-01-01 01:00','2025-01','2025-02',V4/'complete_fleet'),
          ('development','2025-03-01 01:00','2025-03','2025-05',ex.RUN/'development')]
    for part,origin,start,end,base in runs:
        model,names=train_one(data,origin,base,RUN/part)
        ex.predict_new(model,names,'train',pd.period_range(start,end,freq='M'),RUN/part/'predictions.parquet')
        del model;gc.collect()


def select_model(data, run=None):
    root=RUN if run is None else Path(run)
    old=ex.old_oof()
    new=pd.concat([pd.read_parquet(root/part/'predictions.parquet') for part in ['seasonal_proxy','complete_fleet']])
    site=old.merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
    if not np.isfinite(site['new']).all():raise ValueError('Missing OOF predictions')
    g=ex.group_frame(site,data)
    choices=[]
    for weight in [0,.25,.5,.75,1.]:
        rows=[]
        for m,z in g.groupby((g.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m'),sort=True):
            base=ex.loss(z,z.base);candidate=ex.loss(z,(1-weight)*z.base+weight*z['new'])
            rows.append(dict(month=m,base=base,candidate=candidate,ratio=candidate/base))
        score=.5*np.mean([r['ratio'] for r in rows[:6]])+.5*np.mean([r['ratio'] for r in rows[6:]])
        choices.append(dict(new_weight=weight,score=float(score),months=rows))
    selected=min(choices,key=lambda x:x['score']);nw=selected['new_weight']
    p.save_json(root/'frozen_before_spring.json',dict(selected=selected,choices=choices,
        prediction_sha256={part:p.sha256(root/part/'predictions.parquet') for part in ['seasonal_proxy','complete_fleet']}))
    return selected


def evaluate(data, run=None, selection=None):
    root=RUN if run is None else Path(run)
    selected=select_model(data,root) if selection is None else selection
    nw=selected['new_weight']
    old=pd.read_parquet(ex.OLD/'development/site_predictions.parquet').rename(columns={'ensemble':'base'})
    has_new=(root/'development/predictions.parquet').exists()
    if not has_new and nw!=0:raise ValueError('Selected candidate requires development predictions')
    if has_new:
        site=old.merge(pd.read_parquet(root/'development/predictions.parquet'),on=['site_id','ts'],how='left',validate='one_to_one')
        if not np.isfinite(site['new']).all():raise ValueError('Missing development predictions')
    else:site=old
    g=ex.group_frame(site,data)
    rows=[]
    for m in [3,4,5]:
        z=g[(g.ts-pd.Timedelta(hours=1)).dt.month==m]
        pred=(1-nw)*z.base+nw*z['new'] if has_new else z.base
        base=ex.loss(z,z.base);cost=ex.loss(z,pred)
        rows.append(dict(month=m,hours=z.ts.nunique(),sites=253,baseline_penalty_krw=base,
                         candidate_penalty_krw=cost,new_model_only_penalty_krw=ex.loss(z,z['new']) if has_new else None,gain_pct=100*(1-cost/base)))
        if m==4:gate=ex.block_gate(z,pred.to_numpy())
        p.log(str(rows[-1]))
    accepted=bool(gate['accepted'] and rows[-1]['gain_pct']>=0 and rows[0]['gain_pct']>=-1)
    result=dict(accepted=accepted,selected_new_weight=nw,months=rows,april_gate=gate,seasonal=selected,
                standalone_development_evaluated=has_new)
    p.save_json(root/'evaluation.json',result)
    pd.DataFrame(rows).to_csv(root/'monthly_comparison.csv',index=False)
    p.log(f'{root.name} adoption {accepted}; selected new weight {nw}; gate {gate}')


def produce(data):
    result=ex.read(RUN/'evaluation.json')
    if not result['accepted']:
        p.log('Not adopted; official submission preserved.');return
    base_dir=RUN/'production_base'
    ex.train(data,production=True,dest_override=base_dir)
    model,names=train_one(data,'2025-06-01 01:00',base_dir,RUN/'production')
    ex.predict_new(model,names,'test',pd.period_range('2025-06','2026-05',freq='M'),RUN/'production/predictions.parquet')
    d=pd.read_parquet(ex.OLD/'production/site_predictions.parquet')
    d=d.merge(pd.read_parquet(RUN/'production/predictions.parquet'),on=['site_id','ts'],how='left',validate='one_to_one')
    if len(d)!=253*8760 or not np.isfinite(d['new']).all():raise ValueError('Incomplete target grid')
    nw=result['selected_new_weight'];d['forecast_kwh']=(1-nw)*d.prediction_kwh+nw*d['new']
    comp=pd.read_csv(ex.HERE/'제출/composition.csv')
    d['group']=d.site_id.map(comp.set_index('site_id').group)
    out=ex.HERE/'제출/improved_v5';out.mkdir(exist_ok=True)
    comp.to_csv(out/'composition.csv',index=False)
    d.groupby(['group','ts'],as_index=False).forecast_kwh.sum().to_csv(out/'forecast.csv',index=False)
    errors=validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS',official_submitted=False,result=result,
        files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']}))
    p.log(f'Validated {out}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--stage',required=True,choices=['train','evaluate','produce'])
    args=ap.parse_args();protocol();{'train':train,'evaluate':evaluate,'produce':produce}[args.stage](args.data)


if __name__=='__main__':main()
