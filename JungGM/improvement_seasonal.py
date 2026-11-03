"""Second, explicitly exploratory multi-season comparison after rejected v3.

No hidden-year truth or score fitting. All inspected historical months are reused
research data, not an untouched holdout. Frozen choice uses July-Feb only.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as p
import improvement_experiment as ex
import grouping
from validate_submission import validate

RUN = ex.HERE/'_local/improvement_v4_seasonal'


def protocol():
    RUN.mkdir(parents=True, exist_ok=True)
    obj = dict(version=4, motivation='v3 spring-only selection rejected; test all original seasonal folds',
        disclosure='March-May v3 results already inspected; exploratory historical validation, not untouched test',
        model=ex.PARAMS, origins=['2024-07-01 01:00', '2025-01-01 01:00'],
        model_selection='July-Dec monthly relative loss 50%, Jan-Feb 50%; same 225/253 site cohorts as original',
        new_weights=[0,.25,.5,.75,1.], calibrator='none',
        grouping='one candidate fit Jan-Feb time-external residuals; equal monthly total weight; 20000 steps x5 seeds',
        gate='March grouping gain >=1% and block lower90 positive; otherwise keep reference',
        system_gate='April >=1% gain and block lower90 positive; May gain >=0; March degradation <=1%',
        production='same final estimator settings and chosen weights/groups; no residual calibration refit',
        official_score_search=False)
    path=RUN/'protocol_before_results.json'
    if path.exists() and ex.read(path)!=obj:raise ValueError('Protocol mismatch')
    if not path.exists():p.save_json(path,obj)


def fit_seasons(data):
    for part,origin,start,end in [('seasonal_proxy','2024-07-01 01:00','2024-07','2024-12'),
                                  ('complete_fleet','2025-01-01 01:00','2025-01','2025-02')]:
        dest=RUN/part;dest.mkdir(exist_ok=True)
        model,names=ex.train(data,origin_override=origin,dest_override=dest)
        ex.predict_new(model,names,'train',pd.period_range(start,end,freq='M'),dest/'predictions.parquet')
        del model;gc.collect()


def merged_oof():
    old=ex.old_oof()
    new=pd.concat([pd.read_parquet(RUN/part/'predictions.parquet')
                   for part in ['seasonal_proxy','complete_fleet']],ignore_index=True)
    d=old.merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
    if not np.isfinite(d['new']).all():raise ValueError('Missing OOF prediction')
    return d


def panel(d,data,assignment):
    ids=pd.read_csv(data/'sites.csv').sort_values('site_id').site_id.tolist()
    pr=ex.prices(data).reset_index();pr['k']=np.where(pr.ts.dt.hour.between(11,15),3,1)
    ts,y,pp,w,smp=p.panel(d,['base','new'],ids,pr)
    return ts,y,pp,w,smp


def grouped(ts,y,base,pred,w,a):
    # Reuse the independently tested day-block gate on already aggregated values.
    actual=p.group_sum(y,a); b=p.group_sum(base,a); c=p.group_sum(pred,a)
    d=pd.DataFrame({'ts':np.repeat(ts,2),'energy':actual.ravel(),
                    'base':b.ravel(),'weight':np.repeat(w,2)})
    return d,c.ravel()


def evaluate(data):
    protocol()
    site=merged_oof()
    g=ex.group_frame(site,data)
    months=(g.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    choices=[]
    for weight in [0,.25,.5,.75,1.]:
        ratios=[]; rows=[]
        for m,z in g.groupby(months,sort=True):
            base=ex.loss(z,z.base); loss=ex.loss(z,(1-weight)*z.base+weight*z['new'])
            ratios.append(loss/base)
            rows.append(dict(month=m,hours=z.ts.nunique(),sites=int(z.groupby('ts').sites.sum().iloc[0]),
                             baseline_penalty=base,candidate_penalty=loss,gain_pct=100*(1-loss/base)))
        score=.5*np.mean(ratios[:6])+.5*np.mean(ratios[6:])
        choices.append(dict(new_weight=weight,score=float(score),months=rows))
    selected=min(choices,key=lambda x:x['score'])
    nw=selected['new_weight']
    p.save_json(RUN/'frozen_model_before_march.json',dict(selected=selected,choices=choices))
    p.log(f'Seasonal selection: {nw} new, score {selected["score"]:.6f}')
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id')
    ref=p.reference(sites)
    winter=site[(site.ts-pd.Timedelta(hours=1)).dt.year==2025]
    ts,y,pp,w,smp=panel(winter,data,ref)
    prediction=(1-nw)*pp['base']+nw*pp['new']
    # Balance Jan and Feb so one month cannot dominate group fitting by price/mass.
    balanced=w.copy()
    for m in [1,2]:
        mask=(ts-pd.Timedelta(hours=1)).month==m
        balanced[mask]/=w[mask].sum()
    assignment,search=grouping.optimize(prediction-y,balanced,sites.capacity_kw.to_numpy(),ref,
                                       seed=ex.SEED,steps=20000,restarts=5)
    p.save_json(RUN/'frozen_group_before_march.json',dict(assignment=assignment.tolist(),search=search))
    ev=pd.read_parquet(ex.OLD/'development/site_predictions.parquet').rename(columns={'ensemble':'base'})
    ev=ev.merge(pd.read_parquet(ex.RUN/'development/predictions.parquet'),on=['site_id','ts'],how='left',validate='one_to_one')
    month=(ev.ts-pd.Timedelta(hours=1)).dt.month
    march=ev[month==3];ts,y,pp,w,smp=panel(march,data,ref)
    pred=(1-nw)*pp['base']+nw*pp['new']
    # A grouping-only comparison retains identical forecasts, but changes aggregation.
    d,c=grouped(ts,y,pred,pred,w,ref)
    candidate_error=p.group_sum(pred-y,assignment)
    base_error=p.group_sum(pred-y,ref)
    pseudo=pd.DataFrame(dict(ts=np.repeat(ts,2),energy=np.zeros(2*len(ts)),
                              base=abs(base_error).ravel(),weight=np.repeat(w,2)))
    gate=ex.block_gate(pseudo,abs(candidate_error).ravel())
    a=assignment if gate['accepted'] else ref
    p.save_json(RUN/'frozen_system_before_april.json',dict(new_weight=nw,assignment=a.tolist(),
                  grouping_gate=gate,group_changes=int((a!=ref).sum())))
    rows=[]
    for m in [3,4,5]:
        d=ev[month==m];ts,y,pp,w,smp=panel(d,data,ref)
        pred=(1-nw)*pp['base']+nw*pp['new']
        base=.1*p.cost(y,pp['base'],w,ref)
        model_only=.1*p.cost(y,pred,w,ref)
        full=.1*p.cost(y,pred,w,a)
        rows.append(dict(month=m,hours=len(ts),sites=253,baseline_penalty_krw=base,
                         model_only_penalty_krw=model_only,candidate_penalty_krw=full,gain_pct=100*(1-full/base)))
        if m==4:
            pseudo=pd.DataFrame(dict(ts=np.repeat(ts,2),energy=np.zeros(2*len(ts)),
                base=abs(p.group_sum(pp['base']-y,ref)).ravel(),weight=np.repeat(w,2)))
            system_gate=ex.block_gate(pseudo,abs(p.group_sum(pred-y,a)).ravel())
        p.log(str(rows[-1]))
    accepted=bool(system_gate['accepted'] and rows[-1]['gain_pct']>=0 and rows[0]['gain_pct']>=-1)
    result=dict(accepted=accepted,months=rows,system_gate=system_gate,group_gate=gate,
                selected_new_weight=nw,assignment=a.tolist(),seasonal=selected,
                no_untouched_holdout=True)
    p.save_json(RUN/'evaluation.json',result)
    pd.DataFrame(rows).to_csv(RUN/'monthly_comparison.csv',index=False)
    p.log(f'v4 adoption: {accepted}')


def produce(data):
    result=ex.read(RUN/'evaluation.json')
    if not result['accepted']:
        p.log('Not adopted; original submission retained.');return
    nw=result['selected_new_weight']
    dest=RUN/'production';dest.mkdir(exist_ok=True)
    if nw:
        model,names=ex.train(data,production=True,dest_override=dest)
        ex.predict_new(model,names,'test',pd.period_range('2025-06','2026-05',freq='M'),dest/'predictions.parquet')
    d=pd.read_parquet(ex.OLD/'production/site_predictions.parquet').rename(columns={'prediction_kwh':'base'})
    if nw:
        d=d.merge(pd.read_parquet(dest/'predictions.parquet'),on=['site_id','ts'],how='left',validate='one_to_one')
        d['prediction']=(1-nw)*d.base+nw*d['new']
    else:d['prediction']=d.base
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id')
    comp=pd.DataFrame({'site_id':sites.site_id,'group':np.where(np.array(result['assignment'])==1,'G1','G2')})
    d['group']=d.site_id.map(comp.set_index('site_id').group)
    out=ex.HERE/'제출/improved_v4';out.mkdir(exist_ok=True)
    comp.to_csv(out/'composition.csv',index=False)
    forecast=d.groupby(['group','ts'],as_index=False).prediction.sum().rename(columns={'prediction':'forecast_kwh'})
    if len(d)!=253*8760 or not np.isfinite(d.prediction).all():raise ValueError('Incomplete prediction')
    forecast.to_csv(out/'forecast.csv',index=False)
    errors=validate(out/'composition.csv',out/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    p.save_json(out/'validation.json',dict(format_validation='PASS',official_submitted=False,
       files={n:p.sha256(out/n) for n in ['composition.csv','forecast.csv']},result=result))
    p.log(f'Validated candidate: {out}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--stage',required=True,choices=['train','evaluate','produce'])
    args=ap.parse_args();protocol()
    {'train':fit_seasons,'evaluate':evaluate,'produce':produce}[args.stage](args.data)


if __name__=='__main__':main()
