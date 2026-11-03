"""Time-external expert mixture selected before evaluating its spring period."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as p
import improvement_experiment as ex
import fleet_residual as fr
from audit_site_expert import ROOTS
from validate_submission import validate
from submission_guard import ensure_writable


def grid(n,total=8,prefix=()):
    if n==1:
        yield prefix+(total,)
    else:
        for count in range(total+1):yield from grid(n-1,total-count,prefix+(count,))


def panel(data,experts,production=False):
    d=fr.source_panel(production)
    for label in experts:
        root=ex.HERE/'_local'/ROOTS[label]
        parts=['production'] if production else ['seasonal_proxy','complete_fleet','development']
        pred=pd.concat([pd.read_parquet(root/s/'predictions.parquet') for s in parts],ignore_index=True)
        d=d.merge(pred.rename(columns={'new':label}),on=['site_id','ts'],how='left',validate='one_to_one')
    cols=['v8']+experts
    if not np.isfinite(d[cols]).all().all():raise ValueError('Incomplete expert predictions')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    d['group']=d.site_id.map(comp)
    g=d.groupby(['ts','group'])[cols+([] if production else ['energy'])].sum().reset_index()
    if not production:g['weight']=g.ts.map(ex.prices(data).weight)
    return g


def evaluate(data,experts,run):
    run.mkdir(parents=True,exist_ok=True);g=panel(data,experts)
    cols=['v8']+experts;sel=g[g.ts<=p.stamp('2025-03-01 00:00')].copy()
    candidates=[]
    for counts in grid(len(cols)):
        w=np.array(counts)/8;pred=sel[cols].to_numpy()@w;rows=fr.summary(sel,pred)
        score=.5*np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows[:6]])+.5*np.mean([r['candidate_penalty']/r['base_penalty'] for r in rows[6:]])
        candidates.append(dict(weights=dict(zip(cols,w.tolist())),score=float(score),months=rows))
    chosen=min(candidates,key=lambda c:c['score']);p.save_json(run/'selected.json',dict(selected=chosen,choices=candidates))
    z=g[g.ts>=p.stamp('2025-03-01 01:00')].copy();z['candidate']=z[cols].to_numpy()@np.array(list(chosen['weights'].values()))
    rows=fr.summary(z,z.candidate);gain=100*(1-fr.penalty(z,z.candidate)/fr.penalty(z,z.v8))
    z['saving']=.1*z.weight*(abs(z.v8-z.energy)-abs(z.candidate-z.energy))
    daily=z.groupby((z.ts-pd.Timedelta(hours=1)).dt.date).saving.sum().to_numpy();n=len(daily)
    rng=np.random.default_rng(20261104)
    samples=[daily[(rng.integers(n,size=(int(np.ceil(n/3)),1))+np.arange(3))%n].ravel()[:n].mean() for _ in range(2000)]
    lower=float(np.quantile(samples,.05))
    accepted=chosen['score']<1 and gain>=1 and min(r['gain_pct'] for r in rows)>-2 and lower>0
    result=dict(accepted=bool(accepted),selected=chosen,months=rows,total_gain_pct=gain,lower90_daily_saving=lower,
                experts=experts,disclosure='Repeated historical model development; not an untouched final holdout')
    p.save_json(run/'evaluation.json',result);z.to_parquet(run/'development_predictions.parquet',index=False)
    p.log(str(result))


def produce(data,experts,run,candidate):
    ensure_writable(ex.HERE/'제출'/candidate)
    result=ex.read(run/'evaluation.json')
    if not result['accepted'] or result['experts']!=experts:raise ValueError('Rejected or mismatched candidate')
    weights=result['selected']['weights'];active=[e for e in experts if weights[e]>0]
    # Production experts must already have been trained through their own entry points.
    d=panel(data,active,True);d['forecast_kwh']=sum(weights[e]*d[e] for e in ['v8']+active)
    dest=ex.HERE/'제출'/candidate;dest.mkdir(parents=True,exist_ok=True)
    pd.read_csv(ex.HERE/'제출/composition.csv').to_csv(dest/'composition.csv',index=False)
    d[['group','ts','forecast_kwh']].to_csv(dest/'forecast.csv',index=False)
    errors=validate(dest/'composition.csv',dest/'forecast.csv',data/'sites.csv')
    if errors:raise ValueError(errors)
    records={e:ex.read(ex.HERE/'_local'/ROOTS[e]/'production/metadata.json') for e in active}
    p.save_json(dest/'validation.json',dict(format_validation='PASS',official_submitted=False,result=result,models=records,
        files={n:p.sha256(dest/n) for n in ['composition.csv','forecast.csv']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--experts',nargs='+',choices=list(ROOTS),required=True)
    ap.add_argument('--run',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate');a=ap.parse_args()
    if Path(a.run).name!=a.run or Path(a.candidate).name!=a.candidate:raise ValueError('Names cannot contain paths')
    root=ex.HERE/'_local'/a.run
    if a.stage=='evaluate':evaluate(a.data,a.experts,root)
    else:produce(a.data,a.experts,root,a.candidate)
