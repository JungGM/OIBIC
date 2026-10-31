"""Local OIBC pilot: matched-budget baseline, joint weather features, held-out grouping.
No evaluation-period actuals, no external data. Official raw baseline is kept separately.
"""
from __future__ import annotations
import argparse,gc,hashlib,importlib.metadata,json,os,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from baseline_original import solar,WA_COLS,WB_COLS,FEATS

ROOT=Path(__file__).resolve().parent
TZ='Asia/Seoul'
def timestamp(s):return pd.Timestamp(s,tz=TZ)
def log(s):print(time.strftime('%H:%M:%S')+' '+s,flush=True)
def writejson(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def read_period(path,columns,start,end):
    return pd.read_parquet(path,columns=columns,filters=[('ts','>=',start.to_pydatetime()),('ts','<=',end.to_pydatetime())])
def ts_local(s):return pd.DatetimeIndex(pd.to_datetime(s,utc=True)).tz_convert(TZ)
def periods(start,end):
    cur=start.normalize().replace(day=1)
    while cur<=end:
        nxt=cur+pd.offsets.MonthBegin(1)
        lo=max(start,cur+pd.Timedelta(hours=1));hi=min(end,nxt)
        if lo<=hi:yield lo,hi
        cur=nxt

def weather(data,start,end,multiscale=True):
    """Features are computed within station, forecast round AND target day.
    Right-closed 00:00 belongs to preceding delivery day. No following-day issuance leaks.
    """
    tables={}
    for prov,cols in [('wa',WA_COLS),('wb',WB_COLS)]:
        key=prov+'_station'
        d=read_period(data/'train'/f'forecast_{prov}.parquet',[key,'round','ts']+cols,start,end)
        d['ts']=ts_local(d.ts);d=d.sort_values([key,'round','ts'])
        d['_day']=(d.ts-pd.Timedelta(minutes=1)).dt.normalize()
        # Features based only on forecasts belonging to this same delivery day/issuance.
        for col in (([prov+'_cloud'] if prov=='wa' else ['wb_cloud','wb_uv_idx']) if multiscale else []):
            grouped=d.groupby([key,'round','_day'],sort=False)[col]
            window=np.column_stack([grouped.shift(1).to_numpy(),d[col].to_numpy(),grouped.shift(-1).to_numpy()])
            d[col+'_mean3']=np.nanmean(window,axis=1)
            d[col+'_std3']=np.nanstd(window,axis=1,ddof=0)
            d[col+'_daymean']=grouped.transform('mean')
        d=d.drop(columns='_day')
        for r in [10,17]:
            q=d.loc[d['round']==r].drop(columns='round').copy()
            q['_present']=np.float32(1)
            for c in q.columns:
                if c not in [key,'ts']:q[c]=q[c].astype('float32')
            q=q.rename(columns={c:f'{c}_r{r}' for c in q.columns if c not in [key,'ts']})
            tables[(prov,r)]=q
        del d
    return tables

def make_features(base,sites,tables,kind,r=None):
    d=base.merge(sites,on='site_id',how='left',validate='many_to_one',sort=False)
    feature_names=[]
    for prov,cols in [('wa',WA_COLS),('wb',WB_COLS)]:
        key=prov+'_station'
        for rr in ([r] if kind=='baseline' else [10,17]):
            q=tables[(prov,rr)]
            if kind=='baseline':q=q[[key,'ts']+[f'{c}_r{rr}' for c in cols]]
            d=d.merge(q,on=[key,'ts'],how='left',validate='many_to_one',sort=False)
        if kind=='baseline':
            for c in cols:d[c]=d[f'{c}_r{r}'];feature_names.append(c)
        else:
            for rr in [10,17]:
                name=f'{prov}_missing_r{rr}';d[name]=d[f'_present_r{rr}'].isna().astype('float32')
                # Each provider presence is consumed before the next provider merge.
                d=d.drop(columns=f'_present_r{rr}')
                feature_names.append(name)
            for c in cols:
                c10=f'{c}_r10';c17=f'{c}_r17'
                old10=d[c10].copy();old17=d[c17].copy()
                # Other-round fallback only at same station/ts; flags preserve missingness.
                d[c10]=old10.fillna(old17);d[c17]=old17.fillna(old10)
                delta=c+'_revision';d[delta]=old17-old10
                feature_names.extend([c10,c17,delta])
            for rr in [10,17]:
                for c in ([prov+'_cloud'] if prov=='wa' else ['wb_cloud','wb_uv_idx']):
                    feature_names.extend([f'{c}_{s}_r{rr}' for s in ['mean3','std3','daymean']])
    ts=ts_local(d.ts)
    d['elev'],d['azim'],d['cosz']=solar(ts,d.lat.to_numpy(),d.lon.to_numpy())
    doy=ts.dayofyear.to_numpy();d['doy_sin']=np.sin(2*np.pi*doy/365.25);d['doy_cos']=np.cos(2*np.pi*doy/365.25)
    d['hour']=ts.hour.to_numpy();d['cap']=d.capacity_kw
    feature_names+=['elev','azim','cosz','doy_sin','doy_cos','hour','cap','lat','lon']
    if kind!='baseline':
        # Different raw source units remain separate; differences use common units.
        for rr in [10,17]:
            for a,b,scale,name in [('wa_temp','wb_temp',1,'temp'),('wa_humidity','wb_rel_hum',1,'humidity'),('wa_wind_speed','wb_wind_speed',1/3.6,'wind')]:
                c=f'ab_{name}_r{rr}';d[c]=d[f'{a}_r{rr}']-d[f'{b}_r{rr}']*scale;feature_names.append(c)
            for prov in ['wa','wb']:
                c=f'{prov}_cloud_cosz_r{rr}';d[c]=d[f'{prov}_cloud_r{rr}']*d.cosz.clip(lower=0);feature_names.append(c)
    # Source row order is checked, not assumed after joins.
    assert np.array_equal(d['_row'].to_numpy(),base['_row'].to_numpy())
    X=d[feature_names].astype('float32')
    assert not np.isinf(X.to_numpy()).any()
    return X,d.elev.to_numpy(),d.cap.to_numpy()

def select_training(a,idx,sites,start,end,max_rows,seed):
    rng=np.random.default_rng(seed); ti,si=np.where(np.isfinite(a)&((idx>=start)&(idx<=end))[:,None])
    # Sample a pool, retain daylight by the same geometry as official baseline.
    if max_rows and len(ti)>max_rows*3:
        p=rng.choice(len(ti),max_rows*3,replace=False);ti,si=ti[p],si[p]
    elev,_,_=solar(idx[ti],sites.lat.to_numpy()[si],sites.lon.to_numpy()[si])
    good=elev>0;ti,si=ti[good],si[good]
    if max_rows and len(ti)>max_rows:
        p=rng.choice(len(ti),max_rows,replace=False);ti,si=ti[p],si[p]
    order=np.lexsort((si,ti));ti,si=ti[order],si[order]
    return pd.DataFrame({'site_id':sites.site_id.to_numpy()[si],'ts':idx[ti],'energy':a[ti,si],'_row':np.arange(len(ti))})

def build_training(base,sites,data,kind,r):
    pieces=[];names=None
    months=(base.ts-pd.Timedelta(minutes=1)).dt.strftime('%Y-%m')
    for month in sorted(months.unique()):
        b=base.loc[months==month];lo=timestamp(month+'-01')+pd.Timedelta(hours=1);hi=timestamp(month+'-01')+pd.offsets.MonthBegin(1)
        tables=weather(data,lo,hi,multiscale=kind!='baseline');X,_,_=make_features(b,sites,tables,kind,r)
        names=list(X.columns);pieces.append(X);del tables;gc.collect();log(f'{kind} training features {month}')
    X=pd.concat(pieces,ignore_index=True)
    assert len(X)==len(base)
    return X

def predictions(model,sites,data,index,kind,r):
    result=np.empty((len(index),len(sites)),dtype='float32')
    for lo,hi in periods(index[0],index[-1]):
        ids=np.flatnonzero((index>=lo)&(index<=hi));sub=index[ids]
        base=pd.DataFrame({'ts':np.repeat(sub.to_numpy(),len(sites)),'site_id':np.tile(sites.site_id.to_numpy(),len(sub))})
        base['ts']=ts_local(base.ts);base['_row']=np.arange(len(base))
        tables=weather(data,lo,hi,multiscale=kind!='baseline');X,elev,cap=make_features(base,sites,tables,kind,r)
        p=model.predict(X);p=np.where(elev>0,np.maximum(p,0)*cap,0)
        assert np.isfinite(p).all();result[ids]=p.reshape(len(sub),len(sites))
        log(f'{kind} r={r} predicted {str(lo)[:7]}');del tables,X,base;gc.collect()
    return result

def loss_by_hour(errors,z,weights):
    a=errors[:,z].sum(axis=1);b=errors[:,~z].sum(axis=1)
    return weights*(np.abs(a)+np.abs(b))

def optimize(errors,weights,capacities,z0,seed=20261029,steps=6000,restarts=3):
    """Feasible local search, small annealed exploration; best feasible state retained.
    Receives ONLY composition-selection errors, never external holdout errors.
    """
    rng=np.random.default_rng(seed);C=capacities;lo=20000.;hi=C.sum()-20000.
    total=errors.sum(axis=1);bestz=z0.copy();best=float(loss_by_hour(errors,z0,weights).sum());trace=[]
    for restart in range(restarts):
        z=z0.copy() if restart==0 else bestz.copy();cap=float(C[z].sum());a=errors[:,z].sum(axis=1);cur=float((weights*(np.abs(a)+np.abs(total-a))).sum())
        temp0=max(cur*0.0002,1.0)
        for step in range(steps):
            if rng.random()<.85:
                i=int(rng.choice(np.flatnonzero(z)));j=int(rng.choice(np.flatnonzero(~z)));newcap=cap-C[i]+C[j]
                if not lo-1e-7<=newcap<=hi+1e-7:continue
                delta=errors[:,j]-errors[:,i];flip=[i,j]
            else:
                i=int(rng.integers(len(z)));sign=-1 if z[i] else 1;newcap=cap+sign*C[i]
                if not lo-1e-7<=newcap<=hi+1e-7:continue
                delta=sign*errors[:,i];flip=[i]
            aa=a+delta;value=float((weights*(np.abs(aa)+np.abs(total-aa))).sum());dv=value-cur
            temp=temp0*(1-step/steps)**3
            accept=dv<0 or (temp>0 and rng.random()<np.exp(-max(dv,0)/temp))
            if accept:
                z[flip]=~z[flip];cap=newcap;a=aa;cur=value
                if cur<best:best=cur;bestz=z.copy()
        trace.append({'restart':restart,'best_selection_W':best,'capacity_G1':float(C[bestz].sum())})
    assert lo-1e-7<=C[bestz].sum()<=hi+1e-7
    assert best<=float(loss_by_hour(errors,z0,weights).sum())+1e-4
    return bestz,trace

def metric(errors,pred,actual,z,w):
    ge=np.column_stack([errors[:,z].sum(axis=1),errors[:,~z].sum(axis=1)])
    W=float((w[:,None]*abs(ge)).sum())
    return {'hours':len(w),'weighted_loss_W':W,'penalty_krw':.1*W,'group_MAE_kwh':float(abs(ge).mean()),'site_MAE_kwh':float(abs(errors).mean()),'group_bias_kwh':ge.mean(axis=0).tolist()}

def run(args):
    starttime=time.time();data=Path(args.data).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    conf=vars(args).copy();conf.update({'baseline_description':'official feature/parameter structure, matched fixed daylight subsample; not full official baseline','validation_policy':'selection Jan-Feb 2025; locked external holdout Mar-May 2025','packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','lightgbm','scipy']},'python':platform.python_version()})
    # A run directory is immutable with respect to experiment settings.
    if (out/'config.json').exists():
        previous=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if previous!=conf:raise ValueError('Existing run has different settings; choose a new --out directory.')
    else:writejson(out/'config.json',conf)
    # Prediction caches must match code and original input fingerprints.
    paths=[Path(__file__),ROOT/'baseline_original.py',data/'sites.csv',data/'train/generation.parquet',data/'train/forecast_wa.parquet',data/'train/forecast_wb.parquet',data/'train/smp.csv']
    fingerprint={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths}
    fingerprint['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fp=out/'fingerprint.json'
    if fp.exists():
        if json.loads(fp.read_text(encoding='utf-8'))!=fingerprint:raise ValueError('Code/input changed; choose a new --out directory.')
    elif any(out.glob('*_predictions.npy')):
        raise ValueError('Unversioned prediction caches; choose a new --out directory.')
    else:writejson(fp,fingerprint)
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id').reset_index(drop=True)
    audit=ROOT/'runs/audit';a=np.load(audit/'generation.npy',mmap_mode='r');idx=ts_local(pd.read_csv(audit/'time_index.csv').ts)
    base=select_training(a,idx,sites,timestamp(args.train_start),timestamp(args.train_end),args.max_train_rows,args.seed)
    target=(base.energy.to_numpy()/sites.set_index('site_id').capacity_kw.reindex(base.site_id).to_numpy()).clip(0,1.2).astype('float32')
    writejson(out/'training_summary.json',{'rows':len(base),'sites_with_training':int(base.site_id.nunique()),'daylight_rows_per_site_min':int(base.groupby('site_id').size().min()),'min_ts':str(base.ts.min()),'max_ts':str(base.ts.max()),'row_key_hash':hashlib.sha256(pd.util.hash_pandas_object(base[['site_id','ts']],index=False).values.tobytes()).hexdigest()})
    valmask=(idx>=timestamp(args.selection_start))&(idx<=timestamp(args.holdout_end));vi=idx[valmask];actual=np.array(a[valmask],dtype='float64');del a
    # Never impute unavailable ground truth. A stable common mask is used for all candidates.
    complete=np.isfinite(actual).all(axis=1);writejson(out/'evaluation_coverage.json',{'all_hours':len(vi),'complete_hours':int(complete.sum())})
    if not complete.all():raise ValueError('Chosen validation period is not fully observed: revise split before fitting.')
    log(f'training {len(base):,} identical daylight rows, validation {len(vi)} hours, {len(sites)} sites')
    allpred={}
    for kind,rounds in [('baseline',[10,17]),('enhanced',[None])]:
        rp=[]
        for r in rounds:
            label=f'{kind}_{r}';cache=out/f'{label}_predictions.npy'
            if cache.exists():
                log(f'resume cached predictions {label}');rp.append(np.load(cache));continue
            log(f'building {label} training features')
            X=build_training(base,sites,data,kind,r);writejson(out/f'{label}_features.json',list(X.columns));log(f'fit {label} {X.shape}')
            model=lgb.LGBMRegressor(n_estimators=args.trees,learning_rate=.06,num_leaves=127,min_child_samples=200,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbosity=-1,n_jobs=args.threads,random_state=args.seed)
            t=time.time();model.fit(X,target,callbacks=[lgb.log_evaluation(100)]);log(f'fit complete {label}: {time.time()-t:.1f}s')
            model.booster_.save_model(str(out/f'{label}.txt'));pd.DataFrame({'feature':X.columns,'gain':model.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False).to_csv(out/f'{label}_importance.csv',index=False)
            del X;gc.collect();p=predictions(model,sites,data,vi,kind,r);np.save(cache,p);rp.append(p);del model;gc.collect()
        allpred[kind]=np.mean(rp,axis=0,dtype='float64');del rp
    pd.DataFrame({'ts':vi}).to_csv(out/'prediction_time_index.csv',index=False)
    sites[['site_id']].to_csv(out/'prediction_site_order.csv',index=False)
    smp=pd.read_csv(data/'train/smp.csv');smp['ts']=ts_local(smp.ts);s=smp.set_index('ts').smp.reindex(vi).to_numpy()
    assert np.isfinite(s).all() and (s>=0).all();w=s*np.where((vi.hour>=11)&(vi.hour<=15),3.,1.)
    selection=vi<=timestamp(args.selection_end);holdout=vi>=timestamp(args.holdout_start)
    assert not (selection&holdout).any() and (selection|holdout).all()
    z0=sites.current_group.eq('A').to_numpy();assert sites.capacity_kw.to_numpy()[z0].sum()>=20000 and sites.capacity_kw.to_numpy()[~z0].sum()>=20000
    candidate_error=allpred['enhanced']-actual
    log('optimize composition on selection months ONLY')
    z,trace=optimize(candidate_error[selection],w[selection],sites.capacity_kw.to_numpy(),z0,seed=args.seed,steps=args.search_steps,restarts=3)
    writejson(out/'optimizer_trace.json',trace)
    for name,zz in [('reference',z0),('candidate',z)]:
        pd.DataFrame({'site_id':sites.site_id,'group':np.where(zz,'G1','G2')}).to_csv(out/f'composition_{name}.csv',index=False)
    rows=[];monthly=[]
    for name,p in allpred.items():
        e=p-actual;np.save(out/f'{name}_residuals.npy',e.astype('float32'))
        for comp,zz in [('reference',z0),('candidate',z)]:
            for period,mask in [('selection',selection),('holdout',holdout)]:
                r={'model':name,'composition':comp,'period':period,**metric(e[mask],p[mask],actual[mask],zz,w[mask])};rows.append(r)
            for month in sorted(set((vi-pd.Timedelta(minutes=1)).strftime('%Y-%m'))):
                mask=(vi-pd.Timedelta(minutes=1)).strftime('%Y-%m')==month
                monthly.append({'model':name,'composition':comp,'month':month,**metric(e[mask],p[mask],actual[mask],zz,w[mask])})
    metrics=pd.DataFrame(rows)
    for period in ['selection','holdout']:
        ref=float(metrics.loc[(metrics.model=='baseline')&(metrics.composition=='reference')&(metrics.period==period),'weighted_loss_W'].iloc[0]);mask=metrics.period==period
        metrics.loc[mask,'local_score_vs_matched_baseline']=100*(1-metrics.loc[mask,'weighted_loss_W']/ref)
    metrics.to_csv(out/'metrics.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False)
    # Paired day-block bootstrap: uncertainty estimate, not a new selection criterion.
    day=(vi-pd.Timedelta(minutes=1)).strftime('%Y-%m-%d');rng=np.random.default_rng(args.seed)
    b_loss=loss_by_hour(allpred['baseline']-actual,z0,w);c_loss=loss_by_hour(candidate_error,z,w)
    daily=pd.DataFrame({'day':day[holdout],'base':b_loss[holdout],'candidate':c_loss[holdout]}).groupby('day').sum();draw=rng.integers(0,len(daily),size=(2000,len(daily)));boot=100*(1-daily.candidate.to_numpy()[draw].sum(axis=1)/daily.base.to_numpy()[draw].sum(axis=1))
    summary={'runtime_seconds':time.time()-starttime,'capacity_G1_kw':float(sites.capacity_kw.to_numpy()[z].sum()),'capacity_G2_kw':float(sites.capacity_kw.to_numpy()[~z].sum()),'changed_sites':int((z!=z0).sum()),'holdout_days':len(daily),'paired_day_bootstrap_score_95pct':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_caveat':'day blocks assume days can be resampled; multi-day weather dependence may make intervals optimistic','limitations':['100k-row cap is configurable; compare matched-budget baseline, not official server baseline','selection winter, holdout spring; no summer/fall full-253-site validation','candidate composition selected for enhanced model only; baseline+candidate is diagnostic','no submission or test-year inference performed']}
    writejson(out/'summary.json',summary)
    log(metrics[['model','composition','period','penalty_krw','local_score_vs_matched_baseline']].to_string(index=False));log(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'data'));ap.add_argument('--out',default=str(ROOT/'runs/pilot_v2'))
    ap.add_argument('--max-train-rows',type=int,default=100000);ap.add_argument('--trees',type=int,default=600);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=20261029);ap.add_argument('--search-steps',type=int,default=6000)
    ap.add_argument('--train-start',default='2023-01-01 01:00');ap.add_argument('--train-end',default='2025-01-01 00:00');ap.add_argument('--selection-start',default='2025-01-01 01:00');ap.add_argument('--selection-end',default='2025-03-01 00:00');ap.add_argument('--holdout-start',default='2025-03-01 01:00');ap.add_argument('--holdout-end',default='2025-06-01 00:00')
    run(ap.parse_args())
