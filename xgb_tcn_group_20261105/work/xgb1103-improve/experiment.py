"""Controlled local improvement of the downloaded 11.03 XGBoost baseline."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'reference'))
BASE = json.loads((ROOT/'reference/feature_sets.json').read_text(encoding='utf-8'))['enhanced']
ADD = [f'{s}_revision_{v}' for s in ['a','b'] for v in ['cloud','temp','humidity','wind']]
ADD += [f'{s}10_{v}' for s in ['a','b'] for v in ['temp','cloud']]
ADD += [f'b17_{v}_{op}' for v in ['cloud','uv'] for op in ['change','mean3','std3']]
ADD += ['a10_available','b10_available']
SETS = {'baseline52': BASE, 'augmented72': BASE+ADD,
        'without_site_code51': [f for f in BASE if f != 'site_code']}
if (ROOT/'expanded_features.json').exists():
    expanded=json.loads((ROOT/'expanded_features.json').read_text(encoding='utf-8'))
    SETS[expanded['name']]=expanded['features']
PARAMS = dict(n_estimators=300, objective='reg:squarederror', tree_method='hist',
              n_jobs=4, random_state=42, max_depth=8, learning_rate=.03,
              subsample=.8, colsample_bytree=.8, min_child_weight=100, reg_lambda=30)
TRAIN_END = pd.Timestamp('2024-12-31 17:00', tz='Asia/Seoul')
DEV_END = pd.Timestamp('2025-03-31 17:00', tz='Asia/Seoul')
HOLD_START = pd.Timestamp('2025-04-01 01:00', tz='Asia/Seoul')
FINAL_END = pd.Timestamp('2025-05-31 17:00', tz='Asia/Seoul')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def log(**kw): print(json.dumps(kw), flush=True)


def aux(f, source, station, ix):
    key = 'wa_station' if source == 'a' else 'wb_station'
    pre = 'wa_' if source == 'a' else 'wb_'
    raw = f[f[key] == station]
    r = {i: raw[raw['round'] == i].set_index('ts') for i in [10,17]}
    out = pd.DataFrame(index=ix)
    for v, col in [('cloud','cloud'),('temp','temp'),('humidity','humidity' if source=='a' else 'rel_hum'),('wind','wind_speed')]:
        out[f'{source}_revision_{v}'] = r[17][pre+col].reindex(ix)-r[10][pre+col].reindex(ix)
        if source == 'b' and v == 'wind': out[f'{source}_revision_{v}'] /= 3.6
    for v in ['temp','cloud']: out[f'{source}10_{v}'] = r[10][pre+v].reindex(ix)
    out[f'{source}10_available'] = ix.isin(r[10].index).astype(float)
    if source == 'b':
        day = (ix-pd.Timedelta(hours=1)).normalize()
        pos = (ix.hour-1)%24
        for v, col in [('cloud','cloud'),('uv','uv_idx')]:
            a = r[17][pre+col].reindex(ix)
            out[f'b17_{v}_change'] = a.groupby(day).diff()
            out[f'b17_{v}_mean3'] = a.rolling(3,center=True,min_periods=3).mean().where((pos>=1)&(pos<=22))
            out[f'b17_{v}_std3'] = a.rolling(3,center=True,min_periods=3).std(ddof=0).where((pos>=1)&(pos<=22))
    return out


def prepare(a):
    os.environ['OIBC_DATA'] = str(a.data.resolve())
    from run_experiment import enhanced, v1
    a.cache.mkdir(parents=True, exist_ok=True)
    if (a.cache/'manifest.json').exists(): raise ValueError('Completed cache exists')
    sites = pd.read_csv(a.data/'sites.csv');sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    gen = pd.read_parquet(a.data/'train/generation.parquet')
    gen['_row_id'] = np.arange(len(gen), dtype='int64')
    prices = pd.read_csv(a.data/'train/smp.csv');prices.ts=pd.to_datetime(prices.ts)
    report = {'sets':SETS,'params':PARAMS,'years':{},'source_sha256':{p.name:sha(p) for p in (ROOT/'reference').iterdir() if p.is_file()},
              'note':'Original row order restored before fitting; all label values retained in cache, negatives excluded from fits.'}
    for year in range(2021,2026):
        start = pd.Timestamp(f'{year}-01-01 01:00',tz='Asia/Seoul')
        end = pd.Timestamp(f'{year+1}-01-01 00:00',tz='Asia/Seoul') if year<2025 else pd.Timestamp('2025-06-01 00:00',tz='Asia/Seoul')
        g = gen[gen.ts.between(start,end)]
        scope=sites[sites.site_id.isin(g.site_id)]
        ix=pd.date_range(start,end,freq='h')
        fa=pd.read_parquet(a.data/'train/forecast_wa.parquet',filters=[('ts','>=',start),('ts','<=',end)])
        fb=pd.read_parquet(a.data/'train/forecast_wb.parquet',filters=[('ts','>=',start),('ts','<=',end)])
        fa=fa[fa['round'].isin([10,17])];fb=fb[fb['round'].isin([10,17])]
        for f,key in [(fa,'wa_station'),(fb,'wb_station')]:
            if f.duplicated([key,'round','ts']).any(): raise ValueError('Duplicate weather key')
        aa={};bb={};rows=0;dest=a.cache/str(year);dest.mkdir(exist_ok=True)
        log(stage='prepare',year=year,sites=len(scope))
        for j,site in enumerate(scope.itertuples(index=False)):
            y=g[g.site_id==site.site_id].copy().reset_index(drop=True)
            d=v1.b.features(y,sites,fa[fa.wa_station==site.wa_station],fb[fb.wb_station==site.wb_station],None,17)
            if enhanced(d)!=BASE or not d[['site_id','ts']].equals(y[['site_id','ts']]): raise ValueError('Baseline feature/order mismatch')
            if site.wa_station not in aa: aa[site.wa_station]=aux(fa,'a',site.wa_station,ix)
            if site.wb_station not in bb: bb[site.wb_station]=aux(fb,'b',site.wb_station,ix)
            out=d[['site_id','ts']+BASE].copy()
            extra=pd.concat([aa[site.wa_station],bb[site.wb_station]],axis=1).reindex(pd.DatetimeIndex(y.ts))
            for f in ADD: out[f]=extra[f].to_numpy()
            for f in BASE+ADD: out[f]=out[f].astype('float32')
            out['_row_id']=y['_row_id'].to_numpy();out['energy']=y.energy.to_numpy()
            out['capacity_kw']=site.capacity_kw;out['group']=site.group
            pp=prices.set_index('ts').smp.reindex(pd.DatetimeIndex(y.ts)).to_numpy()
            if not np.isfinite(pp).all(): raise ValueError('Missing SMP')
            out['smp']=pp;out['k']=np.where(out.ts.dt.hour.between(11,15),3,1)
            out['target_cf']=(out.energy/out.capacity_kw).clip(0,1.2)
            out['sample_weight']=out.capacity_kw*out.smp*out.k
            if len(out)!=len(y) or out.duplicated(['site_id','ts']).any(): raise ValueError('Bad site join')
            out.to_parquet(dest/f'{site.site_id}.parquet',index=False,compression='zstd')
            rows+=len(out)
            if (j+1)%40==0 or j+1==len(scope): log(year=year,sites_done=j+1,rows=rows)
        report['years'][str(year)]={'rows':rows,'sites':len(scope)}
        save(a.cache/'progress.json',report)
        del fa,fb,aa,bb,g;gc.collect()
    save(a.cache/'manifest.json',report)


def load(a, years, columns, filters):
    chunks=[pd.read_parquet(p,columns=columns,filters=filters) for year in years for p in sorted((a.cache/str(year)).glob('*.parquet'))]
    d=pd.concat(chunks,ignore_index=True).sort_values('_row_id',kind='stable').reset_index(drop=True)
    return d


class Progress(xgb.callback.TrainingCallback):
    def after_iteration(self,model,epoch,evals_log):
        if (epoch+1)%50==0: log(stage='fit',trees=epoch+1)
        return False


def fit(a):
    names=SETS[a.variant];out=a.out/a.variant;out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions.parquet').exists(): raise ValueError('Fit result already exists')
    filters=[('ts','<=',TRAIN_END),('elev','>',0),('energy','>=',0)]
    meta=load(a,range(2021,2025),['target_cf','sample_weight','_row_id'],filters)
    y=meta.target_cf.to_numpy(copy=True);w=meta.sample_weight.to_numpy(copy=True);w=w/w.mean()
    positions=np.full(int(meta._row_id.max())+1,-1,dtype='int32')
    positions[meta._row_id.to_numpy()]=np.arange(len(meta),dtype='int32')
    train=np.lib.format.open_memmap(out/'training_features.npy',mode='w+',dtype='float32',shape=(len(meta),len(names)))
    written=0
    for year in range(2021,2025):
        for path in sorted((a.cache/str(year)).glob('*.parquet')):
            part=pd.read_parquet(path,columns=names+['_row_id'],filters=filters)
            ix=positions[part._row_id.to_numpy()]
            assert (ix>=0).all()
            train[ix]=part[names].to_numpy(dtype='float32');written+=len(part)
        train.flush();log(stage='matrix_ready',variant=a.variant,year=year,rows=written)
    assert written==len(meta)
    del meta,positions,part;gc.collect()
    params={**PARAMS,'callbacks':[Progress()]};model=xgb.XGBRegressor(**params)
    log(stage='fit_start',variant=a.variant,rows=len(train),features=len(names))
    t=time.time();model.fit(train,y,sample_weight=w)
    model.get_booster().feature_names=names
    model.get_booster().feature_types=['float']*len(names)
    model.save_model(out/'model.ubj')
    importance=model.get_booster().get_score(importance_type='gain')
    rows=len(train);del train,y,w;gc.collect()
    while a.wait_expansion and not (ROOT/'expanded_features.json').exists():
        log(stage='waiting_for_feature_cache_completion');time.sleep(10)
    pieces=[];keys=['_row_id','site_id','ts','energy','capacity_kw','group','smp','k']
    for path in sorted((a.cache/'2025').glob('*.parquet')):
        valid=pd.read_parquet(path,columns=names+keys)
        prediction=np.maximum(model.predict(valid[names]),0)*valid.capacity_kw.to_numpy()
        prediction[valid.elev.to_numpy()<=0]=0
        if not np.isfinite(prediction).all(): raise ValueError('Invalid predictions')
        z=valid[keys].copy();z['prediction']=prediction;pieces.append(z)
    result=pd.concat(pieces,ignore_index=True).sort_values('_row_id',kind='stable').reset_index(drop=True)
    result.to_parquet(out/'predictions.parquet',index=False)
    save(out/'run.json',{'variant':a.variant,'features':names,'training_rows':rows,'training_end':str(TRAIN_END),'parameters':PARAMS,'seconds':time.time()-t,'model_sha256':sha(out/'model.ubj'),'feature_gain':importance,'versions':{'xgboost':xgb.__version__,'pandas':pd.__version__,'numpy':np.__version__}})
    log(stage='fit_complete',variant=a.variant,seconds=time.time()-t)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--data',type=Path,default=Path('work/data'));p.add_argument('--cache',type=Path,default=ROOT/'cache');p.add_argument('--out',type=Path,default=ROOT/'runs');p.add_argument('--variant',choices=list(SETS),default='baseline52');p.add_argument('--wait-expansion',action='store_true');a=p.parse_args()
    if a.stage=='prepare': prepare(a)
    else: fit(a)


if __name__=='__main__': main()
