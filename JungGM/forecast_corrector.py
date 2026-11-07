"""Learn forecast bias from provided historical weather, never test actuals.

The auxiliary teachers are frozen at 2022-12-31 17:00, before every power-model
evaluation origin. Their inputs are only already issued weather and calendar.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pipeline as p
import improvement_experiment as ex

RUN=ex.HERE/'_local/weather_teacher_v14'
CUTOFF=p.stamp('2022-12-31 17:00')
FIELDS={'a':'cloud temp humidity ground_press wind_speed rain snow'.split(),
        'b':'temp feel_temp wet_bulb_temp dew_point wind_speed wind_gust_spd rel_hum vis ceiling uv_idx precip_prob total_liq snow cloud'.split()}
TARGETS={'a':['cloud'],'b':['cloud','uv_idx']}


def calendar(d,station):
    d=d.copy();t=d.ts-pd.Timedelta(minutes=30)
    d['hour_sin']=np.sin(2*np.pi*(t.dt.hour+t.dt.minute/60)/24)
    d['hour_cos']=np.cos(2*np.pi*(t.dt.hour+t.dt.minute/60)/24)
    d['year_sin']=np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25)
    d['year_cos']=np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)
    d['station_code']=pd.Categorical(d[station])
    return d


def matrix(data,source):
    prefix='wa' if source=='a' else 'wb';station=prefix+'_station'
    raw=pd.read_parquet(data/'train'/f'forecast_{prefix}.parquet')
    raw=raw[raw.ts<=p.stamp('2024-01-01 00:00')].copy()
    frames=[]
    for rd in [10,17]:
        z=raw[raw['round']==rd][[station,'ts']+[prefix+'_'+f for f in FIELDS[source]]].copy()
        z=z.rename(columns={prefix+'_'+f:f'{source}{rd}_{f}' for f in FIELDS[source]})
        for f in FIELDS[source]:
            c=f'{source}{rd}_{f}'
            if f in ['cloud','humidity','rel_hum']:z[c]=z[c].clip(0,100)
            if f in ['rain','snow','total_liq','wind_speed','wind_gust_spd']:z[c]=z[c].where(z[c]>=0)
            if source=='b' and f in ['wind_speed','wind_gust_spd']:z[c]/=3.6
            if source=='b' and f=='snow':z[c]*=10
        frames.append(z)
    d=frames[0].merge(frames[1],on=[station,'ts'],how='outer',validate='one_to_one')
    obs=pd.read_parquet(data/'train'/f'observed_{prefix}.parquet',columns=[station,'ts']+[prefix+'_'+f for f in TARGETS[source]])
    # Six A-provider station/hour keys have contradictory observations. Exclude
    # every copy of an ambiguous key instead of selecting an arbitrary reading.
    obs=obs.loc[~obs.duplicated([station,'ts'],keep=False)]
    d=d.merge(obs,on=[station,'ts'],how='inner',validate='one_to_one')
    return calendar(d,station)


def train(data):
    RUN.mkdir(parents=True,exist_ok=True)
    for source in ['a','b']:
        if all((RUN/f'{source}_{t}.txt').exists() and (RUN/f'{source}_{t}.json').exists() for t in TARGETS[source]):
            continue
        d=matrix(data,source);prefix='wa' if source=='a' else 'wb'
        names=[f'{source}{rd}_{f}' for rd in [10,17] for f in FIELDS[source]]+['hour_sin','hour_cos','year_sin','year_cos','station_code']
        for target in TARGETS[source]:
            dest=RUN/f'{source}_{target}.txt'
            if dest.exists():continue
            truth=d[prefix+'_'+target];upper=100 if target=='cloud' else 25
            mask=truth.between(0,upper)&d[f'{source}17_{target}'].notna()
            tr=d[mask&(d.ts<=CUTOFF)];val=d[mask&(d.ts>CUTOFF)]
            model=lgb.LGBMRegressor(n_estimators=220,num_leaves=15,learning_rate=.04,
                min_child_samples=1000,reg_lambda=30,colsample_bytree=.9,max_bin=63,
                n_jobs=2,random_state=20261103,verbosity=-1,deterministic=True,force_col_wise=True)
            model.fit(tr[names],tr[prefix+'_'+target]-tr[f'{source}17_{target}'],categorical_feature=['station_code'])
            model.booster_.save_model(str(dest))
            pred=np.clip(val[f'{source}17_{target}']+model.predict(val[names],num_threads=2),0,upper)
            error=(pred-val[prefix+'_'+target]).to_numpy()
            before=(val[f'{source}17_{target}']-val[prefix+'_'+target]).to_numpy()
            record=dict(source=source,target=target,cutoff=str(CUTOFF),last_label=str(tr.ts.max()),
                features=names,station_categories=list(d.station_code.cat.categories),rows=len(tr),
                validation_rows=len(val),validation_period='2023',raw_rmse=float(np.sqrt(np.mean(before**2))),
                corrected_rmse=float(np.sqrt(np.mean(error**2))),model_sha256=p.sha256(dest),
                observed_source_sha256=p.sha256(data/'train'/f'observed_{prefix}.parquet'))
            p.save_json(RUN/f'{source}_{target}.json',record)
            p.log(str({k:v for k,v in record.items() if k not in ['features','station_categories','observed_source_sha256']}))


def load():
    result={}
    for source in ['a','b']:
        for target in TARGETS[source]:
            path=RUN/f'{source}_{target}.txt';meta=ex.read(RUN/f'{source}_{target}.json')
            if meta['cutoff']!=str(CUTOFF) or meta['model_sha256']!=p.sha256(path):raise ValueError('Weather teacher provenance mismatch')
            result[(source,target)]=(lgb.Booster(model_file=str(path)),meta)
    return result


def augment(d,teachers):
    d=d.copy()
    for (source,target),(model,record) in teachers.items():
        station='wa_station' if source=='a' else 'wb_station'
        x=calendar(d,station);x.station_code=pd.Categorical(x[station],categories=record['station_categories'])
        upper=100 if target=='cloud' else 25
        raw=d[f'{source}17_{target}'].to_numpy()
        corrected=np.clip(raw+model.predict(x[record['features']],num_threads=2),0,upper)
        d[f'corrected_{source}_{target}']=corrected
        d[f'corrected_q_{source}_{target}']=corrected*d.solar_q
    return d


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args();train(a.data)
