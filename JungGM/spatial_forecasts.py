"""Neighbor forecast context using only the supplied masked site coordinates.

No station location or original site coordinate is estimated. Neighbor sites
are selected in the supplied coordinate system; shared weather stations count
once. Every value is a forecast issued for the same target day.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as p
import improvement_experiment as ex

RUN=ex.HERE/'_local/spatial_v18'
VARIABLES={'r17_wa_cloud':'wa_station','r17_wb_cloud':'wb_station',
           'r17_wb_uv_idx':'wb_station','r17_wb_temp':'wb_station','r17_wb_rel_hum':'wb_station'}
NAMES=[f'neighbor{k}_{c}_{stat}' for c in VARIABLES for k in [4,12] for stat in ['mean','std','delta']]


def weights(sites,station,k):
    lat=np.deg2rad(sites.lat.to_numpy());lon=np.deg2rad(sites.lon.to_numpy())
    distance=(lat[:,None]-lat[None,:])**2+((lon[:,None]-lon[None,:])*np.cos((lat[:,None]+lat[None,:])/2))**2
    ids=sites[station].to_numpy();w=np.zeros((len(sites),len(sites)))
    for i in range(len(sites)):
        seen={ids[i]};chosen=[]
        for j in np.argsort(distance[i]):
            if ids[j] in seen:continue
            seen.add(ids[j]);chosen.append(j)
            if len(chosen)==k:break
        if not chosen:raise ValueError('No distinct neighboring weather station')
        w[i,chosen]=1/len(chosen)
    return w


def moments(x,w):
    finite=np.isfinite(x);den=finite.astype(float)@w.T
    mean=np.divide(np.where(finite,x,0)@w.T,den,out=np.full_like(den,np.nan),where=den>0)
    sq=np.divide(np.where(finite,x*x,0)@w.T,den,out=np.full_like(den,np.nan),where=den>0)
    return mean,np.sqrt(np.maximum(sq-mean**2,0))


def prepare(data,split):
    dest=RUN/'cache'/split;dest.mkdir(parents=True,exist_ok=True)
    signature={n:p.sha256(data/n) for n in ['sites.csv',f'{split}/forecast_wa.parquet',f'{split}/forecast_wb.parquet']}
    signature['code']=p.sha256(Path(__file__))
    manifest=dest/'manifest.json'
    if manifest.exists():
        saved=ex.read(manifest)
        if saved['signature']!=signature:raise ValueError('Stale spatial forecast cache')
        return
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id').reset_index(drop=True)
    transforms={(s,k):weights(sites,s,k) for s in ['wa_station','wb_station'] for k in [4,12]}
    months=pd.period_range('2022-07','2025-05',freq='M') if split=='train' else pd.period_range('2025-06','2026-05',freq='M')
    buffers={sid:[] for sid in sites.site_id}
    for month in months:
        d=pd.read_parquet(ex.CACHE/split/f'{month}.parquet',columns=['site_id','ts']+list(VARIABLES))
        cubes={c:d.pivot(index='ts',columns='site_id',values=c).reindex(columns=sites.site_id) for c in VARIABLES}
        ix=cubes[next(iter(cubes))].index;values={}
        for c,station in VARIABLES.items():
            x=cubes[c].to_numpy(dtype=float)
            for k in [4,12]:
                mean,std=moments(x,transforms[(station,k)])
                for stat,arr in [('mean',mean),('std',std),('delta',x-mean)]:values[f'neighbor{k}_{c}_{stat}']=arr.astype('float32')
        for j,sid in enumerate(sites.site_id):
            frame=pd.DataFrame({name:v[:,j] for name,v in values.items()});frame['ts']=ix;buffers[sid].append(frame)
        p.log(f'Spatial forecast context {split} {month}')
    for sid,parts in buffers.items():
        pd.concat(parts,ignore_index=True).to_parquet(dest/f'{sid}.parquet',index=False,compression='zstd')
    p.save_json(manifest,dict(signature=signature,features=NAMES,sites=len(sites),labels_used=False,
        coordinate_use='Neighbor relationships in supplied masked site coordinates only; no coordinate recovery'))
    del buffers;gc.collect()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--split',choices=['train','test'],default='train');a=ap.parse_args();prepare(a.data,a.split)
