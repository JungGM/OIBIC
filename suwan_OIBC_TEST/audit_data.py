"""Streaming audit. Raw records remain local; stdout contains aggregate counts only."""
from pathlib import Path
import json,gc,os
import numpy as np,pandas as pd,pyarrow.parquet as pq,psutil
R=Path(__file__).resolve().parent; D=Path(os.environ.get('OIBC_DATA_DIR',str(R/'data'))); O=R/'runs/audit'
O.mkdir(parents=True,exist_ok=True)
def log(s): print(s,flush=True)
def save(x,n): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
sites=pd.read_csv(D/'sites.csv').sort_values('site_id').reset_index(drop=True)
smp=pd.read_csv(D/'train/smp.csv'); idx=pd.DatetimeIndex(pd.to_datetime(smp.ts,utc=True)).tz_convert('Asia/Seoul')
a=np.full((len(idx),len(sites)),np.nan,dtype='float32'); dup=0; n=0; neg=0; invalid=0
for batch in pq.ParquetFile(D/'train/generation.parquet').iter_batches(batch_size=65536):
 b=batch.to_pandas(); ts=pd.DatetimeIndex(pd.to_datetime(b.ts,utc=True)).tz_convert('Asia/Seoul'); ti=idx.get_indexer(ts); si=pd.Categorical(b.site_id,categories=sites.site_id).codes
 assert (ti>=0).all() and (si>=0).all()
 dup+=int(np.isfinite(a[ti,si]).sum())+int(b.duplicated(['site_id','ts']).sum())
 v=b.energy.to_numpy(dtype='float32');invalid+=int((~np.isfinite(v)).sum());neg+=int((v<0).sum());a[ti,si]=v;n+=len(b)
assert dup==0 and invalid==0
np.save(O/'generation.npy',a);pd.DataFrame({'ts':idx}).to_csv(O/'time_index.csv',index=False)
rows=[]
for j,s in enumerate(sites.site_id):
 mask=np.isfinite(a[:,j]);pos=np.flatnonzero(mask)
 rows.append({'site_id':s,'first':str(idx[pos[0]]) if len(pos) else None,'last':str(idx[pos[-1]]) if len(pos) else None,'rows':int(mask.sum()),'zeros':int((a[:,j]==0).sum())})
pd.DataFrame(rows).to_csv(O/'site_coverage.csv',index=False)
counts=np.isfinite(a).sum(axis=1); coverage=pd.DataFrame({'month':(idx-pd.Timedelta(minutes=1)).strftime('%Y-%m'),'n':counts}).groupby('month').agg(hours=('n','size'),fully_observed_hours=('n',lambda x:int((x==253).sum())),mean_sites=('n','mean'),min_sites=('n','min'))
coverage.to_csv(O/'monthly_coverage.csv')
report={'memory_gb':round(psutil.virtual_memory().total/2**30,2),'available_gb':round(psutil.virtual_memory().available/2**30,2),'generation_rows':n,'duplicates':dup,'nonfinite_energy':invalid,'negative_energy':neg,'capacity_kw':float(sites.capacity_kw.sum()),'files':{}}
for period in ['train','test']:
 timeidx=idx if period=='train' else pd.DatetimeIndex(pd.to_datetime(pd.read_csv(D/'smp.csv').ts,utc=True)).tz_convert('Asia/Seoul')
 for prov in ['wa','wb']:
  path=D/period/f'forecast_{prov}.parquet';p=pq.ParquetFile(path); station=prov+'_station'; cats=sorted({v for sb in p.iter_batches(batch_size=65536,columns=[station]) for v in sb.column(0).unique().to_pylist()});seen=np.zeros((len(cats),2,len(timeidx)),bool);nulls={};duplicates=0;nr=0;offgrid=0;offgrid_years={}
  for batch in p.iter_batches(batch_size=65536):
   b=batch.to_pandas();sc=pd.Categorical(b[station],categories=cats).codes;rc=b['round'].map({10:0,17:1}).to_numpy();tc=timeidx.get_indexer(pd.DatetimeIndex(pd.to_datetime(b.ts,utc=True)).tz_convert('Asia/Seoul'))
   assert (sc>=0).all() and np.isfinite(rc).all()
   valid=tc>=0;offgrid+=int((~valid).sum())
   for yy,nn in b.loc[~valid,'ts'].dt.year.value_counts().items():offgrid_years[str(yy)]=offgrid_years.get(str(yy),0)+int(nn)
   duplicates+=int(seen[sc[valid],rc[valid],tc[valid]].sum())+int(b.duplicated([station,'round','ts']).sum());seen[sc[valid],rc[valid],tc[valid]]=True;nr+=len(b)
   for c in b.columns:nulls[c]=nulls.get(c,0)+int(b[c].isna().sum())
  assert duplicates==0
  report['files'][f'{period}_{prov}']={'rows':nr,'off_grid_rows':offgrid,'off_grid_by_year':offgrid_years,'duplicates':duplicates,'columns':p.schema_arrow.names,'null_counts':{k:v for k,v in nulls.items() if v},'missing_keys':int((~seen).sum()),'timestamps_with_missing':int((~seen).any(axis=(0,1)).sum()),'both_rounds_missing':int((~seen).all(axis=1).sum())}
  log(f'audited {period}_{prov}: {nr:,} rows')
  del seen,b;gc.collect()
save(report,'audit.json')
log(json.dumps({'audit':report,'monthly_coverage':coverage.loc['2023-06':].reset_index().to_dict('records')},ensure_ascii=False,default=str))


