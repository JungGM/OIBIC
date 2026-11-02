from pathlib import Path
import sys, json, hashlib, gc
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT
sys.path.insert(0,str(SOURCE))
from run_experiment import DATA, inputs, enhanced, v1, PARAMS, XGBRegressor, pd, np, log

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')
assert not (ROOT/'forecast.csv').exists(), 'Refuse overwrite'
paths=[SOURCE/'run_experiment.py',SOURCE/'baseline_reference.py',SOURCE/'feature_sets.json',DATA/'README.md',DATA/'sites.csv',DATA/'train/generation.parquet',DATA/'train/smp.csv',DATA/'smp.csv']+[DATA/f'{part}/forecast_{provider}.parquet' for part in ['train','test'] for provider in ['wa','wb']]
hashes={str(p):digest(p) for p in paths}
s=inputs(); cutoff=pd.Timestamp('2025-05-31 17:00',tz='Asia/Seoul')
g=pd.read_parquet(DATA/'train/generation.parquet',filters=[('ts','<=',cutoff)])
g=g[g.ts.ge(pd.Timestamp('2021-01-01',tz='Asia/Seoul'))].reset_index(drop=True)
assert not g.duplicated(['site_id','ts']).any()
fa=pd.read_parquet(DATA/'train/forecast_wa.parquet'); fb=pd.read_parquet(DATA/'train/forecast_wb.parquet')
for f,key in [(fa,'wa_station'),(fb,'wb_station')]:
    assert not f.duplicated([key,'round','ts']).any()
d=v1.b.features(g,s,fa,fb,None,17); feats=enhanced(d)
assert feats==json.loads((SOURCE/'feature_sets.json').read_text())['enhanced'] and len(feats)==52
assert d[['site_id','ts']].equals(g[['site_id','ts']])
prices=pd.read_csv(DATA/'train/smp.csv'); prices.ts=pd.to_datetime(prices.ts)
meta=g.merge(s[['site_id','capacity_kw','group']],on='site_id',validate='many_to_one').merge(prices,on='ts',validate='many_to_one')
assert meta[['site_id','ts']].equals(g[['site_id','ts']]) and meta.smp.notna().all()
meta['k']=np.where(meta.ts.dt.hour.between(11,15),3,1)
X=d[feats].astype('float32')
mask=d.elev.gt(0)&meta.energy.notna()&meta.energy.ge(0)&meta.ts.le(cutoff)
y=(meta.energy/meta.capacity_kw).clip(0,1.2)
w=(meta.smp*meta.k*meta.capacity_kw).loc[mask]; w=w/w.mean()
params=dict(n_estimators=300,objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=42,**{**PARAMS,'max_depth':8,'min_child_weight':100,'reg_lambda':30})
training=dict(raw_observed_rows=len(g),fit_rows=int(mask.sum()),first_label=meta.loc[mask,'ts'].min(),last_label=meta.loc[mask,'ts'].max(),cutoff=cutoff,features=feats,parameters=params,round=17,group_scales={'G1':1,'G2':1},ensemble=False,target='clip(energy/capacity_kw,0,1.2), float64',weight='SMP*k*capacity_kw normalized to mean 1',feature_dtype='float32',night='elev<=0 => 0',observed_weather_used=False)
save('training_manifest.json',training)
log(f'Training final standalone XGBoost: {mask.sum()} rows, 52 features')
m=XGBRegressor(**params); m.fit(X.loc[mask,feats],y.loc[mask],sample_weight=w)
assert m.get_booster().num_boosted_rounds()==300
m.save_model(ROOT/'improved_xgb_final.ubj')
del X,d,g,fa,fb,meta,y,w; gc.collect()
log('Model saved; building submission predictions')
target=pd.read_csv(DATA/'smp.csv'); target.ts=pd.to_datetime(target.ts)
ix=pd.date_range('2025-06-01 01:00','2026-06-01 00:00',freq='h',tz='Asia/Seoul')
assert len(target)==8760 and (pd.DatetimeIndex(target.ts)==ix).all()
fa=pd.read_parquet(DATA/'test/forecast_wa.parquet'); fb=pd.read_parquet(DATA/'test/forecast_wb.parquet')
weather_audit={}
for f,key in [(fa,'wa_station'),(fb,'wb_station')]:
    assert set(f['round'])=={10,17} and not f.duplicated([key,'round','ts']).any()
    assert f.ts.isin(ix).all()
    weather_audit[key]={str(r):int(f['round'].eq(r).sum()) for r in [10,17]}
d=v1.b.features(None,s,fa,fb,ix,17); assert enhanced(d)==feats
X=d[feats].astype('float32'); p=np.maximum(m.predict(X),0)*d.capacity_kw.to_numpy(); p[d.elev.to_numpy()<=0]=0
assert len(p)==253*8760 and np.isfinite(p).all() and (p>=0).all()
# Verify persisted model is the sole source of every submitted prediction.
loaded=XGBRegressor(n_jobs=4); loaded.load_model(ROOT/'improved_xgb_final.ubj')
q=np.maximum(loaded.predict(X),0)*d.capacity_kw.to_numpy(); q[d.elev.to_numpy()<=0]=0
assert np.array_equal(p,q)
z=d[['site_id','group','ts']].copy(); z['forecast_kwh']=p
f=z.groupby(['group','ts'],sort=True).forecast_kwh.sum().reset_index()
c=s[['site_id','group']].copy()
c.to_csv(ROOT/'composition.csv',index=False)
f.to_csv(ROOT/'forecast.csv',index=False)
# Validate files reloaded from disk, not only in-memory frames.
c2=pd.read_csv(ROOT/'composition.csv'); f2=pd.read_csv(ROOT/'forecast.csv')
assert list(c2)==['site_id','group'] and list(f2)==['group','ts','forecast_kwh']
assert len(c2)==253 and c2.site_id.nunique()==253 and set(c2.site_id)==set(s.site_id)
assert c2.equals(c) and set(c2.group)=={'G1','G2'}
caps=c2.merge(s[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum()
assert caps.min()>=20000
assert len(f2)==17520 and not f2.duplicated(['group','ts']).any() and not f2.isna().any().any()
assert f2.ts.str.endswith('+09:00').all()
f2.ts=pd.to_datetime(f2.ts)
for group in ['G1','G2']: assert (pd.DatetimeIndex(f2.loc[f2.group.eq(group),'ts'])==ix).all()
assert np.isfinite(f2.forecast_kwh).all() and f2.forecast_kwh.ge(0).all()
assert np.allclose(f2.forecast_kwh,f.forecast_kwh,rtol=1e-14,atol=1e-10)
assert all(digest(Path(p))==h for p,h in hashes.items())
report=dict(passed=True,composition_rows=len(c2),forecast_rows=len(f2),hours_per_group=8760,start=f2.ts.min(),end=f2.ts.max(),capacity_kw=caps.to_dict(),sites_per_group=c2.groupby('group').size().to_dict(),duplicates=0,missing=0,negative=0,timezone='+09:00',source_files_unchanged=True,saved_model_prediction_max_difference=float(abs(p-q).max()),feature_missing_cells=int(X.isna().sum().sum()),weather_round_rows=weather_audit,model='newly fitted standalone XGBoost, no V1 predictions, no LightGBM, no group calibration',uploaded=False,submitted=False,training=training,sha256={name:digest(ROOT/name) for name in ['composition.csv','forecast.csv','improved_xgb_final.ubj']})
save('source_hashes.json',hashes); save('validation_report.json',report)
log(json.dumps(report,ensure_ascii=False,default=str))
