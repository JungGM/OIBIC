"""Reproducible, staged V1 extension. Original inputs are read-only."""
from pathlib import Path
import sys, os, json, hashlib, time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import baseline_reference as v1
DATA=Path(os.environ['OIBC_DATA']).resolve()
OUT=Path(__file__).resolve().parent
WORK=OUT/'cache'

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def log(x): print(time.strftime('%H:%M:%S'),x,flush=True)
def year(ts): return (ts-pd.Timedelta(minutes=30)).dt.year
def inputs():
    s=pd.read_csv(DATA/'sites.csv'); s['group']=s.current_group.map({'A':'G1','B':'G2'})
    return s
def audit():
    s=inputs(); g=pd.read_parquet(DATA/'train/generation.parquet',filters=[('ts','<=',pd.Timestamp('2024-01-01',tz='Asia/Seoul'))]); g=g[year(g.ts).le(2023)].copy()
    g=g.merge(s[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
    rows=[]
    for y,d in g.groupby(year(g.ts)):
        rows.append(dict(year=int(y),rows=len(d),sites=d.site_id.nunique(),expected_site_hours=253*8760,coverage_pct=100*len(d)/(253*8760),duplicate_keys=int(d.duplicated(['site_id','ts']).sum()),null_energy=int(d.energy.isna().sum()),negative=int((d.energy<0).sum()),above_120pct_capacity=int((d.energy>1.2*d.capacity_kw).sum()),zero_pct=float(d.energy.eq(0).mean()*100),complete_fleet_hours=int(d.groupby('ts').site_id.nunique().eq(253).sum())))
    pd.DataFrame(rows).to_csv(OUT/'generation_quality.csv',index=False)
    g['year']=year(g.ts); g['month']=(g.ts-pd.Timedelta(minutes=30)).dt.month
    g.groupby(['year','month']).agg(rows=('energy','size'),sites=('site_id','nunique'),mean_energy=('energy','mean')).to_csv(OUT/'generation_monthly_coverage.csv')
    g.groupby(['year','site_id']).agg(rows=('energy','size'),first=('ts','min'),last=('ts','max'),zero_fraction=('energy',lambda x:x.eq(0).mean())).to_csv(OUT/'generation_site_coverage.csv')
    weather=[]
    for provider in ['wa','wb']:
        d=pd.read_parquet(DATA/f'train/forecast_{provider}.parquet'); d=d[year(d.ts).le(2023)]
        for y,z in d.groupby(year(d.ts)):
            for c in z.select_dtypes(include='number'):
                weather.append(dict(provider=provider,year=int(y),feature=c,rows=len(z),missing=int(z[c].isna().sum()),min=z[c].min(),max=z[c].max(),duplicate_keys=int(z.duplicated([provider+'_station','round','ts']).sum())))
    pd.DataFrame(weather).to_csv(OUT/'weather_quality.csv',index=False)
    paths=[DATA/'sites.csv',DATA/'train/generation.parquet',DATA/'train/smp.csv',DATA/'train/forecast_wa.parquet',DATA/'train/forecast_wb.parquet',OLD/'outputs/xgboost_comparison.py',OLD/'outputs/results/xgb_frozen_selection.json']+[OLD/f'outputs/results/xgb_improved_round{r}.json' for r in [10,17]]
    save('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    log(rows)
def enhanced(d):
    f=v1.extra(d)
    z=d.cosz.clip(lower=0)
    additions={'cloud_mean':(d.wa_cloud+d.wb_cloud)/200,'cloud_disagreement':abs(d.wa_cloud-d.wb_cloud)/100,'humidity_disagreement':d.wa_humidity-d.wb_rel_hum,'sun_cloud_sq':z*(1-(d.wa_cloud+d.wb_cloud)/200)**2,'temp_solar':z*d.wb_temp,'uv_cloud':d.wb_uv_idx*(1-d.wb_cloud/100),'sun_squared':z*z,'wa_distance':d.wa_dist,'wb_distance':d.wb_dist,'weather_missing':d[v1.b.WA_COLS+v1.b.WB_COLS].isna().sum(axis=1)}
    for c in ['wa_wind_dir','wb_wind_dir']:
        additions[c+'_sin']=np.sin(np.deg2rad(d[c])); additions[c+'_cos']=np.cos(np.deg2rad(d[c]))
    for c,x in additions.items(): d[c]=x
    extras=['wa_snow','wb_snow','wb_ceiling','wb_wet_bulb_temp','wb_feel_temp','wb_wind_gust_spd']
    return f+list(additions)+extras
def prepare(split):
    s=inputs()
    # Explicit Parquet predicate prevents 2024 truth from entering selection stage.
    if split=='history':
        g=pd.read_parquet(DATA/'train/generation.parquet',filters=[('ts','<=',pd.Timestamp('2024-01-01',tz='Asia/Seoul'))])
    else:
        start,end=('2024-01-01 01:00','2025-01-01') if split=='2024' else ('2025-06-01 01:00','2026-06-01')
        ix=pd.date_range(start,end,freq='h',tz='Asia/Seoul')
        g=pd.MultiIndex.from_product([s.site_id,ix],names=['site_id','ts']).to_frame(index=False); g['energy']=np.nan
    assert not g.duplicated(['site_id','ts']).any()
    fa=pd.read_parquet(DATA/('test' if split=='submission' else 'train')/'forecast_wa.parquet')
    fb=pd.read_parquet(DATA/('test' if split=='submission' else 'train')/'forecast_wb.parquet')
    fa=fa[fa.ts.isin(g.ts.unique())]; fb=fb[fb.ts.isin(g.ts.unique())]
    meta=g.merge(s[['site_id','capacity_kw','group']],on='site_id',validate='many_to_one')
    prices=pd.read_csv(DATA/('smp.csv' if split=='submission' else 'train/smp.csv')); prices.ts=pd.to_datetime(prices.ts)
    meta=meta.merge(prices[['ts','smp']],on='ts',validate='many_to_one'); meta['k']=np.where(meta.ts.dt.hour.between(11,15),3,1)
    assert meta[['site_id','ts']].equals(g[['site_id','ts']]) and meta.smp.notna().all()
    meta.to_parquet(WORK/f'{split}_meta.parquet',index=False)
    for r in [10,17]:
        d=v1.b.features(g,s,fa,fb,None,r); feats=enhanced(d)
        assert d[['site_id','ts']].equals(g[['site_id','ts']])
        d[feats].astype('float32').to_parquet(WORK/f'{split}_{r}.parquet',index=False)
        save('feature_sets.json',dict(v1=v1.extra(d),enhanced=feats))
        log(f'prepared {split} round {r}: {len(d)} rows, {len(feats)} features')
def aggregate(meta,p):
    z=meta[['ts','group','site_id','energy','smp','k']].copy(); z['pred']=p
    z=z[z.energy.notna()]
    a=z.groupby(['ts','group'],sort=True).agg(actual=('energy','sum'),pred=('pred','sum'),n=('site_id','nunique'),smp=('smp','first'),k=('k','first')).reset_index()
    a['penalty']=.1*a.smp*a.k*abs(a.pred-a.actual)
    return a
def score(meta,p): return float(aggregate(meta,p).penalty.sum())
PARAMS=dict(max_depth=6,learning_rate=.03,subsample=.8,colsample_bytree=.8,min_child_weight=50,reg_lambda=10)
def configs():
    return [dict(name='v1_2022_control',start=2022,features='v1',weighted=False,params=PARAMS,trees=[307,302]),dict(name='v1_2021_extension',start=2021,features='v1',weighted=False,params=PARAMS,trees=[307,302]),dict(name='weighted_v1',start=2021,features='v1',weighted=True,params=PARAMS,trees=[300,600]),dict(name='weather_depth4',start=2021,features='enhanced',weighted=True,params={**PARAMS,'max_depth':4,'learning_rate':.04,'min_child_weight':30,'reg_lambda':20},trees=[300,600]),dict(name='weather_depth6',start=2021,features='enhanced',weighted=True,params=PARAMS,trees=[300,600]),dict(name='weather_depth8',start=2021,features='enhanced',weighted=True,params={**PARAMS,'max_depth':8,'min_child_weight':100,'reg_lambda':30},trees=[300,600])]
def predict(m,X,meta):
    p=np.maximum(m.predict(X),0)*meta.capacity_kw.to_numpy(); p[X.elev.to_numpy()<=0]=0
    return p
def select():
    meta=pd.read_parquet(WORK/'history_meta.parquet'); yy=year(meta.ts); valid=yy.eq(2023); vm=meta[valid].reset_index(drop=True)
    fs=json.loads((OUT/'feature_sets.json').read_text()); results=[]; predictions={}; candidates=configs()
    save('predeclared_search.json',dict(candidates=candidates,selection='2023 partial-observed SMP*k group absolute error; fixed original groups; missing truth never imputed',bias_scales=[.96,.98,1.,1.02,1.04],round_blends=[0.,.5,1.],holdout='2024 truth not loaded before freeze',training_cutoff='Dec 31 round hour; target end labels after cutoff excluded'))
    for c in candidates:
        rp={}; tree_options=[None] if c['name'].startswith('v1_') else c['trees']
        for ri,r in enumerate([10,17]):
            X=pd.read_parquet(WORK/f'history_{r}.parquet'); features=fs[c['features']]
            mask=yy.between(c['start'],2022)&X.elev.gt(0)&meta.energy.notna()&meta.ts.le(pd.Timestamp(f'2022-12-31 {r}:00',tz='Asia/Seoul'))
            if c['weighted']: mask &= meta.energy.ge(0)
            y=(meta.energy/meta.capacity_kw).clip(0,1.2)
            w=(meta.smp*meta.k*meta.capacity_kw).loc[mask] if c['weighted'] else None
            if w is not None: w=w/w.mean()
            for trees in tree_options:
                nt=c['trees'][ri] if trees is None else trees
                m=XGBRegressor(n_estimators=nt,objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=42,**c['params'])
                m.fit(X.loc[mask,features],y.loc[mask],sample_weight=w)
                rp[(r,trees)]=predict(m,X.loc[valid,features],vm)
                log(f"validation {c['name']} r{r} trees{nt} training rows{mask.sum()}")
            del X
        for trees in tree_options:
            # Baseline controls keep original equal blend and no calibration.
            blends=[.5] if c['name'].startswith('v1_') else [0.,.5,1.]
            scales=[1.] if c['name'].startswith('v1_') else [.96,.98,1.,1.02,1.04]
            for blend in blends:
                raw=(1-blend)*rp[(10,trees)]+blend*rp[(17,trees)]
                for scale in scales:
                    p=raw*scale; cost=score(vm,p)
                    key=f"{c['name']}_t{trees}_b{blend}_s{scale}"
                    results.append(dict(key=key,name=c['name'],trees=trees,blend17=blend,scale=scale,penalty=cost))
                    predictions[key]=p
        pd.DataFrame(results).to_csv(OUT/'validation_search.csv',index=False)
    best=min((z for z in results if z['name']!='v1_2022_control'),key=lambda z:z['penalty'])
    ref=next(z for z in results if z['name']=='v1_2022_control')
    cfg=next(c for c in candidates if c['name']==best['name'])
    frozen=dict(winner=best,config=cfg,reference=ref,validation_improved=best['penalty']<ref['penalty'],selection_year=2023,final_training_years=[2021,2022,2023],frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S'),test_truth_used=False)
    save('frozen_selection.json',frozen)
    for name,z in [('v1_control',ref),('selected',best),('v1_extension',next(z for z in results if z['name']=='v1_2021_extension'))]:
        np.save(WORK/f'validation_{name}.npy',predictions[z['key']]); aggregate(vm,predictions[z['key']]).to_csv(OUT/f'validation_{name}_groups.csv',index=False)
    log(frozen)
def final_fit():
    frozen=json.loads((OUT/'frozen_selection.json').read_text()); cfg=frozen['config']; best=frozen['winner']
    meta=pd.read_parquet(WORK/'history_meta.parquet'); yy=year(meta.ts); fs=json.loads((OUT/'feature_sets.json').read_text())
    specs={'v1_2021_2023':dict(features='v1',weighted=False,params=PARAMS,start=2021,trees=[307,302],blend17=.5,scale=1.)}
    for start in [2022,2021]:
        trees=cfg['trees'] if best['trees'] is None else [best['trees']]*2
        specs[f'selected_{start}_2023']=dict(features=cfg['features'],weighted=cfg['weighted'],params=cfg['params'],start=start,trees=trees,blend17=best['blend17'],scale=best['scale'])
    (OUT/'models').mkdir(exist_ok=True); fitted=[]
    for ri,r in enumerate([10,17]):
        X=pd.read_parquet(WORK/f'history_{r}.parquet')
        for name,c in specs.items():
            mask=yy.between(c['start'],2023)&X.elev.gt(0)&meta.energy.notna()&meta.ts.le(pd.Timestamp(f'2023-12-31 {r}:00',tz='Asia/Seoul'))
            if c['weighted']: mask &= meta.energy.ge(0)
            w=(meta.smp*meta.k*meta.capacity_kw).loc[mask] if c['weighted'] else None
            if w is not None: w=w/w.mean()
            m=XGBRegressor(n_estimators=c['trees'][ri],objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=42,**c['params'])
            m.fit(X.loc[mask,fs[c['features']]],(meta.energy/meta.capacity_kw).clip(0,1.2).loc[mask],sample_weight=w)
            m.save_model(OUT/f'models/{name}_r{r}.ubj')
            fitted.append(dict(model=name,round=r,rows=int(mask.sum()),last_label=meta.loc[mask,'ts'].max()))
            log(f'final fit {name} round {r}: {mask.sum()} rows')
        del X
    save('final_model_specs.json',specs); save('training_rows.json',fitted)
def feature_ablation():
    """Separate the effect of added weather features from sample weighting, on 2023 only."""
    frozen=json.loads((OUT/'frozen_selection.json').read_text()); best=frozen['winner']
    c=dict(name='weather_unweighted',start=2021,features='enhanced',weighted=False,params=PARAMS,trees=[307,302])
    save('feature_ablation_plan.json',dict(config=c,declared_at=time.strftime('%Y-%m-%dT%H:%M:%S'),reason='Isolate weather feature contribution without changing sample weights or V1 tree counts',test_truth_used=False))
    meta=pd.read_parquet(WORK/'history_meta.parquet'); yy=year(meta.ts); valid=yy.eq(2023); vm=meta[valid].reset_index(drop=True)
    fs=json.loads((OUT/'feature_sets.json').read_text())['enhanced']; p=np.zeros(len(vm))
    for ri,r in enumerate([10,17]):
        X=pd.read_parquet(WORK/f'history_{r}.parquet'); mask=yy.between(2021,2022)&X.elev.gt(0)&meta.ts.le(pd.Timestamp(f'2022-12-31 {r}:00',tz='Asia/Seoul'))
        m=XGBRegressor(n_estimators=c['trees'][ri],objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=42,**PARAMS)
        m.fit(X.loc[mask,fs],(meta.energy/meta.capacity_kw).clip(0,1.2).loc[mask]); p+=predict(m,X.loc[valid,fs],vm)/2
        log(f'unweighted feature ablation round{r}')
    z=dict(key='weather_unweighted_tNone_b0.5_s1.0',name=c['name'],trees=None,blend17=.5,scale=1.,penalty=score(vm,p))
    df=pd.read_csv(OUT/'validation_search.csv'); pd.concat([df,pd.DataFrame([z])],ignore_index=True).to_csv(OUT/'validation_search.csv',index=False)
    if z['penalty']<best['penalty']:
        frozen.update(winner=z,config=c,validation_improved=z['penalty']<frozen['reference']['penalty'])
        np.save(WORK/'validation_selected.npy',p); aggregate(vm,p).to_csv(OUT/'validation_selected_groups.csv',index=False)
    frozen.update(frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S'),feature_ablation=z)
    save('frozen_selection.json',frozen)
    save('selection_lock_hash.json',dict(Hash=hashlib.sha256((OUT/'frozen_selection.json').read_bytes()).hexdigest().upper(),Algorithm='SHA256'))
    log(frozen)
def infer(split):
    frozen=json.loads((OUT/'frozen_selection.json').read_text()); fs=json.loads((OUT/'feature_sets.json').read_text()); specs=json.loads((OUT/'final_model_specs.json').read_text())
    if split=='submission':
        if not frozen['validation_improved']: raise ValueError('No historically improved candidate; submission inference blocked')
        specs={k:v for k,v in specs.items() if k=='selected_2021_2023'}
    else: specs={'original_v1':dict(features='v1',blend17=.5,scale=1.),**specs}
    meta=pd.read_parquet(WORK/f'{split}_meta.parquet'); predictions={k:np.zeros(len(meta)) for k in specs}
    for r in [10,17]:
        X=pd.read_parquet(WORK/f'{split}_{r}.parquet')
        for name,c in specs.items():
            m=XGBRegressor(n_jobs=4)
            path=OLD/f'outputs/results/xgb_improved_round{r}.json' if name=='original_v1' else OUT/f'models/{name}_r{r}.ubj'
            m.load_model(path)
            predictions[name]+=predict(m,X[fs[c['features']]],meta)*(c['blend17'] if r==17 else 1-c['blend17'])*c['scale']
            log(f'inference {split} {name} round{r}')
        del X
    for name,p in predictions.items():
        np.save(WORK/f'{split}_{name}.npy',p)
        z=meta[['site_id','ts','group']].copy(); z['forecast_kwh']=p
        z.groupby(['group','ts']).forecast_kwh.sum().reset_index().to_csv(OUT/f'{split}_{name}_forecast.csv',index=False)
        if split=='submission':
            assert len(z)==253*8760 and np.isfinite(p).all() and (p>=0).all()
            inputs()[['site_id','group']].to_csv(OUT/'submission_composition.csv',index=False)
            save('submission_candidate_status.json',dict(eligible_on_2023=True,uploaded=False,submitted=False,model=name,validation_reduction_pct=100*(1-frozen['winner']['penalty']/frozen['reference']['penalty']),warning='Candidate only. No claim of improved official settlement.'))
def metrics(a,p):
    e=np.asarray(p)-np.asarray(a); a=np.asarray(a)
    return dict(n=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),wape_pct=float(100*np.abs(e).sum()/a.sum()) if a.sum()!=0 else None,bias=float(e.mean()))
def evaluate():
    assert (OUT/'frozen_selection.json').exists() and (OUT/'final_model_specs.json').exists()
    # First use of 2024 actual generation, AFTER model freeze, final fit and predictions.
    g=pd.read_parquet(DATA/'train/generation.parquet',filters=[('ts','>',pd.Timestamp('2024-01-01',tz='Asia/Seoul')),('ts','<=',pd.Timestamp('2025-01-01',tz='Asia/Seoul'))])
    meta=pd.read_parquet(WORK/'2024_meta.parquet').drop(columns='energy').merge(g,on=['site_id','ts'],how='left',validate='one_to_one')
    observed=meta.energy.notna(); counts=g.groupby('ts').site_id.nunique(); full=counts[counts.eq(253)].index
    save('evaluation_coverage.json',dict(complete_hours=len(full),first_complete=str(full.min()),last_complete=str(full.max()),observed_site_hours=int(observed.sum()),requested_hours=8784))
    site_rows=[]; group_rows=[]; seasonal=[]; per_site=[]; daily=[]
    season_names={12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
    ss=(meta.ts-pd.Timedelta(minutes=30)).dt.month.map(season_names)
    names=['original_v1']+list(json.loads((OUT/'final_model_specs.json').read_text()))
    for name in names:
        p=np.load(WORK/f'2024_{name}.npy'); assert len(p)==len(meta) and np.isfinite(p).all() and (p>=0).all()
        a=aggregate(meta,p); a.to_csv(OUT/f'2024_{name}_observed_groups.csv',index=False)
        site_rows.append(dict(model=name,**metrics(meta.loc[observed,'energy'],p[observed])))
        for scope in ['complete_fleet_178h','observed_sites_proxy']:
            z=a[a.ts.isin(full)] if scope=='complete_fleet_178h' else a
            for group in ['G1','G2','TOTAL']:
                zz=z if group=='TOTAL' else z[z.group.eq(group)]
                group_rows.append(dict(model=name,scope=scope,group=group,penalty_krw=zz.penalty.sum(),**metrics(zz.actual,zz.pred)))
        for season in ['winter','spring','summer','autumn']:
            m=observed&ss.eq(season)
            az=a[(a.ts-pd.Timedelta(minutes=30)).dt.month.map(season_names).eq(season)]
            seasonal.append(dict(model=name,season=season,**metrics(meta.loc[m,'energy'],p[m]),observed_group_penalty_krw=az.penalty.sum(),complete_fleet_hours=int(az[az.ts.isin(full)].ts.nunique())))
        z=meta.loc[observed,['site_id','energy']].copy(); z['pred']=p[observed]
        for sid,d in z.groupby('site_id'):
            per_site.append(dict(model=name,site_id=sid,**metrics(d.energy,d.pred)))
        d=a.copy(); d['date']=(d.ts-pd.Timedelta(minutes=30)).dt.strftime('%Y-%m-%d'); d=d.groupby('date').penalty.sum().rename(name); daily.append(d)
    for fname,rows in [('site_metrics.csv',site_rows),('penalty_comparison.csv',group_rows),('seasonal_metrics.csv',seasonal),('per_site_metrics.csv',per_site)]:
        df=pd.DataFrame(rows)
        if fname=='penalty_comparison.csv':
            ref=df[df.model.eq('original_v1')].set_index(['scope','group']).penalty_krw
            df['reduction_pct_vs_v1']=[100*(1-v/ref.loc[(sc,gr)]) for v,sc,gr in zip(df.penalty_krw,df.scope,df.group)]
        df.to_csv(OUT/fname,index=False)
    pd.concat(daily,axis=1).to_csv(OUT/'daily_proxy_penalties.csv')
    # Check saved V1 predictions rather than silently replacing original baseline.
    original=pd.read_parquet(OLD/'outputs/results/xgb_improved_2024_sites.parquet')
    z=meta[['site_id','ts']].copy(); z['new']=np.load(WORK/'2024_original_v1.npy'); z=z.merge(original[['site_id','ts','prediction_kwh']],on=['site_id','ts'],validate='one_to_one')
    save('baseline_reproduction.json',dict(rows=len(z),max_abs_difference=float(abs(z.new-z.prediction_kwh).max())))
    hashes=json.loads((OUT/'source_hashes.json').read_text()); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    save('verification.json',dict(original_source_hashes_unchanged=True,selection_frozen_before_2024_truth=True,models_retrained_after_evaluation=False,all_predictions_finite=True))
    log(pd.DataFrame(group_rows).query("scope == 'complete_fleet_178h'").to_string(index=False))
if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='audit': audit()
    elif stage=='prepare': prepare(sys.argv[2])
    elif stage=='select': select()
    elif stage=='fit': final_fit()
    elif stage=='feature_ablation': feature_ablation()
    elif stage=='infer': infer(sys.argv[2])
    elif stage=='evaluate': evaluate()
