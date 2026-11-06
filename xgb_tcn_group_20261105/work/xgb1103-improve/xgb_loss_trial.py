"""Frozen 177-feature objective ablation; TCN and 50:50 coefficient fixed."""
import gc,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import xgboost as xgb
from experiment import ROOT,SETS,PARAMS,Progress,load,sha,save,log

RUN=ROOT/'xgb_loss_trial1';OUT=Path('outputs/OIBC_XGBoost_손실함수비교_20261104')
FEATURES=SETS['expanded177']
FOLDS={
 'spring':{'cutoff':'2024-12-31 17:00','year':2025,'dir':ROOT/'runs/expanded177','pred':'predictions.parquet','sites':253},
 'summer':{'cutoff':'2024-05-31 17:00','year':2024,'dir':ROOT/'summer_backtest','pred':'site_predictions.parquet','sites':207}}
OBJECTIVES={'L1':{'objective':'reg:absoluteerror'},'PseudoHuber':{'objective':'reg:pseudohubererror','huber_slope':.1}}

def protected():
    paths=[ROOT/'production177/model.ubj',Path('outputs/OIBC_177_제출용_20261103_v2/forecast.csv'),Path('outputs/OIBC_177_제출용_20261103_v2/composition.csv')]
    for fold,cfg in FOLDS.items():paths += [cfg['dir']/'model.ubj',cfg['dir']/cfg['pred'],ROOT/'tcn_candidate1'/fold/'model.pt',ROOT/'tcn_candidate1'/fold/'predictions.parquet']
    return {str(p):sha(p) for p in paths}

def fit(fold,name):
    cfg=FOLDS[fold];dest=RUN/fold/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'predictions.parquet').exists():return
    if not (dest/'model.ubj').exists():
        cut=pd.Timestamp(cfg['cutoff'],tz='Asia/Seoul');filters=[('ts','<=',cut),('elev','>',0),('energy','>=',0)]
        meta=load(SimpleNamespace(cache=ROOT/'cache'),range(2021,2025),['target_cf','sample_weight','_row_id','ts'],filters)
        assert meta._row_id.is_unique and meta.ts.max()<=cut
        y=meta.target_cf.to_numpy(copy=True);w=meta.sample_weight.to_numpy(copy=True);w/=w.mean()
        X=np.load(cfg['dir']/'training_features.npy',mmap_mode='r')
        assert X.shape==(len(meta),177) and np.isfinite(y).all() and np.isfinite(w).all() and (w>=0).all()
        # Independently verify sampled matrix rows against the original cache.
        rng=np.random.default_rng(42);positions=np.sort(rng.choice(len(meta),min(100,len(meta)),replace=False));wanted=meta.iloc[positions]._row_id.to_numpy();seen={}
        for yr in range(2021,2025):
            for p in sorted((ROOT/'cache'/str(yr)).glob('*.parquet')):
                key=pd.read_parquet(p,columns=['_row_id']);mask=key._row_id.isin(wanted)
                if mask.any():
                    q=pd.read_parquet(p,columns=FEATURES+['_row_id']);q=q[q._row_id.isin(wanted)]
                    for _,r in q.iterrows():seen[int(r['_row_id'])]=r[FEATURES].to_numpy('float32')
        for pos,rid in zip(positions,wanted):assert np.array_equal(X[pos],seen[int(rid)],equal_nan=True)
        last=str(meta.ts.max());n=len(meta);del meta,seen;gc.collect()
        params={**PARAMS,**OBJECTIVES[name]};log(stage='loss_fit_start',fold=fold,loss=name,rows=n,parameters=params)
        start=time.time();model=xgb.XGBRegressor(**params,callbacks=[Progress()]);model.fit(X,y,sample_weight=w)
        model.get_booster().feature_names=FEATURES;model.get_booster().feature_types=['float']*177;model.save_model(dest/'model.ubj')
        save(dest/'training.json',{'rows':n,'last_label':last,'parameters':params,'features':FEATURES,'seconds':time.time()-start,'model_sha256':sha(dest/'model.ubj'),'sampled_matrix_rows_verified':len(positions),'base_score_config':json.loads(model.get_booster().save_config())['learner']['learner_model_param']})
        del model,X,y,w;gc.collect()
    model=xgb.XGBRegressor(n_jobs=4);model.load_model(dest/'model.ubj');ref=pd.read_parquet(cfg['dir']/cfg['pred']);pieces=[]
    for site,z in ref.groupby('site_id',sort=True):
        z=z.sort_values('ts').reset_index(drop=True);d=pd.read_parquet(ROOT/'cache'/str(cfg['year'])/f'{site}.parquet',columns=['ts']+FEATURES).set_index('ts').loc[pd.DatetimeIndex(z.ts)]
        values=d[FEATURES].astype('float32');p=np.maximum(model.predict(values).astype('float64'),0)*z.capacity_kw.to_numpy('float64');p[d.elev.to_numpy()<=0]=0
        assert np.isfinite(p).all() and (p>=0).all();z['prediction']=p;pieces.append(z)
    result=pd.concat(pieces,ignore_index=True);assert len(result)==len(ref);result.to_parquet(dest/'predictions.parquet',index=False)
    log(stage='loss_complete',fold=fold,loss=name)

def metric(d):
    g=d.groupby(['ts','group']).agg(pred=('prediction','sum'),actual=('energy','sum'),smp=('smp','first'),k=('k','first')).reset_index();g['month']=(g.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    def m(z):
        e=z.pred-z.actual
        return {'hours':int(z.ts.nunique()),'penalty':float((.1*z.smp*z.k*abs(e)).sum()),'wape_pct':float(100*abs(e).sum()/z.actual.sum()),'bias_pct':float(100*e.sum()/z.actual.sum())}
    ans={'whole':m(g),'monthly':{k:m(z) for k,z in g.groupby('month')}}
    if d.ts.min().year==2025:
        for label,mask in [('JanMar',g.month<='2025-03'),('AprMay',g.month>='2025-04')]:ans[label]=m(g[mask])
    return ans

def report():
    results={};rows=[];monthrows=[]
    for fold,cfg in FOLDS.items():
        ref=pd.read_parquet(cfg['dir']/cfg['pred']).sort_values(['site_id','ts']).reset_index(drop=True)
        tcn=pd.read_parquet(ROOT/'tcn_candidate1'/fold/'predictions.parquet').sort_values(['site_id','ts']).reset_index(drop=True)
        keys=['site_id','ts','energy','group','smp','k','capacity_kw'];assert ref[keys].equals(tcn[keys])
        variants={'L2':ref}
        for name in OBJECTIVES:
            p=RUN/fold/name/'predictions.parquet'
            if p.exists():variants[name]=pd.read_parquet(p).sort_values(['site_id','ts']).reset_index(drop=True)
        results[fold]={}
        for name,z in variants.items():
            assert z[keys].equals(ref[keys]) and z.groupby('ts').site_id.nunique().eq(cfg['sites']).all()
            for blend in [False,True]:
                tag=name+('_TCN5050' if blend else '_alone');d=z.copy()
                if blend:d.prediction=.5*z.prediction+.5*tcn.prediction
                v=metric(d);results[fold][tag]=v
                for period,stats in v.items():
                    if period=='monthly':
                        for month,s in stats.items():monthrows.append({'fold':fold,'model':tag,'month':month,**s})
                    else:rows.append({'fold':fold,'model':tag,'period':period,**stats})
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'results.json',results);pd.DataFrame(rows).to_csv(OUT/'period_comparison.csv',index=False);pd.DataFrame(monthrows).to_csv(OUT/'monthly_comparison.csv',index=False)
    text=['# XGBoost loss comparison, fixed TCN 50:50','','Identical177 features, label filters, original row order, target clip[0,1.2], normalized SMP*k*capacity weights and300-tree settings. L2 uses original saved baseline. L1 uses reg:absoluteerror; Huber candidate uses smooth Pseudo-Huber delta=.1 CF, not the piecewise classical Huber. Delta and all settings fixed before results. No objective-specific tuning. Same min_child_weight/reg_lambda may behave differently with objective Hessians. Objective-specific default initial intercept estimation may differ.','','Historical periods previously inspected; not pristine untouched tests. Summer207 sites/4905h partial-population proxy; spring253 sites/3624h. Different training cutoffs and cohorts: never add into official annual settlement. No forecast export or production replacement in this experiment.','','|fold|model|whole penalty|change vs L2 alone|change vs L2+TCN|','|---|---|---:|---:|---:|']
    for fold,models in results.items():
        a=models['L2_alone']['whole']['penalty'];b=models['L2_TCN5050']['whole']['penalty']
        for label,v in models.items():
            p=v['whole']['penalty'];text.append(f'|{fold}|{label}|{p:,.0f}|{100*(p/a-1):+.2f}%|{100*(p/b-1):+.2f}%|')
    (OUT/'report.md').write_text('\n'.join(text),encoding='utf-8');log(stage='loss_report',whole={f:{m:v['whole']['penalty'] for m,v in ms.items()} for f,ms in results.items()})

if __name__=='__main__':
    RUN.mkdir(exist_ok=True)
    if not (RUN/'protocol.json').exists():save(RUN/'protocol.json',{'features':FEATURES,'base_parameters':PARAMS,'objectives':OBJECTIVES,'TCN_fixed':True,'blend':[.5,.5],'objective_specific_tuning':False,'promotion_rule':'both blocks >=1% lower penalty than current L2+TCN and no month worse >2%; otherwise experimental only','protected_sha256':protected(),'source_sha256':sha(__file__)})
    protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'));assert protocol['protected_sha256']==protected()
    for fold in ['spring','summer']:
        for name in OBJECTIVES:fit(fold,name);report()
    assert protocol['protected_sha256']==protected();save(OUT/'protected_artifacts_unchanged.json',{'unchanged':True,'sha256':protected()})
