"""Refit frozen 177-feature XGBoost and produce official submission locally."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
import xgboost as xgb
from experiment import ROOT, BASE, ADD, PARAMS, FINAL_END, Progress, load, aux, sha, save, log

DATA=Path('work/data').resolve()
PROD=ROOT/'production177'
OUT=Path('outputs/OIBC_177_제출용_20261103').resolve()
SPEC=json.loads((ROOT/'expanded_features.json').read_text(encoding='utf-8'))
FEATURES=SPEC['features']

def input_hashes():
    paths=[DATA/'sites.csv',DATA/'train/generation.parquet',DATA/'train/smp.csv']
    paths += [DATA/part/f'forecast_{p}.parquet' for part in ['train','test'] for p in ['wa','wb']]
    paths += [ROOT/'experiment.py',ROOT/'expand_features.py',ROOT/'expanded_features.json',Path(__file__).resolve()]
    paths += list((ROOT/'reference').glob('*.py'))
    paths += [Path('work/OIBIC/lgbm_catboost_2024/src/features.py').resolve(),Path('work/OIBIC/lgbm_catboost_2024/configs/features_160.json').resolve()]
    return {str(p):sha(p) for p in paths}

def train():
    PROD.mkdir(exist_ok=True)
    if (PROD/'model.ubj').exists(): raise ValueError('Refuse overwrite of fitted production model')
    assert len(FEATURES)==len(set(FEATURES))==177
    before=input_hashes()
    assert before[str(Path('work/OIBIC/lgbm_catboost_2024/src/features.py').resolve())]==SPEC['prior_engine_sha256']
    review=json.loads((ROOT/'analysis/recommendation_review.json').read_text(encoding='utf-8'))
    assert review['recommendation']['features']=='expanded177' and review['recommendation']['group_scales']==[1.,1.]
    filters=[('ts','<=',FINAL_END),('elev','>',0),('energy','>=',0)]
    class Args: cache=ROOT/'cache'
    meta=load(Args(),range(2021,2026),['target_cf','sample_weight','_row_id','ts'],filters)
    assert len(meta)==2189472 and meta.ts.max()<=FINAL_END and meta._row_id.is_unique
    y=meta.target_cf.to_numpy(copy=True);w=meta.sample_weight.to_numpy(copy=True);w/=w.mean()
    assert np.isfinite(y).all() and np.isfinite(w).all() and (w>=0).all() and w.sum()>0
    first,last=str(meta.ts.min()),str(meta.ts.max())
    positions=np.full(int(meta._row_id.max())+1,-1,dtype='int32')
    positions[meta._row_id.to_numpy()]=np.arange(len(meta),dtype='int32')
    matrix=np.lib.format.open_memmap(PROD/'training_features.npy',mode='w+',dtype='float32',shape=(len(meta),len(FEATURES)))
    rows=len(meta);del meta;gc.collect();written=0
    for year in range(2021,2026):
        for p in sorted((ROOT/'cache'/str(year)).glob('*.parquet')):
            d=pd.read_parquet(p,columns=FEATURES+['_row_id'],filters=filters)
            ix=positions[d._row_id.to_numpy()];assert (ix>=0).all()
            matrix[ix]=d[FEATURES].to_numpy(dtype='float32');written+=len(d)
        matrix.flush();log(stage='production_matrix',year=year,rows=written)
    assert written==rows
    del positions,d;gc.collect()
    save(PROD/'inputs_before.json',before)
    log(stage='production_fit_start',rows=rows,features=len(FEATURES),cutoff=str(FINAL_END))
    t=time.time();model=xgb.XGBRegressor(**PARAMS,callbacks=[Progress()]);model.fit(matrix,y,sample_weight=w)
    model.get_booster().feature_names=FEATURES;model.get_booster().feature_types=['float']*len(FEATURES)
    assert model.get_booster().num_boosted_rounds()==300
    model.save_model(PROD/'model.ubj')
    assert before==input_hashes()
    save(PROD/'training_manifest.json',{'training_rows':rows,'first_label':first,'last_label':last,'cutoff':str(FINAL_END),
        'features':FEATURES,'parameters':PARAMS,'grouping':'Original A/B to G1/G2','scales':[1.,1.],'ensemble':False,
        'model_sha256':sha(PROD/'model.ubj'),'seconds':time.time()-t,'versions':{'xgboost':xgb.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'target':'clip(energy/capacity_kw,0,1.2)','sample_weight':'capacity_kw*SMP*k normalized to mean 1; historical SMP only',
        'source_order':'Original generation row order restored; negative labels and non-daylight rows excluded'})
    log(stage='production_fit_complete',seconds=time.time()-t)

def predict():
    manifest=json.loads((PROD/'training_manifest.json').read_text(encoding='utf-8'))
    before=json.loads((PROD/'inputs_before.json').read_text(encoding='utf-8'))
    assert before==input_hashes() and sha(PROD/'model.ubj')==manifest['model_sha256']
    if (OUT/'forecast.csv').exists(): raise ValueError('Refuse overwrite of submission')
    OUT.mkdir(exist_ok=True)
    os.environ['OIBC_DATA']=str(DATA)
    from run_experiment import enhanced,v1
    old=Path('work/OIBIC/lgbm_catboost_2024/src/features.py').resolve()
    spec=importlib.util.spec_from_file_location('prior160_production',old);ft=importlib.util.module_from_spec(spec);spec.loader.exec_module(ft)
    model=xgb.XGBRegressor(n_jobs=4);model.load_model(PROD/'model.ubj')
    sites=pd.read_csv(DATA/'sites.csv');sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    assert len(sites)==253 and sites.site_id.is_unique and sites.group.notna().all()
    ix=pd.date_range('2025-06-01 01:00','2026-06-01 00:00',freq='h',tz='Asia/Seoul');assert len(ix)==8760
    # Read timestamp column only: official-year actual SMP is not a feature.
    target=pd.read_csv(DATA/'smp.csv',usecols=['ts'])
    assert pd.DatetimeIndex(pd.to_datetime(target.ts)).tz_convert('Asia/Seoul').equals(ix)
    fa=pd.read_parquet(DATA/'test/forecast_wa.parquet');fb=pd.read_parquet(DATA/'test/forecast_wb.parquet')
    for f,key in [(fa,'wa_station'),(fb,'wb_station')]:
        assert not f.duplicated([key,'round','ts']).any() and set(f['round'])=={10,17} and f.ts.isin(ix).all()
    wa=ft.weather(DATA/'test/forecast_wa.parquet','a',ix,sites.wa_station.unique())
    wb=ft.weather(DATA/'test/forecast_wb.parquet','b',ix,sites.wb_station.unique())
    aa={};bb={};pieces=[];missing=0
    dest=PROD/'site_predictions';dest.mkdir(exist_ok=True)
    for j,site in enumerate(sites.itertuples(index=False)):
        s=sites.loc[sites.site_id.eq(site.site_id)]
        d=v1.b.features(None,s,fa.loc[fa.wa_station.eq(site.wa_station)],fb.loc[fb.wb_station.eq(site.wb_station)],ix,17)
        assert enhanced(d)==BASE and pd.DatetimeIndex(d.ts).equals(ix)
        if site.wa_station not in aa: aa[site.wa_station]=aux(fa,'a',site.wa_station,ix)
        if site.wb_station not in bb: bb[site.wb_station]=aux(fb,'b',site.wb_station,ix)
        more=pd.concat([aa[site.wa_station],bb[site.wb_station]],axis=1)
        x=d[BASE].copy()
        for name in ADD: x[name]=more[name].to_numpy()
        previous=ft.build_site(site,ix,wa[site.wa_station],wb[site.wb_station])
        assert pd.DatetimeIndex(previous.ts).equals(ix)
        prior=pd.DataFrame(previous[SPEC['additional_prior_features']].to_numpy(),columns=['old_'+n for n in SPEC['additional_prior_features']])
        x=pd.concat([x.reset_index(drop=True),prior],axis=1)[FEATURES].astype('float32')
        assert not np.isinf(x.to_numpy()).any();missing+=int(x.isna().sum().sum())
        p=np.maximum(model.predict(x),0)*site.capacity_kw;p[x.elev.to_numpy()<=0]=0
        assert np.isfinite(p).all() and (p>=0).all()
        z=pd.DataFrame({'site_id':site.site_id,'group':site.group,'ts':ix,'forecast_kwh':p})
        z.to_parquet(dest/f'{site.site_id}.parquet',index=False)
        pieces.append(z[['group','ts','forecast_kwh']])
        if (j+1)%20==0 or j+1==len(sites): log(stage='production_predict',sites_done=j+1,rows=(j+1)*len(ix))
    whole=pd.concat(pieces,ignore_index=True)
    assert len(whole)==253*8760
    forecast=whole.groupby(['group','ts'],sort=True,as_index=False).forecast_kwh.sum()
    composition=sites[['site_id','group']].copy()
    composition.to_csv(OUT/'composition.csv',index=False);forecast.to_csv(OUT/'forecast.csv',index=False)
    assert before==input_hashes()
    save(PROD/'prediction_manifest.json',{'model_sha256':sha(PROD/'model.ubj'),'site_rows':len(whole),'feature_missing_cells':missing,
         'actual_official_SMP_read':False,'official_forecast_annual_statistics_used':False,'same_issued_day_context_only':True,
         'composition_sha256':sha(OUT/'composition.csv'),'forecast_sha256':sha(OUT/'forecast.csv')})
    log(stage='production_predict_complete')

def validate():
    sites=pd.read_csv(DATA/'sites.csv');c=pd.read_csv(OUT/'composition.csv');f=pd.read_csv(OUT/'forecast.csv')
    assert list(c)==['site_id','group'] and list(f)==['group','ts','forecast_kwh']
    assert len(c)==253 and c.site_id.is_unique and set(c.site_id)==set(sites.site_id) and set(c.group)=={'G1','G2'}
    assert c.group.equals(c.site_id.map(sites.set_index('site_id').current_group.map({'A':'G1','B':'G2'})))
    caps=c.merge(sites[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum()
    assert (caps>=20000).all()
    assert len(f)==17520 and not f.isna().any().any() and not f.duplicated(['group','ts']).any()
    assert np.isfinite(f.forecast_kwh).all() and f.forecast_kwh.ge(0).all() and f.ts.str.endswith('+09:00').all()
    f.ts=pd.to_datetime(f.ts)
    ix=pd.date_range('2025-06-01 01:00','2026-06-01 00:00',freq='h',tz='Asia/Seoul')
    for group in ['G1','G2']:
        assert pd.DatetimeIndex(f.loc[f.group.eq(group),'ts']).tz_convert('Asia/Seoul').equals(ix)
    # Independent aggregation from persisted site-level files, in a fresh process.
    total=pd.concat([pd.read_parquet(p) for p in sorted((PROD/'site_predictions').glob('*.parquet'))],ignore_index=True)
    assert len(total)==253*8760 and not total.duplicated(['site_id','ts']).any()
    assert total.groupby('ts').site_id.nunique().eq(253).all()
    assert total.group.equals(total.site_id.map(c.set_index('site_id').group))
    aggregated=total.groupby(['group','ts']).forecast_kwh.sum().sort_index()
    submitted=f.set_index(['group','ts']).forecast_kwh.sort_index()
    assert np.allclose(aggregated.to_numpy(),submitted.to_numpy(),rtol=1e-14,atol=1e-8)
    manifest=json.loads((PROD/'training_manifest.json').read_text(encoding='utf-8'))
    before=json.loads((PROD/'inputs_before.json').read_text(encoding='utf-8'));assert before==input_hashes()
    report={'passed':True,'composition_rows':len(c),'forecast_rows':len(f),'sites':253,'hours_per_group':8760,
        'capacity_kw':caps.to_dict(),'training_cutoff':str(FINAL_END),'training_rows':manifest['training_rows'],'feature_count':177,
        'model':'XGBoost only, original grouping, no scale correction, no ensemble','source_inputs_unchanged':True,
        'saved_site_sum_max_difference_kwh':float(np.max(abs(aggregated.to_numpy()-submitted.to_numpy()))),
        'start':str(ix.min()),'end':str(ix.max()),'sha256':{n:sha(OUT/n) for n in ['composition.csv','forecast.csv']},
        'uploaded_to_server':False}
    save(OUT/'검증결과.json',report)
    save(OUT/'학습기록.json',manifest)
    (OUT/'제출안내.txt').write_text('제출할 파일은 composition.csv와 forecast.csv 두 개입니다.\n177개 특징 XGBoost 단독, 기존 그룹 유지, 그룹 보정 없음.\n2025-05-31 17:00 KST까지 최종 재학습했습니다.\n기존 제출 파일과 구분해서 사용하세요. 자동 서버 제출은 하지 않았습니다.\n',encoding='utf-8')
    archive=OUT.parent/(OUT.name+'.zip')
    if archive.exists(): raise ValueError('Refuse overwrite of submission ZIP')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['composition.csv','forecast.csv']: z.write(OUT/name,arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==['composition.csv','forecast.csv'] and z.testzip() is None
        for name in z.namelist(): assert z.read(name)==(OUT/name).read_bytes()
    log(stage='submission_validated',composition_rows=len(c),forecast_rows=len(f),capacity_kw=caps.to_dict())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict','validate']);a=p.parse_args()
    {'train':train,'predict':predict,'validate':validate}[a.stage]()
