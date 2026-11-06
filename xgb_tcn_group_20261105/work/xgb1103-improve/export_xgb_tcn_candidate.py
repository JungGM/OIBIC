"""Train the selected TCN design on allowed history and export a local candidate."""
import os,sys
if os.environ.get('OIBC_TORCH_LIBS'):
    sys.path.insert(0,os.environ['OIBC_TORCH_LIBS'])
import argparse,json,gc,time,zipfile,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import tcn_candidate as base
from experiment import ROOT,aux,save,sha,log
from xgb_loss_trial import protected

PROD=ROOT/'production_xgb_tcn_20261105';BASE=PROD/'base/production'
OUT=Path('outputs/OIBC_XGB_TCN_제출후보_20261105')
FINAL_END=pd.Timestamp('2025-05-31 17:00',tz='Asia/Seoul')
ORIGINAL=Path('outputs/OIBC_177_제출용_20261103_v2')

def decide():
    test=json.loads(Path('outputs/OIBC_발전소개별특성_개선_20261105/그룹손실_추가학습_검증.json').read_text(encoding='utf-8'))
    assert all(v['complete'] for v in test['decisions'].values())
    eligible=[n for n,v in test['decisions'].items() if v['promotion_passed']]
    choice=min(eligible,key=lambda n:np.mean([v[n]['all']['penalty_krw']/v['old_5050']['all']['penalty_krw'] for v in test['folds'].values()])) if eligible else 'baseline'
    d={'choice':choice,'eligible_new_variants':eligible,'selection_rule':'Only candidates passing both-period1% improvement and every-month2% safety gate; among passing variants equal-weight mean relative penalty. Otherwise retain original validated50:50 design.','blend':[.5,.5],'groups':'original A->G1/B->G2','new_modifications_adopted':choice!='baseline','baseline_warning':'Original50:50 improved aggregate historical penalties but had worse individual months. Export is a submission candidate, not guaranteed official-year improvement or a winner.','evaluation_warning':'Previously inspected historical periods;2024 partial207-site proxy differs from2025 full253. Production model is refit on longer allowed history, not the same weights used in backtests.','protected':protected(),'validation_result_sha256':sha(Path('outputs/OIBC_발전소개별특성_개선_20261105/그룹손실_추가학습_검증.json'))}
    PROD.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    path=PROD/'decision.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==d
    else:save(path,d)
    return d

def train():
    decision=decide();BASE.mkdir(parents=True,exist_ok=True)
    if not (BASE/'model.pt').exists():
        base.seed();raw=np.load(base.RUN/'data.npz');x=raw['x'];y=raw['y'];w=raw['w'];end=raw['end']
        if end.max()<10**14:end*=1000000
        elif end.max()<10**17:end*=1000
        mask=(end<=FINAL_END.value)&(w.sum(1)>0);assert end[mask].max()<=FINAL_END.value
        mean,std=base.scaling(x[mask]);np.savez(BASE/'scaler.npz',mean=mean,std=std)
        xx=x[mask];yy=y[mask].copy();ww=w[mask].copy();ww/=ww[ww>0].mean();days=int(mask.sum());last=str(pd.Timestamp(int(end[mask].max()),tz='UTC').tz_convert('Asia/Seoul'))
        del raw,x,y,w,end;gc.collect()
        epochs=json.loads((ROOT/'tcn_candidate1/spring/training.json').read_text(encoding='utf-8'))['best_epoch'];assert epochs==10
        dl=DataLoader(TensorDataset(torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(ww)),batch_size=256,shuffle=True,num_workers=0)
        model=base.TCN();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];start=time.time()
        log(stage='production_tcn_start',days=days,epochs=epochs,cutoff=str(FINAL_END))
        for epoch in range(1,epochs+1):
            model.train();total=0.;mass=0.
            for bx,by,bw in dl:
                opt.zero_grad();p=model(torch.from_numpy(base.normalize(bx.numpy(),mean,std)));loss=((p-by).abs()*bw).sum()/bw.sum().clamp_min(1)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();m=float(bw.sum());total+=float(loss.detach())*m;mass+=m
            h={'epoch':epoch,'weighted_cf_mae':total/mass};history.append(h);log(stage='production_tcn_fit',**h)
        model.eval();check=torch.from_numpy(base.normalize(xx[:32],mean,std))
        with torch.no_grad():before=model(check).clone()
        torch.save(model.state_dict(),BASE/'model.pt');stored=base.TCN();stored.load_state_dict(torch.load(BASE/'model.pt',weights_only=True));stored.eval()
        with torch.no_grad():assert torch.equal(before,stored(check))
        save(BASE/'training.json',{'training_days':days,'cutoff':str(FINAL_END),'last_complete_label_day_end':last,'epochs':epochs,'features':base.FEATURES,'target':'clip energy/capacity to[0,1.2]','sample_weight':'historical SMP*k*capacity, positive-weight mean normalized1; invalid/night labels masked','history':history,'reload_exact':True,'model_sha256':sha(BASE/'model.pt'),'seconds':time.time()-start})
        del xx,yy,ww,dl,model,stored,opt;gc.collect()
    chosen=BASE/'model.pt'
    if decision['choice']!='baseline':
        import tcn_group_finetune as tune
        tune.PRETRAIN_ROOT=PROD/'base';tune.RUN=PROD/'tune';tune.FOLDS={'production':{'cutoff':str(FINAL_END)}}
        name=decision['choice'];tune.fit('production',name,tune.VARIANTS[name],predict=False);chosen=tune.RUN/'production'/name/'model.pt'
    assert protected()==decision['protected']
    save(PROD/'chosen_model.json',{'path':str(chosen),'sha256':sha(chosen),'scaler':str(BASE/'scaler.npz'),'choice':decision['choice']})

def build_inputs(site,scope,fa,fb,ix,aa,bb):
    from run_experiment import v1,enhanced
    f=v1.b.features(None,scope,fa[fa.wa_station.eq(site.wa_station)],fb[fb.wb_station.eq(site.wb_station)],ix,17);enhanced(f)
    if site.wa_station not in aa:aa[site.wa_station]=aux(fa,'a',site.wa_station,ix)
    if site.wb_station not in bb:bb[site.wb_station]=aux(fb,'b',site.wb_station,ix)
    for source,extra in [('a',aa[site.wa_station]),('b',bb[site.wb_station])]:
        for name in ['cloud','temp']:f[f'{source}_revision_{name}']=extra[f'{source}_revision_{name}'].to_numpy()
    mid=ix-pd.Timedelta(minutes=30);h=mid.hour+mid.minute/60;f['old_hour_sin']=np.sin(2*np.pi*h/24);f['old_hour_cos']=np.cos(2*np.pi*h/24)
    return f

def feature_audit(sites):
    ix=pd.date_range('2025-01-01 01:00','2025-01-08 00:00',freq='h',tz='Asia/Seoul')
    filters=[('ts','>=',ix.min()),('ts','<=',ix.max())]
    fa=pd.read_parquet('work/data/train/forecast_wa.parquet',filters=filters);fb=pd.read_parquet('work/data/train/forecast_wb.parquet',filters=filters);aa={};bb={};checked=0
    for site in sites[sites.site_id.isin(['PV001','PV011','PV188'])].itertuples(index=False):
        scope=sites[sites.site_id.eq(site.site_id)];f=build_inputs(site,scope,fa,fb,ix,aa,bb)
        cached=pd.read_parquet(ROOT/'cache/2025'/f'{site.site_id}.parquet',columns=['ts']+base.FEATURES).set_index('ts').reindex(ix)
        assert np.allclose(f[base.FEATURES].to_numpy('float32'),cached[base.FEATURES].to_numpy('float32'),rtol=1e-5,atol=1e-5,equal_nan=True);checked+=len(ix)
    assert checked==504;save(PROD/'feature_audit.json',{'passed':True,'site_hours_compared':checked,'features':31,'source':'same production input builder vs historical177-feature cache; no observations or generation inputs'})

def predict():
    decision=decide();mf=json.loads((PROD/'chosen_model.json').read_text(encoding='utf-8'));assert sha(mf['path'])==mf['sha256']
    if (OUT/'forecast.csv').exists():raise ValueError('Preserve completed candidate forecast')
    model=base.TCN();model.load_state_dict(torch.load(mf['path'],weights_only=True));model.eval();sc=np.load(BASE/'scaler.npz');mean=sc['mean'];std=sc['std']
    os.environ['OIBC_DATA']=str(Path('work/data').resolve())
    sites=pd.read_csv('work/data/sites.csv');sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    feature_audit(sites)
    ix=pd.date_range('2025-06-01 01:00','2026-06-01 00:00',freq='h',tz='Asia/Seoul');assert len(ix)==8760 and len(ix)%24==0
    # Timestamp column only: no actual official-year SMP used in features or normalization.
    target=pd.read_csv('work/data/smp.csv',usecols=['ts']);assert pd.DatetimeIndex(pd.to_datetime(target.ts)).tz_convert('Asia/Seoul').equals(ix)
    fa=pd.read_parquet('work/data/test/forecast_wa.parquet');fb=pd.read_parquet('work/data/test/forecast_wb.parquet')
    for weather,key in [(fa,'wa_station'),(fb,'wb_station')]:assert not weather.duplicated([key,'round','ts']).any() and set(weather['round'])=={10,17}
    aa={};bb={};sums={g:np.zeros(len(ix),dtype='float64') for g in ['G1','G2']};xgb_sums={g:np.zeros(len(ix),dtype='float64') for g in ['G1','G2']};missing=0
    dest=PROD/'site_predictions';dest.mkdir(exist_ok=True)
    for j,site in enumerate(sites.itertuples(index=False)):
        scope=sites[sites.site_id.eq(site.site_id)];f=build_inputs(site,scope,fa,fb,ix,aa,bb)
        values=f[base.FEATURES].to_numpy('float32');assert not np.isinf(values).any();missing+=int(np.isnan(values).sum());seq=values.reshape(-1,24,len(base.FEATURES))
        with torch.no_grad():cf=model(torch.from_numpy(base.normalize(seq,mean,std))).numpy().reshape(-1).astype('float64')
        pt=cf*float(site.capacity_kw);pt[f.elev.to_numpy()<=0]=0
        source=ROOT/'production177/site_predictions'/f'{site.site_id}.parquet';old=pd.read_parquet(source);assert old.site_id.eq(site.site_id).all() and old.group.eq(site.group).all() and pd.DatetimeIndex(old.ts).equals(ix)
        px=old.forecast_kwh.to_numpy('float64');p=.5*px+.5*pt;assert np.isfinite(p).all() and (p>=0).all()
        z=pd.DataFrame({'site_id':site.site_id,'group':site.group,'ts':ix,'xgb_kwh':px,'tcn_kwh':pt,'forecast_kwh':p});z.to_parquet(dest/f'{site.site_id}.parquet',index=False)
        sums[site.group]+=p;xgb_sums[site.group]+=px
        if (j+1)%40==0 or j+1==len(sites):log(stage='candidate_forecast',sites=j+1,rows=(j+1)*len(ix))
    forecast=pd.concat([pd.DataFrame({'group':g,'ts':ix,'forecast_kwh':sums[g]}) for g in ['G1','G2']],ignore_index=True)
    original=pd.read_csv(ORIGINAL/'forecast.csv');original.ts=pd.to_datetime(original.ts)
    for g in ['G1','G2']:
        a=original[original.group.eq(g)].sort_values('ts');assert np.allclose(a.forecast_kwh.to_numpy(),xgb_sums[g],rtol=0,atol=1e-6)
    sites[['site_id','group']].to_csv(OUT/'composition.csv',index=False);forecast.to_csv(OUT/'forecast.csv',index=False)
    assert protected()==decision['protected'] and sha(mf['path'])==mf['sha256']
    save(PROD/'prediction.json',{'site_rows':253*8760,'features':base.FEATURES,'feature_missing_cells':missing,'evaluation_year_statistics_used':False,'evaluation_actual_SMP_used':False,'generation_lag_used':False,'same_issued_day24h_forecasts_only':True,'blend':[.5,.5],'xgb_aggregation_matches_original_v2':True,'model':mf})

def validate():
    decision=json.loads((PROD/'decision.json').read_text(encoding='utf-8'));assert protected()==decision['protected']
    sites=pd.read_csv('work/data/sites.csv');c=pd.read_csv(OUT/'composition.csv');f=pd.read_csv(OUT/'forecast.csv');assert f.ts.str.endswith('+09:00').all();f.ts=pd.to_datetime(f.ts).dt.tz_convert('Asia/Seoul')
    assert list(c)==['site_id','group'] and list(f)==['group','ts','forecast_kwh'];assert len(c)==253 and c.site_id.is_unique and set(c.site_id)==set(sites.site_id)
    assert set(c.group)=={'G1','G2'} and c.set_index('site_id').group.equals(pd.read_csv(ORIGINAL/'composition.csv').set_index('site_id').group)
    capacities=c.merge(sites[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum();assert capacities.ge(20000).all()
    ix=pd.date_range('2025-06-01 01:00','2026-06-01 00:00',freq='h',tz='Asia/Seoul');assert len(f)==17520 and not f.duplicated(['group','ts']).any() and not f.isna().any().any() and np.isfinite(f.forecast_kwh).all() and f.forecast_kwh.ge(0).all()
    for g in ['G1','G2']:assert pd.DatetimeIndex(f[f.group.eq(g)].sort_values('ts').ts).equals(ix)
    os.environ['OIBC_DATA']=str(Path('work/data').resolve());from run_experiment import v1
    sums={g:np.zeros(8760,dtype='float64') for g in ['G1','G2']};rows=0;night_rows=0
    for p in sorted((PROD/'site_predictions').glob('*.parquet')):
        z=pd.read_parquet(p);assert len(z)==8760 and z.site_id.nunique()==1 and not z.duplicated(['site_id','ts']).any();assert pd.DatetimeIndex(z.ts).equals(ix)
        assert np.allclose(z.forecast_kwh,.5*z.xgb_kwh+.5*z.tcn_kwh,rtol=0,atol=1e-8);assert np.isfinite(z[['xgb_kwh','tcn_kwh','forecast_kwh']]).all().all()
        site=sites[sites.site_id.eq(z.site_id.iloc[0])].iloc[0];elev=v1.b.solar(ix,np.full(8760,site.lat),np.full(8760,site.lon))[0];night=elev<=0;assert z.loc[night,'forecast_kwh'].eq(0).all();night_rows+=int(night.sum())
        sums[z.group.iloc[0]]+=z.forecast_kwh.to_numpy();rows+=len(z)
    assert rows==253*8760
    diffs={g:float(np.max(abs(f[f.group.eq(g)].sort_values('ts').forecast_kwh.to_numpy()-sums[g]))) for g in sums};assert max(diffs.values())<1e-6
    mf=json.loads((PROD/'chosen_model.json').read_text(encoding='utf-8'));assert sha(mf['path'])==mf['sha256']
    report={'valid':True,'plants':253,'forecast_rows':17520,'site_rows':rows,'nighttime_site_rows_verified_zero':night_rows,'group_capacity_kw':capacities.to_dict(),'first':str(ix[0]),'last':str(ix[-1]),'independent_reaggregation_max_difference_kwh':diffs,'choice':decision['choice'],'blend':[.5,.5],'official_score_known':False,'existing_inputs_unchanged':True,'file_sha256':{n:sha(OUT/n) for n in ['composition.csv','forecast.csv']},'model':mf}
    save(OUT/'검증결과.json',report);shutil.copyfile(BASE/'training.json',OUT/'TCN_학습기록.json');shutil.copyfile(PROD/'decision.json',OUT/'모델선택.json')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for n in ['composition.csv','forecast.csv']:z.write(OUT/n,arcname=n)
    (OUT/'제출안내.txt').write_text('제출 후보 파일: composition.csv와forecast.csv. ZIP은두 CSV만 포함합니다.\n기존 G1/G2 배정을 유지하며두 그룹 모두20MW 이상입니다.\n새 모델은허용된2025-05-31 17:00 이전 역사로학습했습니다. TCN은완료된24시간 묶음만 사용합니다.\nXGBoost50%+TCN50%. 검증결과.json과모델선택.json에서실제 채택한 TCN 변형을확인하십시오.\n과거 검증은공식 평가연도의 실제 점수나순위를 보장하지 않습니다. 기존 원본 제출파일은보존했습니다.\n온라인 대회 제출은자동 실행하지 않았습니다.\n',encoding='utf-8')
    log(stage='candidate_validated',**report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict','validate']);a=p.parse_args()
    {'train':train,'predict':predict,'validate':validate}[a.stage]()
