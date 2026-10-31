"""Finite OIBC improvement experiment. Reads attributed baseline; never mutates it."""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from extras import add_features, EXTRA
import correction as cr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'outputs/suwan_projectV2'
sys.path.insert(0,str(BASE))
import pipeline as p
import features as ft
import grouping
from validate_submission import validate

DATA=ROOT/'work/oibc_data'
CACHE=ROOT/'work/suwan_v2_cache'
BASE_RUN=ROOT/'work/suwan_v2_reproduction'
RUN=ROOT/'work/suwan_final'
RUN.mkdir(parents=True,exist_ok=True)
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):p.save_json(path,value)
def setup():
    protocol=read(HERE/'PROTOCOL.json')
    guard={'protocol':protocol,'source':read(BASE_RUN/'protocol_before_training.json')['source_sha256'],
           'code':{n:p.sha256(HERE/n) for n in ['run.py','correction.py','extras.py']}}
    path=RUN/'guard.json'
    if path.exists():assert read(path)==guard,'Experiment source changed; explicit new run required'
    else:save(path,guard)
    sites=pd.read_csv(DATA/'sites.csv').sort_values('site_id').reset_index(drop=True)
    prices=pd.read_csv(DATA/'train/smp.csv');prices['ts']=pd.to_datetime(prices.ts)
    prices['weight']=prices.smp*np.where(prices.ts.dt.hour.between(11,15),3,1)
    return sites,prices

def group_frame(z,assignment,sites,prices,month,split='train'):
    """Same complete fleet/time grid; forecast-only group weather moments."""
    assert not z.duplicated(['site_id','ts']).any()
    ids=sites.site_id.tolist()
    assert set(z.site_id)==set(ids)
    assert z.groupby('ts').site_id.nunique().eq(len(ids)).all()
    columns=['site_id','ts','cap','cosz_positive','r17_wa_cloud','r17_wb_cloud',
             'r17_wb_uv_idx','r17_wa_humidity','r17_provider_cloud_difference',
             'revision_wb_cloud','hour_sin','hour_cos','r17_weather_missing']
    weather=pd.read_parquet(CACHE/split/f'{month}.parquet',columns=columns)
    d=z[['site_id','ts','energy','base']].merge(weather,on=['site_id','ts'],validate='one_to_one')
    assert len(d)==len(z)
    mapping=dict(zip(ids,np.where(assignment==1,'G1','G2')))
    d['group']=d.site_id.map(mapping)
    result=[]
    for group in ['G1','G2']:
        a=d[d.group==group].copy();by=a.groupby('ts',sort=True)
        times=by.size().index
        assert by.size().eq(sum(x==group for x in mapping.values())).all()
        v=pd.DataFrame({'ts':times,'group':group})
        v['base']=by.base.sum().to_numpy()
        v['actual']=by.energy.sum(min_count=by.size().iloc[0]).to_numpy()
        capacity=sites.loc[sites.site_id.map(mapping)==group,'capacity_kw'].sum()
        v['capacity']=capacity;v['base_util']=v.base/capacity
        def mean(values):
            val=np.asarray(values,float);finite=np.isfinite(val)
            numer=pd.Series(np.where(finite,val,0)*a.cap.to_numpy(),index=a.ts).groupby(level=0).sum()
            denom=pd.Series(finite*a.cap.to_numpy(),index=a.ts).groupby(level=0).sum()
            return np.divide(numer.to_numpy(),denom.to_numpy(),out=np.full(len(times),np.nan),where=denom.to_numpy()>0)
        for name,col in [('solar','cosz_positive'),('cloud_a','r17_wa_cloud'),
                         ('cloud_b','r17_wb_cloud'),('uv','r17_wb_uv_idx'),
                         ('humidity','r17_wa_humidity'),('hour_sin','hour_sin'),
                         ('hour_cos','hour_cos'),('weather_missing','r17_weather_missing')]:
            v[name]=mean(a[col])
        for name,col,avg in [('cloud_sd','r17_wb_cloud','cloud_b'),('uv_sd','r17_wb_uv_idx','uv')]:
            v[name]=np.sqrt(np.maximum(mean(a[col].to_numpy(dtype=float)**2)-v[avg].to_numpy()**2,0))
        v['provider_disagreement']=mean(abs(a.r17_provider_cloud_difference))
        v['revision_disagreement']=mean(abs(a.revision_wb_cloud))
        v['cloud_solar']=v.solar*(1-np.clip(v.cloud_b/100,0,1))**2
        v['uv_solar']=v.solar*v.uv
        v=v.merge(prices[['ts','weight']],on='ts',how='left',validate='one_to_one')
        result.append(v)
    return pd.concat(result,ignore_index=True).sort_values(['ts','group']).reset_index(drop=True)

def dev_sites(month,variant='base'):
    d=pd.read_parquet(BASE_RUN/'development/site_predictions.parquet')
    d=d[(d.ts-pd.Timedelta(hours=1)).dt.month==month].copy()
    d['base']=d.ensemble
    if variant=='feature':
        a=pd.read_parquet(RUN/f'dev_feature_{month:02d}.parquet')
        d=d.merge(a,on=['site_id','ts'],validate='one_to_one')
        d['base']=np.maximum(d.ensemble+.4*(d.new_xgb-d.old_xgb),0)
    return d

def prepare(sites,prices):
    assignment=p.reference(sites)
    d=dev_sites(3)
    frame=group_frame(d,assignment,sites,prices,'2025-03')
    frame.to_parquet(RUN/'march_reference.parquet',index=False)
    recipe,table=cr.select(frame)
    save(OUT/'stage2_correction.json',{'selected':recipe,'March_validation':table})
    print('Stage2 March selection:',[(r['kind'],r['validation_penalty']) for r in table],flush=True)

def extra_cache():
    dest=RUN/'feature_cache';dest.mkdir(exist_ok=True)
    for file in sorted((CACHE/'train').glob('*.parquet')):
        # Only months needed by development/production 24-month windows.
        if file.stem<'2023-03':continue
        path=dest/file.name
        if path.exists():continue
        d=add_features(pd.read_parquet(file))
        d[['site_id','ts']+EXTRA].to_parquet(path,index=False,compression='zstd')
        print('Extra features:',file.stem,flush=True)
        del d;gc.collect()

def train_feature(stage):
    directory=RUN/f'feature_{stage}';directory.mkdir(exist_ok=True)
    path=directory/'model.ubj'
    schema=ft.get_schema(CACHE);names=schema['enhanced']+EXTRA
    origin=p.stamp('2025-03-01 01:00' if stage=='development' else '2025-06-01 01:00')
    if path.exists():
        meta=read(directory/'metadata.json');assert p.sha256(path)==meta['model_sha256']
        assert meta['features']==names and meta['cutoff']==str(p.issue_deadline(origin))
        return
    extra_cache()
    start,end=p.history_start(origin),p.issue_deadline(origin)
    chunks=[]
    for month in pd.period_range((start-pd.Timedelta(hours=1)).tz_localize(None).to_period('M'),
                                 (end-pd.Timedelta(hours=1)).tz_localize(None).to_period('M'),freq='M'):
        d=pd.read_parquet(CACHE/'train'/f'{month}.parquet')
        e=pd.read_parquet(RUN/'feature_cache'/f'{month}.parquet')
        d=d.merge(e,on=['site_id','ts'],validate='one_to_one')
        d=d[(d.ts>=start)&(d.ts<=end)&(d.elev>0)&d.energy.notna()].copy()
        chunks.append(d[names+['energy']])
        del d,e
    d=pd.concat(chunks,ignore_index=True);del chunks;gc.collect()
    y=(d.energy/d.cap).clip(0,1.2).astype('float32')
    w=(d.cap.to_numpy()/d.cap.median())**.5;w/=w.mean()
    count=len(d)
    expected=read(BASE_RUN/('development' if stage=='development' else 'production')/'models/xgb_deep.json')['training_rows']
    assert count==expected,'Different training rows from base XGB'
    x=d[names].astype('float32');del d;gc.collect()
    config=next(c for c in p.CONFIGS if c['name']=='xgb_deep')
    params={k:v for k,v in config.items() if k not in ['name','kind','history_months','cap_weight']}
    model=xgb.XGBRegressor(**params,learning_rate=.035,tree_method='hist',max_bin=128,
          subsample=.85,colsample_bytree=.85,reg_alpha=.1,n_jobs=2,random_state=p.SEED)
    print('Training ONE feature XGB:',stage,count,'rows',len(names),'features',flush=True)
    model.fit(x,y,sample_weight=w)
    model.save_model(path)
    save(directory/'metadata.json',{'features':names,'config':config,'training_rows':count,
          'cutoff':str(end),'model_sha256':p.sha256(path)})
    print('Feature model complete:',stage,flush=True)

def feature_predictions(stage):
    directory=RUN/f'feature_{stage}'
    model=xgb.XGBRegressor(n_jobs=2);model.load_model(directory/'model.ubj')
    old=xgb.XGBRegressor(n_jobs=2)
    old.load_model(BASE_RUN/('development' if stage=='development' else 'production')/'models/xgb_deep.ubj')
    schema=ft.get_schema(CACHE);names=read(directory/'metadata.json')['features']
    months=['2025-03','2025-04','2025-05'] if stage=='development' else [str(x) for x in pd.period_range('2025-06','2026-05',freq='M')]
    for month in months:
        output=RUN/(f'dev_feature_{month[-2:]}.parquet' if stage=='development' else f'production_feature_{month}.parquet')
        if output.exists():continue
        d=add_features(pd.read_parquet(CACHE/('train' if stage=='development' else 'test')/f'{month}.parquet'))
        if stage=='development':
            end=p.stamp(str(pd.Period(month,freq='M').end_time.date())+' 17:00')
            d=d[d.ts<=end].copy()
        z=d[['site_id','ts']].copy();day=d.elev.to_numpy()>0
        for label,m,cols in [('new_xgb',model,names),('old_xgb',old,schema['enhanced'])]:
            arr=np.zeros(len(d));arr[day]=np.maximum(m.predict(d.loc[day,cols].astype('float32')),0)*d.loc[day,'cap'].to_numpy()
            z[label]=arr
        z.to_parquet(output,index=False,compression='zstd')
        print('Feature prediction:',month,flush=True)
        del d,z;gc.collect()

def choose(sites,prices):
    ref=p.reference(sites)
    summaries=[];entries=[]
    for variant in ['base','feature']:
        frame=group_frame(dev_sites(3,variant),ref,sites,prices,'2025-03')
        recipe,table=cr.select(frame)
        score=min(r['validation_penalty'] for r in table)
        entries.append((score,variant,recipe,ref))
        summaries.append({'variant':variant,'selected':recipe,'candidates':table})
    baseline=next(x for x in summaries[0]['candidates'] if x['kind']=='none')['validation_penalty']
    score,variant,recipe,assignment=min(entries,key=lambda a:a[0])
    direct_used=False
    if score>=baseline:
        frame=group_frame(dev_sites(3),ref,sites,prices,'2025-03')
        alt,table=cr.select(frame,allow_direct=True)
        summaries.append({'variant':'direct_alternative','selected':alt,'candidates':table})
        candidate=min(x['validation_penalty'] for x in table)
        if candidate<score:score,variant,recipe=candidate,'base',alt
        direct_used=True
    # One group proposal from earlier March only; correction refitted for proposal.
    z=dev_sites(3,variant)
    early=z[(z.ts-pd.Timedelta(hours=1)).dt.day<=20]
    pr=prices.copy();pr['k']=np.where(pr.ts.dt.hour.between(11,15),3,1)
    times,y,pp,w,smp=p.panel(early,['base'],sites.site_id.tolist(),pr)
    proposal,search=grouping.optimize(pp['base']-y,w,sites.capacity_kw.to_numpy(),ref,
                                     seed=p.SEED,steps=10000,restarts=3)
    save(OUT/'stage5_group_search.json',search)
    group_frame_new=group_frame(z,proposal,sites,prices,'2025-03')
    newrecipe,table=cr.select(group_frame_new,allow_direct=direct_used)
    newscore=min(r['validation_penalty'] for r in table)
    summaries.append({'variant':variant+'_group_proposal','selected':newrecipe,'candidates':table})
    if newscore<score:
        score,recipe,assignment=newscore,newrecipe,proposal
    candidate={'variant':variant,'recipe':recipe,'assignment':assignment.tolist(),
               'March_selection_penalty':score,'reference_March_selection_penalty':baseline,
               'direct_alternative_run':direct_used,'group_changes':int(np.sum(assignment!=ref)),
               'protocol_sha256':p.sha256(HERE/'PROTOCOL.json')}
    save(OUT/'March_candidate_comparisons.json',summaries)
    save(OUT/'candidate_before_April.json',candidate)
    print('Frozen single April candidate:',{k:v for k,v in candidate.items() if k not in ['recipe','assignment']},flush=True)

def daily(frame,pred):
    day=(frame.ts-pd.Timedelta(hours=1)).dt.normalize()
    return pd.Series(.1*frame.weight.to_numpy()*abs(pred-frame.actual.to_numpy()),index=day).groupby(level=0).sum()

def block_lower(before,after):
    savings=np.asarray(before-after);n=len(savings);rng=np.random.default_rng(p.SEED)
    starts=rng.integers(n,size=(2000,(n+2)//3))
    indices=((starts[:,:,None]+np.arange(3))%n).reshape(2000,-1)[:,:n]
    return float(np.quantile(savings[indices].mean(axis=1),.05))

def gate(sites,prices):
    c=read(OUT/'candidate_before_April.json');ref=p.reference(sites);assignment=np.array(c['assignment'])
    base=group_frame(dev_sites(4),ref,sites,prices,'2025-04')
    frame=group_frame(dev_sites(4,c['variant']),assignment,sites,prices,'2025-04')
    pred=cr.predict(frame,c['recipe'])
    before,after=daily(base,base.base.to_numpy()),daily(frame,pred)
    assert before.index.equals(after.index)
    check=grouping.daily_gate(before.to_numpy(),after.to_numpy())
    lower=block_lower(before.to_numpy(),after.to_numpy())
    accepted=bool(check['accepted'] and lower>0 and c['March_selection_penalty']<c['reference_March_selection_penalty'])
    chosen=c if accepted else {'variant':'base','recipe':{'kind':'none','strength':0.},'assignment':ref.tolist()}
    save(OUT/'frozen_before_May.json',{'accepted':accepted,'candidate':c,'selected':chosen,
         'April_gate':check,'April_3day_bootstrap_lower90_krw_per_day':lower,
         'candidate_file_sha256':p.sha256(OUT/'candidate_before_April.json')})
    reports=[]
    for mo in [3,4,5]:
        b=group_frame(dev_sites(mo),ref,sites,prices,f'2025-{mo:02d}')
        f=group_frame(dev_sites(mo,c['variant']),assignment,sites,prices,f'2025-{mo:02d}')
        basepen=cr.loss(b,b.base.to_numpy());candpen=cr.loss(f,cr.predict(f,c['recipe']))
        reports.append({'month':mo,'baseline_penalty':basepen,'candidate_penalty':candpen,
                        'selected_penalty':candpen if accepted else basepen,
                        'candidate_improvement_pct':100*(1-candpen/basepen)})
        if mo==5:
            f['candidate_forecast']=cr.predict(f,c['recipe']);f.to_parquet(RUN/'May_group_audit.parquet',index=False)
    pd.DataFrame(reports).to_csv(OUT/'monthly.csv',index=False)
    print('April accepted:',accepted,'gate',check,'blocklower',lower,flush=True)
    print(pd.DataFrame(reports).to_string(index=False),flush=True)

def produce(sites,prices):
    assert (BASE_RUN/'completion.json').exists()
    frozen=read(OUT/'frozen_before_May.json');c=frozen['selected']
    assignment=np.array(c['assignment']);reference=p.reference(sites)
    output=HERE/'submission';output.mkdir(exist_ok=True)
    basepred=pd.read_parquet(BASE_RUN/'production/site_predictions.parquet')
    generated=[]
    # Only target metadata/forecasts, never target actual generation or price.
    for month in pd.period_range('2025-06','2026-05',freq='M'):
        mo=str(month);op=(basepred.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
        z=basepred[op==mo].copy();z['energy']=np.nan;z['base']=z.prediction_kwh
        if c['variant']=='feature':
            extra=pd.read_parquet(RUN/f'production_feature_{mo}.parquet')
            z=z.merge(extra,on=['site_id','ts'],validate='one_to_one')
            z['base']=np.maximum(z.prediction_kwh+.4*(z.new_xgb-z.old_xgb),0)
        if c['recipe']['kind']=='none':
            mapping=dict(zip(sites.site_id,np.where(assignment==1,'G1','G2')))
            z['group']=z.site_id.map(mapping)
            forecast=z.groupby(['group','ts'],as_index=False).base.sum().rename(columns={'base':'forecast_kwh'})
        else:
            dummy=pd.DataFrame({'ts':z.ts.unique(),'weight':0.})
            frame=group_frame(z,assignment,sites,dummy,mo,split='test')
            forecast=frame[['group','ts']].copy();forecast['forecast_kwh']=cr.predict(frame,c['recipe'])
        generated.append(forecast)
    forecast=pd.concat(generated,ignore_index=True).sort_values(['group','ts']).reset_index(drop=True)
    composition=pd.DataFrame({'site_id':sites.site_id,'group':np.where(assignment==1,'G1','G2')})
    composition.to_csv(output/'composition.csv',index=False);forecast.to_csv(output/'forecast.csv',index=False)
    errors=validate(output/'composition.csv',output/'forecast.csv',DATA/'sites.csv');assert not errors,errors
    assert len(forecast)==17520 and len(composition)==253
    assert (forecast.forecast_kwh>=0).all() and np.isfinite(forecast.forecast_kwh).all()
    assert p.source_manifest(DATA)==read(BASE_RUN/'protocol_before_training.json')['source_sha256']
    assert min(sites.assign(group=composition.group).groupby('group').capacity_kw.sum())>=20000
    save(OUT/'completion.json',{'complete':True,'accepted_improvement':frozen['accepted'],
         'forecast_rows':len(forecast),'composition_rows':len(composition),
         'hashes':{n:p.sha256(output/n) for n in ['forecast.csv','composition.csv']},
         'source_unchanged':True,'official_server_submission':False,
         'group_changes':int(np.sum(assignment!=reference))})
    print('Final submission validated:',output,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','feature-cache','feature-fit-dev','feature-predict-dev','choose','gate','feature-fit-production','feature-predict-production','produce'])
    a=ap.parse_args();sites,prices=setup()
    if a.stage=='prepare':prepare(sites,prices)
    elif a.stage=='feature-cache':extra_cache()
    elif a.stage=='feature-fit-dev':train_feature('development')
    elif a.stage=='feature-predict-dev':feature_predictions('development')
    elif a.stage=='choose':choose(sites,prices)
    elif a.stage=='gate':gate(sites,prices)
    elif a.stage=='feature-fit-production':train_feature('production')
    elif a.stage=='feature-predict-production':feature_predictions('production')
    elif a.stage=='produce':produce(sites,prices)

if __name__=='__main__':main()
