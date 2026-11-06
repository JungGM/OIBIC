"""One predeclared small forecast-only TCN; all data and training stay local."""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys
if os.environ.get('OIBC_TORCH_LIBS'):
    sys.path.insert(0,os.environ['OIBC_TORCH_LIBS'])
import argparse, copy, gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from experiment import ROOT, sha, save, log

FEATURES='wa_cloud wa_temp wa_humidity wa_rain wa_wind_speed wa_ground_press wb_cloud wb_temp wb_rel_hum wb_uv_idx wb_precip_prob wb_total_liq wb_vis wb_wind_speed wb_dew_point elev azim cosz doy_sin doy_cos cap lat lon a_revision_cloud a_revision_temp b_revision_cloud b_revision_temp wa_snow wb_snow old_hour_sin old_hour_cos'.split()
RUN=ROOT/'tcn_candidate1'
OUT=Path('outputs/OIBC_TCN_첫후보_20261104')

class Block(nn.Module):
    def __init__(self,c,d):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(c,c,3,padding=d,dilation=d),nn.GELU(),nn.Dropout(.1),nn.Conv1d(c,c,3,padding=d,dilation=d),nn.GELU())
    def forward(self,x): return x+self.net(x)

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(2*len(FEATURES),16,1),*[Block(16,d) for d in [1,2,4,8]],nn.Conv1d(16,1,1))
    def forward(self,x): return 1.2*torch.sigmoid(self.net(x).squeeze(1))

def seed():
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

def prepare():
    if (RUN/'data.npz').exists(): return
    xs=[];ys=[];ws=[];ds=[];sites=[];kept=0;dropped=0
    cols=list(dict.fromkeys(FEATURES+['ts','energy','capacity_kw','sample_weight']))
    for year in range(2021,2026):
        for f in sorted((ROOT/'cache'/str(year)).glob('*.parquet')):
            z=pd.read_parquet(f,columns=cols).sort_values('ts')
            day=(z.ts-pd.Timedelta(hours=1)).dt.normalize()
            # Only complete 24-hour input rows; missing labels are masked, not filled.
            groups=z.groupby(day,sort=True)
            sizes=groups.size();good=sizes[sizes.eq(24)].index
            dropped+=int((sizes.ne(24)).sum());z=z.loc[day.isin(good)].copy()
            if not len(z):continue
            assert np.all(((z.ts.dt.hour.to_numpy()-1)%24).reshape(-1,24)==np.arange(24))
            x=z[FEATURES].to_numpy('float32').reshape(-1,24,len(FEATURES))
            e=z.energy.to_numpy('float64').reshape(-1,24);cap=z.capacity_kw.to_numpy().reshape(-1,24)
            y=np.clip(e/cap,0,1.2).astype('float32');valid=np.isfinite(y)&(e>=0)&(x[:,:,FEATURES.index('elev')]>0)
            y[~valid]=0
            w=z.sample_weight.to_numpy('float32').reshape(-1,24);w[~valid]=0
            dates=z.ts.iloc[23::24].dt.as_unit('ns').astype('int64').to_numpy()
            xs.append(x);ys.append(y);ws.append(w);ds.append(dates);sites.extend([f.stem]*len(x));kept+=len(x)
        log(stage='tcn_sequence_cache',year=year,days=kept)
    np.savez(RUN/'data.npz',x=np.concatenate(xs),y=np.concatenate(ys),w=np.concatenate(ws),end=np.concatenate(ds),site=np.array(sites))
    save(RUN/'cache_manifest.json',{'complete_site_days':kept,'dropped_partial_site_days':dropped,'features':FEATURES})

def normalize(x,mean,std):
    ok=np.isfinite(x)
    a=np.where(ok,(x-mean)/std,0).clip(-10,10).astype('float32')
    return np.concatenate([a,(~ok).astype('float32')],axis=2).transpose(0,2,1).copy()

def scaling(x):
    x=np.where(np.isfinite(x),x,np.nan)
    mean=np.nanmean(x,axis=(0,1));std=np.nanstd(x,axis=(0,1))
    return np.nan_to_num(mean),np.where(np.isfinite(std)&(std>1e-5),std,1)

def loader(x,y,w,shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y.copy()),torch.from_numpy(w.copy())),batch_size=256,shuffle=shuffle,num_workers=0)

def train_epoch(model,dl,opt):
    model.train();total=0.;den=0.
    for x,y,w in dl:
        opt.zero_grad();p=model(x);loss=((p-y).abs()*w).sum()/w.sum().clamp_min(1)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        total+=float(loss.detach())*float(w.sum());den+=float(w.sum())
    return total/den

def val_loss(model,dl):
    model.eval();num=0.;den=0.
    with torch.no_grad():
        for x,y,w in dl:
            num+=float(((model(x)-y).abs()*w).sum());den+=float(w.sum())
    return num/den

def fit(fold):
    seed();dest=RUN/fold;dest.mkdir(exist_ok=True)
    if (dest/'predictions.parquet').exists(): return
    cutoff=pd.Timestamp('2024-12-31 17:00' if fold=='spring' else '2024-05-31 17:00',tz='Asia/Seoul')
    inner=cutoff-pd.Timedelta(days=60)
    raw=np.load(RUN/'data.npz');x=raw['x'];y=raw['y'];w=raw['w'];end=raw['end']
    if end.max()<10**14:end=end*1000000  # initial cache uses pandas millisecond timestamps
    elif end.max()<10**17:end=end*1000
    train=(end<=cutoff.value)&(w.sum(1)>0);a=train&(end<=inner.value);b=train&(end>inner.value)
    assert a.sum()>0 and b.sum()>0 and end[train].max()<=cutoff.value
    log(stage='tcn_fit_start',fold=fold,train_days=int(a.sum()),inner_validation_days=int(b.sum()))
    mean,std=scaling(x[a]);xx=normalize(x[a],mean,std);ww=w[a]/w[a][w[a]>0].mean()
    dl=loader(xx,y[a],ww,True);vd=loader(normalize(x[b],mean,std),y[b],w[b]/w[a][w[a]>0].mean(),False)
    model=TCN();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');best_epoch=0;history=[];t=time.time()
    for epoch in range(1,21):
        loss=train_epoch(model,dl,opt);v=val_loss(model,vd);history.append({'epoch':epoch,'train':loss,'inner_validation':v})
        log(stage='tcn_inner',fold=fold,epoch=epoch,loss=loss,validation=v,seconds=time.time()-t)
        if v<best-1e-5:best=v;best_epoch=epoch
        if epoch-best_epoch>=4:break
    del dl,vd,xx,ww,model,opt;gc.collect()
    # Refit once on every allowed complete day, for the inner-selected epoch count.
    seed();mean,std=scaling(x[train]);xx=normalize(x[train],mean,std);ww=w[train]/w[train][w[train]>0].mean()
    dl=loader(xx,y[train],ww,True);model=TCN();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for epoch in range(1,best_epoch+1):
        loss=train_epoch(model,dl,opt);log(stage='tcn_refit',fold=fold,epoch=epoch,loss=loss)
    state={k:v.detach().clone() for k,v in model.state_dict().items()};torch.save(state,dest/'model.pt');np.savez(dest/'scaler.npz',mean=mean,std=std)
    model.eval();check=torch.from_numpy(xx[:32]);before=model(check).detach()
    stored=TCN();stored.load_state_dict(torch.load(dest/'model.pt',weights_only=True));stored.eval()
    assert torch.equal(before,stored(check).detach())
    save(dest/'training.json',{'cutoff':str(cutoff),'inner_cutoff':str(inner),'features':FEATURES,'training_days':int(train.sum()),'inner_train_days':int(a.sum()),'inner_validation_days':int(b.sum()),'best_epoch':best_epoch,'history':history,'seconds':time.time()-t,'model_sha256':sha(dest/'model.pt'),'reload_exact':True})
    del x,y,w,xx,ww,dl,raw;gc.collect()
    baseline=ROOT/('runs/expanded177/predictions.parquet' if fold=='spring' else 'summer_backtest/site_predictions.parquet')
    ref=pd.read_parquet(baseline);pieces=[]
    os.environ['OIBC_DATA']=str(Path('work/data').resolve())
    from run_experiment import v1, enhanced
    from experiment import aux
    site_table=pd.read_csv('work/data/sites.csv')
    first=(ref.ts.min()-pd.Timedelta(hours=1)).normalize()+pd.Timedelta(hours=1)
    last=(ref.ts.max()-pd.Timedelta(hours=1)).normalize()+pd.Timedelta(days=1)
    ix=pd.date_range(first,last,freq='h')
    fa=pd.read_parquet('work/data/train/forecast_wa.parquet',filters=[('ts','>=',first),('ts','<=',last)])
    fb=pd.read_parquet('work/data/train/forecast_wb.parquet',filters=[('ts','>=',first),('ts','<=',last)])
    for site,r in ref.groupby('site_id'):
        chunks=[pd.read_parquet(ROOT/'cache'/str(year)/f'{site}.parquet',columns=FEATURES+['ts']) for year in sorted(r.ts.dt.year.unique()) if (ROOT/'cache'/str(year)/f'{site}.parquet').exists()]
        f=pd.concat(chunks).sort_values('ts');f['day']=(f.ts-pd.Timedelta(hours=1)).dt.normalize()
        needed=(r.ts-pd.Timedelta(hours=1)).dt.normalize().unique();f=f.loc[f.day.isin(needed)]
        if not f.groupby('day').size().eq(24).all():
            s=site_table.loc[site_table.site_id.eq(site)]
            full=v1.b.features(None,s,fa.loc[fa.wa_station.eq(s.iloc[0].wa_station)],fb.loc[fb.wb_station.eq(s.iloc[0].wb_station)],ix,17)
            enhanced(full)
            aa=aux(fa,'a',s.iloc[0].wa_station,ix);bb=aux(fb,'b',s.iloc[0].wb_station,ix)
            for name in ['a_revision_cloud','a_revision_temp']:full[name]=aa[name].to_numpy()
            for name in ['b_revision_cloud','b_revision_temp']:full[name]=bb[name].to_numpy()
            h=(ix-pd.Timedelta(minutes=30)).hour+(ix-pd.Timedelta(minutes=30)).minute/60
            full['old_hour_sin']=np.sin(2*np.pi*h/24);full['old_hour_cos']=np.cos(2*np.pi*h/24)
            full['day']=(full.ts-pd.Timedelta(hours=1)).dt.normalize()
            full=full.loc[full.day.isin(needed)]
            # Verify regenerated context has the exact cached input semantics.
            aligned=full.set_index('ts').reindex(f.ts)
            assert np.allclose(aligned[FEATURES].to_numpy(),f[FEATURES].to_numpy(),equal_nan=True,rtol=1e-5,atol=1e-5)
            f=full
        # Read context for full forecast days even where labels are not simultaneous.
        counts=f.groupby('day').size();assert counts.eq(24).all(), 'Incomplete forecast sequence'
        z=normalize(f[FEATURES].to_numpy('float32').reshape(-1,24,len(FEATURES)),mean,std)
        with torch.no_grad():p=np.concatenate([stored(torch.from_numpy(z[i:i+512])).numpy() for i in range(0,len(z),512)]).reshape(-1).astype('float64')
        p[f.elev.to_numpy()<=0]=0
        lookup=pd.Series(p,index=f.ts);out=r.copy();out['prediction']=lookup.reindex(r.ts).to_numpy()*r.capacity_kw.to_numpy('float64')
        assert np.isfinite(out.prediction).all();pieces.append(out)
    pd.concat(pieces,ignore_index=True).to_parquet(dest/'predictions.parquet',index=False)

def metrics(d):
    a=d.groupby(['ts','group']).agg(p=('prediction','sum'),y=('energy','sum'),smp=('smp','first'),k=('k','first')).reset_index()
    a['month']=(a.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    def calc(z):
        e=z.p-z.y
        return {'hours':int(z.ts.nunique()),'penalty_krw':float((.1*z.smp*z.k*abs(e)).sum()),'wape_pct':float(100*abs(e).sum()/z.y.sum()),'bias_pct':float(100*e.sum()/z.y.sum())}
    return {'all':calc(a),'monthly':{m:calc(z) for m,z in a.groupby('month')}}

def report():
    result={}
    for fold,base in [('spring',ROOT/'runs/expanded177/predictions.parquet'),('summer',ROOT/'summer_backtest/site_predictions.parquet')]:
        if not (RUN/fold/'predictions.parquet').exists():continue
        a=pd.read_parquet(base).sort_values(['site_id','ts']).reset_index(drop=True)
        b=pd.read_parquet(RUN/fold/'predictions.parquet').sort_values(['site_id','ts']).reset_index(drop=True)
        assert a[['site_id','ts','energy','group']].equals(b[['site_id','ts','energy','group']])
        blend=b.copy();blend.prediction=.5*a.prediction+.5*b.prediction
        result[fold]={'xgb177':metrics(a),'tcn':metrics(b),'fixed_50_50':metrics(blend)}
    OUT.mkdir(exist_ok=True,parents=True);save(OUT/'metrics.json',result)
    lines=['# TCN first candidate','', 'One fixed small noncausal TCN, 31 forecast/solar/static inputs plus missingness masks. No GRU/attention. No external data. CPU local training.', '', 'Inner last-60-day validation selects epochs only (max20, patience4); normalization fit on inner train, then refit normalization/model on all allowed complete site-days. Weighted site CF MAE training; not direct group-loss training. Original groups and XGB evaluation rows retained.', '', '50:50 blend fixed before external results; no ratio search. Existing historical periods were already inspected: retrospective comparison, not untouched test. Summer is 207-site/4905h proxy; spring is full253. Different training origins/cohorts must not be combined into official-year settlement.', '', '|Fold|Model|Penalty KRW|Change vs XGB|','|---|---|---:|---:|']
    for fold,items in result.items():
        base=items['xgb177']['all']['penalty_krw']
        for model,m in items.items():
            value=m['all']['penalty_krw'];lines.append(f'|{fold}|{model}|{value:,.0f}|{100*(value/base-1):+.2f}%|')
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');log(stage='report',results=result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['spring','summer','all'],default='all');args=p.parse_args()
    RUN.mkdir(exist_ok=True)
    # Freeze the experiment before any fit or external scoring.
    protocol=RUN/'protocol.json'
    if not protocol.exists():save(protocol,{'seed':42,'features':FEATURES,'channels':16,'dilations':[1,2,4,8],'kernel':3,'convolution':'symmetric noncausal within already-issued next-day forecasts','loss':'historical SMP*k*capacity weighted site CF MAE','max_epochs':20,'patience':4,'batch':256,'blend':[.5,.5],'promotion':'both external blocks >=1% penalty improvement, no month worse >2%; otherwise exploratory only','source_sha256':sha(__file__),'torch':torch.__version__})
    prepare()
    for fold in (['spring','summer'] if args.fold=='all' else [args.fold]):fit(fold);report()
