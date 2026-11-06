"""Independent TCN/BiGRU/attention with chronological simplex linear stacking."""
import os,sys
if os.environ.get('OIBC_TORCH_LIBS'):
    sys.path.insert(0,os.environ['OIBC_TORCH_LIBS'])
import gc,json,time,copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize
import tcn_candidate as base
from experiment import ROOT,save,log,sha,aux

RUN=ROOT/'deep_linear_ensemble1';OUT=Path('outputs/OIBC_딥러닝3종_선형앙상블_20261104')
NAMES=['TCN','BiGRU','Attention'];F=base.FEATURES

class BiGRU(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(2*len(F),24);self.gru=nn.GRU(24,16,batch_first=True,bidirectional=True);self.out=nn.Linear(32,1)
    def forward(self,x):
        z=torch.nn.functional.gelu(self.proj(x.transpose(1,2)));z,_=self.gru(z)
        return 1.2*torch.sigmoid(self.out(z).squeeze(-1))

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(2*len(F),32)
        pos=torch.arange(24)[:,None];freq=torch.exp(torch.arange(0,32,2)*(-np.log(10000)/32));p=torch.zeros(24,32)
        p[:,0::2]=torch.sin(pos*freq);p[:,1::2]=torch.cos(pos*freq);self.register_buffer('pos',p)
        self.layers=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,.1,batch_first=True,activation='gelu'),num_layers=2,enable_nested_tensor=False);self.out=nn.Linear(32,1)
    def forward(self,x):
        z=self.proj(x.transpose(1,2))+self.pos
        return 1.2*torch.sigmoid(self.out(self.layers(z)).squeeze(-1))

FACTORY={'TCN':base.TCN,'BiGRU':BiGRU,'Attention':Attention}
def predict(model,x):
    model.eval()
    with torch.no_grad():return np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy() for i in range(0,len(x),256)])

def external_inputs(fold):
    dest=RUN/fold;target=dest/'external.npz'
    if target.exists():return
    path=ROOT/('runs/expanded177/predictions.parquet' if fold=='spring' else 'summer_backtest/site_predictions.parquet')
    ref=pd.read_parquet(path).sort_values(['site_id','ts']).reset_index(drop=True);ref.to_parquet(dest/'reference.parquet',index=False)
    os.environ['OIBC_DATA']=str(Path('work/data').resolve())
    from run_experiment import v1,enhanced
    sites=pd.read_csv('work/data/sites.csv');first=(ref.ts.min()-pd.Timedelta(hours=1)).normalize()+pd.Timedelta(hours=1);last=(ref.ts.max()-pd.Timedelta(hours=1)).normalize()+pd.Timedelta(days=1)
    ix=pd.date_range(first,last,freq='h')
    fa=pd.read_parquet('work/data/train/forecast_wa.parquet',filters=[('ts','>=',first),('ts','<=',last)]);fb=pd.read_parquet('work/data/train/forecast_wb.parquet',filters=[('ts','>=',first),('ts','<=',last)])
    xs=[];select=[];offset=0
    for site,r in ref.groupby('site_id',sort=True):
        chunks=[pd.read_parquet(ROOT/'cache'/str(year)/f'{site}.parquet',columns=F+['ts']) for year in sorted(r.ts.dt.year.unique()) if (ROOT/'cache'/str(year)/f'{site}.parquet').exists()]
        f=pd.concat(chunks).sort_values('ts');f['day']=(f.ts-pd.Timedelta(hours=1)).dt.normalize();needed=(r.ts-pd.Timedelta(hours=1)).dt.normalize().unique();f=f.loc[f.day.isin(needed)]
        if not f.groupby('day').size().eq(24).all():
            s=sites.loc[sites.site_id.eq(site)];full=v1.b.features(None,s,fa.loc[fa.wa_station.eq(s.iloc[0].wa_station)],fb.loc[fb.wb_station.eq(s.iloc[0].wb_station)],ix,17);enhanced(full)
            for source,weather,station in [('a',fa,s.iloc[0].wa_station),('b',fb,s.iloc[0].wb_station)]:
                extra=aux(weather,source,station,ix)
                for v in ['cloud','temp']:full[f'{source}_revision_{v}']=extra[f'{source}_revision_{v}'].to_numpy()
            mid=ix-pd.Timedelta(minutes=30);h=mid.hour+mid.minute/60
            full['old_hour_sin']=np.sin(2*np.pi*h/24);full['old_hour_cos']=np.cos(2*np.pi*h/24);full['day']=(full.ts-pd.Timedelta(hours=1)).dt.normalize();full=full.loc[full.day.isin(needed)]
            assert np.allclose(full.set_index('ts').reindex(f.ts)[F],f[F],equal_nan=True,rtol=1e-5,atol=1e-5);f=full
        assert f.groupby('day').size().eq(24).all()
        indices=pd.Index(f.ts).get_indexer(r.ts);assert (indices>=0).all()
        xs.append(f[F].to_numpy('float32').reshape(-1,24,len(F)));select.extend((offset+indices).tolist());offset+=len(f)
    np.savez(target,x=np.concatenate(xs),select=np.array(select))

def fit_fold(fold):
    dest=RUN/fold;dest.mkdir(exist_ok=True)
    cutoff=pd.Timestamp('2024-12-31 17:00' if fold=='spring' else '2024-05-31 17:00',tz='Asia/Seoul')
    end_train=cutoff-pd.Timedelta(days=60);end_select=cutoff-pd.Timedelta(days=30)
    z=np.load(base.RUN/'data.npz');x=z['x'];y=z['y'];w=z['w'];end=z['end']
    if end.max()<10**14:end=end*1000000
    elif end.max()<10**17:end=end*1000
    all_train=(end<=cutoff.value)&(w.sum(1)>0);a=all_train&(end<=end_train.value);b=all_train&(end>end_train.value)&(end<=end_select.value);c=all_train&(end>end_select.value)
    assert a.sum() and b.sum() and c.sum() and end[all_train].max()<=cutoff.value
    mean,std=base.scaling(x[a]);np.savez(dest/'inner_scaler.npz',mean=mean,std=std)
    xa=base.normalize(x[a],mean,std);xb=base.normalize(x[b],mean,std);xc=base.normalize(x[c],mean,std)
    trainw=w[a]/w[a][w[a]>0].mean();valw=w[b]/w[a][w[a]>0].mean()
    save(dest/'split.json',{'cutoff':str(cutoff),'base_train_end':str(end_train),'epoch_validation_end':str(end_select),'calibration_end':str(cutoff),'days':{'base_train':int(a.sum()),'epoch_validation':int(b.sum()),'calibration':int(c.sum()),'full_refit':int(all_train.sum())}})
    for name in NAMES:
        md=dest/name;md.mkdir(exist_ok=True)
        if (md/'calibration.npy').exists():continue
        base.seed();model=FACTORY[name]();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        dl=base.loader(xa,y[a],trainw,True);vd=base.loader(xb,y[b],valw,False);best=float('inf');epoch_best=0;state=None;hist=[];start=time.time()
        for ep in range(1,13):
            loss=base.train_epoch(model,dl,opt);v=base.val_loss(model,vd);hist.append({'epoch':ep,'train':loss,'validation':v});log(stage='three_inner',fold=fold,model=name,epoch=ep,validation=v)
            if v<best-1e-5:best=v;epoch_best=ep;state=copy.deepcopy(model.state_dict())
            if ep-epoch_best>=3:break
        model.load_state_dict(state);torch.save(state,md/'inner_model.pt');p=predict(model,xc);p[x[c,: ,F.index('elev')]<=0]=0;np.save(md/'calibration.npy',p)
        save(md/'selection.json',{'best_epoch':epoch_best,'history':hist,'seconds':time.time()-start})
        del model,opt,dl,vd,state;gc.collect()
    # Regression data are separate from epoch-selection data; sample weighting is historical only.
    p=np.stack([np.load(dest/n/'calibration.npy').reshape(-1) for n in NAMES],axis=1).astype('float64');target=y[c].reshape(-1).astype('float64');weights=w[c].reshape(-1).astype('float64');ok=weights>0;p=p[ok];target=target[ok];weights=weights[ok];weights/=weights.mean()
    def loss(v):return float(np.average((p@v-target)**2,weights=weights))
    def jac(v):return 2*(p.T@(weights*(p@v-target)))/weights.sum()
    reg=minimize(loss,np.full(3,1/3),jac=jac,method='SLSQP',bounds=[(0,1)]*3,constraints=[{'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(3)}],options={'ftol':1e-12,'maxiter':1000})
    assert reg.success and np.min(reg.x)>=-1e-8 and abs(reg.x.sum()-1)<1e-8
    coef=np.maximum(reg.x,0);coef/=coef.sum();save(dest/'weights.json',{'weights':dict(zip(NAMES,coef.tolist())),'intercept':0,'objective':'historical SMP*k*capacity weighted site CF squared-error simplex regression','calibration_rows':int(ok.sum()),'success':bool(reg.success),'loss':loss(coef),'external_labels_used':False});log(stage='linear_weights',fold=fold,weights=coef.tolist())
    del p,target,weights,xa,xb,xc,trainw,valw;gc.collect()
    mean,std=base.scaling(x[all_train]);np.savez(dest/'final_scaler.npz',mean=mean,std=std);xx=base.normalize(x[all_train],mean,std);yy=y[all_train].copy();ww=w[all_train]/w[all_train][w[all_train]>0].mean()
    del x,y,w,z;gc.collect();external_inputs(fold)
    ext=np.load(dest/'external.npz');raw=ext['x'];ex=base.normalize(raw,mean,std);night=raw[:,:,F.index('elev')].reshape(-1)<=0;select=ext['select'];ref=pd.read_parquet(dest/'reference.parquet')
    for name in NAMES:
        md=dest/name
        if (md/'predictions.parquet').exists():continue
        epochs=json.loads((md/'selection.json').read_text())['best_epoch'];base.seed();model=FACTORY[name]();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);dl=base.loader(xx,yy,ww,True)
        for ep in range(1,epochs+1):
            loss=base.train_epoch(model,dl,opt);log(stage='three_refit',fold=fold,model=name,epoch=ep,loss=loss)
        model.eval();check=torch.from_numpy(xx[:16]);before=model(check).detach();torch.save(model.state_dict(),md/'model.pt');stored=FACTORY[name]();stored.load_state_dict(torch.load(md/'model.pt',weights_only=True));stored.eval();assert torch.equal(before,stored(check).detach())
        out=ref.copy();pred=predict(stored,ex).reshape(-1).astype('float64');pred[night]=0;out['prediction']=pred[select]*ref.capacity_kw.to_numpy('float64');assert np.isfinite(out.prediction).all();out.to_parquet(md/'predictions.parquet',index=False)
        save(md/'final.json',{'epochs':epochs,'reload_exact':True,'model_sha256':sha(md/'model.pt')});del model,stored,opt,dl;gc.collect()
    combined=ref.copy();combined['prediction']=sum(coef[j]*pd.read_parquet(dest/n/'predictions.parquet').prediction.to_numpy() for j,n in enumerate(NAMES));combined.to_parquet(dest/'ensemble_predictions.parquet',index=False)

def report():
    results={};rows=[];monthrows=[]
    for fold in ['spring','summer']:
        d=RUN/fold
        if not (d/'ensemble_predictions.parquet').exists():continue
        ref=pd.read_parquet(d/'reference.parquet');models={'XGBoost177_reference':ref,'previous_TCN':pd.read_parquet(base.RUN/fold/'predictions.parquet')}
        models.update({n:pd.read_parquet(d/n/'predictions.parquet') for n in NAMES});models['linear_ensemble']=pd.read_parquet(d/'ensemble_predictions.parquet')
        results[fold]={'weights':json.loads((d/'weights.json').read_text()),'models':{}}
        basew=base.metrics(ref)['all']['penalty_krw']
        for name,p in models.items():
            assert p[['site_id','ts','energy','group']].reset_index(drop=True).equals(ref[['site_id','ts','energy','group']].reset_index(drop=True)) or name=='previous_TCN'
            m=base.metrics(p);results[fold]['models'][name]=m;v=m['all']['penalty_krw'];rows.append({'fold':fold,'model':name,'penalty_krw':v,'gain_vs_xgb':basew-v,'reduction_pct':100*(basew-v)/basew})
            for month,r in m['monthly'].items():monthrows.append({'fold':fold,'model':name,'month':month,**r})
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'results.json',results);pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False);pd.DataFrame(monthrows).to_csv(OUT/'monthly.csv',index=False)
    lines=['# TCN + BiGRU + Attention: linear-weight ensemble','','Three independent next-day forecasters; NOT the paper single-network fusion. XGBoost is reference only.','', 'First past block: fit; following30 days: epoch selection; last30 days before cutoff: simplex weighted linear regression; external evaluation is later. Base models then refit on all allowed complete days. This transfer from earlier models to refitted models can alter calibration. Weights minimize weighted SITE CF squared error, not official group L1. No external-period weight search. Historical intervals were already inspected. Summer207-site proxy and spring253-site full-fleet results must not be combined as official annual settlement.','','|fold|model|penalty|gain vs XGB|reduction|','|---|---|---:|---:|---:|']
    lines += [f"|{r['fold']}|{r['model']}|{r['penalty_krw']:,.0f}|{r['gain_vs_xgb']:+,.0f}|{r['reduction_pct']:+.2f}%|" for r in rows]
    for fold,v in results.items():lines += ['',fold+' weights: '+json.dumps(v['weights']['weights'])]
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');log(stage='three_report',rows=rows)

if __name__=='__main__':
    RUN.mkdir(exist_ok=True)
    if not (RUN/'protocol.json').exists():save(RUN/'protocol.json',{'features':F,'models':NAMES,'seed':42,'max_epochs':12,'patience':3,'train_cutoff_days_before_outer':60,'selection_end_days_before_outer':30,'meta':'nonnegative sum-to-one, zero intercept weighted least squares; site CF target','source_sha256':sha(__file__),'outer_results_do_not_change_weights':True})
    for fold in ['spring','summer']:fit_fold(fold);report()
