"""One fixed site-aware TCN candidate; support-shrunk gain learned jointly."""
import os,sys
if os.environ.get('OIBC_TORCH_LIBS'):
    sys.path.insert(0,os.environ['OIBC_TORCH_LIBS'])
import json,gc,time,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import tcn_candidate as base
import deep_linear_ensemble as context
from experiment import ROOT,save,sha,log
from xgb_loss_trial import protected

RUN=ROOT/'tcn_site_adaptation1';OUT=Path('outputs/OIBC_발전소개별특성_개선_20261105')
F=base.FEATURES;PRIOR_DAYS=60.;MAX_LOG_GAIN=.5;REGULARIZATION=.005
SITES=pd.read_csv('work/data/sites.csv').site_id.sort_values().tolist();SITE_TO_ID={s:i+1 for i,s in enumerate(SITES)}
FOLDS={'spring':{'cutoff':'2024-12-31 17:00','base':ROOT/'runs/expanded177/predictions.parquet'},'summer':{'cutoff':'2024-05-31 17:00','base':ROOT/'summer_backtest/site_predictions.parquet'}}

class SiteTCN(nn.Module):
    def __init__(self,reliability):
        super().__init__();self.base=base.TCN();self.site_gain=nn.Parameter(torch.zeros(len(SITES)+1));self.register_buffer('reliability',torch.as_tensor(reliability,dtype=torch.float32))
    def gain(self):return torch.exp(MAX_LOG_GAIN*torch.tanh(self.site_gain)*self.reliability)
    def forward(self,x,site):return torch.clamp(self.base(x)*self.gain()[site,None],min=0,max=1.2)

def fit(fold):
    dest=RUN/fold;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'predictions.parquet').exists():return
    cutoff=pd.Timestamp(FOLDS[fold]['cutoff'],tz='Asia/Seoul');previous=ROOT/'tcn_candidate1'/fold
    manifest=json.loads((previous/'training.json').read_text(encoding='utf-8'));epochs=manifest['best_epoch'];assert pd.Timestamp(manifest['cutoff'])==cutoff
    scaler=np.load(previous/'scaler.npz');mean=scaler['mean'];std=scaler['std']
    if not (dest/'model.pt').exists():
        raw=np.load(base.RUN/'data.npz');x=raw['x'];y=raw['y'];w=raw['w'];end=raw['end'];ids=np.array([SITE_TO_ID[s] for s in raw['site']],dtype='int64')
        if end.max()<10**14:end*=1000000
        elif end.max()<10**17:end*=1000
        train=(end<=cutoff.value)&(w.sum(1)>0)
        assert train.sum()==manifest['training_days'] and end[train].max()<=cutoff.value
        poisoned=w.copy();poisoned[end>cutoff.value]=0;assert np.array_equal(train,(end<=cutoff.value)&(poisoned.sum(1)>0));del poisoned
        count=np.bincount(ids[train],minlength=len(SITES)+1).astype('float32');rel=count/(count+PRIOR_DAYS);rel[0]=0
        base.seed();model=SiteTCN(rel)
        # Extra zero-initialized site parameters do not change the initial base network.
        base.seed();reference=base.TCN()
        assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.base.state_dict().items())
        check=base.normalize(x[train][:16],mean,std);model.eval();reference.eval()
        with torch.no_grad():assert torch.equal(model(torch.from_numpy(check),torch.from_numpy(ids[train][:16])),reference(torch.from_numpy(check)))
        del reference,check
        # Normalize per batch using the exact original function to keep RAM bounded.
        xx=x[train];yy=y[train].copy();ww=w[train].copy();ww/=ww[ww>0].mean();ii=ids[train].copy()
        last=str(pd.Timestamp(int(end[train].max()),tz='UTC').tz_convert('Asia/Seoul'));del raw,x,y,w,ids,end;gc.collect()
        dl=DataLoader(TensorDataset(torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(ww),torch.from_numpy(ii)),batch_size=256,shuffle=True,num_workers=0)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];start=time.time()
        log(stage='site_tcn_start',fold=fold,days=int(train.sum()),epochs=epochs,known_sites=int((count>0).sum()))
        for epoch in range(1,epochs+1):
            model.train();num=0.;den=0.
            for bx,by,bw,bi in dl:
                features=torch.from_numpy(base.normalize(bx.numpy(),mean,std));opt.zero_grad();p=model(features,bi)
                data_loss=((p-by).abs()*bw).sum()/bw.sum().clamp_min(1)
                loss=data_loss+REGULARIZATION*model.site_gain.square().mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                mass=float(bw.sum());num+=float(data_loss.detach())*mass;den+=mass
            history.append({'epoch':epoch,'weighted_cf_mae':num/den});log(stage='site_tcn_fit',fold=fold,epoch=epoch,weighted_cf_mae=num/den,seconds=time.time()-start)
        model.eval();test=base.normalize(xx[:32],mean,std);tid=torch.from_numpy(ii[:32]);tx=torch.from_numpy(test)
        with torch.no_grad():before=model(tx,tid).clone()
        torch.save(model.state_dict(),dest/'model.pt');stored=SiteTCN(rel);stored.load_state_dict(torch.load(dest/'model.pt',weights_only=True));stored.eval()
        with torch.no_grad():assert torch.equal(before,stored(tx,tid))
        gains=model.gain().detach().numpy();pd.DataFrame({'site_id':SITES,'training_site_days':count[1:],'reliability':rel[1:],'gain':gains[1:]}).to_csv(dest/'site_gains.csv',index=False)
        save(dest/'training.json',{'cutoff':str(cutoff),'last_complete_training_day_end':last,'training_days':int(train.sum()),'epochs_fixed_from_baseline':epochs,'history':history,'prior_days':PRIOR_DAYS,'maximum_log_gain':MAX_LOG_GAIN,'site_parameter_l2':REGULARIZATION,'features':F,'site_to_id':SITE_TO_ID,'future_label_mask_check_passed':True,'initial_base_network_equal':True,'reload_exact':True,'scaler_source_sha256':sha(previous/'scaler.npz'),'model_sha256':sha(dest/'model.pt'),'seconds':time.time()-start})
        del model,stored,opt,dl,xx,yy,ww,ii;gc.collect()
    # Use the independently checked full-day forecast reconstruction helper in a new folder.
    context.RUN=RUN;context.external_inputs(fold)
    raw=np.load(dest/'external.npz');ex=raw['x'];select=raw['select'];ref=pd.read_parquet(dest/'reference.parquet')
    state=torch.load(dest/'model.pt',weights_only=True);model=SiteTCN(state['reliability']);model.load_state_dict(state);model.eval();pred=[]
    # Each complete-day sequence belongs to one site's run. Derive the site index from row mapping.
    sequence_ids=np.full(len(ex),-1,dtype='int64')
    for site,z in ref.groupby('site_id',sort=True):
        seq=np.unique(select[z.index.to_numpy()]//24);known=sequence_ids[seq];assert ((known==-1)|(known==SITE_TO_ID[site])).all();sequence_ids[seq]=SITE_TO_ID[site]
    assert (sequence_ids>=0).all()
    with torch.no_grad():
        for i in range(0,len(ex),512):
            x=torch.from_numpy(base.normalize(ex[i:i+512],mean,std));sid=torch.from_numpy(sequence_ids[i:i+512]);pred.append(model(x,sid).numpy())
    p=np.concatenate(pred).reshape(-1).astype('float64');p[ex[:,:,F.index('elev')].reshape(-1)<=0]=0
    result=ref.copy();result['prediction']=p[select]*ref.capacity_kw.to_numpy('float64');assert np.isfinite(result.prediction).all() and (result.prediction>=0).all();result.to_parquet(dest/'predictions.parquet',index=False)
    log(stage='site_tcn_complete',fold=fold)

def report():
    results={};rows=[];months=[]
    for fold,cfg in FOLDS.items():
        path=RUN/fold/'predictions.parquet'
        if not path.exists():continue
        x=pd.read_parquet(cfg['base']).sort_values(['site_id','ts']).reset_index(drop=True);old=pd.read_parquet(ROOT/'tcn_candidate1'/fold/'predictions.parquet').sort_values(['site_id','ts']).reset_index(drop=True);new=pd.read_parquet(path).sort_values(['site_id','ts']).reset_index(drop=True)
        keys=['site_id','ts','energy','capacity_kw','group','smp','k'];assert x[keys].equals(old[keys]) and x[keys].equals(new[keys])
        models={'xgb':x,'old_tcn':old,'site_tcn':new}
        for name,z in [('old_5050',old),('site_5050',new)]:
            a=x.copy();a.prediction=.5*x.prediction+.5*z.prediction;models[name]=a
        metrics={name:base.metrics(z) for name,z in models.items()};results[fold]=metrics
        b=metrics['old_5050']['all']['penalty_krw']
        for name,v in metrics.items():
            cost=v['all']['penalty_krw'];rows.append({'fold':fold,'model':name,'penalty':cost,'gain_vs_old5050':b-cost,'reduction_pct':100*(b-cost)/b})
            for month,s in v['monthly'].items():months.append({'fold':fold,'model':name,'month':month,**s})
    promotion=len(results)==2 and all(v['site_5050']['all']['penalty_krw']<=.99*v['old_5050']['all']['penalty_krw'] for v in results.values()) and all(v['site_5050']['monthly'][m]['penalty_krw']<=1.02*v['old_5050']['monthly'][m]['penalty_krw'] for v in results.values() for m in v['site_5050']['monthly'])
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'모델검증.json',{'folds':results,'promotion_passed':bool(promotion),'both_folds_completed':len(results)==2,'site_gain_hyperparameters_fixed_before_fit':True});pd.DataFrame(rows).to_csv(OUT/'모델별_검증.csv',index=False,encoding='utf-8-sig');pd.DataFrame(months).to_csv(OUT/'월별_검증.csv',index=False,encoding='utf-8-sig')
    log(stage='site_tcn_report',rows=rows,promotion_passed=promotion)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['spring','summer','all'],default='all');a=p.parse_args();RUN.mkdir(exist_ok=True)
    if not (RUN/'protocol.json').exists():save(RUN/'protocol.json',{'candidate':'one site-specific multiplicative gain per plant, learned jointly with original TCN; gain exp(0.5*tanh(parameter)*n/(n+60)); rare/unseen sites shrink to1','prior_days':PRIOR_DAYS,'max_log_gain':MAX_LOG_GAIN,'regularization':REGULARIZATION,'training':'same baseline epoch count, scalers, input31, target, raw SMP weights, batch256, AdamW0.001, seed42','blend':'original XGB50% + siteTCN50%','promotion':'both periods>=1% improvement vs old5050 and no month worse>2%','sources':{'script':sha(__file__),'tcn_candidate.py':sha(ROOT/'tcn_candidate.py'),'deep_linear_ensemble.py':sha(ROOT/'deep_linear_ensemble.py')},'protected':protected(),'historical_periods_previously_inspected':True})
    protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'));assert protocol['protected']==protected()
    for fold in (['spring','summer'] if a.fold=='all' else [a.fold]):fit(fold);report()
    assert protocol['protected']==protected();save(OUT/'기존파일_보존검사.json',{'unchanged':True,'sha256':protected()})
