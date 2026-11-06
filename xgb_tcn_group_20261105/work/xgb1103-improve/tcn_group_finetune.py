"""Matched short fine-tunes: site loss control vs available-label group loss."""
import sys
if os.environ.get('OIBC_TORCH_LIBS'):
    sys.path.insert(0,os.environ['OIBC_TORCH_LIBS'])
import json,gc,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import tcn_candidate as base
from experiment import ROOT,save,sha,log
from xgb_loss_trial import protected

RUN=ROOT/'tcn_group_finetune1';OUT=Path('outputs/OIBC_발전소개별특성_개선_20261105')
PRETRAIN_ROOT=ROOT/'tcn_candidate1'
EPOCHS=2;LR=.0001;BATCH_DAYS=2
VARIANTS={'site_control':0.,'group_aligned':.8}
FOLDS={'spring':{'cutoff':'2024-12-31 17:00','base':ROOT/'runs/expanded177/predictions.parquet'},'summer':{'cutoff':'2024-05-31 17:00','base':ROOT/'summer_backtest/site_predictions.parquet'}}

def fit(fold,name,mix,predict=True):
    dest=RUN/fold/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'predictions.parquet').exists():return
    previous=PRETRAIN_ROOT/fold;sc=np.load(previous/'scaler.npz');mean=sc['mean'];std=sc['std']
    if not (dest/'model.pt').exists():
        raw=np.load(base.RUN/'data.npz');x=raw['x'];y=raw['y'];w=raw['w'];end=raw['end'];sites=raw['site']
        if end.max()<10**14:end*=1000000
        elif end.max()<10**17:end*=1000
        cutoff=pd.Timestamp(FOLDS[fold]['cutoff'],tz='Asia/Seoul');mask=(end<=cutoff.value)&(w.sum(1)>0);indices=np.flatnonzero(mask);assert end[indices].max()<=cutoff.value
        support=pd.read_csv('work/data/sites.csv').set_index('site_id').current_group.map({'A':0,'B':1});group=np.array([support[s] for s in sites],dtype='int64')
        ordered=indices[np.argsort(end[indices],kind='stable')];ends=end[ordered];bounds=np.r_[0,np.flatnonzero(np.diff(ends))+1,len(ordered)];days=[ordered[a:b] for a,b in zip(bounds[:-1],bounds[1:])]
        mean_positive=float(w[mask][w[mask]>0].mean());mass=float(w[mask].sum(dtype='float64')/mean_positive);average_day_mass=mass/len(days)
        base.seed();model=base.TCN();model.load_state_dict(torch.load(previous/'model.pt',weights_only=True));opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.0001);rng=np.random.default_rng(42);history=[];start=time.time()
        log(stage='group_tcn_start',fold=fold,variant=name,days=len(days),site_days=len(indices),group_mix=mix)
        for epoch in range(1,EPOCHS+1):
            model.train();order=rng.permutation(len(days));site_num=0.;group_num=0.;den_num=0.
            for j in range(0,len(order),BATCH_DAYS):
                chosen=[days[k] for k in order[j:j+BATCH_DAYS]];ix=np.concatenate(chosen);gid=np.concatenate([i*2+group[a] for i,a in enumerate(chosen)])
                bx=torch.from_numpy(base.normalize(x[ix],mean,std));by=torch.from_numpy(y[ix]);bw=torch.from_numpy((w[ix]/mean_positive).astype('float32'))
                opt.zero_grad();p=model(bx);weighted_residual=(p-by)*bw
                grouped=torch.zeros((len(chosen)*2,24),dtype=p.dtype);grouped.index_add_(0,torch.from_numpy(gid),weighted_residual)
                den=len(chosen)*average_day_mass;site_loss=weighted_residual.abs().sum()/den;group_loss=grouped.abs().sum()/den
                loss=(1-mix)*site_loss+mix*group_loss;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                site_num+=float(weighted_residual.detach().abs().sum());group_num+=float(grouped.detach().abs().sum());den_num+=float(bw.sum())
            h={'epoch':epoch,'site_weighted_cf_error':site_num/den_num,'available_group_weighted_cf_error':group_num/den_num};history.append(h);log(stage='group_tcn_fit',fold=fold,variant=name,seconds=time.time()-start,**h)
        test=torch.from_numpy(base.normalize(x[indices[:32]],mean,std));model.eval()
        with torch.no_grad():before=model(test).clone()
        torch.save(model.state_dict(),dest/'model.pt');stored=base.TCN();stored.load_state_dict(torch.load(dest/'model.pt',weights_only=True));stored.eval()
        with torch.no_grad():assert torch.equal(before,stored(test))
        save(dest/'training.json',{'cutoff':str(cutoff),'site_days':len(indices),'calendar_days':len(days),'site_day_count_quantiles':np.quantile([len(v) for v in days],[0,.25,.5,.75,1]).tolist(),'epochs':EPOCHS,'learning_rate':LR,'group_mix':mix,'batch_days':BATCH_DAYS,'loss_denominator':'fixed average training-day total weight times number of whole days in batch; prevents early tiny cohorts getting equal mass with later full cohorts','masked_missing_labels':True,'partial_group_warning':'Group loss only sums plants with valid labels in each hour. It is an observed-subset proxy, not a full253 official group loss where historical labels are missing.','pretrained_sha256':sha(previous/'model.pt'),'scaler_sha256':sha(previous/'scaler.npz'),'model_sha256':sha(dest/'model.pt'),'reload_exact':True,'history':history,'seconds':time.time()-start})
        del raw,x,y,w,end,sites,model,stored,opt;gc.collect()
    if not predict:return
    # Reuse full-day forecast contexts independently regenerated and checked in site trial.
    context=ROOT/'tcn_site_adaptation1'/fold;ex=np.load(context/'external.npz');rawx=ex['x'];select=ex['select'];ref=pd.read_parquet(context/'reference.parquet')
    model=base.TCN();model.load_state_dict(torch.load(dest/'model.pt',weights_only=True));model.eval();output=[]
    with torch.no_grad():
        for i in range(0,len(rawx),512):output.append(model(torch.from_numpy(base.normalize(rawx[i:i+512],mean,std))).numpy())
    p=np.concatenate(output).reshape(-1).astype('float64');p[rawx[:,:,base.FEATURES.index('elev')].reshape(-1)<=0]=0
    result=ref.copy();result.prediction=p[select]*ref.capacity_kw.to_numpy('float64');assert np.isfinite(result.prediction).all();result.to_parquet(dest/'predictions.parquet',index=False);log(stage='group_tcn_complete',fold=fold,variant=name)

def report():
    all_results={};rows=[];monthly=[]
    for fold,cfg in FOLDS.items():
        x=pd.read_parquet(cfg['base']).sort_values(['site_id','ts']).reset_index(drop=True);old=pd.read_parquet(ROOT/'tcn_candidate1'/fold/'predictions.parquet').sort_values(['site_id','ts']).reset_index(drop=True)
        keys=['site_id','ts','energy','capacity_kw','group','smp','k'];assert x[keys].equals(old[keys]);old_blend=x.copy();old_blend.prediction=.5*x.prediction+.5*old.prediction
        models={'old_5050':base.metrics(old_blend)}
        for name in VARIANTS:
            path=RUN/fold/name/'predictions.parquet'
            if path.exists():
                p=pd.read_parquet(path).sort_values(['site_id','ts']).reset_index(drop=True);assert x[keys].equals(p[keys]);blend=x.copy();blend.prediction=.5*x.prediction+.5*p.prediction;models[name]=base.metrics(blend);models[name+'_TCN_alone']=base.metrics(p)
        all_results[fold]=models;baseline=models['old_5050']['all']['penalty_krw']
        for name,v in models.items():
            cost=v['all']['penalty_krw'];rows.append({'fold':fold,'model':name,'penalty':cost,'gain_vs_old5050':baseline-cost,'reduction_pct':100*(baseline-cost)/baseline})
            for month,z in v['monthly'].items():monthly.append({'fold':fold,'model':name,'month':month,**z})
    decisions={}
    for name in VARIANTS:
        complete=all(name in all_results[f] for f in FOLDS)
        passed=complete and all(v[name]['all']['penalty_krw']<=.99*v['old_5050']['all']['penalty_krw'] for v in all_results.values()) and all(v[name]['monthly'][m]['penalty_krw']<=1.02*v['old_5050']['monthly'][m]['penalty_krw'] for v in all_results.values() for m in v[name]['monthly'])
        decisions[name]={'complete':complete,'promotion_passed':bool(passed)}
    result={'folds':all_results,'decisions':decisions,'historical_periods_previously_inspected':True};save(OUT/'그룹손실_추가학습_검증.json',result);pd.DataFrame(rows).to_csv(OUT/'그룹손실_추가학습_비교.csv',index=False,encoding='utf-8-sig');pd.DataFrame(monthly).to_csv(OUT/'그룹손실_추가학습_월별.csv',index=False,encoding='utf-8-sig');log(stage='group_tcn_report',rows=rows,decisions=decisions)

if __name__=='__main__':
    RUN.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (RUN/'protocol.json').exists():save(RUN/'protocol.json',{'variants':VARIANTS,'epochs':EPOCHS,'learning_rate':LR,'batch_days':BATCH_DAYS,'pretrained':'original frozen TCN at each origin; no site-gain model','group_loss':'absolute sum of capacity*SMP*k weighted CF residuals within each original group/hour, only valid labels. Partial historical cohorts are masked, never zero-filled as truth.','control':'identical warm start, batch order and2 epochs with original site loss','blend':[.5,.5],'promotion':'both periods>=1% improvement, no month worse>2%','protected':protected(),'source_sha256':sha(__file__),'history_previously_inspected':True})
    protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'));assert protocol['protected']==protected()
    for fold in ['spring','summer']:
        for name,mix in VARIANTS.items():fit(fold,name,mix);report()
    assert protocol['protected']==protected();save(OUT/'그룹손실_보존검사.json',{'unchanged':True,'sha256':protected()})
