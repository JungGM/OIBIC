"""One frozen, shrunk per-site TCN calibration rule on chronological OOF blocks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment import ROOT,save,sha,log
from xgb_loss_trial import metric as original_metric,protected

RUN=ROOT/'tcn_site_calibration1';OUT=Path('outputs/OIBC_발전소개별특성_개선_20261105')
PRIOR_HOURS=240.;LOW=.8;HIGH=1.2;MIN_PRED_CF=.02
FOLDS={'spring':{'base':ROOT/'runs/expanded177/predictions.parquet','tcn':ROOT/'tcn_candidate1/spring/predictions.parquet','cutoff':'2025-03-31 17:00','start':'2025-04-01 01:00','end':'2025-06-01 00:00'},'summer':{'base':ROOT/'summer_backtest/site_predictions.parquet','tcn':ROOT/'tcn_candidate1/summer/predictions.parquet','cutoff':'2024-08-31 17:00','start':'2024-09-01 01:00','end':'2025-01-01 00:00'}}
def ts(s):return pd.Timestamp(s,tz='Asia/Seoul')
def metric(d):
    # Audit-only slices may have no Jan-Mar rows; do not report a0/0 metric.
    with np.errstate(divide='ignore',invalid='ignore'):out=original_metric(d)
    return {k:v for k,v in out.items() if k=='monthly' or v.get('hours',0)>0}
def fit_scales(d,cutoff):
    d=d[d.ts<=cutoff];out={}
    for site,z in d.groupby('site_id'):
        ok=(z.prediction>MIN_PRED_CF*z.capacity_kw)&(z.energy>=0)&(z.smp>0)
        p=z.loc[ok,'prediction'].to_numpy('float64');y=z.loc[ok,'energy'].to_numpy('float64');w=(z.loc[ok,'smp']*z.loc[ok,'k']).to_numpy('float64')*p;n=len(p)
        if n:
            ratio=y/p;idx=np.argsort(ratio,kind='stable');median=float(ratio[idx][np.searchsorted(np.cumsum(w[idx]),w.sum()/2)])
        else:median=1.
        reliability=n/(n+PRIOR_HOURS);gain=1+reliability*(np.clip(median,LOW,HIGH)-1)
        out[site]={'gain':float(gain),'valid_hours':n,'reliability':float(reliability),'raw_median_ratio':median}
    return out

def main():
    RUN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True,parents=True)
    if (RUN/'results.json').exists():raise ValueError('Preserve existing calibration results')
    proto={'prior_hours':PRIOR_HOURS,'raw_scale_bounds':[LOW,HIGH],'minimum_predicted_cf':MIN_PRED_CF,'formula':'scale=1+n/(n+240)*(clip(weighted_median(actual/pred),0.8,1.2)-1); median weights=SMP*k*pred','all_sites_same_rule':True,'TCN_only_calibrated':True,'blend':[.5,.5],'forecast_cap':'clip calibrated TCN kWh to [0,1.2*capacity]','promotion':'both future audit blocks >=1% less penalty, no month worse>2%','protected':protected(),'source_sha256':sha(__file__),'historical_periods_previously_inspected':True}
    save(RUN/'protocol.json',proto);results={};rows=[];monthly=[]
    for fold,cfg in FOLDS.items():
        cutoff=ts(cfg['cutoff']);cal=pd.read_parquet(cfg['tcn'],filters=[('ts','<=',cutoff)]);scales=fit_scales(cal,cutoff)
        save(RUN/f'{fold}_scales_frozen.json',{'cutoff':str(cutoff),'calibration_hours':int(cal.ts.nunique()),'scales':scales})
        filters=[('ts','>=',ts(cfg['start'])),('ts','<=',ts(cfg['end']))]
        x=pd.read_parquet(cfg['base'],filters=filters).sort_values(['site_id','ts']).reset_index(drop=True);t=pd.read_parquet(cfg['tcn'],filters=filters).sort_values(['site_id','ts']).reset_index(drop=True)
        keys=['site_id','ts','energy','capacity_kw','group','smp','k'];assert x[keys].equals(t[keys])
        poison=t.copy();poison.energy=999999;assert fit_scales(pd.concat([cal,poison],ignore_index=True),cutoff)==scales
        gain=t.site_id.map({s:v['gain'] for s,v in scales.items()}).fillna(1).to_numpy('float64')
        modified=t.copy();modified.prediction=np.minimum(np.maximum(t.prediction.to_numpy('float64')*gain,0),1.2*t.capacity_kw.to_numpy('float64'))
        old=x.copy();old.prediction=.5*x.prediction+.5*t.prediction;new=x.copy();new.prediction=.5*x.prediction+.5*modified.prediction
        assert np.isfinite(new.prediction).all();new.to_parquet(RUN/f'{fold}_predictions.parquet',index=False)
        a=metric(old);b=metric(new);gain_cost=a['whole']['penalty']-b['whole']['penalty'];pct=100*gain_cost/a['whole']['penalty'];safe=True
        for m,v in b['monthly'].items():
            base=a['monthly'][m]['penalty'];delta=100*(base-v['penalty'])/base;safe &= delta>=-2
            monthly.append({'fold':fold,'month':m,'old5050_penalty':base,'calibrated_penalty':v['penalty'],'reduction_pct':delta})
        results[fold]={'old5050':a,'calibrated5050':b,'gain':gain_cost,'reduction_pct':pct,'months_safe':bool(safe),'future_label_poisoning_check_passed':True,'start':str(x.ts.min()),'end':str(x.ts.max())};rows.append({'fold':fold,'old5050_penalty':a['whole']['penalty'],'calibrated_penalty':b['whole']['penalty'],'gain':gain_cost,'reduction_pct':pct})
        log(stage='site_calibration_result',**rows[-1])
    passed=all(v['reduction_pct']>=1 and v['months_safe'] for v in results.values());assert proto['protected']==protected()
    final={'folds':results,'promotion_passed':bool(passed),'protected_inputs_unchanged':True,'warning':'Retrospective previously inspected history. Different cohorts and origins. Future audit performance only; do not mix fit-period scores into independent evaluation. Production use must keep the same frozen TCN base or separately validate refit calibration transfer.'}
    save(RUN/'results.json',final);save(OUT/'과거예측_개별배율_검증.json',final);pd.DataFrame(rows).to_csv(OUT/'과거예측_개별배율_비교.csv',index=False,encoding='utf-8-sig');pd.DataFrame(monthly).to_csv(OUT/'과거예측_개별배율_월별.csv',index=False,encoding='utf-8-sig')
    log(stage='site_calibration_complete',promotion_passed=passed)

if __name__=='__main__':main()
