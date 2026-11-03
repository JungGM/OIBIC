"""Independent source, time boundary, score and submission identity audit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as p
import improvement_experiment as ex
from validate_submission import validate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path)
    args=ap.parse_args()
    expected=ex.read(ex.OLD/'protocol_before_training.json')['source_sha256']
    actual=p.source_manifest(args.data)
    if actual!=expected:raise ValueError('Original data changed')
    receipt=ex.read(ex.HERE/'제출/공식제출결과_20261102.json')
    for name,record in receipt['files'].items():
        if p.sha256(ex.HERE/'제출'/name)!=record['sha256']:
            raise ValueError('First official submission changed')
    model_records=[]
    v4root=ex.HERE/'_local/improvement_v4_seasonal'
    v5root=ex.HERE/'_local/improvement_v5_group_loss'
    v6root=ex.HERE/'_local/improvement_v6_compact_xgb'
    v7root=ex.HERE/'_local/improvement_v7_team_pair'
    v8root=ex.HERE/'_local/improvement_v8_components'
    for root in [ex.RUN,v4root,v5root,v6root,v7root,v8root]:
        for f in root.glob('*/*metadata.json'):
            m=ex.read(f);origin=pd.Timestamp(m['origin']);cutoff=pd.Timestamp(m['cutoff'])
            if cutoff!=p.issue_deadline(origin) or pd.Timestamp(m['last_target'])>cutoff:
                raise ValueError('Invalid estimator time boundary')
            if f.name=='lightgbm_metadata.json':model_path=f.parent/'lightgbm.txt'
            elif f.name=='catboost_metadata.json':model_path=f.parent/'catboost.cbm'
            else:model_path=next((f.parent/n for n in ['compact_xgb.ubj','group_lgb.txt','cost_lgb.txt'] if (f.parent/n).exists()))
            if 'model_sha256' in m and p.sha256(model_path)!=m['model_sha256']:
                raise ValueError('Model identity changed')
            if 'saved_model_max_abs_difference' in m and m['saved_model_max_abs_difference']!=0:
                raise ValueError('Saved model reproduction failed')
            model_records.append(dict(experiment=root.name,directory=f.parent.name,origin=str(origin),cutoff=str(cutoff),
                                      model_file=model_path.name,model_sha256=p.sha256(model_path)))
    g=pd.read_parquet(ex.RUN/'evaluation_frame.parquet')
    raw=pd.read_parquet(args.data/'train/generation.parquet',columns=['site_id','ts','energy'])
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    raw=raw[raw.ts.isin(g.ts.unique())].copy()
    raw['group']=raw.site_id.map(comp)
    truth=raw.groupby(['ts','group']).energy.sum()
    expected_truth=truth.reindex(pd.MultiIndex.from_frame(g[['ts','group']])).to_numpy()
    if not np.allclose(g.energy,expected_truth,rtol=0,atol=1e-9):
        raise ValueError('Group truth differs from original labels')
    score_checks=[]
    eval3=ex.read(ex.RUN/'evaluation.json')
    for row in eval3['months']:
        z=g[(g.ts-pd.Timedelta(hours=1)).dt.month==row['month']]
        penalty=float(.1*np.dot(z.weight,np.abs(z.base-z.energy)))
        if not np.isclose(penalty,row['baseline_penalty_krw'],rtol=1e-12):raise ValueError('Score mismatch')
        score_checks.append(dict(month=row['month'],hours=z.ts.nunique(),penalty=penalty))
    result=dict(passed=True,source_hashes_unchanged=True,first_official_submission_unchanged=True,
                original_truth_equal=True,score_checks=score_checks,models=model_records,
                code_sha256={n:p.sha256(ex.HERE/n) for n in ['improvement_features.py',
                    'improvement_experiment.py','improvement_seasonal.py','group_loss_experiment.py',
                    'compact_xgb_experiment.py','team_pair_experiment.py','component_ensemble.py','audit_improvement.py']})
    for label,root,newroot in [('v4',v4root,ex.RUN),('v5',v5root,v5root),('v6',v6root,v6root),('v7',v7root,v7root),('v8',v8root,v8root)]:
        if not (root/'evaluation.json').exists():continue
        r=ex.read(root/'evaluation.json')
        old=pd.read_parquet(ex.OLD/'development/site_predictions.parquet')
        nw=r['selected_new_weight']
        new_path=newroot/'development/predictions.parquet'
        if new_path.exists():
            new=pd.read_parquet(new_path)
            z=old[['site_id','ts','ensemble']].merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
            pred=(1-nw)*z.ensemble+nw*z['new']
            if label=='v8' and nw and (v7root/'development/components.parquet').exists():
                weights=r['seasonal']['weights']
                if not np.isclose(sum(weights.values()),1) or not np.isclose(nw,1-weights['base']):
                    raise ValueError('Inconsistent component weights')
                components=pd.read_parquet(v7root/'development/components.parquet',columns=['site_id','ts','team_lgb','team_cat'])
                check=z[['site_id','ts','new']].merge(components,on=['site_id','ts'],how='left',validate='one_to_one')
                expected=(weights['lightgbm']*check.team_lgb+weights['catboost']*check.team_cat)/nw
                if not np.allclose(check['new'],expected,rtol=1e-12,atol=1e-8):
                    raise ValueError('Component forecast differs from frozen expert weights')
                result['v8_component_combination_equal']=True
        else:
            if nw!=0 or r.get('standalone_development_evaluated') is not False:
                raise ValueError('Selected model has no audited spring prediction')
            z=old[['site_id','ts','ensemble']].copy();pred=z.ensemble
        z['candidate_prediction']=pred
        z=z.merge(raw[['site_id','ts','energy']],on=['site_id','ts'],how='left',validate='one_to_one')
        if not np.isfinite(z[['energy','ensemble','candidate_prediction']]).all().all():raise ValueError('Missing audit row')
        ids=pd.read_csv(args.data/'sites.csv').sort_values('site_id').site_id.to_numpy()
        groups=pd.Series(np.where(np.array(r['assignment'])==1,'G1','G2'),index=ids) if 'assignment' in r else comp
        z['group']=z.site_id.map(groups)
        z['error']=z.candidate_prediction-z.energy
        error=z.groupby(['ts','group']).error.sum().reset_index()
        error['weight']=error.ts.map(ex.prices(args.data).weight)
        checked=[]
        for record in r['months']:
            a=error[(error.ts-pd.Timedelta(hours=1)).dt.month==record['month']]
            value=float(.1*np.sum(a.weight*np.abs(a.error)))
            if not np.isclose(value,record['candidate_penalty_krw'],rtol=1e-12,atol=1e-6):
                raise ValueError(f'{label} score differs from raw-truth independent aggregation')
            checked.append(dict(month=record['month'],candidate_penalty=value))
        result[f'{label}_independent_scores']=checked
    for label,root in [('v4',v4root),('v5',v5root),('v6',v6root),('v7',v7root),('v8',v8root)]:
        output=ex.HERE/f'제출/improved_{label}'
        if not (output/'forecast.csv').exists():continue
        errors=validate(output/'composition.csv',output/'forecast.csv',args.data/'sites.csv')
        if errors:raise ValueError(errors)
        v=ex.read(output/'validation.json');r=v['result']
        old=pd.read_parquet(ex.OLD/'production/site_predictions.parquet')
        nw=r['selected_new_weight']
        if nw:
            new=pd.read_parquet(root/'production/predictions.parquet')
            old=old.merge(new,on=['site_id','ts'],how='left',validate='one_to_one')
            old['prediction_kwh']=(1-nw)*old.prediction_kwh+nw*old['new']
        if len(old)!=253*8760 or not np.isfinite(old.prediction_kwh).all():
            raise ValueError('Incomplete audited production panel')
        comp=pd.read_csv(output/'composition.csv').set_index('site_id').group
        old['group']=old.site_id.map(comp)
        agg=old.groupby(['group','ts']).prediction_kwh.sum().sort_index()
        pred=pd.read_csv(output/'forecast.csv');pred.ts=pd.to_datetime(pred.ts).dt.tz_convert('Asia/Seoul')
        actual=pred.set_index(['group','ts']).forecast_kwh.sort_index()
        if not agg.index.equals(actual.index) or not np.allclose(agg,actual,rtol=1e-12,atol=1e-9):
            raise ValueError('Production group sums mismatch')
        result[f'{label}_production_group_sums_equal']=True
        result[f'{label}_submission_sha256']={n:p.sha256(output/n) for n in ['composition.csv','forecast.csv']}
        if result[f'{label}_submission_sha256']!=v['files']:raise ValueError('New submission hashes mismatch')
    p.save_json(ex.HERE/'_local/improvement_audit.json',result)
    print('PASS: original data, first submission, cutoffs, original truth and independent score audit',flush=True)


if __name__=='__main__':main()
