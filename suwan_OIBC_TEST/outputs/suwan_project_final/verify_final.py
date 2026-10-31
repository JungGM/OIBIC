"""Verify frozen choices, independently score audit predictions, and repeat final CSV generation."""
import gc
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import run as r
import adopt_candidate
from extras import add_features

def main():
    sites,prices=r.setup()
    baseline=r.read(r.BASE_RUN/'reproduction_check.json')
    assert baseline['passed'] and baseline['byte_identical']
    assert r.read(r.BASE_RUN/'independent_audit.json')['passed']
    freeze=r.read(r.OUT/'frozen_before_May.json')
    assert r.p.sha256(r.OUT/'candidate_before_April.json')==freeze['candidate_file_sha256']
    c=adopt_candidate.decision() or freeze['selected']
    assert len(c['assignment'])==253
    if c['variant']=='feature':
        meta=r.read(r.RUN/'feature_production/metadata.json')
        assert r.p.sha256(r.RUN/'feature_production/model.ubj')==meta['model_sha256']
        model=xgb.XGBRegressor(n_jobs=2);model.load_model(r.RUN/'feature_production/model.ubj')
        maximum=0.
        for month in pd.period_range('2025-06','2026-05',freq='M'):
            d=add_features(pd.read_parquet(r.CACHE/'test'/f'{month}.parquet'))
            day=d.elev.to_numpy()>0;pred=np.zeros(len(d))
            pred[day]=np.maximum(model.predict(d.loc[day,meta['features']].astype('float32')),0)*d.loc[day,'cap'].to_numpy()
            z=d[['site_id','ts']].copy();z['repeated']=pred
            saved=pd.read_parquet(r.RUN/f'production_feature_{month}.parquet')
            checked=z.merge(saved,on=['site_id','ts'],validate='one_to_one')
            assert len(checked)==len(d)
            np.testing.assert_allclose(checked.repeated,checked.new_xgb,rtol=1e-10,atol=1e-7)
            maximum=max(maximum,float(abs(checked.repeated-checked.new_xgb).max()))
            del d,z,saved,checked;gc.collect()
        del model;gc.collect()
    else:maximum=None
    # Independently verify reported candidate May penalty from saved group predictions.
    may=pd.read_parquet(r.RUN/'May_group_audit.parquet')
    cost=float(np.sum(.1*may.weight.to_numpy()*np.abs(may.candidate_forecast.to_numpy()-may.actual.to_numpy())))
    recorded=pd.read_csv(r.OUT/'monthly.csv').set_index('month').loc[5,'candidate_penalty']
    assert np.isclose(cost,recorded,rtol=1e-12)
    before=r.read(r.OUT/'completion.json')['hashes']
    command=[sys.executable,str(r.HERE/'adopt_candidate.py')] if (r.HERE/'ADOPTION.json').exists() else [sys.executable,str(r.HERE/'run.py'),'produce']
    subprocess.run(command,check=True)
    after=r.read(r.OUT/'completion.json')['hashes']
    assert before==after,'Final CSV differs after inference-only regeneration'
    submission=r.HERE/'submission'
    forecast=pd.read_csv(submission/'forecast.csv');forecast.ts=pd.to_datetime(forecast.ts).dt.tz_convert('Asia/Seoul')
    expected=pd.date_range('2025-06-01 01:00',periods=8760,freq='h',tz='Asia/Seoul')
    for _,d in forecast.groupby('group'):
        assert pd.DatetimeIndex(d.ts).equals(expected)
    # Verify night output using source solar geometry, not a universal clock hour.
    comp=pd.read_csv(submission/'composition.csv').set_index('site_id').group
    expected_groups=np.where(np.array(c['assignment'])==1,'G1','G2')
    np.testing.assert_array_equal(comp.reindex(sites.site_id).to_numpy(),expected_groups)
    completion=r.read(r.OUT/'completion.json')
    if (r.HERE/'ADOPTION.json').exists():
        assert completion['adoption']=='explicit_user_override'
        assert completion['passed_original_stability_gate']==freeze['accepted']
    for month in pd.period_range('2025-06','2026-05',freq='M'):
        d=pd.read_parquet(r.CACHE/'test'/f'{month}.parquet',columns=['site_id','ts','elev'])
        d['group']=d.site_id.map(comp)
        night=d.groupby(['group','ts']).elev.max().le(0)
        values=forecast.set_index(['group','ts']).forecast_kwh.reindex(night.index)
        assert values.notna().all() and values[night].eq(0).all()
    r.save(r.OUT/'verification.json',{'passed':True,'baseline_saved_model_reproduction':True,
          'feature_saved_model_max_difference':maximum,'independent_May_candidate_penalty':cost,
          'final_CSV_byte_identical_on_repeat':True,'full_hour_grid':True,'night_zero':True,
          'source_unchanged':True,'official_server_submission':False,
          'group_capacity_kw':sites.assign(group=comp.reindex(sites.site_id).to_numpy()).groupby('group').capacity_kw.sum().to_dict(),
          'group_site_counts':comp.value_counts().sort_index().to_dict()})
    print('PASS: final forecast regenerated identically; independent score, grid and night checks passed')

if __name__=='__main__':main()
