"""Independent raw-label scoring and submission identity checks for site experts."""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import pipeline as p
import improvement_experiment as ex
import solar_normalized as engine
import team_pair_experiment as team
from validate_submission import validate

ROOTS={'v11':'solar_normalized_v11','v12':'site_regression_v12',
       'v13':'catboost_rmse_v13','v14':'weather_augmented_v14','v17':'catboost_solar_v17','v18':'spatial_v18'}


def audit(data,label,production=False):
    root=ex.HERE/'_local'/ROOTS[label]
    expected=ex.read(ex.OLD/'protocol_before_training.json')['source_sha256']
    if p.source_manifest(data)!=expected:raise ValueError('Original source hashes changed')
    if label=='v14':
        import forecast_corrector as teacher
        for (source,target),(model,m) in teacher.load().items():
            if p.stamp(m['last_label'])>teacher.CUTOFF:raise ValueError('Weather teacher uses future labels')
            prefix='wa' if source=='a' else 'wb'
            if p.sha256(data/'train'/f'observed_{prefix}.parquet')!=m['observed_source_sha256']:
                raise ValueError('Observed weather changed')
    v8=ex.HERE/'제출/improved_v8';record=ex.read(v8/'validation.json')
    for name,digest in record['files'].items():
        if p.sha256(v8/name)!=digest:raise ValueError('Previous official submission changed')
    models=[]
    for part in ['seasonal_proxy','complete_fleet','development']+(['production'] if production else []):
        m=ex.read(root/part/'metadata.json');cutoff=p.issue_deadline(p.stamp(team.PARTS[part][0]))
        if p.stamp(m['cutoff'])!=cutoff or p.stamp(m['last_label'])>cutoff:raise ValueError('Invalid training cutoff')
        modelpath=root/part/('model.cbm' if label in ['v13','v17'] else 'model.txt')
        if p.sha256(modelpath)!=m['model_sha256']:raise ValueError('Model hash mismatch')
        models.append(dict(part=part,cutoff=str(cutoff),sha256=m['model_sha256']))
    result=ex.read(root/'evaluation.json');a=result['selected']['strength']
    raw=pd.read_parquet(data/'train/generation.parquet')
    base=pd.read_parquet(ex.OLD/'development/site_predictions.parquet')[['site_id','ts','ensemble']]
    cats=pd.read_parquet(team.RUN/'development/components.parquet')[['site_id','ts','team_cat']]
    new=pd.read_parquet(root/'development/predictions.parquet')
    d=base.merge(cats,on=['site_id','ts'],validate='one_to_one').merge(new,on=['site_id','ts'],validate='one_to_one')
    d=d.merge(raw,on=['site_id','ts'],how='left',validate='one_to_one')
    if d.site_id.nunique()!=253 or not d.groupby('ts').site_id.nunique().eq(253).all():raise ValueError('Incomplete validation panel')
    if not np.isfinite(d[['ensemble','team_cat','new','energy']]).all().all():raise ValueError('Missing label or prediction')
    comp=pd.read_csv(ex.HERE/'제출/composition.csv').set_index('site_id').group
    d['group']=d.site_id.map(comp);d['v8']=.75*d.ensemble+.25*d.team_cat
    d['candidate']=(1-a)*d.v8+a*d['new']
    g=d.groupby(['ts','group'])[['energy','v8','candidate']].sum().reset_index()
    smp=pd.read_csv(data/'train/smp.csv');smp.ts=pd.to_datetime(smp.ts)
    g['w']=g.ts.map(smp.set_index('ts').smp)*np.where(g.ts.dt.hour.between(11,15),3,1)
    checks=[]
    for r in result['months']:
        z=g[(g.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')==r['month']]
        value=float(.1*np.sum(z.w*np.abs(z.candidate-z.energy)))
        if not np.isclose(value,r['candidate_penalty'],rtol=1e-12,atol=1e-6):raise ValueError('Raw-label penalty mismatch')
        checks.append(dict(month=r['month'],penalty=value,hours=int(z.ts.nunique())))
    g['saving']=.1*g.w*(abs(g.v8-g.energy)-abs(g.candidate-g.energy))
    daily=g.groupby((g.ts-pd.Timedelta(hours=1)).dt.date).saving.sum().to_numpy()
    rng=np.random.default_rng(20261104);n=len(daily)
    samples=[daily[(rng.integers(n,size=(int(np.ceil(n/3)),1))+np.arange(3))%n].ravel()[:n].mean() for _ in range(2000)]
    out=dict(passed=True,source_hashes_unchanged=True,official_v8_unchanged=True,models=models,
             raw_label_scores=checks,lower90_daily_saving=float(np.quantile(samples,.05)))
    if production:
        if not result['accepted']:raise ValueError('Historical validation rejected candidate')
        dest=ex.HERE/'제출'/f'improved_{label}'
        errors=validate(dest/'composition.csv',dest/'forecast.csv',data/'sites.csv')
        if errors:raise ValueError(errors)
        v=ex.read(dest/'validation.json')
        for name,digest in v['files'].items():
            if p.sha256(dest/name)!=digest:raise ValueError('Submission hash mismatch')
        old=pd.read_parquet(ex.OLD/'production/site_predictions.parquet')
        cat=pd.read_parquet(team.RUN/'production/components.parquet')[['site_id','ts','team_cat']]
        new=pd.read_parquet(root/'production/predictions.parquet')
        z=old.merge(cat,on=['site_id','ts'],validate='one_to_one').merge(new,on=['site_id','ts'],validate='one_to_one')
        if len(z)!=253*8760 or z.duplicated(['site_id','ts']).any():raise ValueError('Bad production grid')
        z['candidate']=(1-a)*(.75*z.prediction_kwh+.25*z.team_cat)+a*z['new'];z['group']=z.site_id.map(comp)
        expected=z.groupby(['group','ts']).candidate.sum().sort_index()
        submitted=pd.read_csv(dest/'forecast.csv');submitted.ts=pd.to_datetime(submitted.ts).dt.tz_convert('Asia/Seoul')
        actual=submitted.set_index(['group','ts']).forecast_kwh.sort_index()
        if not expected.index.equals(actual.index) or not np.allclose(expected,actual,rtol=1e-12,atol=1e-8):raise ValueError('CSV aggregation mismatch')
        out['submission_aggregation_equal']=True;out['files']=v['files']
        # Check saved predictions against freshly loaded model + original forecast cache.
        meta,names,categories=engine.inputs(data)
        if label=='v14':
            import weather_augmented
            weather_augmented.configure(data);meta,names,categories=engine.inputs(data)
        if label=='v18':
            import spatial_expert
            spatial_expert.configure(data,True);meta,names,categories=engine.inputs(data)
        if label in ['v13','v17']:
            from catboost import CatBoostRegressor
            model=engine.CatPredictor(CatBoostRegressor().load_model(str(root/'production/model.cbm')))
        else:model=lgb.Booster(model_file=str(root/'production/model.txt'))
        differences=[]
        for sid in [meta.index[0],meta.index[len(meta)//2],meta.index[-1]]:
            x=pd.read_parquet(team.RUN/'cache/test'/f'{sid}.parquet').iloc[::37].copy()
            x=engine.design(x,sid,meta,categories)
            norm=np.maximum(x.solar_q.to_numpy(),.15) if label in ['v11','v14','v17','v18'] else np.ones(len(x))
            check=np.where(x.solar_q>0,np.maximum(model.predict(x[names],num_threads=4),0)*norm*float(meta.loc[sid,'capacity_kw']),0)
            saved=new[new.site_id==sid].set_index('ts')['new'].reindex(x.ts).to_numpy()
            differences.append(float(np.max(abs(check-saved))))
        if max(differences)>1e-8:raise ValueError('Saved model cannot reproduce prediction sample')
        out['saved_model_probe_max_difference']=max(differences)
    p.save_json(root/'audit.json',out);print(out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=engine.Path,required=True)
    ap.add_argument('--candidate',choices=list(ROOTS),required=True);ap.add_argument('--production',action='store_true')
    a=ap.parse_args();audit(a.data,a.candidate,a.production)
