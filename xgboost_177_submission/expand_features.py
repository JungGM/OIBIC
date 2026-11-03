"""Add the prior 160-feature engine to the original observed-row cache."""
import importlib.util
from pathlib import Path
import json
import gc
import numpy as np
import pandas as pd
from experiment import ROOT, BASE, ADD, sha, save

OLD=Path('work/OIBIC/lgbm_catboost_2024').resolve()
names=[f['name'] for f in json.loads((OLD/'configs/features_160.json').read_text())['features']]
skip={'capacity_kw','log_capacity','wa_station','wb_station','wa_dist','wb_dist',
      'a10_temp','a10_cloud','b10_temp','b10_cloud','a10_available','b10_available',
      'ab17_diff_cloud','ab17_diff_temp','ab17_diff_humidity','ab17_absdiff_cloud','b17_dewpoint_depression'}
skip.update('a17_'+x for x in 'cloud temp humidity ground_press wind_speed wind_sin wind_cos rain snow'.split())
skip.update('b17_'+x for x in 'temp feel_temp wet_bulb_temp dew_point wind_speed wind_sin wind_cos wind_gust_spd rel_hum vis ceiling uv_idx precip_prob total_liq snow'.split())
skip.update(s+'_revision_'+v for s in ['a','b'] for v in ['cloud','temp','humidity','wind_speed'])
skip.update('b17_'+v+'_'+op for v in ['cloud','uv_idx'] for op in ['lagdiff','mean3','std3'])
extra=[n for n in names if n not in skip]

def main():
    spec=importlib.util.spec_from_file_location('old160',OLD/'src/features.py');ft=importlib.util.module_from_spec(spec);spec.loader.exec_module(ft)
    data=Path('work/data');sites=pd.read_csv(data/'sites.csv')
    manifest={'name':'expanded'+str(len(BASE+ADD+extra)), 'features':BASE+ADD+['old_'+n for n in extra],
              'additional_prior_features':extra,'omitted_prior_features':[n for n in names if n in skip],
              'note':'Known equivalent raw variables, scales and IDs omitted. Existing baseline numeric semantics preserved.',
              'prior_engine_sha256':sha(OLD/'src/features.py')}
    if (ROOT/'expanded_features.json').exists(): raise ValueError('Expansion already complete')
    for year in range(2021,2026):
        start=pd.Timestamp(f'{year}-01-01 01:00',tz='Asia/Seoul')
        end=pd.Timestamp(f'{year+1}-01-01 00:00',tz='Asia/Seoul') if year<2025 else pd.Timestamp('2025-06-01 00:00',tz='Asia/Seoul')
        ix=pd.date_range(start,end,freq='h')
        files=sorted((ROOT/'cache'/str(year)).glob('*.parquet'));ids=[p.stem for p in files];scope=sites[sites.site_id.isin(ids)]
        wa=ft.weather(data/'train/forecast_wa.parquet','a',ix,scope.wa_station.unique())
        wb=ft.weather(data/'train/forecast_wb.parquet','b',ix,scope.wb_station.unique())
        for j,site in enumerate(scope.itertuples(index=False)):
            path=ROOT/'cache'/str(year)/f'{site.site_id}.parquet';base=pd.read_parquet(path)
            old=ft.build_site(site,ix,wa[site.wa_station],wb[site.wb_station]).set_index('ts')
            old=old[extra].reindex(pd.DatetimeIndex(base.ts))
            extra_frame=pd.DataFrame(old.to_numpy(dtype='float32'),columns=['old_'+n for n in extra],index=base.index)
            pd.concat([base,extra_frame],axis=1).to_parquet(path,index=False,compression='zstd')
            if (j+1)%40==0 or j+1==len(scope): print(json.dumps({'stage':'expand','year':year,'sites_done':j+1,'features':len(manifest['features'])}),flush=True)
        del wa,wb,old,base;gc.collect()
    save(ROOT/'expanded_features.json',manifest)
    print(json.dumps({'expanded_features':len(manifest['features']),'extra_from_prior160':len(extra)}),flush=True)

if __name__=='__main__': main()
