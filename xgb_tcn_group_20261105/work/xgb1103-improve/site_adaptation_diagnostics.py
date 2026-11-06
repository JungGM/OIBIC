"""Local-only NWP key/unit/missingness audit and regional retrospective checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path('outputs/OIBC_발전소개별특성_개선_20261105');OUT.mkdir(parents=True,exist_ok=True)
sites=pd.read_csv('work/data/sites.csv').merge(pd.read_csv('outputs/OIBC_발전소_지역분류_20261104/발전소별_지역분류.csv')[['site_id','region_id','region_name']],on='site_id',validate='one_to_one')
start=pd.Timestamp('2025-01-01 01:00',tz='Asia/Seoul');end=pd.Timestamp('2025-06-01 00:00',tz='Asia/Seoul');hours=len(pd.date_range(start,end,freq='h'))
rows=[];key_checks=[]
for source in ['a','b']:
    key='w'+source+'_station';prefix='w'+source+'_';variables=['cloud','temp','humidity' if source=='a' else 'rel_hum','wind_speed']
    cols=[prefix+v for v in variables]
    f=pd.read_parquet(f'work/data/train/forecast_w{source}.parquet',columns=[key,'ts','round']+cols,filters=[('ts','>=',start),('ts','<=',end),('round','==',17)])
    o=pd.read_parquet(f'work/data/train/observed_w{source}.parquet',columns=[key,'ts']+cols,filters=[('ts','>=',start),('ts','<=',end)])
    dup_f=int(f.duplicated([key,'ts']).sum());dup_o=int(o.duplicated([key,'ts']).sum());assert dup_f==0
    ambiguous=o.duplicated([key,'ts'],keep=False);excluded=int(o.loc[ambiguous,[key,'ts']].drop_duplicates().shape[0]);o=o.loc[~ambiguous].copy()
    joined=f.merge(o,on=[key,'ts'],how='left',suffixes=('_forecast','_observed'),validate='one_to_one')
    key_checks.append({'source':source,'forecast_duplicate_keys':dup_f,'observed_duplicate_extra_rows':dup_o,'observed_ambiguous_keys_excluded_for_diagnostics_only':excluded,'used_stations_absent_from_forecast':sorted(set(sites[key])-set(f[key])),'used_stations_absent_from_observed':sorted(set(sites[key])-set(o[key]))})
    for (rid,name),scope in sites.groupby(['region_id','region_name']):
        mass=scope.groupby(key).capacity_kw.sum();z=joined[joined[key].isin(mass.index)].copy();weight=z[key].map(mass).to_numpy('float64')
        for period,mask in [('all_hours',np.ones(len(z),dtype=bool)),('ending08_18',z.ts.dt.hour.between(8,18).to_numpy())]:
            for var in variables:
                p=z[prefix+var+'_forecast'].to_numpy('float64');y=z[prefix+var+'_observed'].to_numpy('float64');scale=3.6 if source=='b' and var=='wind_speed' else 1
                p=p/scale;y=y/scale;valid=mask&np.isfinite(p)&np.isfinite(y);den=weight[valid].sum()
                rows.append({'region_id':rid,'region_name':name,'source':source,'variable':var,'period':period,'unit':'m/s' if var=='wind_speed' else 'degC' if var=='temp' else 'percentage_points','matched_station_hours':int(valid.sum()),'mae':float(np.dot(weight[valid],abs(p[valid]-y[valid]))/den) if den else None,'bias':float(np.dot(weight[valid],p[valid]-y[valid])/den) if den else None,'forecast_value_coverage_pct':float(100*weight[mask&np.isfinite(p)].sum()/(scope.capacity_kw.sum()*(hours if period=='all_hours' else int(pd.Series(pd.date_range(start,end,freq='h')).dt.hour.between(8,18).sum()))))})
pd.DataFrame(rows).to_csv(OUT/'지역별_예보실측_진단.csv',index=False,encoding='utf-8-sig')
distance=sites.groupby(['region_id','region_name'])[['wa_dist','wb_dist']].agg(['median','max']);distance.to_csv(OUT/'지역별_관측소거리.csv',encoding='utf-8-sig')
peer_rows=[]
for site in ['PV011','PV188']:
    s=sites[sites.site_id.eq(site)].iloc[0];peer=sites[sites.wa_station.eq(s.wa_station)&sites.wb_station.eq(s.wb_station)]
    peer_rows.append({'site_id':site,'region_id':s.region_id,'wa_station':s.wa_station,'wb_station':s.wb_station,'same_forecast_pair_sites':peer.site_id.tolist(),'wa_distance_m':float(s.wa_dist),'wb_distance_m':float(s.wb_dist)})
result={'period':'2025 Jan-May retrospective diagnostics only','key_checks':key_checks,'units':'A wind m/s; B forecast and observed wind km/h, divided by3.6 in this comparison. Temperature degC and cloud/RH percentage points.','diagnostic_observations_not_used_as_future_forecast_inputs':True,'forecast_station_assignment_changed':False,'known_site_identity':'Existing TCN uses31 forecast/solar/static inputs including lat/lon/capacity; no explicit site identity parameter.','selected_site_forecast_peers':peer_rows,'caution':'Observation at a weather station is not the true weather at a plant. Differences in sites/coordinates and observational definitions remain. These checks cannot establish a causal root cause.'}
(OUT/'예보점검.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks':key_checks,'peers':peer_rows},ensure_ascii=True))
print(pd.DataFrame(rows).query("period=='ending08_18' and variable=='cloud'")[['region_id','source','mae','bias','forecast_value_coverage_pct']].round(3).to_string(index=False))
