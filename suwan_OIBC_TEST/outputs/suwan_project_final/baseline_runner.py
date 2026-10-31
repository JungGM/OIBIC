"""Resumable local execution of the attributed upstream experiment."""
import argparse,gc,json,platform,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import features as ft
import pipeline as p

ORIGINAL_FIT=p.fit_one

def cached_fit(d,schema,config,origin,directory):
 directory=Path(directory);name=config['name'];record=directory/(name+'_checkpoint.json')
 if record.exists():
  r=json.loads(record.read_text());assert r['config']==config and r['schema']==schema and r['origin']==str(origin),'Checkpoint settings mismatch'
  for f,h in r['model_sha256'].items():assert p.sha256(directory/f)==h,'Model checkpoint hash mismatch'
  if config['kind']=='baseline':model={rr:lgb.Booster(model_file=str(directory/f'{name}_r{rr}.txt')) for rr in [10,17]}
  elif config['kind']=='lgb':model=lgb.Booster(model_file=str(directory/f'{name}.txt'))
  else:model=xgb.XGBRegressor(n_jobs=p.THREADS);model.load_model(directory/f'{name}.ubj')
  p.log('Verified checkpoint reused: '+name);return model
 model=ORIGINAL_FIT(d,schema,config,origin,directory)
 paths=[directory/f'{name}_r{rr}.txt' for rr in [10,17]] if config['kind']=='baseline' else [directory/f'{name}.txt' if config['kind']=='lgb' else directory/f'{name}.ubj']
 p.save_json(record,{'config':config,'origin':str(origin),'schema':schema,'model_sha256':{x.name:p.sha256(x) for x in paths}})
 return model

p.fit_one=cached_fit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--submission',type=Path,required=True);ap.add_argument('--group-steps',type=int,default=20000);a=ap.parse_args()
 for k in ['data','cache','run','submission']:setattr(a,k,getattr(a,k).resolve())
 for x in [a.cache,a.run,a.submission]:assert x!=a.data and a.data not in x.parents
 a.run.mkdir(parents=True,exist_ok=True)
 code={x.name:p.sha256(x) for x in Path(__file__).parent.glob('*.py')};manifest=p.source_manifest(a.data)
 protocol={'configs':p.CONFIGS,'seed':p.SEED,'threads':p.THREADS,'source_sha256':manifest,'code_sha256':code,'upstream':'JungGM/OIBIC@73d269f69ce059cabbee1c4f68a2cb05ce4d1aac','adaptations':['2 CPU threads','hash-verified model and stage resume'],'model_selection':'original two-origin seasonal225 proxy and winter253 equal block weights; 1456 <=3 model 0.1 simplex grid','group_selection':'full253 March search; April >=1% and paired-day bootstrap gate; fixed May audit','final_fit_cutoff':'2025-05-31 17:00+09:00','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__,'xgboost':xgb.__version__}}
 path=a.run/'protocol_before_training.json'
 if path.exists():assert json.loads(path.read_text())==protocol,'Sources or runner changed; choose fresh run'
 else:p.save_json(path,protocol)
 sites=pd.read_csv(a.data/'sites.csv').sort_values('site_id').reset_index(drop=True);assert len(sites)==253 and sites.site_id.is_unique
 prices=pd.read_csv(a.data/'train/smp.csv');prices['ts']=pd.to_datetime(prices.ts).dt.tz_convert(p.TZ);prices['k']=np.where(prices.ts.dt.hour.between(11,15),3,1)
 if (a.run/'completion.json').exists():p.log('Already complete');return
 p.save_json(a.run/'status.json',{'stage':'features','completed':False})
 schema=p.ensure_train_cache(a)
 selection=a.run/'selection/selection.json'
 if selection.exists():
  chosen=json.loads(selection.read_text())['selected']['weights'];weights=np.array([chosen[c['name']] for c in p.CONFIGS]);p.log('Completed selection reused')
 else:
  p.save_json(a.run/'status.json',{'stage':'selection','completed':False});weights=p.select_models(a,schema,sites,prices)
 gc.collect()
 done=a.run/'development_done.json'
 if done.exists():
  saved=json.loads(done.read_text());groups=np.array(saved['groups']);reports=saved['reports'];p.log('Completed development reused')
 else:
  p.save_json(a.run/'status.json',{'stage':'development','completed':False});groups,reports=p.development(a,schema,sites,prices,weights);p.save_json(done,{'groups':groups.tolist(),'reports':reports})
 gc.collect();p.save_json(a.run/'status.json',{'stage':'production','completed':False});p.production(a,schema,sites,weights,groups)
 assert p.source_manifest(a.data)==manifest;p.save_json(a.run/'source_integrity.json',{'unchanged':True,'source_sha256':manifest});p.save_json(a.run/'status.json',{'stage':'complete','completed':True})

if __name__=='__main__':main()
