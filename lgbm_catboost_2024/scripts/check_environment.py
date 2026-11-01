"""Check dependencies using synthetic values only; no competition data."""
import importlib.metadata as md
import json
import numpy as np
import pandas as pd
import pyarrow
import pvlib
import lightgbm as lgb
from catboost import CatBoostRegressor

x=np.arange(40,dtype=float).reshape(20,2)
y=np.maximum(0,x[:,0]/40)
a=lgb.LGBMRegressor(n_estimators=3,min_child_samples=2,n_jobs=1,verbosity=-1,objective="regression_l1").fit(x,y)
b=CatBoostRegressor(iterations=3,depth=2,loss_function="MAE",thread_count=1,verbose=False,allow_writing_files=False).fit(x,y)
assert np.isfinite(a.predict(x)).all() and np.isfinite(b.predict(x)).all()
t=pd.date_range("2024-01-01 01:00","2025-01-01 00:00",freq="h",tz="Asia/Seoul")
assert len(t)==8784
solar=pvlib.solarposition.get_solarposition(t[:2],latitude=35,longitude=128)
assert np.isfinite(solar.zenith).all()
print(json.dumps({"status":"passed","synthetic_model_smoke":True,"forecast_hours":len(t),"versions":{n:md.version(n) for n in ["numpy","pandas","pyarrow","lightgbm","catboost","scikit-learn","pvlib"]}},indent=2))
