import unittest,itertools
import numpy as np,pandas as pd
from pipeline import weather,make_features,optimize,loss_by_hour,TZ
from baseline_original import features,FEATS,WA_COLS,WB_COLS
import tempfile
from pathlib import Path
(Path(__file__).parent/'runs').mkdir(exist_ok=True)

class PipelineTests(unittest.TestCase):
 def test_optimizer_matches_exhaustive_toy(self):
  e=np.array([[10,10,-10,-10],[8,8,-8,-8]],float);w=np.array([1,3.]);c=np.full(4,10000.)
  z0=np.array([1,1,0,0],bool);z,_=optimize(e,w,c,z0,seed=7,steps=300,restarts=2)
  feasible=[np.array(v,bool) for v in itertools.product([0,1],repeat=4) if sum(v)==2]
  exact=min(loss_by_hour(e,s,w).sum() for s in feasible)
  self.assertAlmostEqual(loss_by_hour(e,z,w).sum(),exact)
  self.assertEqual(c[z].sum(),20000)
 def test_label_symmetry_and_no_cross_group_cancellation(self):
  e=np.array([[10,-5]],float);z=np.array([1,0],bool);w=np.array([3.])
  self.assertEqual(loss_by_hour(e,z,w)[0],45)
  np.testing.assert_equal(loss_by_hour(e,z,w),loss_by_hour(e,~z,w))
 def test_issuance_day_boundary_and_baseline_parity(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as tmp:
   data=Path(tmp);(data/'train').mkdir();ts=pd.date_range('2025-01-01 01:00',periods=48,freq='h',tz=TZ)
   sites=pd.DataFrame({'site_id':['S1'],'capacity_kw':[100.],'lat':[36.],'lon':[127.],'wa_station':['A1'],'wb_station':['B1']})
   frames={}
   for prov,cols,key in [('wa',WA_COLS,'A1'),('wb',WB_COLS,'B1')]:
    rows=[]
    for r in [10,17]:
     d=pd.DataFrame({'ts':ts,prov+'_station':key,'round':r})
     for c in cols:d[c]=np.arange(48,dtype=float)+r
     rows.append(d)
    frames[prov]=pd.concat(rows,ignore_index=True);frames[prov].to_parquet(data/'train'/f'forecast_{prov}.parquet',index=False)
   b=pd.DataFrame({'site_id':'S1','ts':ts,'energy':1.,'_row':np.arange(48)})
   tables=weather(data,ts[0],ts[-1]);x,_,_=make_features(b,sites,tables,'enhanced')
   for offset in [0,24]:
    expected=pd.Series(np.arange(offset,offset+24,dtype=float)+10).rolling(3,center=True,min_periods=1)
    np.testing.assert_allclose(x['wa_cloud_mean3_r10'].iloc[offset:offset+24],expected.mean(),rtol=1e-6)
    np.testing.assert_allclose(x['wa_cloud_std3_r10'].iloc[offset:offset+24],expected.std(ddof=0),rtol=1e-6)
   old=features(b,sites,frames['wa'],frames['wb'],None,10)
   bx,_,_=make_features(b,sites,tables,'baseline',10)
   np.testing.assert_allclose(bx[FEATS],old[FEATS].astype('float32'),rtol=1e-6,atol=1e-6)
   for prov,cols in [('wa',WA_COLS),('wb',WB_COLS)]:
    d=frames[prov].copy();mask=d.ts>pd.Timestamp('2025-01-02 00:00',tz=TZ);d.loc[mask,cols]=999.
    d.to_parquet(data/'train'/f'forecast_{prov}.parquet',index=False)
   x2,_,_=make_features(b,sites,weather(data,ts[0],ts[-1]),'enhanced')
   np.testing.assert_allclose(x.iloc[:24],x2.iloc[:24],equal_nan=True)
   self.assertFalse(np.allclose(x.iloc[24:],x2.iloc[24:],equal_nan=True))

if __name__=='__main__':unittest.main()
