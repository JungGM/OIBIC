import unittest
import numpy as np
import pandas as pd
from extras import add_features, EXTRA, RAW
import correction as cr

class Contracts(unittest.TestCase):
    def test_feature_day_and_label_isolation(self):
        d=pd.DataFrame({'site_id':['s']*48,'ts':pd.date_range('2025-03-01 01:00',periods=48,freq='h',tz='Asia/Seoul'),
                        'cosz':np.ones(48),'energy':np.arange(48,dtype=float)})
        for c in RAW:d[c]=np.arange(48,dtype=float)
        expected=add_features(d)
        changed=d.copy();changed.loc[24:,RAW]=10000;changed.energy=-10000
        actual=add_features(changed)
        np.testing.assert_allclose(expected.loc[:23,EXTRA],actual.loc[:23,EXTRA],equal_nan=True)
        self.assertTrue(np.isnan(expected.loc[24,RAW[0]+'_delta1']))
        missing=d.copy();missing.loc[2,RAW[0]]=np.nan
        x=add_features(missing)
        self.assertTrue(np.isnan(x.loc[3,RAW[0]+'_delta1']))
        self.assertTrue(np.isnan(x.loc[3,RAW[0]+'_std3']))

    def test_correction_bounds_night_and_target_independence(self):
        d=pd.DataFrame({'group':['G1','G2','G1'],'base':[100.,200.,0.],'capacity':[20000]*3,
                        'solar':[.5,.5,0.],'weight':[1.,3.,1.],'actual':[90.,210.,0.],
                        'base_util':[.005,.01,0.]})
        model=dict(features=['base_util'],median=[0.],mean=[0.],scale=[1.],coef=[0.],intercept=10.)
        recipe=dict(kind='residual',strength=1.,models=[model,model])
        v=cr.predict(d,recipe)
        np.testing.assert_allclose(v,[90.,180.,0.])
        changed=d.copy();changed.actual=-999
        np.testing.assert_array_equal(cr.predict(changed,recipe),v)
        self.assertAlmostEqual(cr.loss(d,v),9.)

if __name__=='__main__':unittest.main()
