"""Synthetic tests of the imported team's feature boundary and missing-weather path."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from test_features import sample
import compact_xgb_experiment as c


class CompactFeatureContracts(unittest.TestCase):
    def test_truth_price_round10_and_other_day_do_not_enter_features(self):
        src=c.source(Path('.'));sites,a,b,ix=sample()
        g=pd.MultiIndex.from_product([sites.site_id,ix],names=['site_id','ts']).to_frame(index=False)
        g['energy']=1.;g['smp']=10.
        first,names=c.build(src,g,sites,a,b)
        g['energy']=1e9;g['smp']=-1e9
        a.loc[(a['round']==10)|(a.ts>ix[23]),'wa_cloud']=9999.
        b.loc[(b['round']==10)|(b.ts>ix[23]),'wb_uv_idx']=9999.
        second,_=c.build(src,g,sites,a,b)
        pd.testing.assert_frame_equal(first.loc[first.ts<=ix[23],names],second.loc[second.ts<=ix[23],names])
        self.assertNotIn('energy',names);self.assertNotIn('smp',names)
        self.assertEqual(len(names),52)

    def test_missing_weather_retains_complete_target_grid(self):
        src=c.source(Path('.'));sites,a,b,ix=sample()
        a=a[~((a.ts==ix[12])&(a['round']==17))]
        d,names=c.build(src,None,sites,a,b,ix)
        self.assertEqual(len(d),len(sites)*len(ix))
        self.assertEqual(d.loc[d.ts==ix[12],'wa_cloud'].isna().sum(),2)
        self.assertTrue(d.loc[d.ts==ix[12],'weather_missing'].ge(6).all())
        self.assertTrue(np.isfinite(d.elev).all())


if __name__=='__main__':unittest.main()
