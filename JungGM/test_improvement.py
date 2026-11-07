"""Leakage and numerical contracts for improvement features/calibration."""
import unittest
import numpy as np
import pandas as pd
import features as ft
from test_features import sample
from improvement_features import add_features
from improvement_experiment import apply, weighted_median, block_gate


class ImprovementContracts(unittest.TestCase):
    def test_day_and_site_isolation_and_truth_invariance(self):
        sites, a, b, ix = sample()
        raw = ft.build_month(sites, a, b, ix)
        first, names = add_features(raw)
        raw.loc[raw.ts > ix[23], 'r17_wb_cloud'] = 1000
        raw['energy'] = 1e9
        second, _ = add_features(raw)
        pd.testing.assert_frame_equal(first.loc[first.ts <= ix[23], names],
                                      second.loc[second.ts <= ix[23], names])
        raw.loc[raw.site_id == 'PV002', 'r17_wa_snow'] = 9999
        third, _ = add_features(raw)
        pd.testing.assert_frame_equal(second.loc[second.site_id == 'PV001', names],
                                      third.loc[third.site_id == 'PV001', names])

    def test_missing_hour_rejected(self):
        sites, a, b, ix = sample()
        d = ft.build_month(sites, a, b, ix)
        with self.assertRaises(ValueError):
            add_features(d.iloc[1:])

    def test_correction_cannot_create_night_generation(self):
        d = pd.DataFrame({'base':[0.,100.], 'group':['G1','G1'],
                          'ts':pd.date_range('2025-01-01', periods=2, freq='h', tz=ft.TZ)})
        result = apply(d, {'kind':'scale','strength':1., 'factors':{'G1':2.}})
        np.testing.assert_allclose(result, [0,112])

    def test_weighted_median_loss_optimality(self):
        x,w=np.array([1,2,8]),np.array([1,10,1])
        m=weighted_median(x,w)
        self.assertEqual(m,2)
        self.assertLessEqual(np.sum(w*abs(x-m)),np.sum(w*abs(x-x.mean())))

    def test_gate_accepts_paired_persistent_gain_not_a_large_single_day(self):
        ts=pd.date_range('2025-04-01 01:00',periods=30*24,freq='h',tz=ft.TZ)
        d=pd.DataFrame({'ts':ts,'base':100.,'energy':0.,'weight':1.})
        self.assertTrue(block_gate(d,np.full(len(d),98.))['accepted'])
        candidate=np.full(len(d),101.)
        candidate[:24]=0.
        self.assertFalse(block_gate(d,candidate)['accepted'])


if __name__=='__main__':unittest.main()
