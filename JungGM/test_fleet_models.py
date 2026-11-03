"""Synthetic leakage and aggregation invariants for the fleet experiments."""
import unittest
import numpy as np
import pandas as pd
import direct_fleet as df
import fleet_residual as fr
import pipeline as p
import tempfile
from pathlib import Path
from submission_guard import ensure_writable
import spatial_forecasts as spatial


class FleetContracts(unittest.TestCase):
    def test_neighbor_graph_counts_weather_stations_once(self):
        s=pd.DataFrame({'lat':[35.,35.01,35.02,35.03], 'lon':[127.,127.01,127.02,127.03],
                        'wb_station':['B1','B1','B2','B3']})
        w=spatial.weights(s,'wb_station',2)
        np.testing.assert_allclose(w.sum(axis=1),1)
        self.assertEqual(float(w[0,:2].sum()),0)
        self.assertEqual(int(np.count_nonzero(w[2,:2])),1)
    def test_neighbor_missing_forecast_is_not_treated_as_zero(self):
        x=np.array([[np.nan,10.,30.]])
        w=np.array([[.5,.5,0.]])
        mean,std=spatial.moments(x,w)
        self.assertEqual(float(mean[0,0]),10.)
        self.assertEqual(float(std[0,0]),0.)
    def test_official_artifacts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);ensure_writable(path)
            (path/'validation.json').write_text('{"official_submitted":true}',encoding='utf-8')
            with self.assertRaises(ValueError):ensure_writable(path)
    def test_receipt_blocks_overwrite_even_with_missing_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'공식제출결과_20260101.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):ensure_writable(path)
    def test_aggregate_features_do_not_depend_on_label_values(self):
        ts=p.stamp('2025-01-01 12:00')
        meta=pd.DataFrame({'site_id':['PV001','PV002'],'capacity_kw':[100.,300.],
                           'current_group':['A','A']}).set_index('site_id')
        rows=pd.DataFrame({'site_id':['PV001','PV002'],'ts':[ts,ts],'energy':[10.,90.]})
        rows=pd.concat([rows,pd.DataFrame({c:[1.,3.] for c in df.FEATURES})],axis=1)
        a=df.aggregate(rows,meta,False)
        rows.energy=[9999.,1.]
        b=df.aggregate(rows,meta,False)
        pd.testing.assert_frame_equal(a,b)
        self.assertAlmostEqual(a['avg_cosz_positive'].iloc[0],2.5)
        self.assertEqual(a.capacity.iloc[0],400)

    def test_future_labels_cannot_change_calibrator(self):
        ix=pd.date_range(p.stamp('2024-12-01 01:00'),p.stamp('2025-01-03 00:00'),freq='h')
        rng=np.random.default_rng(1);x=rng.uniform(.1,.9,len(ix))
        d=pd.DataFrame({'ts':ix,'v8':x*1000,'capacity':2000.,'weight':100.,
                        'energy':x*1000+rng.normal(0,10,len(ix)),'v8_util':x/2})
        cfg=dict(objective='regression',rounds=5,leaves=3)
        a,record=fr.fit(d,'2025-01-01 01:00',cfg)
        d.loc[d.ts>p.stamp('2024-12-31 17:00'),'energy']=1e12
        b,_=fr.fit(d,'2025-01-01 01:00',cfg)
        np.testing.assert_array_equal(a.predict(d[fr.names(d)]),b.predict(d[fr.names(d)]))
        self.assertEqual(record['last_label'],str(p.stamp('2024-12-31 17:00')))

    def test_direct_model_training_cutoff(self):
        ix=pd.date_range(p.stamp('2024-12-01 01:00'),p.stamp('2025-01-03 00:00'),freq='h')
        x=np.linspace(.1,.8,len(ix))
        d=pd.DataFrame({'ts':ix,'avg_cosz_positive':x,'capacity':20000.,'sites':100,
                        'weight':100.,'energy':x*12000})
        cfg=dict(objective='regression',solar_normalized=True,leaves=7)
        a,record=df.fit(d,'2025-01-01 01:00',cfg)
        d.loc[d.ts>p.stamp('2024-12-31 17:00'),'energy']=1e12
        b,_=df.fit(d,'2025-01-01 01:00',cfg)
        np.testing.assert_array_equal(df.predict(a,d,cfg),df.predict(b,d,cfg))
        self.assertEqual(record['last_label'],str(p.stamp('2024-12-31 17:00')))


if __name__=='__main__':unittest.main()
