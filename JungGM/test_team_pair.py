"""Optional dependency tests; run in the documented team environment."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from test_features import sample

AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ['catboost','pvlib'])


@unittest.skipUnless(AVAILABLE,'requires requirements-team.txt')
class TeamPairFeatures(unittest.TestCase):
    def test_day_boundary_isolation_and_forecast_units(self):
        import team_pair_experiment as t
        src=t.source();sites,a,b,ix=sample()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a.to_parquet(root/'a.parquet');b.to_parquet(root/'b.parquet')
            wa=src.weather(root/'a.parquet','a',ix,sites.wa_station.unique())
            wb=src.weather(root/'b.parquet','b',ix,sites.wb_station.unique())
            site=next(sites.itertuples(index=False))
            full=src.build_site(site,ix,wa[site.wa_station],wb[site.wb_station])
            self.assertAlmostEqual(float(full.b17_wind_speed.iloc[0]),30/3.6,places=5)
            self.assertEqual(float(full.b17_snow.iloc[0]),300.)
            wb[site.wb_station].loc[ix[24]:,'b17_cloud']=9999.
            second=src.build_site(site,ix,wa[site.wa_station],wb[site.wb_station])
            pd.testing.assert_frame_equal(full.iloc[:24],second.iloc[:24])

    def test_missing_forecast_never_drops_targets(self):
        import team_pair_experiment as t
        src=t.source();sites,a,b,ix=sample()
        a=a[~((a['round']==17)&(a.ts==ix[12]))]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a.to_parquet(root/'a.parquet');b.to_parquet(root/'b.parquet')
            wa=src.weather(root/'a.parquet','a',ix,sites.wa_station.unique())
            wb=src.weather(root/'b.parquet','b',ix,sites.wb_station.unique())
            site=next(sites.itertuples(index=False))
            d=src.build_site(site,ix,wa[site.wa_station],wb[site.wb_station])
            self.assertEqual(len(d),len(ix));self.assertTrue(pd.isna(d.a17_cloud.iloc[12]))
            self.assertEqual(len(src.NAMES),160)


if __name__=='__main__':unittest.main()
