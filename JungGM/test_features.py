"""Synthetic feature contracts; no competition records are needed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import features as f


def sample(days=2):
    index = pd.date_range("2025-01-31 01:00", periods=days*24, freq="h", tz=f.TZ)
    sites = pd.DataFrame({"site_id": ["PV002", "PV001"], "capacity_kw": [100.0, 200.0],
        "lat": [36.0, 36.5], "lon": [127.0, 127.5], "wa_station": ["WA001"]*2,
        "wb_station": ["WB001"]*2, "wa_dist": [1000.0, 2000.0], "wb_dist": [500.0, 1500.0]})
    frames = []
    for station, value, columns in (("wa_station", "WA001", f.WA), ("wb_station", "WB001", f.WB)):
        a = pd.MultiIndex.from_product([[value], [10, 17], index], names=[station, "round", "ts"]).to_frame(index=False)
        for c in columns:
            a[c] = 30.0
        frames.append(a)
    return sites, frames[0], frames[1], index


class FeatureContracts(unittest.TestCase):
    def test_target_day_and_site_isolation(self):
        a = np.r_[np.ones(24), np.ones(24)*100, np.ones(24)*1000]
        z = f.issued_day_mean(a, 7)
        np.testing.assert_array_equal(z, a)
        a[:24] = np.nan
        z = f.issued_day_mean(a, 3)
        self.assertTrue(np.isnan(z[:24]).all())
        self.assertEqual(z[24], 100)

    def test_no_generation_effect_and_missing_row_retention(self):
        sites, fa, fb, ix = sample()
        fa = fa.drop(index=0)
        y = pd.DataFrame({"site_id": ["PV001"], "ts": [ix[12]], "energy": [20.0]})
        a = f.build_month(sites, fa, fb, ix, y)
        y["energy"] = 999.0
        b = f.build_month(sites, fa, fb, ix, y)
        self.assertEqual(len(a), 2*len(ix))
        self.assertEqual(a.energy.notna().sum(), 1)
        self.assertEqual(a.r10_wa_cloud.isna().sum(), 2)
        self.assertTrue(a[f.feature_schema()["enhanced"]].equals(b[f.feature_schema()["enhanced"]]))
        self.assertNotIn("energy", f.feature_schema()["enhanced"])
        self.assertFalse(a.energy.equals(b.energy))

    def test_midnight_bounds_and_feature_units(self):
        left, right = f.month_bounds("2025-01")
        self.assertEqual(left, pd.Timestamp("2025-01-01 01:00", tz=f.TZ))
        self.assertEqual(right, pd.Timestamp("2025-02-01 00:00", tz=f.TZ))
        self.assertEqual(str(f._months(right, right)[0]), "2025-01")
        sites, fa, fb, ix = sample()
        d = f.build_month(sites, fa, fb, ix)
        self.assertTrue(np.isnan(d.energy).all())
        np.testing.assert_allclose(d.r17_wb_wind_speed, 30/3.6, rtol=1e-6)
        np.testing.assert_allclose(d.r17_wb_snow, 300)
        self.assertEqual(len(f.feature_schema()["base10"]), 24)

    def test_cache_read_exact_cutoff(self):
        sites, fa, fb, ix = sample()
        d = f.build_month(sites, fa, fb, ix)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            (cache/"train").mkdir()
            (cache/"schema.json").write_text(__import__("json").dumps(f.feature_schema()), encoding="utf-8")
            for m in ("2025-01", "2025-02"):
                left, right = f.month_bounds(m)
                d.loc[d.ts.between(left, right)].to_parquet(cache/"train"/f"{m}.parquet", index=False)
            cutoff = pd.Timestamp("2025-01-31 17:00", tz=f.TZ)
            out = f.load_rows(cache, ix[0], cutoff)
            self.assertEqual(out.ts.max(), cutoff)
            self.assertEqual(len(out), 34)
            self.assertEqual(len(f.load_rows(cache, ix[0], cutoff, labeled=True)), 0)

    def test_forecast_changes_do_not_cross_target_day(self):
        sites, fa, fb, ix = sample()
        a = f.build_month(sites, fa, fb, ix)
        fa.loc[fa.ts > ix[23], "wa_cloud"] = 90
        b = f.build_month(sites, fa, fb, ix)
        cols = f.feature_schema()["enhanced"]
        self.assertTrue(a.loc[a.ts <= ix[23], cols].equals(b.loc[b.ts <= ix[23], cols]))
        self.assertFalse(a.loc[a.ts > ix[23], cols].equals(b.loc[b.ts > ix[23], cols]))


if __name__ == "__main__":
    unittest.main()
