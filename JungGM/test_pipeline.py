"""Synthetic contracts for deadlines, complete-fleet scoring and model choice."""
import unittest

import numpy as np
import pandas as pd

import pipeline as p


class PipelineContracts(unittest.TestCase):
    def test_issue_deadline_right_closed_midnight(self):
        cases = {
            "2025-06-01 00:00": "2025-05-30 17:00",
            "2025-06-01 01:00": "2025-05-31 17:00",
            "2025-06-01 23:00": "2025-05-31 17:00",
            "2025-06-02 00:00": "2025-05-31 17:00",
            "2025-06-02 01:00": "2025-06-01 17:00",
        }
        for target, deadline in cases.items():
            self.assertEqual(p.issue_deadline(p.stamp(target)), p.stamp(deadline))
        self.assertEqual(p.issue_deadline(pd.Timestamp("2025-05-31 16:00Z")),
                         p.stamp("2025-05-31 17:00"))
        with self.assertRaises(ValueError):
            p.issue_deadline("2025-06-01 01:00")

    def test_loss_cancels_within_each_group_only(self):
        actual = np.zeros((2, 4))
        prediction = np.array([[10., -10., 3., -3.], [2., 2., -2., -2.]])
        assignment = np.array([1, 1, 0, 0])
        # First hour cancels inside each group. Second hour's fleet error
        # cancels across groups, but the two group errors still cost 16.
        self.assertEqual(p.cost(actual, prediction, np.array([1., 2.]), assignment), 16.)
        times = pd.date_range("2025-01-01 01:00", periods=2, freq="h", tz=p.TZ)
        metrics = p.metrics(times, actual, prediction, np.array([1., 2.]),
                            np.ones(2), assignment)
        self.assertEqual(metrics["weighted_error"], 16.)
        self.assertEqual(metrics["fleet_error_lower_bound"], 0.)
        self.assertEqual(metrics["penalty_krw"], 1.6)

    def test_simplex_blend_finds_declared_grid_optimum(self):
        names = ["baseline", "opposite"]
        folds = []
        # A=3 and B=-7; their constrained minimum is .7*A+.3*B=0.
        for multiplier in [1., 100.]:
            gp = np.array([[[3., 3.]], [[-7., -7.]]]) * multiplier
            folds.append({"losses": [6.*multiplier, 14.*multiplier],
                          "group_predictions": gp, "group_actual": np.zeros((1, 2)),
                          "w": np.ones(1)})
        weights, selection, history = p.select_ensemble(folds, names)
        np.testing.assert_allclose(weights, [.7, .3], atol=1e-12)
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertLess(selection["mean_monthly_relative_loss"], 1e-12)
        self.assertEqual(len(history), 11)

    def test_block_importance_does_not_count_proxy_months_as_extra_evidence(self):
        # Three proxy summer months favor the alternative, but its winter
        # loss is three times baseline. Equal block weighting keeps baseline;
        # default equal-month weighting would choose the alternative instead.
        summer = {"losses": [4., 0.],
                  "group_predictions": np.array([[[2., 2.]], [[0., 0.]]]),
                  "group_actual": np.zeros((1, 2)), "w": np.ones(1),
                  "importance": 1/6}
        winter = {"losses": [4., 12.],
                  "group_predictions": np.array([[[2., 2.]], [[6., 6.]]]),
                  "group_actual": np.zeros((1, 2)), "w": np.ones(1),
                  "importance": .5}
        folds = [summer.copy() for _ in range(3)] + [winter]
        weighted, _, _ = p.select_ensemble(folds, ["baseline", "alternative"])
        np.testing.assert_array_equal(weighted, [1., 0.])
        unweighted = [{k: v for k, v in fold.items() if k != "importance"} for fold in folds]
        default, _, _ = p.select_ensemble(unweighted, ["baseline", "alternative"])
        np.testing.assert_array_equal(default, [0., 1.])

    def test_complete_panel_reorders_by_time_then_site(self):
        times = pd.date_range("2025-01-01 01:00", periods=2, freq="h", tz=p.TZ)
        d = pd.DataFrame({"site_id": ["B", "A", "B", "A"],
                          "ts": [times[1], times[0], times[0], times[1]],
                          "energy": [4., 1., 2., 3.], "prediction": [40., 10., 20., 30.]})
        prices = pd.DataFrame({"ts": times, "smp": [100., 50.], "k": [1, 3]})
        ordered, actual, pp, w, smp = p.panel(d, ["prediction"], ["A", "B"], prices)
        np.testing.assert_array_equal(actual, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(pp["prediction"], [[10, 20], [30, 40]])
        np.testing.assert_array_equal(w, [100, 150])
        self.assertTrue(ordered.equals(times))

    def test_missing_truth_or_price_and_duplicates_fail(self):
        ts = p.stamp("2025-01-01 01:00")
        d = pd.DataFrame({"site_id": ["A", "B"], "ts": [ts, ts],
                          "energy": [1., 2.], "prediction": [1., 2.]})
        prices = pd.DataFrame({"ts": [ts], "smp": [100.], "k": [1]})
        for bad in [d.iloc[:1], d.assign(energy=[1., np.nan]), pd.concat([d, d.iloc[:1]])]:
            with self.assertRaises(ValueError):
                p.panel(bad, ["prediction"], ["A", "B"], prices)
        with self.assertRaises(ValueError):
            p.panel(d, ["prediction"], ["A", "B"], prices.assign(smp=np.nan))

    def test_daily_losses_assign_midnight_to_previous_operating_day(self):
        times = pd.DatetimeIndex([p.stamp("2025-02-01 00:00"), p.stamp("2025-02-01 01:00")])
        actual = np.zeros((2, 2))
        pred = np.array([[1., 2.], [3., 4.]])
        losses = p.daily_losses(times, actual, pred, np.ones(2), np.array([1, 0]))
        self.assertEqual(losses.loc[p.stamp("2025-01-31 00:00")], 3.)
        self.assertEqual(losses.loc[p.stamp("2025-02-01 00:00")], 7.)


if __name__ == "__main__":
    unittest.main()
