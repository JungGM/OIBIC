import unittest

import numpy as np

from grouping import daily_gate, objective, optimize


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.c = np.repeat(10_000., 4)
        self.ref = np.array([1, 1, 0, 0])
        self.e = np.array([[10., 10., -10., -10.], [-3., -3., 3., 3.]])
        self.w = np.array([2., 1.])

    def test_absolute_is_after_group_sum(self):
        self.assertEqual(objective(self.e, self.w, self.ref), 92.)
        self.assertEqual(objective(self.e, self.w, [1, 0, 1, 0]), 0.)

    def test_search_seed_incumbent_and_capacity(self):
        a, report = optimize(self.e, self.w, self.c, self.ref, steps=150, restarts=3)
        b, other = optimize(self.e, self.w, self.c, self.ref, steps=150, restarts=3)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(report, other)
        self.assertGreaterEqual(self.c @ a, 20000)
        self.assertGreaterEqual(self.c @ (1 - a), 20000)
        self.assertLessEqual(objective(self.e, self.w, a), objective(self.e, self.w, self.ref))
        self.assertEqual(report["best_weighted_error"], 0.)

    def test_missing_and_infeasible_rejected(self):
        e = self.e.copy()
        e[0, 0] = np.nan
        for args in [(e, self.w, self.c, self.ref),
                     (self.e, [-1, 1], self.c, self.ref),
                     (self.e, self.w, self.c, [1, 0, 0, 0]),
                     (self.e, self.w, self.c, [1, 1, 0, 2])]:
            with self.assertRaises(ValueError):
                optimize(*args, steps=10)

    def test_zero_weight_returns_reference(self):
        a, report = optimize(self.e, [0, 0], self.c, self.ref, steps=10)
        np.testing.assert_array_equal(a, self.ref)
        self.assertEqual(report["best_weighted_error"], 0.)

    def test_gate_requires_stable_later_period(self):
        self.assertTrue(daily_gate(np.ones(30) * 100, np.ones(30) * 95)["accepted"])
        self.assertFalse(daily_gate(np.ones(30) * 100, np.ones(30) * 100)["accepted"])
        self.assertFalse(daily_gate(np.ones(3) * 100, np.ones(3) * 90)["accepted"])


if __name__ == "__main__":
    unittest.main()
