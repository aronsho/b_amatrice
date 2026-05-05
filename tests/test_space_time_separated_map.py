import unittest

import numpy as np

from seismostats.analysis.b_significant import est_morans_i

from functions.space_time_separated_map import est_morans_i_temporal


def build_temporal_dense_w(n_time: int, n_space: int) -> np.ndarray:
    """Replicate the original upper-triangular temporal neighbor matrix."""
    total = n_time * n_space
    w = np.zeros((total, total), dtype=float)
    rows = np.arange(total - n_space)
    cols = rows + n_space
    w[rows, cols] = 1.0
    return w


class TestTemporalMoransI(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(12345)

    def assert_matches_dense(
            self,
            values: np.ndarray,
            n_time: int,
            n_space: int,
            mean_v=None) -> None:
        w = build_temporal_dense_w(n_time=n_time, n_space=n_space)
        ac_dense, n_dense, n_p_dense = est_morans_i(
            values, w=w, mean_v=mean_v)
        ac_fast, n_fast, n_p_fast = est_morans_i_temporal(
            values, n_space=n_space, mean_v=mean_v)

        self.assertEqual(n_dense, n_fast)
        self.assertEqual(n_p_dense, n_p_fast)
        np.testing.assert_allclose(ac_dense, ac_fast, rtol=1e-12, atol=1e-12)

    def test_matches_dense_without_explicit_mean(self) -> None:
        for n_time, n_space in [(4, 8), (8, 16), (16, 32)]:
            with self.subTest(n_time=n_time, n_space=n_space):
                values = self.rng.normal(size=n_time * n_space)
                values[self.rng.random(values.size) < 0.12] = np.nan
                self.assert_matches_dense(values, n_time, n_space)

    def test_matches_dense_with_scalar_mean(self) -> None:
        for n_time, n_space in [(4, 8), (8, 16), (32, 64)]:
            with self.subTest(n_time=n_time, n_space=n_space):
                values = self.rng.normal(loc=1.5, scale=0.7,
                                         size=n_time * n_space)
                values[self.rng.random(values.size) < 0.08] = np.nan
                self.assert_matches_dense(
                    values, n_time, n_space, mean_v=0.25)

    def test_matches_dense_with_vector_mean(self) -> None:
        for n_time, n_space in [(4, 8), (8, 64), (32, 64)]:
            with self.subTest(n_time=n_time, n_space=n_space):
                values = self.rng.normal(size=n_time * n_space)
                values[self.rng.random(values.size) < 0.05] = np.nan
                mean_v = self.rng.normal(scale=0.3, size=n_time * n_space)
                self.assert_matches_dense(
                    values, n_time, n_space, mean_v=mean_v)

    def test_rejects_non_multiple_of_n_space(self) -> None:
        with self.assertRaises(ValueError):
            est_morans_i_temporal(np.arange(10, dtype=float), n_space=4)

    def test_returns_nan_when_no_valid_temporal_pairs_exist(self) -> None:
        values = np.array([1.0, np.nan, np.nan, 2.0], dtype=float)
        ac, n, n_p = est_morans_i_temporal(values, n_space=2)
        self.assertTrue(np.isnan(ac))
        self.assertEqual(n, 2)
        self.assertEqual(n_p, 0)

    def test_returns_nan_for_zero_variance(self) -> None:
        values = np.array([3.0, 3.0, 3.0, 3.0], dtype=float)
        ac, n, n_p = est_morans_i_temporal(values, n_space=2)
        self.assertTrue(np.isnan(ac))
        self.assertEqual(n, 4)
        self.assertEqual(n_p, 2)


if __name__ == "__main__":
    unittest.main()
