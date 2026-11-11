"""Small numerical checks against direct grounded-system calculations."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numba import set_num_threads
import egwc as E


class NumericalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(1)
        cls.g = E.Graph(6, [0, 0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 5, 2, 3, 3, 5, 4, 5])

    def test_singleton_formula(self):
        g = self.g
        direct = []
        for edge in range(g.m):
            mask = np.arange(g.m) == edge
            direct.append(E.egwc(g, mask))
        np.testing.assert_allclose(E.exact_singleton_scores(g), direct, rtol=1e-10)

    def test_local_gain_formula(self):
        g = self.g
        selected = np.zeros(g.m, dtype=bool)
        selected[[0, 4]] = True
        before = E.egwc(g, selected)
        ids, gains = E.exact_gains(g, selected)
        direct = [before - E.egwc(g, selected | (np.arange(g.m) == i)) for i in ids]
        np.testing.assert_allclose(gains, direct, rtol=1e-10, atol=1e-12)

    def test_terminal_recovery(self):
        g = self.g
        selected = np.arange(g.m) == 0
        ids, exact = E.exact_gains(g, selected)
        candidates, recovered, stats = E.schur_round(
            g, selected, 0.1, 0.01, 8, np.random.default_rng(0), b=g.n, threads=1)
        np.testing.assert_array_equal(candidates, ids)
        np.testing.assert_allclose(recovered, exact, rtol=1e-10, atol=1e-12)
        self.assertEqual(stats['nonterminals'], 0)
        self.assertEqual(stats['forests'], 0)

    def test_sampling_and_refinement(self):
        g = self.g
        first, _, _ = E.approx_singleton(g, q=8, seed=12345)
        chosen, elapsed, rounds = E.schur_greedy(
            g, 4, first, eps_s=0.1, q=8, seed=0, threads=1,
            b=2, r_min=32, r_max=64, refine=3)
        self.assertEqual(len(set(chosen)), 4)
        self.assertEqual(chosen[0], first)
        self.assertEqual(len(rounds), 3)
        self.assertGreaterEqual(elapsed, 0)
        values = [E.egwc(g, np.isin(np.arange(g.m), chosen[:k])) for k in range(1, 5)]
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(all(values[i + 1] < values[i] for i in range(3)))
        for rd in rounds:
            self.assertGreater(rd['forests'], 0)
            self.assertLessEqual(rd['refine_rhs'], 6)

    def test_loader_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'edges.txt'
            path.write_text('# graph\n10 20\n20 10\n20 30\n30 30\n100 101\n', encoding='utf-8')
            g, info = E.load(path)
        self.assertEqual((g.n, g.m), (3, 2))
        self.assertEqual(info['self_loops'], 1)
        self.assertEqual(info['components'], 2)
        np.testing.assert_array_equal(g.u, [0, 1])
        np.testing.assert_array_equal(g.v, [1, 2])


if __name__ == '__main__':
    unittest.main()
