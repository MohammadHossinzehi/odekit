import unittest
from fractions import Fraction as F

from odekit import trees, tableaux as T


class TreeEnumeration(unittest.TestCase):
    def test_counts_match_oeis_a000081(self):
        expected = [1, 1, 2, 4, 9, 20, 48, 115, 286]
        got = [len(trees.trees_of_order(n)) for n in range(1, 10)]
        self.assertEqual(got, expected)

    def test_trees_are_canonical_and_have_right_order(self):
        for n in range(1, 8):
            for t in trees.trees_of_order(n):
                self.assertEqual(trees.order(t), n)
                self.assertEqual(t, trees.canon(t))

    def test_gamma_known_values(self):
        leaf = ()
        cherry = ((), ())          # root with two leaves: gamma = 3
        chain3 = (((),),)          # path of 3 vertices: gamma = 6
        self.assertEqual(trees.gamma(leaf), 1)
        self.assertEqual(trees.gamma(cherry), 3)
        self.assertEqual(trees.gamma(chain3), 6)
        self.assertEqual(trees.tree_to_str(cherry), "[* *]")

    def test_chain_tree_gamma_is_factorial(self):
        # the tall tree of n vertices encodes the condition b A^(n-2) c = 1/n!
        for n in range(1, 7):
            chain = ()
            for _ in range(n - 1):
                chain = (chain,)
            self.assertEqual(trees.gamma(chain), _fact(n))


def _fact(n):
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


class TableauOrders(unittest.TestCase):
    def test_every_tableau_has_its_claimed_order(self):
        for tab in T.ALL.values():
            with self.subTest(tab=tab.name):
                self.assertEqual(tab.verified_order(), tab.order)
                if tab.embedded:
                    self.assertEqual(tab.verified_err_order(), tab.err_order)

    def test_row_sum_condition(self):
        for tab in T.EXPLICIT.values():
            for i in range(tab.stages):
                self.assertEqual(sum(tab.a_exact[i]), tab.c_exact[i])

    def test_fsal_last_row_equals_b(self):
        for tab in (T.BS32, T.DOPRI54):
            self.assertEqual(tab.a_exact[-1], tab.b_exact)
            self.assertEqual(tab.c_exact[-1], 1)

    def test_a_single_typo_is_caught(self):
        a = [list(r) for r in T.DOPRI54.a_exact]
        a[4][2] += F(1, 10 ** 9)
        p = trees.check_order(a, T.DOPRI54.b_exact, T.DOPRI54.c_exact)
        self.assertLess(p, 5)

    def test_rk4_fails_exactly_the_order5_conditions(self):
        bad = trees.failing_conditions(T.RK4.a_exact, T.RK4.b_exact,
                                       T.RK4.c_exact, 5)
        self.assertTrue(bad)
        self.assertEqual(trees.failing_conditions(
            T.RK4.a_exact, T.RK4.b_exact, T.RK4.c_exact, 4), [])


class Stability(unittest.TestCase):
    def test_explicit_stability_polynomial_of_rk4(self):
        z = complex(-0.7, 0.3)
        expect = 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24
        self.assertAlmostEqual(abs(T.RK4.stability_function(z) - expect), 0,
                               places=13)

    def test_trbdf2_is_l_stable(self):
        tab = T.TRBDF2
        self.assertLess(abs(tab.stability_function(complex(-1e12, 0))), 1e-9)
        # A-stable: |R| <= 1 on a sample of the left half plane
        for x in (0.1, 1, 10, 100, 1e4):
            for y in (-1e3, -10, 0, 1, 50):
                self.assertLessEqual(
                    abs(tab.stability_function(complex(-x, y))), 1 + 1e-12)

    def test_known_real_stability_boundaries(self):
        from odekit.__main__ import stability_boundary
        self.assertAlmostEqual(stability_boundary(T.EULER), 2.0, places=6)
        self.assertAlmostEqual(stability_boundary(T.RK4), 2.785293, places=5)


if __name__ == "__main__":
    unittest.main()
