import random
import unittest

from odekit import linalg


class LU(unittest.TestCase):
    def test_random_systems(self):
        rng = random.Random(7)
        for n in (1, 2, 3, 6, 12):
            a = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
            x = [rng.uniform(-5, 5) for _ in range(n)]
            b = linalg.matvec(a, x)
            got = linalg.solve(a, b)
            for u, v in zip(got, x):
                self.assertAlmostEqual(u, v, places=8)

    def test_requires_pivoting(self):
        a = [[0.0, 1.0], [1.0, 0.0]]
        self.assertEqual(linalg.solve(a, [3.0, 4.0]), [4.0, 3.0])

    def test_singular(self):
        with self.assertRaises(linalg.SingularMatrixError):
            linalg.lu_factor([[1.0, 2.0], [2.0, 4.0]])

    def test_factors_reusable(self):
        a = [[4.0, 1.0], [2.0, 3.0]]
        f = linalg.lu_factor(a)
        for b in ([1.0, 0.0], [0.0, 1.0], [5.0, -2.0]):
            x = linalg.lu_solve(f, b)
            r = linalg.matvec(a, x)
            self.assertAlmostEqual(r[0], b[0])
            self.assertAlmostEqual(r[1], b[1])


class Jacobian(unittest.TestCase):
    def test_finite_difference_matches_analytic(self):
        def f(t, y):
            return [y[0] * y[1], y[0] ** 2 - 3 * y[1], t * y[1]]

        y = [1.5, -2.0, 0.3]
        jac, nfev = linalg.fd_jacobian(f, 2.0, y)
        exact = [[y[1], y[0], 0.0], [2 * y[0], -3.0, 0.0], [0.0, 2.0, 0.0]]
        self.assertEqual(nfev, 4)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(jac[i][j], exact[i][j], places=6)

    def test_rms_norm_is_one_at_tolerance(self):
        y = [10.0, 0.0]
        err = [1e-3 * 10 + 1e-6, 1e-6]
        self.assertAlmostEqual(linalg.rms_norm(err, y, y, 1e-3, 1e-6), 1.0)


if __name__ == "__main__":
    unittest.main()
