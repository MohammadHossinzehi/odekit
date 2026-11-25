import math
import unittest

from odekit import solve_ivp, problems

HARM = problems.harmonic()


def fixed_run(method, h, t_end=2.0):
    """Integrate the harmonic oscillator with a forced constant step."""
    r = solve_ivp(HARM["fun"], (0.0, t_end), HARM["y0"], method=method,
                  first_step=h, max_step=h, rtol=1.0, atol=1e9,
                  dense_output=True)
    end_err = abs(r.y[-1][0] - math.cos(t_end))
    pts = [(i + 0.37) * h for i in range(int(round(t_end / h)))]
    dense_err = max(abs(r.sol(t)[0] - math.cos(t)) for t in pts)
    return end_err, dense_err, r


class Convergence(unittest.TestCase):
    """Empirical order: halve h and look at log2(err(h) / err(h/2))."""

    expected = {"Euler": 1, "HeunEuler21": 2, "BS32": 3, "RK4": 4,
                "RKF45": 5, "DOPRI54": 5, "TRBDF2": 2}

    def test_empirical_orders(self):
        for m, p in self.expected.items():
            with self.subTest(method=m):
                h = 0.05 if p < 5 else 0.1
                e1, _, _ = fixed_run(m, h, t_end=4.0)
                e2, _, _ = fixed_run(m, h / 2, t_end=4.0)
                self.assertAlmostEqual(math.log2(e1 / e2), p, delta=0.35)

    def test_dense_output_orders(self):
        # DOPRI's continuous extension is 4th order (local error h^5);
        # Hermite interpolation gives at least 3 on every other method.
        _, d1, _ = fixed_run("DOPRI54", 0.1)
        _, d2, _ = fixed_run("DOPRI54", 0.05)
        self.assertGreater(math.log2(d1 / d2), 3.7)
        _, d1, _ = fixed_run("RK4", 0.05)
        _, d2, _ = fixed_run("RK4", 0.025)
        self.assertGreater(math.log2(d1 / d2), 3.5)


class Adaptivity(unittest.TestCase):
    def test_tolerance_proportionality(self):
        for m in ("BS32", "RKF45", "DOPRI54"):
            errs = []
            for tol in (1e-4, 1e-7, 1e-10):
                r = solve_ivp(HARM["fun"], (0, 10), HARM["y0"], method=m,
                              rtol=tol, atol=tol)
                errs.append(max(abs(a - b) for a, b in
                                zip(r.y[-1], HARM["exact"](10))))
            with self.subTest(method=m):
                self.assertLess(errs[1], errs[0])
                self.assertLess(errs[2], errs[1])
                self.assertLess(errs[2], 1e-7)

    def test_fsal_costs_six_evaluations_per_dopri_step(self):
        pr = problems.exponential()
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], rtol=1e-8,
                      atol=1e-10)
        s = r.stats
        # 1 initial f, 1 for the starting step heuristic, 6 per attempt
        self.assertEqual(s.nfev, 2 + 6 * (s.naccept + s.nreject))

    def test_dopri_dense_output_accuracy(self):
        r = solve_ivp(HARM["fun"], (0, 10), HARM["y0"], rtol=1e-9,
                      atol=1e-12, dense_output=True)
        worst = max(abs(r.sol(t)[0] - math.cos(t))
                    for t in [0.0137 * i for i in range(700)])
        self.assertLess(worst, 1e-7)
        # at breakpoints the interpolant reproduces the steps exactly
        for t, y in zip(r.t, r.y):
            self.assertAlmostEqual(r.sol(t)[0], y[0], places=12)

    def test_backward_integration_and_t_eval(self):
        y_end = [math.cos(10), -math.sin(10)]
        te = [9.0, 5.0, 0.5, 0.0]
        r = solve_ivp(HARM["fun"], (10, 0), y_end, rtol=1e-10, atol=1e-12,
                      t_eval=te)
        self.assertEqual(r.t, te)
        for t, y in zip(r.t, r.y):
            self.assertAlmostEqual(y[0], math.cos(t), places=8)

    def test_t_eval_does_not_shrink_steps(self):
        a = solve_ivp(HARM["fun"], (0, 10), HARM["y0"], rtol=1e-6)
        b = solve_ivp(HARM["fun"], (0, 10), HARM["y0"], rtol=1e-6,
                      t_eval=[0.01 * i for i in range(1001)])
        self.assertEqual(a.stats.naccept, b.stats.naccept)
        self.assertEqual(len(b.t), 1001)

    def test_vector_atol(self):
        pr = problems.lorenz()
        r = solve_ivp(pr["fun"], (0, 1), pr["y0"], atol=[1e-8, 1e-8, 1e-8],
                      rtol=1e-8)
        self.assertTrue(r.success)

    def test_methods_agree_on_lorenz_short_horizon(self):
        pr = problems.lorenz()
        ends = []
        for m in ("BS32", "RKF45", "DOPRI54"):
            r = solve_ivp(pr["fun"], (0, 2), pr["y0"], method=m, rtol=1e-10,
                          atol=1e-10)
            ends.append(r.y[-1])
        for e in ends[1:]:
            for a, b in zip(e, ends[0]):
                self.assertAlmostEqual(a, b, delta=1e-5)

    def test_scalar_initial_value_and_empty_interval(self):
        r = solve_ivp(lambda t, y: [-y[0]], (0, 1), 2.0)
        self.assertAlmostEqual(r.y[-1][0], 2 * math.exp(-1), places=6)
        r = solve_ivp(lambda t, y: [-y[0]], (1, 1), [1.0])
        self.assertEqual(r.t, [1.0])

    def test_fixed_step_method_requires_step(self):
        with self.assertRaises(ValueError):
            solve_ivp(HARM["fun"], (0, 1), HARM["y0"], method="RK4")
        with self.assertRaises(ValueError):
            solve_ivp(HARM["fun"], (0, 1), HARM["y0"], method="nope")


if __name__ == "__main__":
    unittest.main()
