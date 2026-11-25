import math
import unittest

from odekit import solve_ivp, problems


class Robertson(unittest.TestCase):
    # widely published reference values at t = 40
    REF40 = [0.7158270687, 9.185534764e-6, 0.2841637457]
    ATOL = [1e-8, 1e-12, 1e-8]

    def test_matches_reference_at_t40(self):
        pr = problems.robertson()
        r = solve_ivp(pr["fun"], (0, 40), pr["y0"], method="TRBDF2",
                      rtol=1e-8, atol=[1e-10, 1e-14, 1e-10], jac=pr["jac"])
        for got, ref in zip(r.y[-1], self.REF40):
            self.assertAlmostEqual(got / ref, 1.0, delta=1e-6)

    def test_mass_conservation_to_roundoff(self):
        # TR-BDF2 is a linear method, so it preserves linear invariants
        pr = problems.robertson()
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-6, atol=self.ATOL)
        for y in r.y:
            self.assertAlmostEqual(sum(y), 1.0, places=12)

    def test_cheap_with_finite_difference_jacobian(self):
        pr = problems.robertson()
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-6, atol=self.ATOL)
        self.assertTrue(r.success)
        self.assertLess(r.stats.naccept, 2000)
        self.assertLess(r.stats.njev, 100)


class VanDerPol(unittest.TestCase):
    def test_mu_1e6_against_hairer_wanner_reference(self):
        pr = problems.van_der_pol(1e6)
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-8, atol=1e-8, jac=pr["jac"])
        for got, ref in zip(r.y[-1], pr["reference"]):
            self.assertAlmostEqual(got, ref, delta=1e-5)

    def test_auto_switches_to_stiff_solver(self):
        pr = problems.van_der_pol(1e6)
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="AUTO",
                      rtol=1e-6, atol=1e-6)
        self.assertTrue(r.success)
        self.assertEqual(len(r.switches), 1)
        self.assertEqual(r.switches[0][1:], ("DOPRI54", "TRBDF2"))
        self.assertLess(r.switches[0][0], 1e-3)

    def test_auto_stays_explicit_on_nonstiff_problem(self):
        pr = problems.kepler()
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="AUTO",
                      rtol=1e-8)
        self.assertEqual(r.switches, [])
        self.assertEqual(r.stats.njev, 0)


class StiffLinear(unittest.TestCase):
    def test_large_steps_on_stiff_decay(self):
        lam = -1e8
        r = solve_ivp(lambda t, y: [lam * (y[0] - math.cos(t))], (0, 10),
                      [1.0], method="TRBDF2", rtol=1e-6, atol=1e-9,
                      jac=lambda t, y: [[lam]])
        self.assertLess(r.stats.naccept, 300)
        self.assertAlmostEqual(r.y[-1][0], math.cos(10), delta=1e-6)

    def test_explicit_method_is_stability_bound(self):
        lam = -1e4
        f = lambda t, y: [lam * (y[0] - math.cos(t))]
        e = solve_ivp(f, (0, 1), [1.0], method="DOPRI54", rtol=1e-4)
        i = solve_ivp(f, (0, 1), [1.0], method="TRBDF2", rtol=1e-4)
        # step size of an explicit method is capped near 3.3/|lambda|
        self.assertGreater(e.stats.naccept, 2000)
        self.assertLess(i.stats.naccept, 200)

    def test_prothero_robinson_no_catastrophic_order_reduction(self):
        pr = problems.prothero_robinson()
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-6, atol=1e-8, jac=pr["jac"])
        self.assertLess(abs(r.y[-1][0] - math.sin(10)), 1e-5)

    def test_analytic_and_fd_jacobian_agree(self):
        pr = problems.van_der_pol(1e3)
        a = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-6, atol=1e-6, jac=pr["jac"])
        b = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-6, atol=1e-6)
        for u, v in zip(a.y[-1], b.y[-1]):
            self.assertAlmostEqual(u, v, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
