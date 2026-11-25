import math
import unittest

from odekit import solve_ivp, problems, symplectic


def event(fn, terminal=False, direction=0):
    fn.terminal = terminal
    fn.direction = direction
    return fn


class Events(unittest.TestCase):
    def test_falling_ball_hits_ground_at_analytic_time(self):
        g, h0 = 9.81, 10.0
        hit = event(lambda t, y: y[0], terminal=True, direction=-1)
        r = solve_ivp(lambda t, y: [y[1], -g], (0, 100), [h0, 0.0],
                      events=hit, rtol=1e-10, atol=1e-12)
        self.assertEqual(r.status, 1)
        t_hit = math.sqrt(2 * h0 / g)
        self.assertAlmostEqual(r.t_events[0][0], t_hit, places=10)
        self.assertAlmostEqual(r.t[-1], t_hit, places=10)
        self.assertAlmostEqual(r.y[-1][0], 0.0, places=9)

    def test_all_zero_crossings_of_cosine(self):
        zero = event(lambda t, y: y[0])
        pr = problems.harmonic()
        r = solve_ivp(pr["fun"], (0, 20), pr["y0"], events=zero, rtol=1e-9,
                      atol=1e-12)
        expect = [math.pi / 2 + k * math.pi for k in range(6)]  # < 20
        self.assertEqual(len(r.t_events[0]), len(expect))
        for a, b in zip(r.t_events[0], expect):
            self.assertAlmostEqual(a, b, places=7)

    def test_direction_filter(self):
        up = event(lambda t, y: y[0], direction=1)
        down = event(lambda t, y: y[0], direction=-1)
        pr = problems.harmonic()
        r = solve_ivp(pr["fun"], (0, 20), pr["y0"], events=[up, down],
                      rtol=1e-9, atol=1e-12)
        for t in r.t_events[0]:
            self.assertLess(math.sin(t), 0)   # cos rising => sin < 0
        for t in r.t_events[1]:
            self.assertGreater(math.sin(t), 0)
        self.assertEqual(len(r.t_events[0]) + len(r.t_events[1]), 6)

    def test_events_on_stiff_solver(self):
        pr = problems.robertson()
        half = event(lambda t, y: y[0] - 0.5, terminal=True)
        r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method="TRBDF2",
                      rtol=1e-6, atol=[1e-8, 1e-12, 1e-8], events=half)
        self.assertEqual(r.status, 1)
        self.assertAlmostEqual(r.y_events[0][0][0], 0.5, places=6)


class Symplectic(unittest.TestCase):
    def setUp(self):
        self.pr = problems.kepler(0.5)
        ref = solve_ivp(self.pr["fun"], (0, 1.0), self.pr["y0"], rtol=1e-13,
                        atol=1e-14)
        self.ref = ref.y[-1]

    def _err(self, order, n):
        pr = self.pr
        ts, qs, ps = symplectic.integrate(pr["dT"], pr["dV"], pr["y0"][:2],
                                          pr["y0"][2:], 1.0 / n, n,
                                          order=order)
        return math.hypot(qs[-1][0] - self.ref[0], qs[-1][1] - self.ref[1])

    def test_orders(self):
        for order in (1, 2, 4, 6):
            with self.subTest(order=order):
                p = math.log2(self._err(order, 100) / self._err(order, 200))
                self.assertAlmostEqual(p, order, delta=0.25)

    def test_triple_jump_weights(self):
        for order in (2, 4, 6, 8):
            w = symplectic.triple_jump_weights(order)
            self.assertEqual(len(w), 3 ** (order // 2 - 1))
            self.assertAlmostEqual(sum(w), 1.0, places=14)
            self.assertEqual(w, w[::-1])  # symmetric composition
        with self.assertRaises(ValueError):
            symplectic.triple_jump_weights(3)

    def test_energy_error_bounded_over_long_times(self):
        pr = problems.kepler(0.6)
        H = pr["invariant"]
        e0 = H(pr["y0"])
        h = 2 * math.pi / 100
        ts, qs, ps = symplectic.integrate(pr["dT"], pr["dV"], pr["y0"][:2],
                                          pr["y0"][2:], h, 100 * 200)
        dh = [abs(H(q + p) - e0) for q, p in zip(qs, ps)]
        first, last = max(dh[:2000]), max(dh[-2000:])
        self.assertLess(last, 1.1 * first)

    def test_verlet_is_time_reversible(self):
        pr = self.pr
        ts, qs, ps = symplectic.integrate(pr["dT"], pr["dV"], pr["y0"][:2],
                                          pr["y0"][2:], 0.01, 500)
        back = symplectic.integrate(pr["dT"], pr["dV"], qs[-1],
                                    [-p for p in ps[-1]], 0.01, 500)
        for a, b in zip(back[1][-1], pr["y0"][:2]):
            self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()
