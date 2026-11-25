"""Butcher tableaux, stored with exact rational coefficients.

Each tableau is checked against the rooted tree order conditions in the test
suite, so a single mistyped coefficient fails loudly instead of quietly
degrading accuracy.
"""

from fractions import Fraction as F
import math

from . import trees


class Tableau:
    """An explicit (or diagonally implicit) Runge-Kutta tableau.

    ``b`` is the propagated solution, ``b_hat`` the embedded one used only
    for error estimation. ``order`` refers to ``b`` and ``err_order`` to
    ``b_hat``. ``fsal`` marks First Same As Last methods where the last stage
    is f(t + h, y_new) and can be reused as the first stage of the next step.
    """

    def __init__(self, name, c, a, b, b_hat=None, order=None, err_order=None,
                 fsal=False, description=""):
        s = len(c)
        self.name = name
        self.c_exact = list(c)
        self.a_exact = [list(row) + [0] * (s - len(row)) for row in a]
        while len(self.a_exact) < s:
            self.a_exact.insert(0, [0] * s)
        self.b_exact = list(b)
        self.b_hat_exact = list(b_hat) if b_hat is not None else None
        self.order = order
        self.err_order = err_order
        self.fsal = fsal
        self.description = description
        # float copies used by the integrators
        self.c = [float(x) for x in self.c_exact]
        self.a = [[float(x) for x in row] for row in self.a_exact]
        self.b = [float(x) for x in self.b_exact]
        self.b_hat = ([float(x) for x in self.b_hat_exact]
                      if self.b_hat_exact is not None else None)
        self.e = ([bi - bh for bi, bh in zip(self.b, self.b_hat)]
                  if self.b_hat is not None else None)
        self.stages = s

    @property
    def embedded(self):
        return self.b_hat is not None

    @property
    def explicit(self):
        return all(self.a_exact[i][j] == 0
                   for i in range(self.stages) for j in range(i, self.stages))

    def verified_order(self, max_order=8):
        return trees.check_order(self.a_exact, self.b_exact, self.c_exact,
                                 max_order)

    def verified_err_order(self, max_order=8):
        if self.b_hat_exact is None:
            return None
        return trees.check_order(self.a_exact, self.b_hat_exact, self.c_exact,
                                 max_order)

    def stability_function(self, z):
        """R(z) = 1 + z b^T (I - z A)^{-1} 1 for complex z.

        For an explicit method this is a polynomial; for DIRK methods it is
        rational. Solved by forward substitution since A is lower triangular.
        """
        s = self.stages
        k = [0j] * s
        # stage values for y' = z y, y0 = 1, h = 1:  Y_i = 1 + z sum_j a_ij Y_j
        for i in range(s):
            acc = 1 + z * sum(self.a[i][j] * k[j] for j in range(i))
            aii = self.a[i][i]
            k[i] = acc / (1 - z * aii)
        return 1 + z * sum(self.b[i] * k[i] for i in range(s))

    def __repr__(self):
        return "Tableau(%s, order=%s)" % (self.name, self.order)


EULER = Tableau(
    "Euler", c=[0], a=[[0]], b=[1], order=1,
    description="Forward Euler, the order 1 baseline")

HEUN_EULER = Tableau(
    "HeunEuler21", c=[0, 1], a=[[0, 0], [1, 0]],
    b=[F(1, 2), F(1, 2)], b_hat=[1, 0], order=2, err_order=1,
    description="Heun's method with an embedded Euler estimate")

RK4 = Tableau(
    "RK4", c=[0, F(1, 2), F(1, 2), 1],
    a=[[0, 0, 0, 0], [F(1, 2), 0, 0, 0], [0, F(1, 2), 0, 0], [0, 0, 1, 0]],
    b=[F(1, 6), F(1, 3), F(1, 3), F(1, 6)], order=4,
    description="The classical fourth order method (fixed step only)")

BS32 = Tableau(
    "BS32", c=[0, F(1, 2), F(3, 4), 1],
    a=[[0, 0, 0, 0], [F(1, 2), 0, 0, 0], [0, F(3, 4), 0, 0],
       [F(2, 9), F(1, 3), F(4, 9), 0]],
    b=[F(2, 9), F(1, 3), F(4, 9), 0],
    b_hat=[F(7, 24), F(1, 4), F(1, 3), F(1, 8)],
    order=3, err_order=2, fsal=True,
    description="Bogacki-Shampine 3(2), FSAL; the method behind ode23")

RKF45 = Tableau(
    "RKF45", c=[0, F(1, 4), F(3, 8), F(12, 13), 1, F(1, 2)],
    a=[[0] * 6,
       [F(1, 4), 0, 0, 0, 0, 0],
       [F(3, 32), F(9, 32), 0, 0, 0, 0],
       [F(1932, 2197), F(-7200, 2197), F(7296, 2197), 0, 0, 0],
       [F(439, 216), -8, F(3680, 513), F(-845, 4104), 0, 0],
       [F(-8, 27), 2, F(-3544, 2565), F(1859, 4104), F(-11, 40), 0]],
    b=[F(16, 135), 0, F(6656, 12825), F(28561, 56430), F(-9, 50), F(2, 55)],
    b_hat=[F(25, 216), 0, F(1408, 2565), F(2197, 4104), F(-1, 5), 0],
    order=5, err_order=4,
    description="Runge-Kutta-Fehlberg 4(5), propagating the 5th order "
                "solution (local extrapolation)")

DOPRI54 = Tableau(
    "DOPRI54", c=[0, F(1, 5), F(3, 10), F(4, 5), F(8, 9), 1, 1],
    a=[[0] * 7,
       [F(1, 5), 0, 0, 0, 0, 0, 0],
       [F(3, 40), F(9, 40), 0, 0, 0, 0, 0],
       [F(44, 45), F(-56, 15), F(32, 9), 0, 0, 0, 0],
       [F(19372, 6561), F(-25360, 2187), F(64448, 6561), F(-212, 729),
        0, 0, 0],
       [F(9017, 3168), F(-355, 33), F(46732, 5247), F(49, 176),
        F(-5103, 18656), 0, 0],
       [F(35, 384), 0, F(500, 1113), F(125, 192), F(-2187, 6784),
        F(11, 84), 0]],
    b=[F(35, 384), 0, F(500, 1113), F(125, 192), F(-2187, 6784), F(11, 84),
       0],
    b_hat=[F(5179, 57600), 0, F(7571, 16695), F(393, 640),
           F(-92097, 339200), F(187, 2100), F(1, 40)],
    order=5, err_order=4, fsal=True,
    description="Dormand-Prince 5(4), FSAL, with a 4th order continuous "
                "extension; the method behind ode45")

# Dense output coefficients for DOPRI54 (Hairer, Norsett & Wanner, DOPRI5).
DOPRI_DENSE = {
    0: -12715105075.0 / 11282082432.0,
    2: 87487479700.0 / 32700410799.0,
    3: -10690763975.0 / 1880347072.0,
    4: 701980252875.0 / 199316789632.0,
    5: -1453857185.0 / 822651844.0,
    6: 69997945.0 / 29380423.0,
}


def _trbdf2():
    """TR-BDF2 written as a three stage ESDIRK tableau.

    gamma = 2 - sqrt(2) is the only choice that makes both implicit stages
    share the diagonal coefficient d = gamma/2, so one LU factorisation of
    I - h d J serves the trapezoidal stage and the BDF2 stage. The embedded
    weights are the third order companion of Hosea and Shampine (1996).
    Coefficients are irrational, so this tableau is float only.
    """
    g = 2.0 - math.sqrt(2.0)
    d = g / 2.0
    w = math.sqrt(2.0) / 4.0
    t = Tableau(
        "TRBDF2", c=[0.0, g, 1.0],
        a=[[0.0, 0.0, 0.0], [d, d, 0.0], [w, w, d]],
        b=[w, w, d],
        b_hat=[(1.0 - w) / 3.0, (3.0 * w + 1.0) / 3.0, d / 3.0],
        order=2, err_order=3,
        description="TR-BDF2: trapezoidal stage then BDF2 stage, L-stable, "
                    "stiffly accurate; the method behind ode23tb")
    t.gamma = g
    t.d = d
    return t


TRBDF2 = _trbdf2()

EXPLICIT = {t.name: t for t in (EULER, HEUN_EULER, RK4, BS32, RKF45, DOPRI54)}
ALL = dict(EXPLICIT, TRBDF2=TRBDF2)
