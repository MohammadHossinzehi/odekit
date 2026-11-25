"""A small zoo of classic test problems, each with what is known about it.

Every problem is a dict with ``fun``, ``y0``, ``t_span`` and optionally
``jac``, ``exact`` (closed form solution) and ``invariant`` (a quantity the
true solution conserves).
"""

import math


def exponential(lam=-1.0):
    return dict(
        name="exponential", fun=lambda t, y: [lam * y[0]], y0=[1.0],
        t_span=(0.0, 5.0), exact=lambda t: [math.exp(lam * t)],
        jac=lambda t, y: [[lam]])


def harmonic(omega=1.0):
    return dict(
        name="harmonic",
        fun=lambda t, y: [y[1], -omega * omega * y[0]], y0=[1.0, 0.0],
        t_span=(0.0, 10.0),
        exact=lambda t: [math.cos(omega * t), -omega * math.sin(omega * t)],
        invariant=lambda y: 0.5 * (y[1] ** 2 + omega ** 2 * y[0] ** 2))


def kepler(e=0.5):
    """Two body problem in the plane with eccentricity e, period 2*pi."""
    q0 = [1.0 - e, 0.0]
    p0 = [0.0, math.sqrt((1.0 + e) / (1.0 - e))]

    def fun(t, y):
        x, yy, vx, vy = y
        r3 = (x * x + yy * yy) ** 1.5
        return [vx, vy, -x / r3, -yy / r3]

    def energy(y):
        x, yy, vx, vy = y
        return 0.5 * (vx * vx + vy * vy) - 1.0 / math.sqrt(x * x + yy * yy)

    return dict(name="kepler", fun=fun, y0=q0 + p0,
                t_span=(0.0, 2 * math.pi), invariant=energy,
                exact_period=2 * math.pi,
                dT=lambda p: list(p),
                dV=lambda q: [q[0] / (q[0] ** 2 + q[1] ** 2) ** 1.5,
                              q[1] / (q[0] ** 2 + q[1] ** 2) ** 1.5])


def van_der_pol(mu=1e6):
    """Van der Pol in Hairer & Wanner's scaling (their epsilon is 1/mu):

        y1' = y2,   y2' = mu ((1 - y1^2) y2 - y1)

    The relaxation period is about 3 - 2 ln 2 = 1.614 whatever mu is, while
    the Jacobian eigenvalue on the slow manifold grows like -mu (y1^2 - 1),
    so mu dials stiffness without changing the time scale of interest.
    """
    def fun(t, y):
        return [y[1], mu * ((1.0 - y[0] ** 2) * y[1] - y[0])]

    def jac(t, y):
        return [[0.0, 1.0],
                [mu * (-2.0 * y[0] * y[1] - 1.0), mu * (1.0 - y[0] ** 2)]]

    out = dict(name="vanderpol", fun=fun, jac=jac,
               y0=[2.0, -2.0 / 3.0 + 10.0 / (81.0 * mu)],
               t_span=(0.0, 2.0))
    if mu == 1e6:
        # reference solution at t = 2 from the Hairer & Wanner test set
        out["reference"] = [1.706167732170483, -0.8928097010247975]
    return out


def robertson():
    """Robertson's autocatalytic chemistry: the classic stiff benchmark.

    Rate constants span 9 orders of magnitude; y1 + y2 + y3 = 1 exactly.
    """
    def fun(t, y):
        a, b, c = y
        return [-0.04 * a + 1e4 * b * c,
                0.04 * a - 1e4 * b * c - 3e7 * b * b,
                3e7 * b * b]

    def jac(t, y):
        a, b, c = y
        return [[-0.04, 1e4 * c, 1e4 * b],
                [0.04, -1e4 * c - 6e7 * b, -1e4 * b],
                [0.0, 6e7 * b, 0.0]]

    return dict(name="robertson", fun=fun, jac=jac, y0=[1.0, 0.0, 0.0],
                t_span=(0.0, 1e5), invariant=lambda y: y[0] + y[1] + y[2])


def lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    def fun(t, y):
        x, yy, z = y
        return [sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z]
    return dict(name="lorenz", fun=fun, y0=[1.0, 1.0, 1.0],
                t_span=(0.0, 20.0))


def brusselator(a=1.0, b=3.0):
    def fun(t, y):
        u, v = y
        return [a + u * u * v - (b + 1.0) * u, b * u - u * u * v]
    return dict(name="brusselator", fun=fun, y0=[1.5, 3.0],
                t_span=(0.0, 20.0))


def prothero_robinson(lam=-1e6):
    """y' = lam (y - g(t)) + g'(t) with g = sin: exact solution sin(t).

    A severe test of order reduction in stiff solvers.
    """
    return dict(
        name="prothero_robinson",
        fun=lambda t, y: [lam * (y[0] - math.sin(t)) + math.cos(t)],
        jac=lambda t, y: [[lam]], y0=[0.0], t_span=(0.0, 10.0),
        exact=lambda t: [math.sin(t)])


REGISTRY = {
    "exponential": exponential, "harmonic": harmonic, "kepler": kepler,
    "vanderpol": van_der_pol, "robertson": robertson, "lorenz": lorenz,
    "brusselator": brusselator, "prothero": prothero_robinson,
}
