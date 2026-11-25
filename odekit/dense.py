"""Continuous extensions (dense output) for accepted steps.

Every accepted step produces an interpolant over [t_old, t_new]. Events are
located on these interpolants and ``t_eval`` points are produced from them,
so the step size is never forced down just to hit an output time.
"""

import bisect


class Hermite:
    """Cubic Hermite interpolant from (y0, f0) and (y1, f1).

    Third order accurate, C1 across steps; used for every method that has
    no native continuous extension.
    """

    __slots__ = ("t0", "h", "y0", "y1", "f0", "f1")

    def __init__(self, t0, t1, y0, y1, f0, f1):
        self.t0 = t0
        self.h = t1 - t0
        self.y0, self.y1, self.f0, self.f1 = y0, y1, f0, f1

    def __call__(self, t):
        h = self.h
        th = (t - self.t0) / h
        th1 = th - 1.0
        out = []
        for a, b, fa, fb in zip(self.y0, self.y1, self.f0, self.f1):
            out.append((1.0 - th) * a + th * b + th * th1 * (
                (1.0 - 2.0 * th) * (b - a) + th1 * h * fa + th * h * fb))
        return out


class DopriDense:
    """Fourth order continuous extension of Dormand-Prince 5(4).

    This is the formulation used in Hairer's DOPRI5 code: five coefficient
    vectors are precomputed at the end of the step so evaluation costs a
    handful of multiply-adds per component.
    """

    __slots__ = ("t0", "h", "r1", "r2", "r3", "r4", "r5")

    def __init__(self, t0, t1, y0, y1, k, dcoef):
        h = t1 - t0
        self.t0, self.h = t0, h
        n = len(y0)
        self.r1 = list(y0)
        self.r2 = [y1[i] - y0[i] for i in range(n)]
        self.r3 = [h * k[0][i] - self.r2[i] for i in range(n)]
        self.r4 = [self.r2[i] - h * k[6][i] - self.r3[i] for i in range(n)]
        self.r5 = [h * sum(dc * k[j][i] for j, dc in dcoef.items())
                   for i in range(n)]

    def __call__(self, t):
        th = (t - self.t0) / self.h
        th1 = 1.0 - th
        return [a + th * (b + th1 * (c + th * (d + th1 * e)))
                for a, b, c, d, e in zip(self.r1, self.r2, self.r3, self.r4,
                                         self.r5)]


class OdeSolution:
    """Piecewise dense solution over the whole integration interval."""

    def __init__(self, direction):
        self.direction = direction
        self.ts = []      # breakpoints, monotone in integration direction
        self._keys = []   # direction * t, always increasing (for bisect)
        self.pieces = []

    def append(self, t0, t1, interp):
        if not self.ts:
            self.ts.append(t0)
            self._keys.append(self.direction * t0)
        self.ts.append(t1)
        self._keys.append(self.direction * t1)
        self.pieces.append(interp)

    @property
    def t_min(self):
        return min(self.ts[0], self.ts[-1])

    @property
    def t_max(self):
        return max(self.ts[0], self.ts[-1])

    def _piece(self, t):
        i = bisect.bisect_right(self._keys, self.direction * t) - 1
        return self.pieces[min(max(i, 0), len(self.pieces) - 1)]

    def __call__(self, t):
        """Evaluate at a scalar t, or at each t in an iterable."""
        if hasattr(t, "__iter__"):
            return [self._piece(x)(x) for x in t]
        return self._piece(t)(t)
