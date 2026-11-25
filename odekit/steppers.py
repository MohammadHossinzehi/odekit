"""Adaptive one step integrators.

Each stepper owns its state (t, y, f(t, y), current step size) and exposes
``step()``, which advances by exactly one *accepted* step, retrying with a
smaller step as often as necessary. After a successful ``step()`` the
attributes ``t_old``, ``y_old``, ``t``, ``y`` describe the step and
``dense()`` returns its interpolant.
"""

import math

from . import linalg
from .dense import Hermite, DopriDense
from .tableaux import DOPRI_DENSE

EPS = linalg.EPS
SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 5.0


class StepSizeTooSmall(RuntimeError):
    pass


class Stats:
    __slots__ = ("nfev", "njev", "nlu", "naccept", "nreject", "nnewton",
                 "nnewton_fail")

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, 0)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return " ".join("%s=%d" % kv for kv in self.as_dict().items())


class _Base:
    def __init__(self, fun, t0, y0, t_bound, rtol, atol, first_step,
                 max_step, stats):
        self.fun_raw = fun
        self.stats = stats if stats is not None else Stats()
        self.t = float(t0)
        self.y = [float(v) for v in y0]
        self.n = len(self.y)
        self.t_bound = float(t_bound)
        self.direction = 1.0 if t_bound >= t0 else -1.0
        self.rtol = rtol
        if isinstance(atol, (list, tuple)):
            if len(atol) != self.n:
                raise ValueError("atol must be a scalar or have len(y0)")
            self.atol = [float(a) for a in atol]
        else:
            self.atol = [float(atol)] * self.n
        self.max_step = max_step
        self.f = self.fun(self.t, self.y)
        self.first_step = first_step
        self.t_old = None
        self.y_old = None
        self.f_old = None
        self.err_old = 1e-4  # PI controller memory
        self.rejected_last = False

    def fun(self, t, y):
        self.stats.nfev += 1
        out = self.fun_raw(t, y)
        return [float(v) for v in out]

    # ------------------------------------------------------------------
    def norm(self, err, y_new):
        return linalg.rms_norm(err, self.y, y_new, self.rtol, self.atol)

    def initial_step(self, order):
        """Hairer, Norsett & Wanner's automatic starting step (II.4)."""
        if self.first_step is not None:
            return abs(self.first_step)
        y, f = self.y, self.f
        scale = [self.atol[i] + self.rtol * abs(y[i]) for i in range(self.n)]
        d0 = math.sqrt(sum((y[i] / scale[i]) ** 2 for i in range(self.n))
                       / max(self.n, 1))
        d1 = math.sqrt(sum((f[i] / scale[i]) ** 2 for i in range(self.n))
                       / max(self.n, 1))
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        h0 = min(h0, abs(self.t_bound - self.t))
        y1 = [y[i] + self.direction * h0 * f[i] for i in range(self.n)]
        f1 = self.fun(self.t + self.direction * h0, y1)
        d2 = math.sqrt(sum(((f1[i] - f[i]) / scale[i]) ** 2
                           for i in range(self.n)) / max(self.n, 1)) / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
        return min(100 * h0, h1, self.max_step)

    def controller(self, err, k):
        """PI step size controller (Gustafsson; constants from DOPRI5).

        Returns the factor by which to multiply the step that produced
        ``err``. The integral part reacts to the current error, the
        proportional part to the trend, which damps the oscillating
        accept/reject cycles a pure I controller shows on stiff-ish problems.
        """
        beta = 0.4 / k
        alpha = 0.7 / k
        if err == 0.0:
            return MAX_FACTOR
        fac = SAFETY * err ** (-alpha) * self.err_old ** beta
        return min(MAX_FACTOR, max(MIN_FACTOR, fac))

    def _next_h(self):
        h_abs = min(self.h_abs, self.max_step)
        min_step = 10 * EPS * max(abs(self.t), 1.0)
        if h_abs < min_step:
            raise StepSizeTooSmall(
                "step size %.3e below minimum at t=%.17g" % (h_abs, self.t))
        remaining = abs(self.t_bound - self.t)
        last = False
        # stretch the final step slightly to avoid a sliver step at the end
        if h_abs >= remaining * (1 - 1e-12) or h_abs * 1.01 >= remaining:
            h_abs = remaining
            last = True
        return self.direction * h_abs, last

    def _commit(self, t_new, y_new, f_new):
        self.t_old, self.y_old, self.f_old = self.t, self.y, self.f
        self.t, self.y, self.f = t_new, y_new, f_new
        self.stats.naccept += 1

    def dense(self):
        return Hermite(self.t_old, self.t, self.y_old, self.y, self.f_old,
                       self.f)


class ExplicitRK(_Base):
    """Embedded explicit Runge-Kutta with PI control and FSAL reuse.

    With a tableau that has no embedded weights (e.g. RK4) the stepper runs
    at the fixed ``first_step``. ``detect_stiffness`` enables Hairer's test
    for DOPRI54: the last two stages are evaluated at the same time t + h,
    so ||k7 - k6|| / ||y7 - y6|| estimates the dominant eigenvalue of the
    Jacobian along the step for free.
    """

    def __init__(self, fun, t0, y0, t_bound, tableau, rtol=1e-6, atol=1e-9,
                 first_step=None, max_step=math.inf, detect_stiffness=False,
                 stats=None):
        super().__init__(fun, t0, y0, t_bound, rtol, atol, first_step,
                         max_step, stats)
        self.tab = tableau
        if not tableau.embedded and first_step is None:
            raise ValueError("%s has no error estimator: pass first_step"
                             % tableau.name)
        self.k_exp = (min(tableau.order, tableau.err_order) + 1
                      if tableau.embedded else None)
        self.h_abs = self.initial_step(tableau.order)
        self.detect_stiffness = detect_stiffness and tableau.name == "DOPRI54"
        self.stiff_hits = 0
        self.nonstiff_hits = 0
        self.stiff = False
        self.h_lambda = 0.0
        self._k = None

    def _stages(self, h):
        tab, t, y, n = self.tab, self.t, self.y, self.n
        k = [self.f]
        y_stage = y
        for i in range(1, tab.stages):
            ai = tab.a[i]
            y_stage = [y[m] + h * sum(ai[j] * k[j][m] for j in range(i)
                                      if ai[j] != 0.0)
                       for m in range(n)]
            k.append(self.fun(t + tab.c[i] * h, y_stage))
        return k

    def step(self):
        tab = self.tab
        n = self.n
        rejected = False
        while True:
            h, last = self._next_h()
            t_new = self.t_bound if last else self.t + h
            k = self._stages(h)
            y_new = [self.y[m] + h * sum(tab.b[j] * k[j][m]
                                         for j in range(tab.stages)
                                         if tab.b[j] != 0.0)
                     for m in range(n)]
            if not tab.embedded:
                err = 0.0
            else:
                e = [h * sum(tab.e[j] * k[j][m] for j in range(tab.stages)
                             if tab.e[j] != 0.0) for m in range(n)]
                err = self.norm(e, y_new)
            if err <= 1.0:
                break
            self.stats.nreject += 1
            rejected = True
            self.h_abs = abs(h) * max(MIN_FACTOR,
                                      SAFETY * err ** (-1.0 / self.k_exp))
        if tab.fsal:
            f_new = k[-1]
        else:
            f_new = self.fun(t_new, y_new)
        if self.detect_stiffness:
            self._stiffness_test(h, k, y_new)
        self._k = k
        if tab.embedded:
            fac = self.controller(err, self.k_exp)
            if rejected:
                fac = min(1.0, fac)
            self.h_abs = abs(h) * fac
            self.err_old = max(err, 1e-4)
        self._commit(t_new, y_new, f_new)

    def _stiffness_test(self, h, k, y_new):
        # DOPRI: stage 6 at (t+h, y6), stage 7 at (t+h, y_new)
        k6, k7 = k[5], k[6]
        y6 = [self.y[m] + h * sum(self.tab.a[5][j] * k[j][m]
                                  for j in range(5)) for m in range(self.n)]
        num = math.sqrt(sum((a - b) ** 2 for a, b in zip(k7, k6)))
        den = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_new, y6)))
        if den > 0:
            self.h_lambda = abs(h) * num / den
            if self.h_lambda > 3.25:
                self.nonstiff_hits = 0
                self.stiff_hits += 1
                if self.stiff_hits >= 15:
                    self.stiff = True
            else:
                self.nonstiff_hits += 1
                if self.nonstiff_hits >= 6:
                    self.stiff_hits = 0

    def dense(self):
        if self.tab.name == "DOPRI54":
            return DopriDense(self.t_old, self.t, self.y_old, self.y, self._k,
                              DOPRI_DENSE)
        return Hermite(self.t_old, self.t, self.y_old, self.y, self.f_old,
                       self.f)


class ESDIRK(_Base):
    """Adaptive TR-BDF2 (or any ESDIRK with explicit first stage and a
    constant diagonal) for stiff problems.

    Design points:

    * One matrix ``I - h*d*J`` serves every implicit stage. Its LU factors
      are reused for as long as h is unchanged, and the controller holds h
      fixed when it would only grow by up to 20 percent, so refactorisations
      are rare. The Jacobian is kept until Newton actually fails, so on
      smooth stiff problems Jacobian evaluations are rarer still.
    * Stage derivatives are recovered algebraically from the converged stage
      values, k_i = (z_i - rhs_i) / (h d), instead of re-evaluating f; this
      avoids amplifying the Newton residual by the stiff Jacobian.
    * The local error estimate is filtered through (I - h d J)^{-1}
      (Shampine), which keeps it bounded as h*lambda -> -infinity instead of
      growing like h*lambda and needlessly throttling the step.
    """

    NEWTON_MAXITER = 7
    NEWTON_TOL = 0.03

    def __init__(self, fun, t0, y0, t_bound, tableau, rtol=1e-6, atol=1e-9,
                 first_step=None, max_step=math.inf, jac=None, stats=None):
        super().__init__(fun, t0, y0, t_bound, rtol, atol, first_step,
                         max_step, stats)
        self.tab = tableau
        self.d = tableau.a[1][1]
        self.jac_fun = jac
        self.J = None
        self.J_current = False
        self.lu = None
        self.h_lu = None
        self.eta = 1.0
        self.k_exp = min(tableau.order, tableau.err_order) + 1
        self.h_abs = self.initial_step(tableau.order)

    def _jacobian(self):
        if self.jac_fun is not None:
            self.J = [[float(v) for v in row]
                      for row in self.jac_fun(self.t, self.y)]
        else:
            self.J, nf = linalg.fd_jacobian(self.fun_raw, self.t, self.y,
                                            self.f)
            self.stats.nfev += nf
        self.stats.njev += 1
        self.J_current = True
        self.lu = None

    def _factor(self, h):
        n = self.n
        hd = h * self.d
        m = [[(1.0 if i == j else 0.0) - hd * self.J[i][j] for j in range(n)]
             for i in range(n)]
        self.lu = linalg.lu_factor(m)
        self.h_lu = h
        self.stats.nlu += 1

    def _newton(self, t_stage, rhs, z0, h):
        """Solve z - h d f(t_stage, z) = rhs by simplified Newton.

        Returns the converged z or None. Uses the factors of I - h_lu d J.
        """
        n = self.n
        hd = h * self.d
        z = list(z0)
        scale = [self.atol[i] + self.rtol * abs(self.y[i]) for i in range(n)]
        dz_old = None
        eta = max(self.eta, EPS) ** 0.8
        for it in range(self.NEWTON_MAXITER):
            self.stats.nnewton += 1
            fz = self.fun(t_stage, z)
            res = [rhs[i] + hd * fz[i] - z[i] for i in range(n)]
            if any(not math.isfinite(r) for r in res):
                return None
            dz = linalg.lu_solve(self.lu, res)
            z = [z[i] + dz[i] for i in range(n)]
            dz_norm = math.sqrt(sum((dz[i] / scale[i]) ** 2
                                    for i in range(n)) / n)
            if dz_old is not None:
                rate = dz_norm / dz_old if dz_old > 0 else 0.0
                if rate >= 0.99:
                    return None
                # predicted remaining error too large for remaining budget
                if (rate ** (self.NEWTON_MAXITER - it) / (1 - rate)
                        * dz_norm > self.NEWTON_TOL):
                    return None
                eta = rate / (1.0 - rate)
            if dz_norm == 0.0 or eta * dz_norm <= self.NEWTON_TOL:
                self.eta = eta
                return z
            dz_old = dz_norm
        return None

    def step(self):
        tab, n = self.tab, self.n
        rejected = False
        if self.J is None:
            self._jacobian()
        while True:
            h, last = self._next_h()
            t_new = self.t_bound if last else self.t + h
            if self.lu is None or self.h_lu != h:
                self._factor(h)
            hd = h * self.d
            k = [self.f]
            zs = [self.y]
            ok = True
            for i in range(1, tab.stages):
                ai = tab.a[i]
                rhs = [self.y[m] + h * sum(ai[j] * k[j][m] for j in range(i))
                       for m in range(n)]
                # predictor: extrapolate the last stage trend
                if i == 1:
                    z0 = [self.y[m] + tab.c[1] * h * self.f[m]
                          for m in range(n)]
                else:
                    ci, cp = tab.c[i], tab.c[i - 1]
                    z0 = [self.y[m] + (zs[-1][m] - self.y[m]) * ci / cp
                          for m in range(n)]
                z = self._newton(self.t + tab.c[i] * h, rhs, z0, h)
                if z is None:
                    ok = False
                    break
                zs.append(z)
                k.append([(z[m] - rhs[m]) / hd for m in range(n)])
            if not ok:
                self.stats.nnewton_fail += 1
                rejected = True
                if not self.J_current:
                    self._jacobian()
                else:
                    self.h_abs = abs(h) * 0.5
                    self.lu = None
                continue
            y_new = zs[-1]  # stiffly accurate: last stage is the solution
            e = [h * sum(tab.e[j] * k[j][m] for j in range(tab.stages))
                 for m in range(n)]
            e = linalg.lu_solve(self.lu, e)
            err = self.norm(e, y_new)
            if err <= 1.0:
                break
            self.stats.nreject += 1
            rejected = True
            self.h_abs = abs(h) * max(MIN_FACTOR,
                                      SAFETY * err ** (-1.0 / self.k_exp))
        f_new = self.fun(t_new, y_new)
        fac = self.controller(err, self.k_exp)
        if rejected:
            fac = min(1.0, fac)
        # hysteresis: keep h (and therefore the LU) for small increases
        if 1.0 <= fac <= 1.2:
            fac = 1.0
        self.h_abs = abs(h) * fac
        self.err_old = max(err, 1e-4)
        self.J_current = False
        self._commit(t_new, y_new, f_new)
