"""``solve_ivp``: the driver that ties steppers, dense output and events
together."""

import math

from . import tableaux
from .dense import OdeSolution
from .steppers import ExplicitRK, ESDIRK, Stats, StepSizeTooSmall

METHODS = ("Euler", "HeunEuler21", "RK4", "BS32", "RKF45", "DOPRI54",
           "TRBDF2", "AUTO")


class Result:
    """Outcome of an integration.

    Attributes: ``t`` and ``y`` (accepted steps, or ``t_eval`` points),
    ``sol`` (dense OdeSolution or None), ``t_events``/``y_events`` (one list
    per event function), ``stats`` (counters), ``status`` (0 reached the
    end, 1 terminal event, -1 failure), ``message``, ``method`` and
    ``switches`` (list of (t, from, to) for the AUTO method).
    """

    def __init__(self):
        self.t = []
        self.y = []
        self.sol = None
        self.t_events = []
        self.y_events = []
        self.stats = None
        self.status = 0
        self.message = ""
        self.method = None
        self.switches = []

    @property
    def success(self):
        return self.status >= 0

    def __repr__(self):
        return ("Result(status=%d, steps=%d, t_end=%.6g, %s)"
                % (self.status, len(self.t), self.t[-1] if self.t else 0.0,
                   self.stats))


# ----------------------------------------------------------------------
# events

def _event_props(g):
    terminal = getattr(g, "terminal", False)
    direction = getattr(g, "direction", 0)
    return terminal, direction


def _find_root(g, interp, ta, tb, ga, gb, xtol):
    """Illinois variant of regula falsi on the step interpolant.

    Superlinear like secant, but bracketing, so it never leaves [ta, tb].
    """
    side = 0
    for _ in range(100):
        if abs(tb - ta) <= xtol:
            break
        tc = (ta * gb - tb * ga) / (gb - ga)
        if not (min(ta, tb) < tc < max(ta, tb)):
            tc = 0.5 * (ta + tb)
        gc = g(tc, interp(tc))
        if gc == 0.0:
            return tc
        if (gc > 0) == (gb > 0):
            tb, gb = tc, gc
            if side == -1:
                ga *= 0.5
            side = -1
        else:
            ta, ga = tc, gc
            if side == 1:
                gb *= 0.5
            side = 1
    return tb if abs(gb) < abs(ga) else ta


def _crossed(g0, g1, direction):
    if g0 == 0.0:
        return False
    up = g0 < 0.0 <= g1
    down = g0 > 0.0 >= g1
    if direction > 0:
        return up
    if direction < 0:
        return down
    return up or down


# ----------------------------------------------------------------------

def _make_stepper(method, fun, t, y, t_bound, rtol, atol, first_step,
                  max_step, jac, stats, detect_stiffness=False):
    if method == "TRBDF2":
        return ESDIRK(fun, t, y, t_bound, tableaux.TRBDF2, rtol, atol,
                      first_step, max_step, jac=jac, stats=stats)
    tab = tableaux.EXPLICIT[method]
    return ExplicitRK(fun, t, y, t_bound, tab, rtol, atol, first_step,
                      max_step, detect_stiffness=detect_stiffness,
                      stats=stats)


def solve_ivp(fun, t_span, y0, method="DOPRI54", rtol=1e-6, atol=1e-9,
              first_step=None, max_step=math.inf, t_eval=None,
              dense_output=False, events=None, jac=None, max_steps=1000000):
    """Integrate y' = fun(t, y) from t_span[0] to t_span[1].

    ``fun`` takes a float and a list and returns a sequence. ``method`` is
    one of METHODS; "AUTO" starts with DOPRI54 and switches permanently to
    TR-BDF2 when Hairer's stiffness test fires. ``events`` is a callable or
    list of callables g(t, y) that may carry ``terminal`` (bool) and
    ``direction`` (-1, 0, +1) attributes. ``jac(t, y)`` optionally gives the
    Jacobian for TR-BDF2 (otherwise forward differences are used).
    """
    if method not in METHODS:
        raise ValueError("unknown method %r; choose from %s"
                         % (method, ", ".join(METHODS)))
    t0, tf = float(t_span[0]), float(t_span[1])
    if hasattr(y0, "__iter__"):
        y0 = [float(v) for v in y0]
    else:
        y0 = [float(y0)]
    if rtol < 100 * 2.22e-16:
        rtol = 100 * 2.22e-16
    if events is None:
        events = []
    elif callable(events):
        events = [events]

    stats = Stats()
    res = Result()
    res.method = method
    res.stats = stats
    res.t_events = [[] for _ in events]
    res.y_events = [[] for _ in events]

    if t0 == tf:
        res.t, res.y = [t0], [list(y0)]
        res.message = "empty interval"
        return res

    current = "DOPRI54" if method == "AUTO" else method
    stepper = _make_stepper(current, fun, t0, y0, tf, rtol, atol, first_step,
                            max_step, jac, stats,
                            detect_stiffness=(method == "AUTO"))
    direction = stepper.direction
    need_dense = dense_output or t_eval is not None or bool(events)
    sol = OdeSolution(direction) if need_dense else None

    if t_eval is not None:
        t_eval = sorted((float(x) for x in t_eval),
                        reverse=(direction < 0))
        lo, hi = min(t0, tf), max(t0, tf)
        if any(x < lo - 1e-14 or x > hi + 1e-14 for x in t_eval):
            raise ValueError("t_eval must lie within t_span")
        te_idx = 0
        while te_idx < len(t_eval) and t_eval[te_idx] == t0:
            res.t.append(t0)
            res.y.append(list(y0))
            te_idx += 1
    else:
        res.t.append(t0)
        res.y.append(list(y0))

    g_vals = [g(t0, y0) for g in events]
    xtol = 4 * 2.22e-16 * max(abs(t0), abs(tf), 1.0)

    nsteps = 0
    while True:
        try:
            stepper.step()
        except StepSizeTooSmall as exc:
            res.status = -1
            res.message = str(exc)
            break
        nsteps += 1
        t_old, t_new = stepper.t_old, stepper.t
        interp = stepper.dense() if need_dense else None

        # ---- events --------------------------------------------------
        terminate = False
        if events:
            new_vals = [g(t_new, stepper.y) for g in events]
            found = []
            for i, g in enumerate(events):
                terminal, d = _event_props(g)
                if _crossed(g_vals[i], new_vals[i], d):
                    tr = _find_root(g, interp, t_old, t_new, g_vals[i],
                                    new_vals[i], xtol)
                    found.append((direction * tr, i, tr, terminal))
            found.sort()
            for _, i, tr, terminal in found:
                res.t_events[i].append(tr)
                res.y_events[i].append(interp(tr))
                if terminal:
                    terminate = True
                    t_new = tr
                    break
            g_vals = new_vals

        y_new = interp(t_new) if terminate else stepper.y
        if sol is not None:
            sol.append(t_old, t_new, interp)

        # ---- output --------------------------------------------------
        if t_eval is None:
            res.t.append(t_new)
            res.y.append(list(y_new))
        else:
            while (te_idx < len(t_eval)
                   and direction * (t_eval[te_idx] - t_new) <= 0):
                te = t_eval[te_idx]
                res.t.append(te)
                res.y.append(stepper.y if te == stepper.t else interp(te))
                te_idx += 1

        if terminate:
            res.status = 1
            res.message = "terminal event"
            break
        if stepper.t == tf:
            res.message = "reached end of interval"
            break
        if nsteps >= max_steps:
            res.status = -1
            res.message = "max_steps exceeded"
            break

        # ---- automatic stiffness switching ---------------------------
        if method == "AUTO" and current == "DOPRI54" and stepper.stiff:
            res.switches.append((stepper.t, "DOPRI54", "TRBDF2"))
            current = "TRBDF2"
            h_abs = stepper.h_abs
            stepper = _make_stepper("TRBDF2", fun, stepper.t, stepper.y, tf,
                                    rtol, atol, h_abs, max_step, jac, stats)

    res.sol = sol if dense_output else None
    return res
