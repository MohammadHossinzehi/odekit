"""Command line interface: ``python -m odekit <command>``.

Commands
  methods                 tableau report: tree-verified orders, stability
  solve PROBLEM [opts]    integrate a built in problem, print stats / CSV
  bench                   compare methods on the problem zoo
  energy [--orbits N]     long time energy drift: RK vs symplectic on Kepler
"""

import argparse
import math
import sys
import time

from . import problems, symplectic, tableaux
from .ivp import solve_ivp, METHODS


def stability_boundary(tab, lo=0.0, hi=50.0):
    """Largest x such that |R(-s)| <= 1 for all s in [0, x] (scanned, then
    refined by bisection). Returns inf for methods stable on the whole
    negative real axis."""
    step = 0.01
    x = step
    while x < hi:
        if abs(tab.stability_function(complex(-x, 0))) > 1 + 1e-12:
            a, b = x - step, x
            for _ in range(60):
                m = 0.5 * (a + b)
                if abs(tab.stability_function(complex(-m, 0))) > 1 + 1e-12:
                    b = m
                else:
                    a = m
            return a
        x += step
    return math.inf


def cmd_methods(args):
    print("%-12s %6s %7s %9s %8s %5s %10s %10s" % (
        "method", "stages", "order", "verified", "err ord", "FSAL",
        "real stab", "|R(-inf)|"))
    for tab in tableaux.ALL.values():
        vo = tab.verified_order()
        ve = tab.verified_err_order()
        sb = stability_boundary(tab)
        rinf = abs(tab.stability_function(complex(-1e12, 0)))
        print("%-12s %6d %7d %9d %8s %5s %10s %10s" % (
            tab.name, tab.stages, tab.order, vo,
            "-" if ve is None else str(ve), "yes" if tab.fsal else "no",
            "inf" if sb == math.inf else "%.4f" % sb,
            "%.2e" % rinf if rinf < 1e6 else "unbounded"))
    print("\n'verified' is the order proven from the tableau by checking every"
          "\nrooted tree order condition (exactly, with rational arithmetic)."
          "\n'real stab' is the stability interval [-x, 0] on the real axis.")


def _make_problem(args):
    name = args.problem
    if name not in problems.REGISTRY:
        sys.exit("unknown problem %r; choose from %s"
                 % (name, ", ".join(problems.REGISTRY)))
    kwargs = {}
    if args.mu is not None and name == "vanderpol":
        kwargs["mu"] = args.mu
    return problems.REGISTRY[name](**kwargs)


def cmd_solve(args):
    pr = _make_problem(args)
    t_span = pr["t_span"] if args.t_end is None else (pr["t_span"][0],
                                                      args.t_end)
    t0 = time.perf_counter()
    res = solve_ivp(pr["fun"], t_span, pr["y0"], method=args.method,
                    rtol=args.rtol, atol=args.atol, jac=pr.get("jac"),
                    dense_output=bool(args.csv), max_steps=args.max_steps)
    dt = time.perf_counter() - t0
    print("problem   %s   method %s   rtol %g   atol %g"
          % (pr["name"], args.method, args.rtol, args.atol))
    print("status    %d (%s)   wall %.3fs" % (res.status, res.message, dt))
    print("stats     %s" % res.stats)
    for t, frm, to in res.switches:
        print("switch    %s -> %s at t = %.6g (stiffness detected)"
              % (frm, to, t))
    print("t_end     %.10g" % res.t[-1])
    print("y_end     %s" % " ".join("%.12g" % v for v in res.y[-1]))
    if "exact" in pr:
        ex = pr["exact"](res.t[-1])
        print("abs err   %.3e" % max(abs(a - b) for a, b in
                                     zip(res.y[-1], ex)))
    if "invariant" in pr:
        print("invariant drift %.3e" % (pr["invariant"](res.y[-1])
                                        - pr["invariant"](pr["y0"])))
    if args.csv:
        a, b = res.t[0], res.t[-1]
        n = args.points
        with open(args.csv, "w") as fh:
            fh.write("t," + ",".join("y%d" % i for i in range(len(pr["y0"])))
                     + "\n")
            for i in range(n + 1):
                t = a + (b - a) * i / n
                fh.write("%.12g," % t + ",".join("%.12g" % v
                                                 for v in res.sol(t)) + "\n")
        print("wrote %d dense samples to %s" % (n + 1, args.csv))


def cmd_bench(args):
    cases = [
        ("exponential", problems.exponential(), ["BS32", "RKF45", "DOPRI54",
                                                  "TRBDF2"], 1e-8),
        ("kepler e=0.5", problems.kepler(), ["BS32", "RKF45", "DOPRI54"],
         1e-8),
        ("vanderpol mu=1e6", problems.van_der_pol(1e6), ["TRBDF2", "AUTO"],
         1e-6),
        ("robertson", problems.robertson(), ["TRBDF2", "AUTO"], 1e-6),
        ("prothero-robinson", problems.prothero_robinson(), ["TRBDF2"], 1e-6),
    ]
    print("%-18s %-8s %7s %7s %6s %6s %10s %8s" % (
        "problem", "method", "steps", "nfev", "njev", "nlu", "error",
        "wall s"))
    for label, pr, methods, tol in cases:
        for m in methods:
            t0 = time.perf_counter()
            r = solve_ivp(pr["fun"], pr["t_span"], pr["y0"], method=m,
                          rtol=tol, atol=tol * 1e-2, jac=pr.get("jac"))
            dt = time.perf_counter() - t0
            if "exact" in pr:
                err = max(abs(a - b) for a, b in
                          zip(r.y[-1], pr["exact"](r.t[-1])))
            elif "reference" in pr:
                err = max(abs(a - b) for a, b in
                          zip(r.y[-1], pr["reference"]))
            elif label.startswith("kepler"):
                err = max(abs(a - b) for a, b in zip(r.y[-1], pr["y0"]))
            elif "invariant" in pr:
                err = abs(pr["invariant"](r.y[-1]) - 1.0)
            else:
                err = float("nan")
            print("%-18s %-8s %7d %7d %6d %6d %10.2e %8.3f" % (
                label, m, r.stats.naccept, r.stats.nfev, r.stats.njev,
                r.stats.nlu, err, dt))


def cmd_energy(args):
    pr = problems.kepler(0.6)
    H = pr["invariant"]
    T = 2 * math.pi * args.orbits
    n_per = args.steps_per_orbit
    h = 2 * math.pi / n_per
    n = n_per * args.orbits
    e0 = H(pr["y0"])
    print("Kepler e=0.6, %d orbits, fixed methods use %d steps per orbit"
          % (args.orbits, n_per))
    print("max |H - H0| over the first and the last tenth of the run")
    print("%-24s %8s %14s %14s %8s" % ("method", "f evals", "first 10%",
                                       "last 10%", "growth"))

    def row(label, evals, ts, states):
        first = max(abs(H(y) - e0) for t, y in zip(ts, states)
                    if t <= 0.1 * T)
        last = max(abs(H(y) - e0) for t, y in zip(ts, states)
                   if t >= 0.9 * T)
        print("%-24s %8d %14.3e %14.3e %7.1fx" % (label, evals, first, last,
                                                  last / first))

    r = solve_ivp(pr["fun"], (0, T), pr["y0"], method="RK4", first_step=h)
    row("RK4 (fixed)", r.stats.nfev, r.t, r.y)
    r = solve_ivp(pr["fun"], (0, T), pr["y0"], method="DOPRI54", rtol=1e-6,
                  atol=1e-9)
    row("DOPRI54 rtol=1e-6", r.stats.nfev, r.t, r.y)
    for order in (2, 4):
        ts, qs, ps = symplectic.integrate(pr["dT"], pr["dV"], pr["y0"][:2],
                                          pr["y0"][2:], h, n, order=order)
        evals = n * len(symplectic.triple_jump_weights(order))
        row("Verlet" if order == 2 else "Yoshida order %d" % order, evals,
            ts, [q + p for q, p in zip(qs, ps)])
    print("\nRK energy error grows linearly with time; the symplectic error "
          "stays bounded.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="odekit", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("methods")
    s = sub.add_parser("solve")
    s.add_argument("problem")
    s.add_argument("--method", default="AUTO", choices=METHODS)
    s.add_argument("--rtol", type=float, default=1e-6)
    s.add_argument("--atol", type=float, default=1e-9)
    s.add_argument("--mu", type=float, default=None)
    s.add_argument("--t-end", type=float, default=None)
    s.add_argument("--max-steps", type=int, default=1000000)
    s.add_argument("--csv", default=None)
    s.add_argument("--points", type=int, default=500)
    sub.add_parser("bench")
    e = sub.add_parser("energy")
    e.add_argument("--orbits", type=int, default=100)
    e.add_argument("--steps-per-orbit", type=int, default=200)
    args = ap.parse_args(argv)
    {"methods": cmd_methods, "solve": cmd_solve, "bench": cmd_bench,
     "energy": cmd_energy}[args.cmd](args)


if __name__ == "__main__":
    main()
