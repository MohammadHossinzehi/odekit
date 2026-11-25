"""Butcher's rooted tree theory: exact order conditions for Runge-Kutta methods.

A Runge-Kutta method with tableau (A, b, c) has order p exactly when, for
every rooted tree t with at most p vertices,

    Phi(t) = b . g(t) = 1 / gamma(t)

where g(t) is the vector of internal elementary weights defined recursively
by g(leaf) = (1, ..., 1) and g([t1, ..., tm]) = prod_k (A g(tk)) taken
componentwise, and gamma(t) is the tree "density" |t| * prod gamma(tk).

Trees are stored canonically as sorted tuples of their children, so the
single vertex tree is ``()`` and the tree with a root and two leaves is
``((), ())``. Using ``fractions.Fraction`` coefficients the whole check is
exact: no floating point tolerance is involved anywhere.
"""

from fractions import Fraction
from functools import lru_cache


def canon(children):
    return tuple(sorted(children))


def order(t):
    return 1 + sum(order(u) for u in t)


def gamma(t):
    g = order(t)
    for u in t:
        g *= gamma(u)
    return g


def _attach_everywhere(t):
    """All trees obtained from t by attaching one new leaf to one vertex."""
    out = {canon(t + ((),))}
    for i, u in enumerate(t):
        for v in _attach_everywhere(u):
            rest = t[:i] + t[i + 1:]
            out.add(canon(rest + (v,)))
    return out


@lru_cache(maxsize=None)
def trees_of_order(n):
    """Tuple of all distinct rooted trees with exactly n vertices.

    Counts follow OEIS A000081: 1, 1, 2, 4, 9, 20, 48, 115, ...
    """
    if n < 1:
        return ()
    if n == 1:
        return ((),)
    found = set()
    for t in trees_of_order(n - 1):
        found |= _attach_everywhere(t)
    return tuple(sorted(found, key=lambda t: (repr(t))))


def internal_weights(t, a, c):
    s = len(c)
    if t == ():
        return [1] * s
    g = [1] * s
    for u in t:
        gu = internal_weights(u, a, c)
        agu = [sum(a[i][j] * gu[j] for j in range(s)) for i in range(s)]
        g = [g[i] * agu[i] for i in range(s)]
    return g


def elementary_weight(t, a, b, c):
    g = internal_weights(t, a, c)
    return sum(bi * gi for bi, gi in zip(b, g))


def residual(t, a, b, c):
    """Phi(t) - 1/gamma(t); zero exactly when the condition holds."""
    return elementary_weight(t, a, b, c) - Fraction(1, gamma(t))


def check_order(a, b, c, max_order=8):
    """Return the largest p such that every order condition up to p holds.

    ``a``, ``b``, ``c`` may hold Fractions (exact) or floats (in which case a
    tolerance of 1e-12 is used).
    """
    exact = all(isinstance(x, (int, Fraction)) for x in b)
    p = 0
    for n in range(1, max_order + 1):
        for t in trees_of_order(n):
            r = residual(t, a, b, c)
            if (r != 0) if exact else (abs(r) > 1e-12):
                return p
        p = n
    return p


def failing_conditions(a, b, c, n):
    """List the trees of order n whose order condition is violated."""
    return [t for t in trees_of_order(n) if residual(t, a, b, c) != 0]


def tree_to_str(t):
    """Bracket notation: '*' is a leaf, '[* *]' is a root with two leaves."""
    if t == ():
        return "*"
    return "[" + " ".join(tree_to_str(u) for u in t) + "]"
