"""Small dense linear algebra kernel used by the implicit solvers.

Everything works on plain Python lists so the package has zero dependencies.
The systems we factor are the Newton matrices ``I - h*gamma*J`` of the
implicit methods, which are small (the dimension of the ODE) and are reused
across many Newton iterations, so a straightforward LU with partial pivoting
is the right tool.
"""

import math

EPS = 2.220446049250313e-16


class SingularMatrixError(ArithmeticError):
    """Raised when a pivot is exactly zero during LU factorisation."""


def lu_factor(a):
    """Factor a square matrix in place style (a copy is made).

    Returns ``(lu, piv)`` where ``lu`` holds L (unit lower, below the
    diagonal) and U (on and above the diagonal) and ``piv`` is the row
    permutation, so that ``P A = L U``.
    """
    n = len(a)
    lu = [list(row) for row in a]
    piv = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if lu[p][k] == 0.0:
            raise SingularMatrixError("matrix is singular at column %d" % k)
        if p != k:
            lu[k], lu[p] = lu[p], lu[k]
            piv[k], piv[p] = piv[p], piv[k]
        pivot = lu[k][k]
        row_k = lu[k]
        for i in range(k + 1, n):
            row_i = lu[i]
            m = row_i[k] / pivot
            if m != 0.0:
                row_i[k] = m
                for j in range(k + 1, n):
                    row_i[j] -= m * row_k[j]
            else:
                row_i[k] = 0.0
    return lu, piv


def lu_solve(factors, b):
    """Solve ``A x = b`` given ``factors = lu_factor(A)``."""
    lu, piv = factors
    n = len(lu)
    x = [b[piv[i]] for i in range(n)]
    for i in range(n):
        row = lu[i]
        s = x[i]
        for j in range(i):
            s -= row[j] * x[j]
        x[i] = s
    for i in range(n - 1, -1, -1):
        row = lu[i]
        s = x[i]
        for j in range(i + 1, n):
            s -= row[j] * x[j]
        x[i] = s / row[i]
    return x


def solve(a, b):
    """Convenience wrapper: factor and solve in one go."""
    return lu_solve(lu_factor(a), b)


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def fd_jacobian(f, t, y, f0=None):
    """Forward difference Jacobian ``J[i][j] = d f_i / d y_j``.

    The increment for column j is ``sqrt(eps) * max(|y_j|, 1e-5)`` which is
    the classic choice balancing truncation error against cancellation.
    Returns ``(J, nfev)``.
    """
    n = len(y)
    nfev = 0
    if f0 is None:
        f0 = f(t, y)
        nfev += 1
    cols = []
    yp = list(y)
    for j in range(n):
        delta = math.sqrt(EPS) * max(abs(y[j]), 1e-5)
        yp[j] = y[j] + delta
        delta = yp[j] - y[j]  # the representable increment
        fj = f(t, yp)
        nfev += 1
        cols.append([(fj[i] - f0[i]) / delta for i in range(n)])
        yp[j] = y[j]
    jac = [[cols[j][i] for j in range(n)] for i in range(n)]
    return jac, nfev


def rms_norm(err, y0, y1, rtol, atol):
    """Weighted RMS norm used for every error test in the package.

    The weight for component i is ``atol_i + rtol * max(|y0_i|, |y1_i|)``,
    so a norm of 1.0 means "exactly at tolerance".
    """
    n = len(err)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        a = atol[i] if isinstance(atol, (list, tuple)) else atol
        w = a + rtol * max(abs(y0[i]), abs(y1[i]))
        s += (err[i] / w) ** 2
    return math.sqrt(s / n)
