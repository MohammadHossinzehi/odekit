"""Symplectic splitting integrators for separable Hamiltonians
H(q, p) = T(p) + V(q).

General purpose integrators (even very accurate ones) let the energy of a
conservative system drift without bound over long times. A symplectic map
instead exactly conserves a nearby "shadow" Hamiltonian, so the energy
error stays bounded for exponentially long times. The price is a fixed
step size.

Higher orders are built by Yoshida/Suzuki triple jump composition: if S_2k
is a symmetric method of order 2k then

    S_2k+2(h) = S_2k(x1 h) S_2k(x0 h) S_2k(x1 h),
    x1 = 1 / (2 - 2^(1/(2k+1))),  x0 = 1 - 2 x1

is symmetric of order 2k + 2. Composing recursively gives any even order.
"""


def triple_jump_weights(order):
    """Step weights w such that applying Verlet with steps w_i h in sequence
    gives a symmetric method of the requested even order."""
    if order < 2 or order % 2:
        raise ValueError("order must be an even integer >= 2")
    w = [1.0]
    k = 1
    while 2 * k < order:
        x1 = 1.0 / (2.0 - 2.0 ** (1.0 / (2 * k + 1)))
        x0 = 1.0 - 2.0 * x1
        w = [x1 * a for a in w] + [x0 * a for a in w] + [x1 * a for a in w]
        k += 1
    return w


def _verlet_step(dT, dV, q, p, h):
    g = dV(q)
    p = [pi - 0.5 * h * gi for pi, gi in zip(p, g)]
    v = dT(p)
    q = [qi + h * vi for qi, vi in zip(q, v)]
    g = dV(q)
    p = [pi - 0.5 * h * gi for pi, gi in zip(p, g)]
    return q, p


def symplectic_euler_step(dT, dV, q, p, h):
    g = dV(q)
    p = [pi - h * gi for pi, gi in zip(p, g)]
    v = dT(p)
    q = [qi + h * vi for qi, vi in zip(q, v)]
    return q, p


def integrate(dT, dV, q0, p0, h, n_steps, order=2, record_every=1):
    """Integrate a separable Hamiltonian system for ``n_steps`` steps.

    ``dT(p)`` returns dT/dp (the velocity), ``dV(q)`` returns dV/dq (minus
    the force). ``order`` 1 selects symplectic Euler; any even order uses
    composed Verlet. Returns lists ``(ts, qs, ps)`` sampled every
    ``record_every`` steps (including the initial state).
    """
    q, p = [float(x) for x in q0], [float(x) for x in p0]
    ts, qs, ps = [0.0], [list(q)], [list(p)]
    if order == 1:
        stepper = lambda q, p: symplectic_euler_step(dT, dV, q, p, h)
    else:
        weights = triple_jump_weights(order)

        def stepper(q, p):
            for w in weights:
                q, p = _verlet_step(dT, dV, q, p, w * h)
            return q, p
    for i in range(1, n_steps + 1):
        q, p = stepper(q, p)
        if i % record_every == 0 or i == n_steps:
            ts.append(i * h)
            qs.append(list(q))
            ps.append(list(p))
    return ts, qs, ps
