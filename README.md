# odekit

Adaptive, stiff and symplectic ODE integrators written from first principles in pure Python. No NumPy, no SciPy, no dependencies at all.

`odekit` solves initial value problems `y' = f(t, y)` with the same machinery production solvers use: embedded Runge Kutta pairs with PI step control, a fourth order continuous extension for dense output, event location, an L stable implicit method with simplified Newton and Jacobian reuse for stiff systems, automatic stiffness detection that switches solvers mid integration, and symplectic composition methods of any even order for Hamiltonian systems.

Every Butcher tableau in the package is **proven** to have its advertised order by enumerating Butcher's rooted trees and checking each order condition in exact rational arithmetic. A single mistyped coefficient fails the test suite.

## Why this exists

Calling `scipy.integrate.solve_ivp` is easy. Knowing why it takes 300 steps on one problem and 3 million on another, why `RK45` silently grinds on a stiff chemistry model, or why a planet simulated with a "very accurate" method slowly spirals into its sun, is not. This project implements those ideas end to end in about 1,500 lines so every one of them can be read, tested and measured:

* **Accuracy vs cost.** Seven methods from Euler to Dormand and Prince 5(4), with exact order verification and empirical convergence tests.
* **Stiffness.** Explicit methods are stability bound: their step can never exceed roughly 3.3/|λ|. The implicit TRBDF2 solver takes steps a million times larger on the same problem.
* **Geometry.** Runge Kutta energy error grows linearly forever; a symplectic integrator's stays bounded. The `energy` command shows it in numbers.

## Quick start

Requires Python 3.8 or newer. Nothing to install.

```bash
git clone https://github.com/MohammadHossinzehi/odekit.git
cd odekit
python -m unittest discover -s tests     # 47 tests, about one second
python -m odekit methods                 # tableau report
python -m odekit bench                   # methods vs the problem zoo
python -m odekit energy --orbits 1000    # symplectic vs Runge Kutta
python -m odekit solve vanderpol --method AUTO --csv vdp.csv
```

As a library:

```python
from odekit import solve_ivp

def ball(t, y):                 # height, velocity
    return [y[1], -9.81]

def ground(t, y):
    return y[0]
ground.terminal = True          # stop integrating at the event
ground.direction = -1           # only while falling

res = solve_ivp(ball, (0, 100), [10.0, 0.0], events=ground,
                rtol=1e-10, atol=1e-12, dense_output=True)
print(res.t_events[0][0])       # 1.4278431229270645 = sqrt(2*10/9.81)
print(res.sol(0.5))             # dense output anywhere in the interval
print(res.stats)                # nfev, njev, nlu, naccept, nreject, ...
```

`method` is one of `Euler`, `HeunEuler21`, `RK4`, `BS32`, `RKF45`, `DOPRI54` (default), `TRBDF2`, `AUTO`. Pass `jac=lambda t, y: [[...]]` to give TRBDF2 an analytic Jacobian; otherwise it uses forward differences.

## What is inside

| module | contents |
| --- | --- |
| `trees.py` | rooted tree enumeration (OEIS A000081), tree density γ(t), elementary weights Φ(t), exact order checking |
| `tableaux.py` | exact rational tableaux: Euler, Heun/Euler 2(1), RK4, Bogacki and Shampine 3(2), Fehlberg 4(5), Dormand and Prince 5(4), TRBDF2 as an ESDIRK; stability functions R(z) |
| `steppers.py` | `ExplicitRK` (embedded pairs, FSAL, PI control, Hairer's stiffness test) and `ESDIRK` (simplified Newton, LU and Jacobian reuse, filtered error estimate) |
| `dense.py` | DOPRI's native continuous extension, cubic Hermite for everything else, piecewise `OdeSolution` |
| `ivp.py` | `solve_ivp` driver: events (Illinois root finding on the interpolant), `t_eval`, backward integration, AUTO switching |
| `symplectic.py` | symplectic Euler, Störmer Verlet, Yoshida/Suzuki triple jump composition to any even order |
| `problems.py` | test zoo with exact solutions, invariants or published reference values: exponential, harmonic, Kepler, Van der Pol, Robertson, Lorenz, Brusselator, Prothero and Robinson |
| `linalg.py` | LU with partial pivoting, finite difference Jacobians, weighted RMS error norm |

## Results

`python -m odekit methods`:

```
method       stages   order  verified  err ord  FSAL  real stab  |R(-inf)|
Euler             1       1         1        -    no     2.0000  unbounded
HeunEuler21       2       2         2        1    no     2.0000  unbounded
RK4               4       4         4        -    no     2.7853  unbounded
BS32              4       3         3        2   yes     2.5127  unbounded
RKF45             6       5         5        4    no     3.6777  unbounded
DOPRI54           7       5         5        4   yes     3.3066  unbounded
TRBDF2            3       2         2        3    no        inf   4.83e-12
```

`python -m odekit bench` (rtol shown in the code, error against exact solutions, invariants, or the Hairer and Wanner reference values):

```
problem            method     steps    nfev   njev    nlu      error
exponential        DOPRI54       71     428      0      0   2.42e-11
kepler e=0.5       DOPRI54      118     710      0      0   1.74e-07
vanderpol mu=1e6   TRBDF2      3887   22975     29   3051   3.84e-05
vanderpol mu=1e6   AUTO        3914   22792     30   3065   3.84e-05
robertson          TRBDF2       775    4112     41    231   2.33e-15
prothero-robinson  TRBDF2        21      74      1     18   3.16e-07
```

Robertson's chemistry runs from t = 0 to t = 100,000 in 775 steps with only 41 Jacobians and conserves total mass to machine precision. On `y' = -1e4 (y - cos t)` DOPRI54 needs more than 2,000 steps per time unit just to stay stable, TRBDF2 fewer than 200 for the whole run.

`python -m odekit energy --orbits 1000` (Kepler problem, eccentricity 0.6):

```
method                    f evals      first 10%       last 10%   growth
RK4 (fixed)                800001      3.356e-04      3.323e-03     9.9x
DOPRI54 rtol=1e-6          326894      4.278e-04      4.509e-03    10.5x
Verlet                     200000      3.676e-03      3.676e-03     1.0x
Yoshida order 4            600000      4.771e-05      4.771e-05     1.0x
```

Ten times longer means ten times more energy error for both Runge Kutta methods. The symplectic methods do not drift at all.

## Design decisions

**Exact order verification.** Coefficients are stored as `fractions.Fraction`. `trees.check_order` enumerates every rooted tree up to the requested order (17 trees for order 5) and checks `Φ(t) = 1/γ(t)` with no tolerance. The tests also perturb one DOPRI coefficient by 1e-9 and assert the verified order drops, so the check is known to have teeth.

**PI step control.** The step factor is `0.9 · err^(-0.7/k) · err_prev^(0.4/k)` clamped to [0.2, 5], with no growth allowed straight after a rejection. The proportional term reacts to the error trend and damps the accept/reject oscillation an integral only controller shows when a step is limited by stability rather than accuracy.

**Dense output is first class.** Every accepted step yields an interpolant. `t_eval` and events are served from interpolants, so asking for 1,000 output points does not change a single step (there is a test for exactly that).

**TRBDF2 internals.** With γ = 2 minus √2 both implicit stages share the diagonal coefficient, so one LU of `I - h d J` serves the whole step. The factors are reused while h is unchanged, and the controller deliberately holds h fixed when it would only grow by up to 20 percent. The Jacobian is kept until Newton actually fails. Stage derivatives are recovered algebraically from converged stage values rather than by calling f again, which avoids multiplying the Newton residual by a huge stiff Jacobian. The embedded error estimate is filtered through `(I - h d J)^-1` (Shampine's trick) so it stays bounded as hλ goes to minus infinity instead of throttling the step.

**Automatic stiffness switching.** DOPRI54 evaluates its last two stages at the same time t + h, so `‖k7 − k6‖ / ‖y7 − y6‖` estimates the dominant Jacobian eigenvalue for free. If hλ exceeds 3.25 on 15 accepted steps (with 6 non stiff steps resetting the counter, as in Hairer's DOPRI5), `AUTO` hands the current state and step size to TRBDF2. The switch is one way and is reported in `res.switches`.

**Symplectic composition.** Rather than hard coding Yoshida's constants, `triple_jump_weights(order)` builds them recursively: if S is a symmetric method of order 2k then `S(x1 h) S(x0 h) S(x1 h)` with `x1 = 1/(2 − 2^(1/(2k+1)))` is symmetric of order 2k + 2. Tests confirm orders 1, 2, 4 and 6 empirically, symmetry of the weights, bounded energy over 200 orbits and exact time reversibility of Verlet.

## Testing

`python -m unittest discover -s tests` runs 47 tests in about a second:

* **Theory:** tree counts match OEIS A000081, every tableau has its claimed order and embedded order, FSAL rows, stability polynomials, A and L stability of TRBDF2, known real stability intervals.
* **Convergence:** halving the step must change the error by 2^p for each method, and DOPRI's continuous extension must converge at order 4 or better.
* **Adaptivity:** tolerance proportionality, exact FSAL evaluation accounting (6 per DOPRI attempt), backward integration, vector `atol`, agreement between methods on Lorenz.
* **Stiff:** Robertson against published values at t = 40 to 1e-6 relative, mass conservation to 1e-12, Van der Pol with μ = 1e6 against the Hairer and Wanner reference, AUTO switching on stiff and not on non stiff problems, analytic vs finite difference Jacobians.
* **Events:** a falling ball hits the ground at exactly `sqrt(2h/g)`, all zero crossings of cosine, direction filters, terminal events on the stiff solver.

## Limitations

TRBDF2 is second order, so very tight tolerances on stiff problems cost many steps; a variable order BDF or Radau IIA method would be the next addition. The linear algebra is dense, which is right for systems with tens of equations and wrong for discretised PDEs with thousands. Pure Python means each f evaluation dominates; the point here is clarity and verifiability, not speed.

## References

* E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, Springer.
* E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential Algebraic Problems*, Springer.
* M. E. Hosea, L. F. Shampine, *Analysis and implementation of TR BDF2*, Applied Numerical Mathematics 20 (1996).
* H. Yoshida, *Construction of higher order symplectic integrators*, Physics Letters A 150 (1990).

## License

MIT
