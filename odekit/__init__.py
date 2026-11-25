"""odekit: adaptive, stiff and symplectic ODE integrators in pure Python."""

from .ivp import solve_ivp, METHODS, Result
from .tableaux import (Tableau, EULER, HEUN_EULER, RK4, BS32, RKF45, DOPRI54,
                       TRBDF2)
from .dense import OdeSolution
from . import problems, symplectic, trees

__all__ = ["solve_ivp", "METHODS", "Result", "Tableau", "OdeSolution",
           "EULER", "HEUN_EULER", "RK4", "BS32", "RKF45", "DOPRI54", "TRBDF2",
           "problems", "symplectic", "trees"]
__version__ = "1.0.0"
