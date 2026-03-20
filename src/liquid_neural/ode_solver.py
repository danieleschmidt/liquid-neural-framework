"""Simple ODE solvers: Euler and RK4."""

import numpy as np
from typing import Callable, Tuple


def euler_step(
    f: Callable[[np.ndarray, float], np.ndarray],
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    """
    One Euler integration step.

    y_{n+1} = y_n + dt * f(y_n, t_n)

    Args:
        f:  Derivative function f(y, t) -> dy/dt
        y:  Current state
        t:  Current time
        dt: Step size

    Returns:
        y_new: Updated state
    """
    return y + dt * f(y, t)


def rk4_step(
    f: Callable[[np.ndarray, float], np.ndarray],
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    """
    One Runge-Kutta 4th order integration step.

    Args:
        f:  Derivative function f(y, t) -> dy/dt
        y:  Current state
        t:  Current time
        dt: Step size

    Returns:
        y_new: Updated state
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ode(
    f: Callable[[np.ndarray, float], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    method: str = "rk4",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE from t_span[0] to t_span[1].

    Args:
        f:       Derivative function f(y, t) -> dy/dt
        y0:      Initial state
        t_span:  (t_start, t_end)
        dt:      Integration step size
        method:  'euler' or 'rk4' (default: 'rk4')

    Returns:
        t_arr: time array, shape (n_steps+1,)
        y_arr: state array, shape (n_steps+1, len(y0))
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))

    step_fn = euler_step if method == "euler" else rk4_step

    t_arr = np.zeros(n_steps + 1)
    y_arr = np.zeros((n_steps + 1, len(y0)))

    y_arr[0] = y0
    t_arr[0] = t_start

    y = y0.copy()
    t = t_start
    for i in range(n_steps):
        y = step_fn(f, y, t, dt)
        t = t_start + (i + 1) * dt
        y_arr[i + 1] = y
        t_arr[i + 1] = t

    return t_arr, y_arr
