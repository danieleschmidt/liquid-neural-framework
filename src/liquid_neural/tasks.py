"""Toy tasks for training and evaluating liquid neural networks."""

import numpy as np
from typing import Tuple

from .ode_solver import solve_ode


class PendulumTask:
    """
    Simple damped pendulum task.

    Dynamics:  theta'' = -sin(theta) - b * theta'

    State:  [theta, omega]  (angle, angular velocity)
    Target: next-step prediction of [theta, omega]
    """

    def __init__(self, dt: float = 0.05, n_steps: int = 100, damping: float = 0.1):
        self.dt = dt
        self.n_steps = n_steps
        self.damping = damping

    def _pendulum_ode(self, b: float):
        """Return the pendulum ODE function with damping b."""
        def f(y, t):
            theta, omega = y
            dtheta = omega
            domega = -np.sin(theta) - b * omega
            return np.array([dtheta, domega])
        return f

    def generate_trajectory(
        self, theta0: float = 0.5, omega0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single pendulum trajectory.

        Args:
            theta0: Initial angle (radians)
            omega0: Initial angular velocity

        Returns:
            states:  (n_steps+1, 2)  — [theta, omega] at each step
            targets: (n_steps, 2)    — next-step states (shifted by 1)
        """
        y0 = np.array([theta0, omega0])
        t_span = (0.0, self.n_steps * self.dt)
        _, y_arr = solve_ode(self._pendulum_ode(self.damping), y0, t_span, dt=self.dt)

        states = y_arr[:-1]   # (n_steps, 2)
        targets = y_arr[1:]   # (n_steps, 2)  — next state
        return states, targets

    def generate_dataset(
        self, n_trajectories: int = 50, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset of multiple trajectories.

        Each trajectory is a sequence of (state, next_state) pairs.

        Returns:
            X: (n_trajectories, n_steps, 2)
            y: (n_trajectories, n_steps, 2)
        """
        rng = np.random.default_rng(seed)
        X_list, y_list = [], []
        for _ in range(n_trajectories):
            theta0 = rng.uniform(-np.pi / 2, np.pi / 2)
            omega0 = rng.uniform(-1.0, 1.0)
            states, targets = self.generate_trajectory(theta0, omega0)
            X_list.append(states)
            y_list.append(targets)
        return np.stack(X_list), np.stack(y_list)


def make_sine_regression_data(
    n_samples: int = 200, noise: float = 0.1, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy sine regression data.

    Returns:
        X: (n_samples, 1)  — input values in [0, 2*pi]
        y: (n_samples, 1)  — sin(x) + noise
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2 * np.pi, (n_samples, 1))
    y = np.sin(X) + rng.normal(0, noise, (n_samples, 1))
    return X, y


def make_step_response_data(
    n_samples: int = 200, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate step-response data (unit step input, first-order system output).

    System: dy/dt = (u - y) / tau  with tau=0.5

    Returns:
        X: (n_samples, 1)  — time values
        y: (n_samples, 1)  — step response
    """
    tau = 0.5
    t = np.linspace(0, 3.0, n_samples).reshape(-1, 1)
    y = 1.0 - np.exp(-t / tau)  # analytical step response
    return t, y
