"""Tests for ODE solvers."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liquid_neural.ode_solver import euler_step, rk4_step, solve_ode


def _linear_ode(y, t):
    """dy/dt = -y  => exact solution: y(t) = y0 * exp(-t)"""
    return -y


def test_euler_step():
    """Euler step should move in the correct direction."""
    y0 = np.array([1.0])
    y1 = euler_step(_linear_ode, y0, t=0.0, dt=0.1)
    # y1 should be y0 + dt * (-y0) = 0.9
    assert np.isclose(y1[0], 0.9), f"Expected 0.9, got {y1[0]}"


def test_rk4_step():
    """RK4 step should be more accurate than Euler for the same dt."""
    y0 = np.array([1.0])
    dt = 0.1
    y_exact = np.exp(-dt)  # exact solution at t=dt

    y_euler = euler_step(_linear_ode, y0, t=0.0, dt=dt)
    y_rk4 = rk4_step(_linear_ode, y0, t=0.0, dt=dt)

    err_euler = abs(y_euler[0] - y_exact)
    err_rk4 = abs(y_rk4[0] - y_exact)

    assert err_rk4 < err_euler, "RK4 should be more accurate than Euler"
    assert err_rk4 < 1e-6, f"RK4 error too large: {err_rk4}"


def test_ode_solve_shape():
    """solve_ode should return arrays of correct shape."""
    y0 = np.array([1.0, 0.0])
    t_arr, y_arr = solve_ode(_linear_ode, y0, t_span=(0.0, 1.0), dt=0.1)

    assert t_arr.ndim == 1
    assert y_arr.ndim == 2
    assert y_arr.shape[1] == 2  # same as y0 dimension
    assert len(t_arr) == y_arr.shape[0]


def test_ode_solve_euler_method():
    """solve_ode with euler method should work."""
    y0 = np.array([1.0])
    t_arr, y_arr = solve_ode(_linear_ode, y0, t_span=(0.0, 0.5), dt=0.05, method="euler")
    assert y_arr.shape[1] == 1
    assert y_arr[0, 0] == pytest.approx(1.0)


def test_ode_solve_accuracy():
    """RK4 solution should closely match exact solution for simple ODE."""
    y0 = np.array([2.0])
    t_arr, y_arr = solve_ode(_linear_ode, y0, t_span=(0.0, 1.0), dt=0.01)
    y_exact = 2.0 * np.exp(-t_arr)
    max_err = np.max(np.abs(y_arr[:, 0] - y_exact))
    assert max_err < 1e-4, f"RK4 max error {max_err} too large"
