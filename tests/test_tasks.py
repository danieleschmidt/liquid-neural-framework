"""Tests for PendulumTask and dataset helpers."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liquid_neural.tasks import PendulumTask, make_sine_regression_data, make_step_response_data


def test_pendulum_task_generates_data():
    """generate_trajectory returns states and targets."""
    task = PendulumTask(dt=0.05, n_steps=50)
    states, targets = task.generate_trajectory(theta0=0.3, omega0=0.0)
    assert states.shape[1] == 2, "States should have 2 columns (theta, omega)"
    assert targets.shape[1] == 2
    assert len(states) == len(targets)


def test_pendulum_task_dataset_shape():
    """generate_dataset returns correct shape arrays."""
    task = PendulumTask(dt=0.05, n_steps=40)
    X, y = task.generate_dataset(n_trajectories=10)
    assert X.shape == (10, 40, 2), f"Expected (10, 40, 2), got {X.shape}"
    assert y.shape == (10, 40, 2), f"Expected (10, 40, 2), got {y.shape}"


def test_pendulum_trajectory_physics():
    """Pendulum starting at rest near 0 should stay bounded."""
    task = PendulumTask(dt=0.02, n_steps=200, damping=0.3)
    states, _ = task.generate_trajectory(theta0=0.1, omega0=0.0)
    # Angle should stay within pi for small initial conditions with damping
    assert np.all(np.abs(states[:, 0]) < np.pi), "Angle should stay bounded"


def test_sine_regression_data_shape():
    """make_sine_regression_data returns correct shapes."""
    X, y = make_sine_regression_data(n_samples=100)
    assert X.shape == (100, 1)
    assert y.shape == (100, 1)


def test_sine_regression_data_range():
    """Input values should be in [0, 2*pi]."""
    X, _ = make_sine_regression_data(n_samples=500)
    assert np.all(X >= 0) and np.all(X <= 2 * np.pi)


def test_step_response_data():
    """make_step_response_data should produce values approaching 1."""
    t, y = make_step_response_data(n_samples=200)
    assert t.shape == (200, 1)
    assert y.shape == (200, 1)
    # Final value should be close to 1
    assert y[-1, 0] > 0.99, f"Step response should approach 1, got {y[-1, 0]}"
    # Initial value should be near 0
    assert y[0, 0] < 0.05, f"Step response should start near 0, got {y[0, 0]}"
