"""Tests for LiquidTimeConstantCell."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liquid_neural.cells import LiquidTimeConstantCell


def test_ltc_cell_forward_shape():
    """Output shape matches hidden_dim."""
    cell = LiquidTimeConstantCell(input_dim=3, hidden_dim=8)
    x = np.zeros(3)
    h = cell.initial_state()
    h_new = cell.forward(x, h)
    assert h_new.shape == (8,), f"Expected (8,), got {h_new.shape}"


def test_ltc_cell_hidden_state_changes():
    """Hidden state should change after a forward pass with non-zero input."""
    cell = LiquidTimeConstantCell(input_dim=2, hidden_dim=4, seed=42)
    x = np.array([1.0, -1.0])
    h = cell.initial_state()
    h_new = cell.forward(x, h)
    assert not np.allclose(h, h_new), "Hidden state should change after non-zero input"


def test_ltc_cell_time_constants_positive():
    """Time constants must always be positive."""
    cell = LiquidTimeConstantCell(input_dim=4, hidden_dim=16, seed=7)
    assert np.all(cell.tau > 0), "All time constants must be positive"


def test_ltc_cell_get_set_params():
    """get_params / set_params round-trip."""
    cell = LiquidTimeConstantCell(input_dim=2, hidden_dim=4, seed=1)
    params = cell.get_params()
    # Modify one param
    params["bias"] = params["bias"] + 1.0
    cell.set_params(params)
    assert np.allclose(cell.bias, params["bias"])


def test_ltc_cell_initial_state_zero():
    """Initial state should be all zeros."""
    cell = LiquidTimeConstantCell(input_dim=3, hidden_dim=5)
    h0 = cell.initial_state()
    assert np.all(h0 == 0.0)
    assert h0.shape == (5,)
