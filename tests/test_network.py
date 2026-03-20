"""Tests for LiquidNetwork."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liquid_neural.network import LiquidNetwork


def test_liquid_network_forward_shape():
    """Output shape should be (seq_len, output_dim)."""
    net = LiquidNetwork(input_dim=2, hidden_dims=[8, 4], output_dim=2, dt=0.01)
    inputs = np.random.randn(20, 2)
    outputs, hidden_states = net.forward(inputs)
    assert outputs.shape == (20, 2), f"Expected (20, 2), got {outputs.shape}"


def test_liquid_network_hidden_states_collected():
    """hidden_states_list should have one entry per layer with correct shapes."""
    net = LiquidNetwork(input_dim=3, hidden_dims=[6, 4], output_dim=1)
    inputs = np.random.randn(15, 3)
    outputs, hidden_states = net.forward(inputs)
    assert len(hidden_states) == 2, "Should have 2 layers"
    assert hidden_states[0].shape == (15, 6), f"Layer 0: expected (15,6), got {hidden_states[0].shape}"
    assert hidden_states[1].shape == (15, 4), f"Layer 1: expected (15,4), got {hidden_states[1].shape}"


def test_network_predict():
    """predict() should return same outputs as forward()."""
    net = LiquidNetwork(input_dim=2, hidden_dims=[4], output_dim=1)
    inputs = np.random.randn(10, 2)
    outputs_forward, _ = net.forward(inputs)
    outputs_predict = net.predict(inputs)
    assert np.allclose(outputs_forward, outputs_predict), "predict() should match forward() outputs"


def test_network_single_layer():
    """Single-layer network should work."""
    net = LiquidNetwork(input_dim=1, hidden_dims=[5], output_dim=1)
    inputs = np.linspace(0, 1, 30).reshape(-1, 1)
    outputs, hidden = net.forward(inputs)
    assert outputs.shape == (30, 1)
    assert len(hidden) == 1
    assert hidden[0].shape == (30, 5)


def test_network_get_set_params():
    """get_params / set_params should allow parameter manipulation."""
    net = LiquidNetwork(input_dim=2, hidden_dims=[4], output_dim=1)
    params = net.get_params()
    assert "W_out" in params
    assert "b_out" in params
    # Modify and set
    params["b_out"] = params["b_out"] + 5.0
    net.set_params(params)
    assert np.allclose(net.b_out, params["b_out"])
