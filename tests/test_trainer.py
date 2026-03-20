"""Tests for SimpleTrainer."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liquid_neural.network import LiquidNetwork
from liquid_neural.trainer import SimpleTrainer


def _make_net_and_data(seq_len=10, input_dim=2, hidden_dim=4, output_dim=1, seed=0):
    rng = np.random.default_rng(seed)
    net = LiquidNetwork(input_dim=input_dim, hidden_dims=[hidden_dim], output_dim=output_dim)
    X = rng.normal(0, 0.3, (seq_len, input_dim))
    y = rng.normal(0, 0.3, (seq_len, output_dim))
    return net, X, y


def test_trainer_fit_returns_losses():
    """fit() should return a list of losses with one entry per epoch."""
    net, X, y = _make_net_and_data()
    trainer = SimpleTrainer(net, lr=0.001)
    losses = trainer.fit(X, y, epochs=3)
    assert isinstance(losses, list), "fit() should return a list"
    assert len(losses) == 3, f"Expected 3 loss values, got {len(losses)}"
    assert all(isinstance(l, float) for l in losses), "Each loss should be a float"


def test_trainer_mse_loss():
    """mse_loss should compute mean squared error correctly."""
    net, _, _ = _make_net_and_data()
    trainer = SimpleTrainer(net)
    y_pred = np.array([[1.0], [2.0]])
    y_true = np.array([[1.5], [1.5]])
    loss = trainer.mse_loss(y_pred, y_true)
    expected = 0.25  # ((0.5)^2 + (0.5)^2) / 2
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_trainer_evaluate():
    """evaluate() should return a non-negative float."""
    net, X, y = _make_net_and_data()
    trainer = SimpleTrainer(net)
    mse = trainer.evaluate(X, y)
    assert isinstance(mse, float)
    assert mse >= 0.0


def test_trainer_loss_is_finite():
    """Losses from fit() should all be finite numbers."""
    net, X, y = _make_net_and_data(seq_len=5)
    trainer = SimpleTrainer(net, lr=1e-4)
    losses = trainer.fit(X, y, epochs=2)
    for i, l in enumerate(losses):
        assert np.isfinite(l), f"Loss at epoch {i} is not finite: {l}"
