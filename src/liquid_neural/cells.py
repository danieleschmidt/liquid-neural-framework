"""LiquidTimeConstantCell - ODE-inspired RNN cell with learned time constants."""

import numpy as np


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class LiquidTimeConstantCell:
    """
    Liquid Time-Constant (LTC) cell.

    Dynamics:
        dx/dt = (-x/tau) + f(x, u, t) * (A - x)

    where:
        tau  : learned time constants (positive)
        A    : attractor bias
        f    : gating function = sigmoid(W_in @ u + W_rec @ x + bias)

    Simplified form used here:
        dh/dt = (-h + f * A) / tau
        h_new = h + dt * dh/dt  (Euler step)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 0.01, seed: int = 0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt

        rng = np.random.default_rng(seed)
        scale = 0.1

        # Input projection weights
        self.W_in = rng.normal(0, scale, (hidden_dim, input_dim))
        # Recurrent weights
        self.W_rec = rng.normal(0, scale, (hidden_dim, hidden_dim))
        # Bias
        self.bias = np.zeros(hidden_dim)
        # Time constants: positive, initialized in [0.1, 1.0]
        self.tau = rng.uniform(0.1, 1.0, hidden_dim)
        # Attractor bias
        self.A = rng.normal(0, scale, hidden_dim)

    def forward(self, x: np.ndarray, h: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        One step forward.

        Args:
            x: input vector, shape (input_dim,)
            h: hidden state, shape (hidden_dim,)
            t: current time (unused in this simplified form, kept for API compat)

        Returns:
            h_new: updated hidden state, shape (hidden_dim,)
        """
        gate = _sigmoid(self.W_in @ x + self.W_rec @ h + self.bias)
        dh_dt = (-h + gate * self.A) / self.tau
        h_new = h + self.dt * dh_dt
        return h_new

    def initial_state(self) -> np.ndarray:
        """Return zero initial hidden state."""
        return np.zeros(self.hidden_dim)

    def get_params(self) -> dict:
        """Return a dict of all trainable parameters."""
        return {
            "W_in": self.W_in.copy(),
            "W_rec": self.W_rec.copy(),
            "bias": self.bias.copy(),
            "tau": self.tau.copy(),
            "A": self.A.copy(),
        }

    def set_params(self, params: dict) -> None:
        """Set trainable parameters from a dict."""
        self.W_in = params["W_in"].copy()
        self.W_rec = params["W_rec"].copy()
        self.bias = params["bias"].copy()
        self.tau = np.abs(params["tau"])  # enforce positivity
        self.A = params["A"].copy()
