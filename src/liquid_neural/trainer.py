"""SimpleTrainer - finite-difference gradient descent for LiquidNetwork."""

import numpy as np
from typing import List

from .network import LiquidNetwork


class SimpleTrainer:
    """
    Minimal trainer using finite-difference gradient approximation.

    For each parameter, perturbs it by a small epsilon, measures the change
    in loss, and updates in the negative gradient direction.

    This is intentionally simple — suitable for small networks and
    demonstration purposes.
    """

    def __init__(self, network: LiquidNetwork, lr: float = 0.001, epsilon: float = 1e-4):
        """
        Args:
            network: LiquidNetwork to train
            lr:      Learning rate
            epsilon: Finite-difference perturbation size
        """
        self.network = network
        self.lr = lr
        self.epsilon = epsilon

    def mse_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean squared error loss."""
        diff = y_pred - y_true
        return float(np.mean(diff ** 2))

    def _compute_loss_for_params(self, params: dict, X: np.ndarray, y: np.ndarray) -> float:
        """Set params, run forward, return loss."""
        self.network.set_params(params)
        y_pred = self.network.predict(X)
        # y may be (seq_len, output_dim); align shapes
        if y_pred.shape != y.shape:
            # If y is 1-D target per step
            y_pred = y_pred.reshape(y.shape)
        return self.mse_loss(y_pred, y)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        One training step: finite-difference gradient estimate + parameter update.

        Args:
            X: input sequence (seq_len, input_dim)
            y: target sequence (seq_len, output_dim)

        Returns:
            loss: MSE loss before the update
        """
        params = self.network.get_params()

        # Current loss
        base_loss = self._compute_loss_for_params(params, X, y)

        # Compute finite-diff gradients and update
        new_params = {}
        for name, param in params.items():
            flat = param.flatten()
            grad_flat = np.zeros_like(flat)
            for i in range(len(flat)):
                p_plus = params.copy()
                arr = flat.copy()
                arr[i] += self.epsilon
                p_plus[name] = arr.reshape(param.shape)
                loss_plus = self._compute_loss_for_params(p_plus, X, y)
                grad_flat[i] = (loss_plus - base_loss) / self.epsilon
            new_params[name] = param - self.lr * grad_flat.reshape(param.shape)

        self.network.set_params(new_params)
        return base_loss

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> List[float]:
        """
        Train the network for a number of epochs.

        For efficiency, only one finite-diff step per epoch is performed
        (suitable for demonstration).

        Args:
            X:      Input, shape (seq_len, input_dim) or batch
            y:      Target, matching output shape
            epochs: Number of training epochs

        Returns:
            losses: List of loss values, one per epoch
        """
        losses = []
        for _ in range(epochs):
            loss = self.train_step(X, y)
            losses.append(loss)
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate MSE on a dataset.

        Args:
            X: input sequence
            y: targets

        Returns:
            mse: mean squared error
        """
        y_pred = self.network.predict(X)
        if y_pred.shape != y.shape:
            y_pred = y_pred.reshape(y.shape)
        return self.mse_loss(y_pred, y)
