"""LiquidNetwork - Stack of LiquidTimeConstantCells with a linear output layer."""

import numpy as np
from typing import List, Tuple

from .cells import LiquidTimeConstantCell


class LiquidNetwork:
    """
    Multi-layer liquid neural network.

    Stacks LiquidTimeConstantCells and adds a final linear projection
    to produce outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dt: float = 0.01,
        seed: int = 0,
    ):
        """
        Args:
            input_dim:   Dimensionality of the input at each time step.
            hidden_dims: List of hidden sizes, one per layer.
            output_dim:  Number of output units.
            dt:          Integration time step for each cell.
            seed:        Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dt = dt

        rng = np.random.default_rng(seed)

        # Build cells
        self.cells: List[LiquidTimeConstantCell] = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            cell = LiquidTimeConstantCell(in_dim, h_dim, dt=dt, seed=int(rng.integers(0, 2**31)))
            self.cells.append(cell)
            in_dim = h_dim

        # Output linear layer
        self.W_out = rng.normal(0, 0.1, (output_dim, hidden_dims[-1]))
        self.b_out = np.zeros(output_dim)

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run the network over a sequence.

        Args:
            inputs: (seq_len, input_dim) array

        Returns:
            outputs:           (seq_len, output_dim)
            hidden_states_list: list of (seq_len, hidden_dim) arrays, one per layer
        """
        seq_len = inputs.shape[0]

        # Initialize hidden states
        hidden = [cell.initial_state() for cell in self.cells]

        # Storage
        all_hidden: List[List[np.ndarray]] = [[] for _ in self.cells]
        outputs = []

        for t_idx in range(seq_len):
            x = inputs[t_idx]
            t = t_idx * self.dt

            # Pass through each layer
            for layer_idx, cell in enumerate(self.cells):
                x = cell.forward(x, hidden[layer_idx], t)
                hidden[layer_idx] = x
                all_hidden[layer_idx].append(x.copy())

            # Linear output from last layer
            out = self.W_out @ hidden[-1] + self.b_out
            outputs.append(out)

        outputs_arr = np.stack(outputs, axis=0)
        hidden_states_list = [np.stack(h, axis=0) for h in all_hidden]
        return outputs_arr, hidden_states_list

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience method: returns only the outputs.

        Args:
            inputs: (seq_len, input_dim)

        Returns:
            outputs: (seq_len, output_dim)
        """
        outputs, _ = self.forward(inputs)
        return outputs

    def get_params(self) -> dict:
        """Return all parameters as a dict."""
        params = {
            "W_out": self.W_out.copy(),
            "b_out": self.b_out.copy(),
        }
        for i, cell in enumerate(self.cells):
            for k, v in cell.get_params().items():
                params[f"cell_{i}_{k}"] = v
        return params

    def set_params(self, params: dict) -> None:
        """Set all parameters from a dict."""
        self.W_out = params["W_out"].copy()
        self.b_out = params["b_out"].copy()
        for i, cell in enumerate(self.cells):
            cell_params = {
                k.replace(f"cell_{i}_", ""): v
                for k, v in params.items()
                if k.startswith(f"cell_{i}_")
            }
            cell.set_params(cell_params)
