"""Visualization utilities for liquid neural networks."""

import numpy as np
from typing import Optional


def plot_hidden_dynamics(
    hidden_states: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Hidden State Dynamics",
    max_units: int = 10,
) -> None:
    """
    Plot hidden state trajectories over time.

    Args:
        hidden_states: (seq_len, hidden_dim) array
        save_path:     If given, saves figure to this path instead of showing
        title:         Plot title
        max_units:     Maximum number of hidden units to plot
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    seq_len, hidden_dim = hidden_states.shape
    n_plot = min(hidden_dim, max_units)
    t = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_plot):
        ax.plot(t, hidden_states[:, i], label=f"h[{i}]", alpha=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Activation")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small", ncol=min(n_plot, 5))
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    else:
        plt.show()
    plt.close(fig)


def plot_trajectory(
    states: np.ndarray,
    predicted: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Pendulum Trajectory",
) -> None:
    """
    Plot pendulum trajectory (angle and angular velocity over time).

    Args:
        states:    (seq_len, 2) array of [theta, omega]
        predicted: Optional (seq_len, 2) array of predicted states
        save_path: If given, saves figure to this path
        title:     Plot title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(states))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    labels = ["θ (angle)", "ω (angular velocity)"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, states[:, i], label="True", color="steelblue")
        if predicted is not None:
            ax.plot(t, predicted[:, i], label="Predicted", color="tomato", linestyle="--")
        ax.set_ylabel(label)
        ax.legend()
    axes[-1].set_xlabel("Time step")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    else:
        plt.show()
    plt.close(fig)
