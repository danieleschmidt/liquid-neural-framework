#!/usr/bin/env python3
"""
Pendulum demo: train a LiquidNetwork to predict pendulum dynamics.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from liquid_neural import LiquidNetwork, SimpleTrainer, PendulumTask
from liquid_neural.visualization import plot_hidden_dynamics, plot_trajectory


def main():
    print("=== Liquid Neural Network - Pendulum Demo ===\n")

    # 1. Generate data
    task = PendulumTask(dt=0.05, n_steps=80)
    states, targets = task.generate_trajectory(theta0=0.8, omega0=0.0)
    print(f"Trajectory shape: {states.shape}")

    # 2. Build network
    net = LiquidNetwork(input_dim=2, hidden_dims=[16, 8], output_dim=2, dt=0.05)
    print(f"Network: 2 -> [16, 8] -> 2")

    # 3. Train (small number of epochs for demo speed)
    trainer = SimpleTrainer(net, lr=5e-4)
    print("\nTraining (5 epochs)...")
    losses = trainer.fit(states, targets, epochs=5)
    for i, l in enumerate(losses):
        print(f"  Epoch {i+1}: loss = {l:.6f}")

    # 4. Predict
    predicted, hidden_states = net.forward(states)
    print(f"\nPrediction MSE: {trainer.mse_loss(predicted, targets):.6f}")

    # 5. Save plots
    out_dir = "/tmp/liquid_neural_plots"
    os.makedirs(out_dir, exist_ok=True)

    plot_trajectory(
        targets, predicted,
        save_path=os.path.join(out_dir, "pendulum_trajectory.png"),
    )
    print(f"Saved trajectory plot to {out_dir}/pendulum_trajectory.png")

    plot_hidden_dynamics(
        hidden_states[0],
        save_path=os.path.join(out_dir, "hidden_dynamics.png"),
        title="Layer 0 Hidden Dynamics",
    )
    print(f"Saved hidden dynamics plot to {out_dir}/hidden_dynamics.png")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
