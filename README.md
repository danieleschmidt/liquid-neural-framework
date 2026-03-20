# Liquid Neural Framework

Continuous-time adaptive neural networks for robotics and control, implemented in Python/NumPy.

## Overview

This framework implements **Liquid Time-Constant (LTC) networks** — a class of neural ODEs where each neuron has a learned time constant governing the speed of its dynamics. LTC networks are particularly well-suited for control problems and time-series prediction with irregular timing.

### Key Concepts

The core ODE for each LTC cell:

```
dh/dt = (-h + f(x, h) * A) / tau
```

where:
- `h` — hidden state
- `tau` — learned time constants (always positive)
- `A` — attractor bias
- `f(x, h) = sigmoid(W_in @ x + W_rec @ h + bias)` — gating function

## Components

| Module | Description |
|--------|-------------|
| `cells.py` | `LiquidTimeConstantCell` — single ODE-inspired neuron layer |
| `network.py` | `LiquidNetwork` — stacked cells with linear output |
| `ode_solver.py` | Euler and RK4 integrators |
| `tasks.py` | `PendulumTask` + synthetic datasets |
| `trainer.py` | `SimpleTrainer` with finite-difference gradients |
| `visualization.py` | Hidden state and trajectory plots |

## Quick Start

```python
from liquid_neural import LiquidNetwork, SimpleTrainer, PendulumTask

# Generate pendulum data
task = PendulumTask(dt=0.05, n_steps=100)
X, y = task.generate_trajectory(theta0=0.5)

# Build and train network
net = LiquidNetwork(input_dim=2, hidden_dims=[16, 8], output_dim=2)
trainer = SimpleTrainer(net, lr=1e-3)
losses = trainer.fit(X, y, epochs=20)

# Predict
outputs = net.predict(X)
```

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

## Running the Demo

```bash
~/anaconda3/bin/python3 examples/pendulum_demo.py
```

## References

- Hasani et al., "Liquid Time-constant Networks" (AAAI 2021)
- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
