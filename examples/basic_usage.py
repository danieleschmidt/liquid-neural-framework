"""
Basic usage examples for the liquid neural framework.

This script demonstrates how to:
1. Create and configure liquid neural networks
2. Generate synthetic data for training
3. Train the networks
4. Evaluate performance
5. Visualize results
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.liquid_neural_network import LiquidNeuralNetwork
from models.continuous_time_rnn import ContinuousTimeRNN
from algorithms.training import LiquidNetworkTrainer
from experiments.synthetic_tasks import SyntheticTaskGenerator


def basic_liquid_network_example():
    """Basic example of creating and using a liquid neural network."""
    print("=== Basic Liquid Neural Network Example ===")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Create a liquid neural network
    model = LiquidNeuralNetwork(
        input_size=1,
        hidden_size=20,
        output_size=1,
        time_constant_init=1.0,
        leak_rate=0.1,
        key=key
    )
    
    # Generate simple test data (sine wave prediction)
    t = jnp.linspace(0, 4*jnp.pi, 200)
    inputs = jnp.sin(t).reshape(-1, 1)
    targets = jnp.sin(t + 0.1).reshape(-1, 1)  # Slightly phase-shifted
    
    print(f"Created model with {sum(p.size for p in model.params.values())} parameters")
    print(f"Input data shape: {inputs.shape}")
    print(f"Target data shape: {targets.shape}")
    
    # Forward pass
    outputs, states = model.forward(inputs)
    initial_loss = jnp.mean((outputs - targets) ** 2)
    print(f"Initial loss (before training): {initial_loss:.6f}")
    
    # Create trainer and train
    trainer = LiquidNetworkTrainer(
        model=model,
        learning_rate=1e-3,
        optimizer_name='adam',
        loss_fn='mse'
    )
    
    print("\\nTraining for 50 epochs...")
    history = trainer.fit(
        train_data=(inputs, targets),
        epochs=50,
        verbose=False
    )
    
    # Evaluate after training
    final_outputs, _ = model.forward(inputs)
    final_loss = jnp.mean((final_outputs - targets) ** 2)
    print(f"Final loss (after training): {final_loss:.6f}")
    print(f"Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    
    # Show time constants
    time_constants = model.get_time_constants()
    print(f"Time constants - Mean: {jnp.mean(time_constants):.3f}, Std: {jnp.std(time_constants):.3f}")
    
    return model, history


def continuous_time_rnn_example():
    """Example of using continuous-time RNN."""
    print("\\n=== Continuous-Time RNN Example ===")
    
    key = random.PRNGKey(123)
    
    # Create continuous-time RNN
    model = ContinuousTimeRNN(
        input_size=2,
        hidden_size=16,
        output_size=2,
        activation='tanh',
        solver='rk4',  # Use RK4 for better accuracy
        key=key
    )
    
    # Generate chaotic data (simple 2D system)
    n_steps = 150
    dt = 0.05
    
    # Initial state
    state = jnp.array([1.0, 0.5])
    trajectory = [state]
    
    # Simple nonlinear dynamics for test
    for _ in range(n_steps - 1):
        # dx/dt = -0.1*x + 0.5*y + input
        # dy/dt = -0.2*y - 0.3*x*y + input
        input_val = jnp.array([0.1 * jnp.sin(len(trajectory) * 0.1), 
                              0.1 * jnp.cos(len(trajectory) * 0.1)])
        
        x, y = state
        dx = -0.1 * x + 0.5 * y + input_val[0]
        dy = -0.2 * y - 0.3 * x * y + input_val[1]
        state = state + dt * jnp.array([dx, dy])
        trajectory.append(state)
    
    trajectory = jnp.array(trajectory)
    
    # Prepare training data
    inputs = trajectory[:-1]  # All but last
    targets = trajectory[1:]  # All but first
    
    print(f"Generated trajectory with {len(trajectory)} points")
    print(f"Training on {len(inputs)} input-target pairs")
    
    # Train the model
    trainer = LiquidNetworkTrainer(
        model=model,
        learning_rate=5e-4,
        optimizer_name='adam',
        loss_fn='mse'
    )
    
    print("Training continuous-time RNN for 100 epochs...")
    history = trainer.fit(
        train_data=(inputs, targets),
        epochs=100,
        dt=0.02,  # Integration time step
        verbose=False
    )
    
    # Test prediction
    test_outputs, _ = model.forward(inputs, dt=0.02)
    final_loss = jnp.mean((test_outputs - targets) ** 2)
    print(f"Final prediction loss: {final_loss:.6f}")
    
    # Show dynamics info
    dynamics_info = model.get_dynamics_info()
    print(f"Dynamics - Alpha mean: {dynamics_info['alpha_mean']:.3f}, Solver: {dynamics_info['solver']}")
    
    return model, trajectory


def benchmark_tasks_example():
    """Example of using synthetic benchmark tasks."""
    print("\\n=== Benchmark Tasks Example ===")
    
    # Create task generator
    task_gen = SyntheticTaskGenerator(seed=456)
    
    # Generate temporal pattern recognition task
    print("Generating temporal pattern recognition task...")
    inputs, labels = task_gen.generate_temporal_patterns(
        n_sequences=50,
        seq_length=40,
        pattern_length=8,
        n_patterns=3,
        noise_level=0.1
    )
    
    print(f"Generated {inputs.shape[0]} sequences of length {inputs.shape[1]}")
    print(f"Patterns embedded in sequences with {jnp.max(labels)} different pattern types")
    
    # Show task statistics
    stats = task_gen.get_task_statistics(inputs.reshape(inputs.shape[0], inputs.shape[1], 1), 
                                       labels.reshape(labels.shape[0], labels.shape[1], 1))
    print(f"Input statistics - Mean: {stats['input_mean']:.3f}, Std: {stats['input_std']:.3f}")
    
    # Generate memory task
    print("\\nGenerating memory task...")
    mem_inputs, mem_targets = task_gen.generate_memory_task(
        n_sequences=30,
        seq_length=80,
        memory_length=15,
        n_symbols=5
    )
    
    print(f"Memory task: {mem_inputs.shape[0]} sequences, each with {mem_inputs.shape[2]} input symbols")
    print(f"Memory length: 15 steps, total sequence length: {mem_inputs.shape[1]}")
    
    # Generate Lorenz system
    print("\\nGenerating Lorenz chaotic system...")
    lorenz_inputs, lorenz_targets = task_gen.generate_lorenz_system(
        n_sequences=10,
        seq_length=200,
        noise_level=0.01
    )
    
    print(f"Lorenz system: {lorenz_inputs.shape[0]} trajectories of {lorenz_inputs.shape[1]} time steps")
    print(f"Each state has {lorenz_inputs.shape[2]} dimensions (x, y, z)")
    
    return {
        'temporal_patterns': (inputs, labels),
        'memory_task': (mem_inputs, mem_targets),
        'lorenz_system': (lorenz_inputs, lorenz_targets)
    }


def model_comparison_example():
    """Compare different model architectures on a simple task."""
    print("\\n=== Model Comparison Example ===")
    
    key = random.PRNGKey(789)
    
    # Generate test data (multi-scale temporal pattern)
    task_gen = SyntheticTaskGenerator(seed=789)
    inputs, targets = task_gen.generate_multi_scale_temporal_task(
        n_sequences=20,
        seq_length=100
    )
    
    # Use first sequence for comparison
    test_inputs = inputs[0]
    test_targets = targets[0]
    
    print(f"Test data: sequence length = {test_inputs.shape[0]}")
    
    # Create different models
    models = {}
    
    # Liquid Neural Network
    key, subkey = random.split(key)
    models['Liquid NN'] = LiquidNeuralNetwork(
        input_size=1, hidden_size=20, output_size=1,
        time_constant_init=1.0, key=subkey
    )
    
    # Continuous-Time RNN with Euler solver
    key, subkey = random.split(key)
    models['CT-RNN (Euler)'] = ContinuousTimeRNN(
        input_size=1, hidden_size=20, output_size=1,
        solver='euler', key=subkey
    )
    
    # Continuous-Time RNN with RK4 solver
    key, subkey = random.split(key)
    models['CT-RNN (RK4)'] = ContinuousTimeRNN(
        input_size=1, hidden_size=20, output_size=1,
        solver='rk4', key=subkey
    )
    
    # Train and compare models
    results = {}
    
    for model_name, model in models.items():
        print(f"\\nTraining {model_name}...")
        
        trainer = LiquidNetworkTrainer(
            model=model,
            learning_rate=1e-3,
            optimizer_name='adam'
        )
        
        # Quick training (30 epochs)
        history = trainer.fit(
            train_data=(test_inputs, test_targets),
            epochs=30,
            verbose=False
        )
        
        # Evaluate
        predictions, _ = model.forward(test_inputs)
        final_loss = float(jnp.mean((predictions - test_targets) ** 2))
        
        results[model_name] = {
            'final_loss': final_loss,
            'training_history': history,
            'n_params': sum(p.size for p in model.params.values())
        }
        
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Parameters: {results[model_name]['n_params']}")
    
    # Show comparison
    print("\\n--- Model Comparison Results ---")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['final_loss'])
    
    for i, (model_name, result) in enumerate(sorted_models):
        rank = i + 1
        print(f"{rank}. {model_name}: Loss = {result['final_loss']:.6f}, Params = {result['n_params']}")
    
    return results


def main():
    """Run all examples."""
    print("Liquid Neural Framework - Basic Usage Examples")
    print("=" * 50)
    
    # Basic liquid network
    liquid_model, liquid_history = basic_liquid_network_example()
    
    # Continuous-time RNN
    ct_rnn_model, trajectory = continuous_time_rnn_example()
    
    # Benchmark tasks
    benchmark_data = benchmark_tasks_example()
    
    # Model comparison
    comparison_results = model_comparison_example()
    
    print("\\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\\nNext steps:")
    print("- Try the benchmark suite for comprehensive evaluation")
    print("- Experiment with different hyperparameters")
    print("- Apply to your own time series data")
    print("- Explore the validation experiments for research-grade analysis")


if __name__ == "__main__":
    main()