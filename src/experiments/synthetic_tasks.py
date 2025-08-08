import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt


class SyntheticTaskGenerator:
    """
    Generator for synthetic tasks to validate liquid neural network capabilities.
    
    Creates various benchmark tasks including:
    - Temporal pattern recognition
    - Memory tasks
    - Chaotic system prediction
    - Adaptive control scenarios
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.key = random.PRNGKey(seed)
    
    def generate_temporal_patterns(
        self,
        n_sequences: int = 1000,
        seq_length: int = 100,
        pattern_length: int = 10,
        n_patterns: int = 5,
        noise_level: float = 0.1
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate temporal pattern recognition task.
        
        Creates sequences with embedded patterns that need to be recognized
        and predicted.
        """
        self.key, subkey = random.split(self.key)
        
        # Create base patterns
        patterns = []
        for i in range(n_patterns):
            pattern = random.normal(subkey, (pattern_length,))
            patterns.append(pattern)
            subkey = random.split(subkey)[0]
        
        sequences = []
        labels = []
        
        for _ in range(n_sequences):
            # Choose random pattern
            pattern_idx = random.randint(subkey, (), 0, n_patterns)
            chosen_pattern = patterns[pattern_idx]
            
            # Generate sequence with embedded pattern
            sequence = random.normal(subkey, (seq_length,)) * noise_level
            
            # Embed pattern at random location
            start_pos = random.randint(subkey, (), 0, seq_length - pattern_length)
            sequence = sequence.at[start_pos:start_pos + pattern_length].set(
                chosen_pattern + random.normal(subkey, (pattern_length,)) * noise_level * 0.1
            )
            
            sequences.append(sequence)
            
            # Label is the pattern index at each time step
            label = jnp.zeros(seq_length, dtype=jnp.int32)
            label = label.at[start_pos:start_pos + pattern_length].set(pattern_idx)
            labels.append(label)
            
            subkey = random.split(subkey)[0]
        
        return jnp.array(sequences), jnp.array(labels)
    
    def generate_memory_task(
        self,
        n_sequences: int = 1000,
        seq_length: int = 200,
        memory_length: int = 50,
        n_symbols: int = 10
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate memory task requiring long-term dependencies.
        
        The model must remember information from early in the sequence
        to make predictions at the end.
        """
        self.key, subkey = random.split(self.key)
        
        sequences = []
        targets = []
        
        for _ in range(n_sequences):
            # Generate sequence
            sequence = random.randint(subkey, (seq_length,), 0, n_symbols)
            
            # Memory cue at beginning
            memory_cue = sequence[:memory_length]
            
            # Target is some function of the memory cue
            target = jnp.sum(memory_cue) % n_symbols
            
            # One-hot encode sequence
            sequence_onehot = jax.nn.one_hot(sequence, n_symbols)
            
            # Target at the end of sequence
            target_sequence = jnp.zeros(seq_length)
            target_sequence = target_sequence.at[-1].set(target)
            
            sequences.append(sequence_onehot)
            targets.append(target_sequence)
            
            subkey = random.split(subkey)[0]
        
        return jnp.array(sequences), jnp.array(targets)
    
    def generate_lorenz_system(
        self,
        n_sequences: int = 100,
        seq_length: int = 500,
        dt: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        noise_level: float = 0.02
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate Lorenz chaotic system trajectories for prediction.
        
        Tests the model's ability to learn and predict chaotic dynamics.
        """
        self.key, subkey = random.split(self.key)
        
        def lorenz_step(state, dt):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return state + dt * jnp.array([dx, dy, dz])
        
        sequences = []
        targets = []
        
        for _ in range(n_sequences):
            # Random initial condition
            initial_state = random.normal(subkey, (3,)) * 5.0
            
            # Generate trajectory
            trajectory = [initial_state]
            current_state = initial_state
            
            for _ in range(seq_length):
                current_state = lorenz_step(current_state, dt)
                trajectory.append(current_state)
            
            trajectory = jnp.array(trajectory)
            
            # Add noise
            trajectory += random.normal(subkey, trajectory.shape) * noise_level
            
            # Input is trajectory, target is next state
            input_seq = trajectory[:-1]
            target_seq = trajectory[1:]
            
            sequences.append(input_seq)
            targets.append(target_seq)
            
            subkey = random.split(subkey)[0]
        
        return jnp.array(sequences), jnp.array(targets)
    
    def generate_adaptive_control_task(
        self,
        n_episodes: int = 500,
        episode_length: int = 200,
        system_changes: bool = True,
        disturbance_level: float = 0.1
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate adaptive control task.
        
        The model must learn to control a system that may change its
        dynamics during operation.
        """
        self.key, subkey = random.split(self.key)
        
        states = []
        actions = []
        targets = []
        
        for episode in range(n_episodes):
            # System parameters (may change during episode)
            if system_changes and episode > n_episodes // 2:
                # Change dynamics halfway through training
                A = jnp.array([[0.8, 0.2], [-0.1, 0.9]])  # Different dynamics
                B = jnp.array([[0.8], [0.6]])
            else:
                # Original dynamics
                A = jnp.array([[0.9, 0.1], [-0.2, 0.8]])
                B = jnp.array([[1.0], [0.5]])
            
            # Target trajectory (sine wave)
            t = jnp.linspace(0, 4*jnp.pi, episode_length)
            target_traj = jnp.column_stack([jnp.sin(t), jnp.cos(t)])
            
            # Initialize state
            state = jnp.array([0.0, 0.0])
            
            episode_states = []
            episode_actions = []
            
            for step in range(episode_length):
                # Simple PD controller with adaptation
                target = target_traj[step]
                error = target - state
                
                # Control action (to be learned by the network)
                control_input = random.normal(subkey, (1,)) * 0.1
                
                # System dynamics: x[k+1] = A*x[k] + B*u[k] + disturbance
                disturbance = random.normal(subkey, (2,)) * disturbance_level
                next_state = A @ state + (B @ control_input).flatten() + disturbance
                
                episode_states.append(jnp.concatenate([state, target, error]))
                episode_actions.append(control_input)
                
                state = next_state
                subkey = random.split(subkey)[0]
            
            states.append(jnp.array(episode_states))
            actions.append(jnp.array(episode_actions))
            targets.append(target_traj)
        
        return jnp.array(states), jnp.array(actions), jnp.array(targets)
    
    def generate_multi_scale_temporal_task(
        self,
        n_sequences: int = 500,
        seq_length: int = 300,
        short_scale: int = 5,
        long_scale: int = 50
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate task with multiple temporal scales.
        
        Tests the model's ability to capture both short-term and long-term
        temporal dependencies simultaneously.
        """
        self.key, subkey = random.split(self.key)
        
        sequences = []
        targets = []
        
        for _ in range(n_sequences):
            # Generate base sequence
            sequence = random.normal(subkey, (seq_length,))
            
            # Add short-scale pattern
            for i in range(0, seq_length - short_scale, short_scale * 2):
                pattern = jnp.sin(jnp.arange(short_scale) * 0.5) * 0.5
                sequence = sequence.at[i:i+short_scale].add(pattern)
            
            # Add long-scale modulation
            long_modulation = jnp.sin(jnp.arange(seq_length) * 2 * jnp.pi / long_scale)
            sequence = sequence * (1.0 + 0.3 * long_modulation)
            
            # Target is prediction of next value
            target = jnp.roll(sequence, -1)
            target = target.at[-1].set(0.0)  # Last target is zero
            
            sequences.append(sequence.reshape(-1, 1))
            targets.append(target.reshape(-1, 1))
            
            subkey = random.split(subkey)[0]
        
        return jnp.array(sequences), jnp.array(targets)
    
    def create_benchmark_suite(self) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create a complete benchmark suite with all tasks."""
        print("Generating benchmark suite...")
        
        suite = {}
        
        # Temporal patterns
        print("  - Generating temporal pattern recognition task...")
        suite['temporal_patterns'] = self.generate_temporal_patterns(
            n_sequences=200, seq_length=50
        )
        
        # Memory task
        print("  - Generating memory task...")
        suite['memory_task'] = self.generate_memory_task(
            n_sequences=300, seq_length=100
        )
        
        # Chaotic dynamics
        print("  - Generating Lorenz system prediction...")
        suite['lorenz_prediction'] = self.generate_lorenz_system(
            n_sequences=50, seq_length=200
        )
        
        # Multi-scale temporal
        print("  - Generating multi-scale temporal task...")
        suite['multi_scale'] = self.generate_multi_scale_temporal_task(
            n_sequences=200, seq_length=150
        )
        
        # Control task
        print("  - Generating adaptive control task...")
        control_data = self.generate_adaptive_control_task(
            n_episodes=100, episode_length=100
        )
        suite['adaptive_control'] = (control_data[0], control_data[2])  # States and targets
        
        print("Benchmark suite generation complete!")
        return suite
    
    def visualize_task(self, task_name: str, inputs: jnp.ndarray, targets: jnp.ndarray, n_samples: int = 3):
        """Visualize examples from a task."""
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(min(n_samples, inputs.shape[0])):
            ax = axes[i]
            
            if inputs[i].ndim == 1:
                # 1D sequences
                ax.plot(inputs[i], label='Input', alpha=0.7)
                ax.plot(targets[i], label='Target', alpha=0.7)
            elif inputs[i].shape[1] <= 3:
                # Multi-dimensional sequences (plot first few dimensions)
                for j in range(min(3, inputs[i].shape[1])):
                    ax.plot(inputs[i][:, j], label=f'Input dim {j}', alpha=0.7)
                    if targets[i].shape[1] > j:
                        ax.plot(targets[i][:, j], '--', label=f'Target dim {j}', alpha=0.7)
            
            ax.set_title(f'{task_name} - Sample {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_task_statistics(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> Dict[str, float]:
        """Compute statistics for a task dataset."""
        stats = {
            'n_sequences': inputs.shape[0],
            'sequence_length': inputs.shape[1],
            'input_dim': inputs.shape[2] if inputs.ndim > 2 else 1,
            'target_dim': targets.shape[2] if targets.ndim > 2 else 1,
            'input_mean': float(jnp.mean(inputs)),
            'input_std': float(jnp.std(inputs)),
            'target_mean': float(jnp.mean(targets)),
            'target_std': float(jnp.std(targets)),
            'input_range': float(jnp.max(inputs) - jnp.min(inputs)),
            'target_range': float(jnp.max(targets) - jnp.min(targets))
        }
        return stats