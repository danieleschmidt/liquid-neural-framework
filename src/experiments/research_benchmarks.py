"""
Research-grade benchmarking suite for liquid neural networks.

This module provides comprehensive benchmarks for evaluating liquid neural networks
against state-of-the-art baselines on standard tasks and novel challenges.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit
    import equinox as eqx
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp

from typing import Dict, Any, Tuple, Optional, List, Callable
import time
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import our models
from ..models.liquid_neural_network import LiquidNeuralNetwork
from ..models.continuous_time_rnn import ContinuousTimeRNN, MultiScaleCTRNN
from ..research.novel_algorithms import MetaAdaptiveLiquidNetwork, MultiScaleTemporalNetwork
from .statistical_validation import ExperimentResult, ReproducibilityValidator


@dataclass
class BenchmarkTask:
    """Definition of a benchmarking task."""
    name: str
    description: str
    input_dimension: int
    output_dimension: int
    sequence_length: int
    difficulty_level: str  # 'easy', 'medium', 'hard'
    evaluation_metrics: List[str]
    data_generator: Callable
    
    
class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""
    
    def __init__(
        self,
        task: BenchmarkTask,
        num_trials: int = 10,
        random_seed: int = 42
    ):
        self.task = task
        self.num_trials = num_trials
        self.random_seed = random_seed
        
    @abstractmethod
    def generate_data(self, num_samples: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate benchmark data."""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate a model on the benchmark task."""
        pass


class MemoryCapacityBenchmark(BaseBenchmark):
    """
    Memory capacity benchmark for testing short-term memory capabilities.
    
    Tests how well models can remember and recall information over varying delays.
    """
    
    def __init__(self, max_delay: int = 20, **kwargs):
        task = BenchmarkTask(
            name="Memory Capacity",
            description="Tests short-term memory capacity with delayed recall tasks",
            input_dimension=1,
            output_dimension=1,
            sequence_length=100,
            difficulty_level='medium',
            evaluation_metrics=['memory_capacity', 'recall_accuracy', 'delay_robustness'],
            data_generator=self.generate_data
        )
        super().__init__(task, **kwargs)
        self.max_delay = max_delay
    
    def generate_data(self, num_samples: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate memory capacity task data."""
        keys = random.split(key, num_samples)
        
        sequences = []
        targets = []
        
        for i in range(num_samples):
            # Generate random input sequence
            input_seq = random.uniform(keys[i], (self.task.sequence_length, 1), minval=-1, maxval=1)
            
            # Create target: delayed version of input
            delay = random.randint(keys[i], (), 1, self.max_delay + 1)
            target_seq = jnp.concatenate([
                jnp.zeros((delay, 1)),
                input_seq[:-delay]
            ])
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return jnp.array(sequences), jnp.array(targets)
    
    def evaluate_model(self, model: Any, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate memory capacity performance."""
        batch_size, seq_len, input_dim = X.shape
        
        # Initialize model state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        predictions = []
        
        # Run model through sequences
        for t in range(seq_len):
            if hasattr(model, '__call__'):
                output, hidden_state = model(X[:, t], hidden_state)
            else:
                # Fallback for different model interfaces
                output = model.forward(X[:, t], hidden_state)
                
            predictions.append(output)
        
        predictions = jnp.array(predictions).transpose(1, 0, 2)  # [batch, time, dim]
        
        # Compute metrics
        mse = float(jnp.mean((predictions - y) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - y)))
        
        # Memory capacity metric (correlation with delayed input)
        correlations = []
        for delay in range(1, min(self.max_delay + 1, seq_len // 2)):
            pred_delayed = predictions[:, delay:, :]
            true_delayed = X[:, :-delay, :]
            
            # Compute correlation coefficient
            pred_flat = pred_delayed.flatten()
            true_flat = true_delayed.flatten()
            
            corr = jnp.corrcoef(pred_flat, true_flat)[0, 1]
            if not jnp.isnan(corr):
                correlations.append(float(corr))
        
        memory_capacity = float(jnp.mean(jnp.array(correlations))) if correlations else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'memory_capacity': max(0.0, memory_capacity),
            'recall_accuracy': float(1.0 / (1.0 + mse)),  # Transformed accuracy
            'delay_robustness': float(jnp.std(jnp.array(correlations))) if len(correlations) > 1 else 0.0
        }


class NonlinearSystemIdentificationBenchmark(BaseBenchmark):
    """
    Nonlinear system identification benchmark.
    
    Tests ability to identify and model nonlinear dynamical systems.
    """
    
    def __init__(self, system_type: str = 'lorenz', **kwargs):
        task = BenchmarkTask(
            name="Nonlinear System ID",
            description=f"Identify {system_type} nonlinear dynamical system",
            input_dimension=3,
            output_dimension=3,
            sequence_length=200,
            difficulty_level='hard',
            evaluation_metrics=['prediction_accuracy', 'lyapunov_error', 'phase_space_fidelity'],
            data_generator=self.generate_data
        )
        super().__init__(task, **kwargs)
        self.system_type = system_type
    
    def lorenz_system(self, state: jnp.ndarray, dt: float = 0.01) -> jnp.ndarray:
        """Lorenz attractor dynamics."""
        x, y, z = state[0], state[1], state[2]
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        return state + dt * jnp.array([dx, dy, z])
    
    def rossler_system(self, state: jnp.ndarray, dt: float = 0.01) -> jnp.ndarray:
        """Rössler attractor dynamics."""
        x, y, z = state[0], state[1], state[2]
        a, b, c = 0.2, 0.2, 5.7
        
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        
        return state + dt * jnp.array([dx, dy, dz])
    
    def generate_data(self, num_samples: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate nonlinear system identification data."""
        keys = random.split(key, num_samples)
        
        sequences = []
        targets = []
        
        system_func = self.lorenz_system if self.system_type == 'lorenz' else self.rossler_system
        
        for i in range(num_samples):
            # Random initial condition
            initial_state = random.uniform(keys[i], (3,), minval=-1, maxval=1)
            
            # Generate trajectory
            trajectory = [initial_state]
            current_state = initial_state
            
            for _ in range(self.task.sequence_length):
                next_state = system_func(current_state)
                trajectory.append(next_state)
                current_state = next_state
            
            trajectory = jnp.array(trajectory)
            
            # Input is current state, target is next state
            input_seq = trajectory[:-1]
            target_seq = trajectory[1:]
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return jnp.array(sequences), jnp.array(targets)
    
    def evaluate_model(self, model: Any, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate system identification performance."""
        batch_size, seq_len, input_dim = X.shape
        
        # Initialize model state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        predictions = []
        
        # Single-step predictions
        for t in range(seq_len):
            if hasattr(model, '__call__'):
                output, hidden_state = model(X[:, t], hidden_state)
            else:
                output = model.forward(X[:, t], hidden_state)
            predictions.append(output)
        
        predictions = jnp.array(predictions).transpose(1, 0, 2)
        
        # Compute metrics
        mse = float(jnp.mean((predictions - y) ** 2))
        
        # Prediction accuracy
        prediction_accuracy = float(1.0 / (1.0 + mse))
        
        # Phase space fidelity (simplified)
        # Compare attractor shape in phase space
        pred_flat = predictions.reshape(-1, input_dim)
        true_flat = y.reshape(-1, input_dim)
        
        # Compute correlation in each dimension
        dim_correlations = []
        for d in range(input_dim):
            corr = jnp.corrcoef(pred_flat[:, d], true_flat[:, d])[0, 1]
            if not jnp.isnan(corr):
                dim_correlations.append(float(corr))
        
        phase_space_fidelity = float(jnp.mean(jnp.array(dim_correlations))) if dim_correlations else 0.0
        
        # Approximate Lyapunov exponent error (simplified)
        # Compare trajectory divergence rates
        trajectory_diff = jnp.linalg.norm(predictions - y, axis=-1)
        lyapunov_error = float(jnp.mean(trajectory_diff))
        
        return {
            'mse': mse,
            'prediction_accuracy': max(0.0, prediction_accuracy),
            'phase_space_fidelity': max(0.0, phase_space_fidelity),
            'lyapunov_error': lyapunov_error
        }


class AdaptationBenchmark(BaseBenchmark):
    """
    Benchmark for testing adaptation capabilities.
    
    Tests how quickly and effectively models can adapt to changing dynamics.
    """
    
    def __init__(self, num_regime_changes: int = 3, **kwargs):
        task = BenchmarkTask(
            name="Adaptation Benchmark",
            description="Tests adaptation to changing task dynamics",
            input_dimension=2,
            output_dimension=1,
            sequence_length=300,
            difficulty_level='hard',
            evaluation_metrics=['adaptation_speed', 'final_performance', 'forgetting_resistance'],
            data_generator=self.generate_data
        )
        super().__init__(task, **kwargs)
        self.num_regime_changes = num_regime_changes
    
    def generate_data(self, num_samples: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate adaptation benchmark data with changing dynamics."""
        keys = random.split(key, num_samples)
        
        sequences = []
        targets = []
        
        for i in range(num_samples):
            sequence = []
            target = []
            
            # Generate sequence with regime changes
            regime_length = self.task.sequence_length // (self.num_regime_changes + 1)
            
            for regime in range(self.num_regime_changes + 1):
                # Different dynamics for each regime
                regime_key = random.fold_in(keys[i], regime)
                
                # Generate regime-specific parameters
                A = random.normal(regime_key, (2, 2)) * 0.5
                b = random.normal(regime_key, (2,)) * 0.1
                
                for t in range(regime_length):
                    if t == 0 and regime == 0:
                        # Initial input
                        x = random.uniform(regime_key, (2,), minval=-1, maxval=1)
                    else:
                        # Evolve according to current regime dynamics
                        x = jnp.dot(A, x) + b + 0.1 * random.normal(regime_key, (2,))
                        x = jnp.tanh(x)  # Keep bounded
                    
                    # Target is a nonlinear function of input
                    y = jnp.tanh(x[0] * x[1] + regime * 0.5)  # Regime-dependent target
                    
                    sequence.append(x)
                    target.append(y)
            
            sequences.append(jnp.array(sequence))
            targets.append(jnp.array(target).reshape(-1, 1))
        
        return jnp.array(sequences), jnp.array(targets)
    
    def evaluate_model(self, model: Any, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate adaptation performance."""
        batch_size, seq_len, input_dim = X.shape
        
        # Initialize model state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        predictions = []
        errors = []
        
        for t in range(seq_len):
            if hasattr(model, '__call__'):
                output, hidden_state = model(X[:, t], hidden_state)
            else:
                output = model.forward(X[:, t], hidden_state)
            
            predictions.append(output)
            error = jnp.mean((output - y[:, t]) ** 2)
            errors.append(float(error))
        
        predictions = jnp.array(predictions).transpose(1, 0, 2)
        errors = jnp.array(errors)
        
        # Analyze adaptation at regime boundaries
        regime_length = seq_len // (self.num_regime_changes + 1)
        adaptation_speeds = []
        
        for regime in range(1, self.num_regime_changes + 1):
            regime_start = regime * regime_length
            regime_end = min(regime_start + regime_length // 2, seq_len)
            
            # Measure error reduction after regime change
            initial_error = errors[regime_start]
            adapted_error = jnp.mean(errors[regime_start:regime_end])
            
            if initial_error > 0:
                adaptation_speed = float(1.0 - adapted_error / (initial_error + 1e-8))
                adaptation_speeds.append(max(0.0, adaptation_speed))
        
        avg_adaptation_speed = float(jnp.mean(jnp.array(adaptation_speeds))) if adaptation_speeds else 0.0
        
        # Final performance (last 20% of sequence)
        final_portion = int(0.8 * seq_len)
        final_performance = float(1.0 / (1.0 + jnp.mean(errors[final_portion:])))
        
        # Forgetting resistance (variance in performance across regimes)
        regime_performances = []
        for regime in range(self.num_regime_changes + 1):
            regime_start = regime * regime_length
            regime_end = min((regime + 1) * regime_length, seq_len)
            regime_error = jnp.mean(errors[regime_start:regime_end])
            regime_performances.append(float(regime_error))
        
        forgetting_resistance = float(1.0 / (1.0 + jnp.std(jnp.array(regime_performances))))
        
        return {
            'adaptation_speed': avg_adaptation_speed,
            'final_performance': final_performance,
            'forgetting_resistance': forgetting_resistance,
            'overall_mse': float(jnp.mean(errors))
        }


class ComprehensiveBenchmarkSuite:
    """
    Comprehensive benchmarking suite for liquid neural networks.
    
    Runs multiple benchmarks and provides comparative analysis.
    """
    
    def __init__(
        self,
        benchmarks: Optional[List[BaseBenchmark]] = None,
        models_to_test: Optional[Dict[str, Any]] = None,
        num_trials: int = 5,
        random_seed: int = 42
    ):
        self.num_trials = num_trials
        self.random_seed = random_seed
        
        # Default benchmarks
        if benchmarks is None:
            self.benchmarks = [
                MemoryCapacityBenchmark(max_delay=15, num_trials=num_trials, random_seed=random_seed),
                NonlinearSystemIdentificationBenchmark(system_type='lorenz', num_trials=num_trials, random_seed=random_seed),
                AdaptationBenchmark(num_regime_changes=2, num_trials=num_trials, random_seed=random_seed)
            ]
        else:
            self.benchmarks = benchmarks
        
        # Default models to test
        if models_to_test is None:
            self.models_to_test = self._get_default_models()
        else:
            self.models_to_test = models_to_test
        
        self.results = {}
        
    def _get_default_models(self) -> Dict[str, Callable]:
        """Get default set of models to benchmark."""
        def create_liquid_network(input_size, hidden_size, output_size, key):
            return LiquidNeuralNetwork(input_size, hidden_size, output_size, key=key)
        
        def create_ctrnn(input_size, hidden_size, output_size, key):
            return ContinuousTimeRNN(input_size, hidden_size, output_size, key=key)
        
        def create_multiscale_ctrnn(input_size, hidden_size, output_size, key):
            return MultiScaleCTRNN(input_size, hidden_size, output_size, key=key)
        
        def create_meta_adaptive(input_size, hidden_size, output_size, key):
            return MetaAdaptiveLiquidNetwork(input_size, hidden_size, output_size, key=key)
        
        return {
            'LiquidNeuralNetwork': create_liquid_network,
            'ContinuousTimeRNN': create_ctrnn,
            'MultiScaleCTRNN': create_multiscale_ctrnn,
            'MetaAdaptiveLiquidNetwork': create_meta_adaptive
        }
    
    def run_benchmark(
        self, 
        benchmark: BaseBenchmark, 
        model_name: str, 
        model_factory: Callable
    ) -> List[ExperimentResult]:
        """Run a single benchmark on a single model."""
        
        key = random.PRNGKey(self.random_seed)
        results = []
        
        for trial in range(self.num_trials):
            trial_key = random.fold_in(key, trial)
            keys = random.split(trial_key, 3)
            
            # Create model
            try:
                model = model_factory(
                    benchmark.task.input_dimension,
                    32,  # hidden size
                    benchmark.task.output_dimension,
                    keys[0]
                )
            except Exception as e:
                warnings.warn(f"Failed to create model {model_name}: {str(e)}")
                continue
            
            # Generate data
            try:
                X, y = benchmark.generate_data(10, keys[1])  # Small batch for testing
            except Exception as e:
                warnings.warn(f"Failed to generate data for {benchmark.task.name}: {str(e)}")
                continue
            
            # Run evaluation
            start_time = time.time()
            try:
                metrics = benchmark.evaluate_model(model, X, y)
                execution_time = time.time() - start_time
                
                result = ExperimentResult(
                    method_name=model_name,
                    performance_metrics=metrics,
                    execution_time=execution_time,
                    memory_usage=None,
                    hyperparameters={'hidden_size': 32, 'trial': trial},
                    random_seed=trial,
                    timestamp=time.time()
                )
                results.append(result)
                
            except Exception as e:
                warnings.warn(f"Evaluation failed for {model_name} on {benchmark.task.name}: {str(e)}")
                continue
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, List[ExperimentResult]]]:
        """Run all benchmarks on all models."""
        
        all_results = {}
        
        for benchmark in self.benchmarks:
            benchmark_results = {}
            
            print(f"Running benchmark: {benchmark.task.name}")
            
            for model_name, model_factory in self.models_to_test.items():
                print(f"  Testing {model_name}...")
                
                model_results = self.run_benchmark(benchmark, model_name, model_factory)
                benchmark_results[model_name] = model_results
            
            all_results[benchmark.task.name] = benchmark_results
        
        self.results = all_results
        return all_results
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        if not self.results:
            return {'error': 'No benchmark results available. Run benchmarks first.'}
        
        report = {
            'summary': {},
            'detailed_results': self.results,
            'statistical_analysis': {},
            'rankings': {}
        }
        
        # Generate summary statistics for each benchmark
        for benchmark_name, benchmark_results in self.results.items():
            benchmark_summary = {}
            
            for model_name, model_results in benchmark_results.items():
                if model_results:
                    # Get all metric names from first result
                    metric_names = list(model_results[0].performance_metrics.keys())
                    model_summary = {}
                    
                    for metric_name in metric_names:
                        values = [r.performance_metrics[metric_name] for r in model_results]
                        model_summary[metric_name] = {
                            'mean': float(jnp.mean(jnp.array(values))),
                            'std': float(jnp.std(jnp.array(values))),
                            'min': float(jnp.min(jnp.array(values))),
                            'max': float(jnp.max(jnp.array(values)))
                        }
                    
                    # Add execution time statistics
                    exec_times = [r.execution_time for r in model_results]
                    model_summary['execution_time'] = {
                        'mean': float(jnp.mean(jnp.array(exec_times))),
                        'std': float(jnp.std(jnp.array(exec_times)))
                    }
                    
                    benchmark_summary[model_name] = model_summary
            
            report['summary'][benchmark_name] = benchmark_summary
        
        # Generate rankings for each benchmark
        for benchmark_name, benchmark_results in self.results.items():
            if benchmark_results:
                # Get primary metric (assume first metric is primary)
                first_model_results = list(benchmark_results.values())[0]
                if first_model_results:
                    primary_metric = list(first_model_results[0].performance_metrics.keys())[0]
                    
                    model_scores = {}
                    for model_name, model_results in benchmark_results.items():
                        if model_results:
                            values = [r.performance_metrics[primary_metric] for r in model_results]
                            model_scores[model_name] = float(jnp.mean(jnp.array(values)))
                    
                    # Rank by score (assume higher is better for most metrics)
                    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                    report['rankings'][benchmark_name] = ranked_models
        
        return report