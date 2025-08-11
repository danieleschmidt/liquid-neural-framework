"""
Comprehensive benchmarking suite for liquid neural network research.

This module provides state-of-the-art benchmarking tools for evaluating
liquid neural networks across multiple dimensions:
1. Computational performance
2. Memory efficiency  
3. Learning dynamics
4. Temporal processing capabilities
5. Statistical significance testing
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp

import time
import json
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    model_name: str
    task_name: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class ComprehensiveBenchmarkSuite:
    """
    Research-grade benchmarking suite with statistical rigor.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.synthetic_tasks = {}
        
        # Initialize synthetic task generators
        self._initialize_synthetic_tasks()
    
    def _initialize_synthetic_tasks(self):
        """Initialize synthetic benchmark tasks."""
        if not HAS_JAX:
            warnings.warn("JAX not available, synthetic tasks limited")
            return
            
        key = random.PRNGKey(42)
        keys = random.split(key, 10)
        
        # Task 1: Temporal Memory - Remember inputs from T steps ago
        def temporal_memory_task(seq_length: int = 100, memory_delay: int = 10):
            inputs = random.normal(keys[0], (seq_length, 1, 5))
            # Target is input from memory_delay steps ago
            targets = jnp.roll(inputs[:, :, 0:1], memory_delay, axis=0)
            targets = targets.at[:memory_delay].set(0)  # Zero out first few steps
            return inputs, targets
        
        # Task 2: Temporal Pattern Recognition
        def pattern_recognition_task(seq_length: int = 200, pattern_length: int = 5):
            # Create sequences with embedded patterns
            inputs = random.normal(keys[1], (seq_length, 1, 3))
            pattern = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]).reshape(-1, 1, 1)
            
            # Randomly insert pattern
            pattern_starts = random.choice(keys[2], jnp.arange(seq_length - pattern_length), (5,))
            targets = jnp.zeros((seq_length, 1, 1))
            
            for start in pattern_starts:
                inputs = inputs.at[start:start+pattern_length, :, 0:1].set(pattern)
                targets = targets.at[start+pattern_length-1].set(1.0)  # Mark end of pattern
                
            return inputs, targets
        
        # Task 3: Nonlinear System Identification
        def nonlinear_system_task(seq_length: int = 150):
            inputs = random.uniform(keys[3], (seq_length, 1, 2), minval=-1, maxval=1)
            # Nonlinear dynamical system: y[t] = tanh(0.5*x[t] + 0.3*x[t-1]*y[t-1])
            targets = jnp.zeros((seq_length, 1, 1))
            
            for t in range(1, seq_length):
                prev_target = targets[t-1, 0, 0] if t > 0 else 0.0
                prev_input = inputs[t-1, 0, 0]
                current_input = inputs[t, 0, 0]
                
                target_val = jnp.tanh(0.5 * current_input + 0.3 * prev_input * prev_target)
                targets = targets.at[t, 0, 0].set(target_val)
                
            return inputs, targets
        
        # Task 4: Multi-Scale Temporal Dependencies
        def multiscale_task(seq_length: int = 300):
            inputs = random.normal(keys[4], (seq_length, 1, 4))
            targets = jnp.zeros((seq_length, 1, 2))
            
            # Fast dependency (delay = 2)
            targets = targets.at[:, :, 0].set(
                jnp.roll(inputs[:, :, 0], 2, axis=0)
            )
            
            # Slow dependency (delay = 20)  
            targets = targets.at[:, :, 1].set(
                jnp.roll(inputs[:, :, 1], 20, axis=0)
            )
            
            return inputs, targets
        
        # Task 5: Chaotic Time Series Prediction (Mackey-Glass)
        def chaotic_prediction_task(seq_length: int = 500, tau: int = 17):
            # Simplified Mackey-Glass equation: x(t+1) = x(t) + dt * (ax(t-τ)/(1+x(t-τ)^10) - bx(t))
            dt, a, b = 0.1, 0.2, 0.1
            
            # Initialize with random values
            x = jnp.zeros(seq_length + tau)
            x = x.at[:tau].set(random.uniform(keys[5], (tau,), minval=0.1, maxval=1.0))
            
            # Generate chaotic series
            for t in range(tau, seq_length + tau - 1):
                x_tau = x[t - tau]
                x_t = x[t]
                dx = a * x_tau / (1 + x_tau**10) - b * x_t
                x = x.at[t + 1].set(x_t + dt * dx)
            
            # Create input-target pairs (predict next value)
            inputs = x[tau:-1].reshape(-1, 1, 1)
            targets = x[tau+1:].reshape(-1, 1, 1)
            
            return inputs, targets
        
        self.synthetic_tasks = {
            'temporal_memory': temporal_memory_task,
            'pattern_recognition': pattern_recognition_task,
            'nonlinear_system': nonlinear_system_task,
            'multiscale_dependencies': multiscale_task,
            'chaotic_prediction': chaotic_prediction_task
        }
    
    def benchmark_model(self, model, model_name: str, num_trials: int = 5) -> List[BenchmarkResult]:
        """
        Comprehensive benchmark of a model across all synthetic tasks.
        """
        model_results = []
        
        for task_name, task_generator in self.synthetic_tasks.items():
            print(f"🔄 Benchmarking {model_name} on {task_name}...")
            
            # Run multiple trials for statistical significance
            task_results = []
            
            for trial in range(num_trials):
                try:
                    # Generate task data
                    inputs, targets = task_generator()
                    
                    # Initialize model state
                    if hasattr(model, 'init_hidden_state'):
                        hidden_state = model.init_hidden_state(1)
                    elif hasattr(model, 'init_state'):
                        hidden_state = model.init_state(1)
                    else:
                        hidden_state = jnp.zeros((1, 32))  # Default hidden size
                    
                    # Timing benchmark
                    start_time = time.time()
                    
                    # Forward pass through sequence
                    predictions = []
                    current_hidden = hidden_state
                    
                    for t in range(inputs.shape[0]):
                        if hasattr(model, '__call__'):
                            if isinstance(model, type(None)):  # Handle different model types
                                pred, current_hidden = model(inputs[t:t+1], current_hidden)
                            else:
                                pred, current_hidden = model(inputs[t:t+1], current_hidden)
                        else:
                            pred = inputs[t:t+1]  # Fallback
                            
                        predictions.append(pred)
                    
                    predictions = jnp.concatenate(predictions, axis=0)
                    processing_time = time.time() - start_time
                    
                    # Compute metrics
                    mse = float(jnp.mean((predictions - targets)**2))
                    mae = float(jnp.mean(jnp.abs(predictions - targets)))
                    
                    # Temporal correlation (measure of temporal understanding)
                    pred_flat = predictions.reshape(-1)
                    target_flat = targets.reshape(-1)
                    
                    if len(pred_flat) > 1:
                        correlation = float(jnp.corrcoef(pred_flat, target_flat)[0, 1])
                        correlation = 0.0 if jnp.isnan(correlation) else correlation
                    else:
                        correlation = 0.0
                    
                    # Memory capacity (for temporal memory task)
                    memory_capacity = 0.0
                    if task_name == 'temporal_memory':
                        # Compute memory capacity as correlation with delayed inputs
                        for delay in range(1, min(20, inputs.shape[0]//2)):
                            delayed_input = jnp.roll(inputs[:, 0, 0], delay)
                            delay_corr = float(jnp.corrcoef(pred_flat, delayed_input)[0, 1])
                            if not jnp.isnan(delay_corr):
                                memory_capacity += abs(delay_corr)
                    
                    # Stability measure (variance of predictions)
                    stability = float(1.0 / (1.0 + jnp.var(predictions)))
                    
                    # Efficiency (samples per second)
                    efficiency = inputs.shape[0] / processing_time
                    
                    metrics = {
                        'mse': mse,
                        'mae': mae,
                        'correlation': correlation,
                        'memory_capacity': memory_capacity,
                        'stability': stability,
                        'processing_time': processing_time,
                        'efficiency': efficiency,
                        'sequence_length': inputs.shape[0]
                    }
                    
                    result = BenchmarkResult(
                        model_name=model_name,
                        task_name=task_name,
                        metrics=metrics,
                        metadata={
                            'trial': trial,
                            'input_shape': inputs.shape,
                            'target_shape': targets.shape
                        },
                        timestamp=time.time(),
                        success=True
                    )
                    
                    task_results.append(result)
                    
                except Exception as e:
                    error_result = BenchmarkResult(
                        model_name=model_name,
                        task_name=task_name,
                        metrics={},
                        metadata={'trial': trial},
                        timestamp=time.time(),
                        success=False,
                        error_message=str(e)
                    )
                    
                    task_results.append(error_result)
                    warnings.warn(f"Trial {trial} failed for {task_name}: {e}")
            
            model_results.extend(task_results)
            
            # Compute summary statistics for this task
            successful_results = [r for r in task_results if r.success]
            if successful_results:
                self._print_task_summary(model_name, task_name, successful_results)
        
        self.results.extend(model_results)
        return model_results
    
    def _print_task_summary(self, model_name: str, task_name: str, results: List[BenchmarkResult]):
        """Print summary statistics for a task."""
        if not results:
            return
            
        print(f"📊 {model_name} - {task_name} Summary:")
        
        # Aggregate metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Print statistics
        for metric_name, values in all_metrics.items():
            if values and not all(jnp.isnan(v) for v in values):
                mean_val = jnp.mean(jnp.array([v for v in values if not jnp.isnan(v)]))
                std_val = jnp.std(jnp.array([v for v in values if not jnp.isnan(v)]))
                print(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")
        
        print()
    
    def compare_models(self, models: Dict[str, Any], num_trials: int = 3) -> Dict[str, Any]:
        """
        Compare multiple models across all benchmark tasks.
        """
        print("🚀 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 60)
        
        comparison_results = {}
        
        # Benchmark each model
        for model_name, model in models.items():
            print(f"\n🔬 Benchmarking {model_name}...")
            model_results = self.benchmark_model(model, model_name, num_trials)
            comparison_results[model_name] = model_results
        
        # Generate comparison report
        report = self._generate_comparison_report(comparison_results)
        
        # Save results
        self._save_results(comparison_results, report)
        
        return report
    
    def _generate_comparison_report(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            'summary': {},
            'task_analysis': {},
            'statistical_significance': {},
            'recommendations': []
        }
        
        # Aggregate results by model and task
        aggregated = {}
        for model_name, model_results in results.items():
            aggregated[model_name] = {}
            
            # Group by task
            for result in model_results:
                if not result.success:
                    continue
                    
                task_name = result.task_name
                if task_name not in aggregated[model_name]:
                    aggregated[model_name][task_name] = []
                    
                aggregated[model_name][task_name].append(result.metrics)
        
        # Compute summary statistics
        for model_name, task_results in aggregated.items():
            model_summary = {}
            
            for task_name, metrics_list in task_results.items():
                if not metrics_list:
                    continue
                    
                task_summary = {}
                
                # Get all metric names
                all_metric_names = set()
                for metrics in metrics_list:
                    all_metric_names.update(metrics.keys())
                
                # Compute statistics for each metric
                for metric_name in all_metric_names:
                    values = [m[metric_name] for m in metrics_list if metric_name in m]
                    values = [v for v in values if not jnp.isnan(v)]
                    
                    if values:
                        task_summary[metric_name] = {
                            'mean': float(jnp.mean(jnp.array(values))),
                            'std': float(jnp.std(jnp.array(values))),
                            'min': float(jnp.min(jnp.array(values))),
                            'max': float(jnp.max(jnp.array(values))),
                            'count': len(values)
                        }
                
                model_summary[task_name] = task_summary
            
            report['summary'][model_name] = model_summary
        
        # Task-specific analysis
        for task_name in self.synthetic_tasks.keys():
            task_analysis = {}
            
            # Find best model for each metric
            for metric_name in ['mse', 'mae', 'correlation', 'memory_capacity', 'efficiency']:
                best_model = None
                best_value = None
                
                for model_name in aggregated.keys():
                    if task_name in aggregated[model_name]:
                        task_data = report['summary'][model_name].get(task_name, {})
                        if metric_name in task_data:
                            value = task_data[metric_name]['mean']
                            
                            # For MSE and MAE, lower is better
                            if metric_name in ['mse', 'mae']:
                                if best_value is None or value < best_value:
                                    best_value = value
                                    best_model = model_name
                            else:
                                # For other metrics, higher is better
                                if best_value is None or value > best_value:
                                    best_value = value
                                    best_model = model_name
                
                if best_model:
                    task_analysis[f'best_{metric_name}'] = {
                        'model': best_model,
                        'value': best_value
                    }
            
            report['task_analysis'][task_name] = task_analysis
        
        # Generate recommendations
        if aggregated:
            # Overall best model (based on multiple criteria)
            model_scores = {}
            for model_name in aggregated.keys():
                score = 0
                
                for task_name, task_analysis in report['task_analysis'].items():
                    for metric_result in task_analysis.values():
                        if metric_result['model'] == model_name:
                            score += 1
                
                model_scores[model_name] = score
            
            if model_scores:
                best_overall = max(model_scores, key=model_scores.get)
                report['recommendations'].append(
                    f"Overall best performing model: {best_overall} "
                    f"(won {model_scores[best_overall]} metric comparisons)"
                )
            
            # Task-specific recommendations
            for task_name, task_analysis in report['task_analysis'].items():
                if 'best_correlation' in task_analysis:
                    best_model = task_analysis['best_correlation']['model']
                    report['recommendations'].append(
                        f"For {task_name}: {best_model} shows best temporal understanding"
                    )
        
        return report
    
    def _save_results(self, results: Dict[str, List[BenchmarkResult]], report: Dict[str, Any]):
        """Save benchmark results and report to files."""
        timestamp = int(time.time())
        
        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = []
            for result in model_results:
                serializable_results[model_name].append({
                    'model_name': result.model_name,
                    'task_name': result.task_name,
                    'metrics': result.metrics,
                    'metadata': result.metadata,
                    'timestamp': result.timestamp,
                    'success': result.success,
                    'error_message': result.error_message
                })
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")
        print(f"📊 Report saved to {report_file}")


class PerformanceAnalyzer:
    """Advanced performance analysis tools."""
    
    @staticmethod
    def memory_profile_model(model, input_sequence: jnp.ndarray) -> Dict[str, float]:
        """Profile memory usage of model."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run model
        if hasattr(model, 'init_hidden_state'):
            hidden = model.init_hidden_state(1)
        else:
            hidden = jnp.zeros((1, 32))
        
        for t in range(input_sequence.shape[0]):
            if hasattr(model, '__call__'):
                _, hidden = model(input_sequence[t:t+1], hidden)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_usage_mb': peak_memory - baseline_memory,
            'memory_per_timestep_kb': (peak_memory - baseline_memory) * 1024 / input_sequence.shape[0]
        }
    
    @staticmethod
    def gradient_flow_analysis(model, inputs: jnp.ndarray, targets: jnp.ndarray) -> Dict[str, float]:
        """Analyze gradient flow characteristics."""
        if not HAS_JAX:
            return {'error': 'JAX required for gradient analysis'}
        
        try:
            def loss_fn(model_params, inputs, targets):
                # Simplified loss computation
                if hasattr(model, 'init_hidden_state'):
                    hidden = model.init_hidden_state(1)
                else:
                    hidden = jnp.zeros((1, 32))
                    
                total_loss = 0.0
                for t in range(inputs.shape[0]):
                    pred, hidden = model(inputs[t:t+1], hidden)
                    total_loss += jnp.mean((pred - targets[t:t+1])**2)
                    
                return total_loss / inputs.shape[0]
            
            # Compute gradients
            grad_fn = grad(loss_fn)
            gradients = grad_fn(model, inputs, targets)
            
            # Analyze gradient properties
            if hasattr(gradients, 'W_rec'):
                grad_norm = float(jnp.linalg.norm(gradients.W_rec))
                grad_mean = float(jnp.mean(jnp.abs(gradients.W_rec)))
                grad_std = float(jnp.std(gradients.W_rec))
            else:
                grad_norm = grad_mean = grad_std = 0.0
            
            return {
                'gradient_norm': grad_norm,
                'gradient_mean': grad_mean,
                'gradient_std': grad_std,
                'gradient_vanishing_score': 1.0 / (1.0 + grad_norm)  # Higher means more vanishing
            }
            
        except Exception as e:
            return {'error': f'Gradient analysis failed: {e}'}


class StatisticalValidator:
    """Statistical significance testing for benchmark results."""
    
    @staticmethod
    def paired_t_test(results1: List[float], results2: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Perform paired t-test between two result sets."""
        if len(results1) != len(results2) or len(results1) < 2:
            return {'error': 'Insufficient or mismatched data for t-test'}
        
        try:
            # Compute differences
            differences = jnp.array(results1) - jnp.array(results2)
            
            # t-statistic
            mean_diff = jnp.mean(differences)
            std_diff = jnp.std(differences, ddof=1)
            n = len(differences)
            
            t_stat = mean_diff / (std_diff / jnp.sqrt(n))
            
            # Degrees of freedom
            df = n - 1
            
            # Critical value (approximation for large samples)
            t_critical = 1.96 if n > 30 else 2.576  # Rough approximation
            
            # P-value approximation (simplified)
            p_value = 2 * (1 - 0.95) if abs(t_stat) > t_critical else 2 * 0.95
            
            significant = p_value < alpha
            
            return {
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                't_statistic': float(t_stat),
                'degrees_freedom': df,
                'p_value': p_value,
                'significant': significant,
                'alpha': alpha
            }
            
        except Exception as e:
            return {'error': f'Statistical test failed: {e}'}
    
    @staticmethod
    def effect_size_cohen_d(results1: List[float], results2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        mean1, mean2 = jnp.mean(jnp.array(results1)), jnp.mean(jnp.array(results2))
        std1, std2 = jnp.std(jnp.array(results1), ddof=1), jnp.std(jnp.array(results2), ddof=1)
        
        pooled_std = jnp.sqrt(((len(results1) - 1) * std1**2 + (len(results2) - 1) * std2**2) / 
                              (len(results1) + len(results2) - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return float(cohens_d)