"""
Simplified benchmarks that work without external dependencies.

This module provides basic benchmarking functionality using only Python built-ins
and can serve as a fallback when JAX/NumPy are not available.
"""

import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


@dataclass 
class SimpleExperimentResult:
    """Simplified experiment result without external dependencies."""
    method_name: str
    performance_metrics: Dict[str, float] 
    execution_time: float
    random_seed: int
    timestamp: float


class SimpleStatistics:
    """Basic statistical functions without external dependencies."""
    
    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean_val = SimpleStatistics.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    @staticmethod
    def min_max(values: List[float]) -> Tuple[float, float]:
        return min(values), max(values) if values else (0.0, 0.0)
    
    @staticmethod
    def coefficient_of_variation(values: List[float]) -> float:
        mean_val = SimpleStatistics.mean(values)
        std_val = SimpleStatistics.std(values)
        return std_val / abs(mean_val) if abs(mean_val) > 1e-8 else float('inf')
    
    @staticmethod
    def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
        """Approximate 95% confidence interval using t-distribution."""
        if len(values) < 2:
            return 0.0, 0.0
        
        mean_val = SimpleStatistics.mean(values)
        std_val = SimpleStatistics.std(values)
        n = len(values)
        
        # Approximate t-critical value for 95% CI
        t_critical_approx = 2.0 if n > 30 else (2.5 if n > 10 else 3.0)
        
        margin_of_error = t_critical_approx * std_val / math.sqrt(n)
        return mean_val - margin_of_error, mean_val + margin_of_error


class SimpleReproducibilityValidator:
    """Simplified reproducibility validator."""
    
    def __init__(self, num_runs: int = 10, significance_level: float = 0.05):
        self.num_runs = num_runs
        self.significance_level = significance_level
    
    def compute_reproducibility_metrics(
        self, 
        results: List[SimpleExperimentResult]
    ) -> Dict[str, Any]:
        """Compute reproducibility metrics."""
        
        if not results:
            raise ValueError("No results provided")
        
        # Extract metrics by name
        metric_names = list(results[0].performance_metrics.keys())
        reproducibility_stats = {}
        
        for metric_name in metric_names:
            values = [r.performance_metrics[metric_name] for r in results]
            
            mean_val = SimpleStatistics.mean(values)
            std_val = SimpleStatistics.std(values)
            cv = SimpleStatistics.coefficient_of_variation(values)
            min_val, max_val = SimpleStatistics.min_max(values)
            ci_lower, ci_upper = SimpleStatistics.confidence_interval_95(values)
            
            reproducibility_stats[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'coefficient_of_variation': cv,
                'min': min_val,
                'max': max_val,
                'confidence_interval_95': (ci_lower, ci_upper),
                'num_runs': len(values)
            }
        
        # Overall reproducibility score (lower CV = more reproducible)
        cvs = [stats['coefficient_of_variation'] for stats in reproducibility_stats.values()]
        overall_cv = SimpleStatistics.mean(cvs)
        reproducibility_score = 1.0 / (1.0 + overall_cv)
        
        reproducibility_stats['overall_reproducibility_score'] = reproducibility_score
        
        # Execution time statistics
        exec_times = [r.execution_time for r in results]
        reproducibility_stats['execution_times'] = {
            'mean': SimpleStatistics.mean(exec_times),
            'std': SimpleStatistics.std(exec_times)
        }
        
        return reproducibility_stats


class SimpleComparativeAnalyzer:
    """Simple comparative statistical analysis."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def simple_t_test(self, values_a: List[float], values_b: List[float]) -> Dict[str, Any]:
        """Simplified two-sample t-test."""
        
        mean_a = SimpleStatistics.mean(values_a)
        mean_b = SimpleStatistics.mean(values_b)
        std_a = SimpleStatistics.std(values_a) 
        std_b = SimpleStatistics.std(values_b)
        n_a, n_b = len(values_a), len(values_b)
        
        # Pooled standard error
        pooled_se = math.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
        
        # t-statistic
        t_stat = (mean_a - mean_b) / (pooled_se + 1e-8)
        
        # Approximate p-value using normal distribution (for large samples)
        # This is a simplification - real t-test requires t-distribution
        z_score = abs(t_stat)
        # Rough approximation of p-value from z-score
        if z_score > 2.58:  # 99% confidence
            p_value = 0.01
        elif z_score > 1.96:  # 95% confidence
            p_value = 0.05
        elif z_score > 1.28:  # 80% confidence
            p_value = 0.20
        else:
            p_value = 0.50
        
        # Effect size (Cohen's d approximation)
        pooled_std = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)
        
        is_significant = p_value < self.significance_level
        
        # Interpretation
        if is_significant:
            if abs(cohens_d) > 0.8:
                interpretation = f"Large effect size detected (d={cohens_d:.3f})"
            elif abs(cohens_d) > 0.5:
                interpretation = f"Medium effect size detected (d={cohens_d:.3f})"  
            else:
                interpretation = f"Small effect size detected (d={cohens_d:.3f})"
        else:
            interpretation = f"No significant difference (p≈{p_value:.2f}, d={cohens_d:.3f})"
        
        return {
            'test_name': 'Simplified t-test',
            't_statistic': t_stat,
            'p_value_approx': p_value,
            'effect_size': cohens_d,
            'is_significant': is_significant,
            'interpretation': interpretation,
            'mean_difference': mean_a - mean_b
        }
    
    def compare_multiple_methods(
        self,
        all_results: Dict[str, List[SimpleExperimentResult]],
        metric_name: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Compare multiple methods."""
        
        method_names = list(all_results.keys())
        pairwise_comparisons = {}
        
        # Compute pairwise comparisons
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method_a = method_names[i]
                method_b = method_names[j]
                
                values_a = [r.performance_metrics[metric_name] for r in all_results[method_a]]
                values_b = [r.performance_metrics[metric_name] for r in all_results[method_b]]
                
                comparison = self.simple_t_test(values_a, values_b)
                pairwise_comparisons[f"{method_a}_vs_{method_b}"] = comparison
        
        # Rank methods by mean performance
        method_means = {}
        for method_name, results in all_results.items():
            values = [r.performance_metrics[metric_name] for r in results]
            method_means[method_name] = SimpleStatistics.mean(values)
        
        ranked_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'pairwise_comparisons': pairwise_comparisons,
            'method_rankings': ranked_methods,
            'method_means': method_means
        }


class SimpleMemoryTask:
    """Simple memory task for benchmarking."""
    
    def __init__(self, sequence_length: int = 20, delay: int = 5):
        self.sequence_length = sequence_length
        self.delay = delay
    
    def generate_sequence(self, seed: int = 42) -> Tuple[List[float], List[float]]:
        """Generate a simple memory task sequence."""
        random.seed(seed)
        
        # Generate random input sequence  
        inputs = [random.uniform(-1, 1) for _ in range(self.sequence_length)]
        
        # Target is delayed version of input
        targets = [0.0] * self.delay + inputs[:-self.delay]
        
        return inputs, targets
    
    def evaluate_memory_capacity(
        self,
        predictions: List[float],
        targets: List[float]
    ) -> Dict[str, float]:
        """Evaluate memory performance."""
        
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        # Mean squared error
        mse = SimpleStatistics.mean([(p - t)**2 for p, t in zip(predictions, targets)])
        
        # Mean absolute error  
        mae = SimpleStatistics.mean([abs(p - t) for p, t in zip(predictions, targets)])
        
        # Simple correlation approximation
        pred_mean = SimpleStatistics.mean(predictions)
        target_mean = SimpleStatistics.mean(targets)
        
        numerator = sum((p - pred_mean) * (t - target_mean) for p, t in zip(predictions, targets))
        pred_var = sum((p - pred_mean)**2 for p in predictions)
        target_var = sum((t - target_mean)**2 for t in targets)
        
        correlation = numerator / (math.sqrt(pred_var * target_var) + 1e-8)
        
        # Memory capacity approximation
        memory_capacity = max(0.0, correlation)
        
        return {
            'mse': mse,
            'mae': mae,
            'memory_capacity': memory_capacity,
            'correlation': correlation
        }


class SimpleModelEvaluator:
    """Simple model evaluator for basic benchmarking."""
    
    def __init__(self):
        self.tasks = {
            'memory': SimpleMemoryTask(),
            'prediction': self.create_prediction_task()
        }
    
    def create_prediction_task(self) -> Callable:
        """Create a simple prediction task."""
        def prediction_task(seed: int = 42) -> Tuple[List[float], List[float]]:
            random.seed(seed)
            
            # Simple AR(1) process: x_t = 0.8 * x_{t-1} + noise
            length = 50
            inputs = []
            targets = []
            
            x = 0.0
            for _ in range(length):
                noise = random.gauss(0, 0.1)
                x_new = 0.8 * x + noise
                
                inputs.append(x)
                targets.append(x_new)
                
                x = x_new
            
            return inputs, targets
        
        return prediction_task
    
    def evaluate_dummy_model(
        self, 
        model_name: str, 
        task_name: str = 'memory',
        num_runs: int = 5
    ) -> List[SimpleExperimentResult]:
        """Evaluate a dummy model (for testing framework)."""
        
        results = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            if task_name == 'memory':
                inputs, targets = self.tasks['memory'].generate_sequence(seed=run)
                
                # Dummy predictions (simple moving average)
                predictions = []
                window = 3
                for i in range(len(inputs)):
                    if i < window:
                        pred = 0.0  # Cold start
                    else:
                        pred = SimpleStatistics.mean(inputs[i-window:i])
                    predictions.append(pred)
                
                metrics = self.tasks['memory'].evaluate_memory_capacity(predictions, targets)
                
            elif task_name == 'prediction':
                inputs, targets = self.tasks['prediction'](seed=run)
                
                # Dummy predictions (persistence model)
                predictions = inputs  # Simple persistence
                
                mse = SimpleStatistics.mean([(p - t)**2 for p, t in zip(predictions, targets)])
                mae = SimpleStatistics.mean([abs(p - t) for p, t in zip(predictions, targets)])
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'prediction_accuracy': 1.0 / (1.0 + mse)
                }
            
            else:
                raise ValueError(f"Unknown task: {task_name}")
            
            execution_time = time.time() - start_time
            
            # Add some noise to make results more realistic
            for key in metrics:
                if random.random() < 0.3:  # 30% chance of noise
                    noise_factor = 1.0 + random.gauss(0, 0.05)  # 5% noise
                    metrics[key] *= noise_factor
            
            result = SimpleExperimentResult(
                method_name=model_name,
                performance_metrics=metrics,
                execution_time=execution_time,
                random_seed=run,
                timestamp=time.time()
            )
            
            results.append(result)
        
        return results


def run_simple_benchmark_demo():
    """Demonstrate the simple benchmarking framework."""
    
    print("🚀 Running Simple Benchmark Demo")
    print("=" * 50)
    
    # Create evaluator
    evaluator = SimpleModelEvaluator()
    
    # Evaluate dummy models
    print("📊 Evaluating models...")
    
    model_results = {}
    model_results['SimpleModel'] = evaluator.evaluate_dummy_model('SimpleModel', 'memory', 5)
    model_results['AdvancedModel'] = evaluator.evaluate_dummy_model('AdvancedModel', 'memory', 5)
    
    # Compute reproducibility metrics
    print("\\n📈 Computing reproducibility metrics...")
    
    validator = SimpleReproducibilityValidator()
    
    for model_name, results in model_results.items():
        print(f"\\n{model_name}:")
        metrics = validator.compute_reproducibility_metrics(results)
        
        for metric_name, stats in metrics.items():
            if metric_name == 'overall_reproducibility_score':
                print(f"  Overall Reproducibility Score: {stats:.3f}")
            elif metric_name == 'execution_times':
                print(f"  Execution Time: {stats['mean']:.3f}±{stats['std']:.3f}s")
            elif isinstance(stats, dict) and 'mean' in stats:
                print(f"  {metric_name}: {stats['mean']:.3f}±{stats['std']:.3f}")
    
    # Comparative analysis
    print("\\n🔍 Comparative Analysis...")
    
    analyzer = SimpleComparativeAnalyzer()
    comparison_results = analyzer.compare_multiple_methods(model_results, 'memory_capacity')
    
    print("\\nMethod Rankings (by memory capacity):")
    for i, (method, score) in enumerate(comparison_results['method_rankings'], 1):
        print(f"  {i}. {method}: {score:.3f}")
    
    print("\\nPairwise Comparisons:")
    for pair, result in comparison_results['pairwise_comparisons'].items():
        print(f"  {pair}: {result['interpretation']}")
    
    print("\\n✅ Demo completed successfully!")
    return True


if __name__ == "__main__":
    run_simple_benchmark_demo()