import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import matplotlib.pyplot as plt
from ..models.liquid_neural_network import LiquidNeuralNetwork
from ..models.continuous_time_rnn import ContinuousTimeRNN
from ..algorithms.training import LiquidNetworkTrainer
from .synthetic_tasks import SyntheticTaskGenerator


class ValidationExperiments:
    """
    Comprehensive validation experiments for liquid neural networks.
    
    Implements systematic evaluation protocols including:
    - Baseline comparisons
    - Ablation studies  
    - Performance benchmarking
    - Statistical significance testing
    """
    
    def __init__(self, seed: int = 123):
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.task_generator = SyntheticTaskGenerator(seed)
        
        # Results storage
        self.experiment_results = {}
        self.benchmark_results = {}
        
    def create_baseline_models(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Create baseline models for comparison."""
        models = {}
        
        # Liquid Neural Network
        self.key, subkey = random.split(self.key)
        models['liquid_nn'] = LiquidNeuralNetwork(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            key=subkey
        )
        
        # Continuous Time RNN
        self.key, subkey = random.split(self.key)
        models['ct_rnn'] = ContinuousTimeRNN(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            activation='tanh',
            key=subkey
        )
        
        # Continuous Time RNN with different activation
        self.key, subkey = random.split(self.key)
        models['ct_rnn_swish'] = ContinuousTimeRNN(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            activation='swish',
            key=subkey
        )
        
        return models
    
    def run_single_task_experiment(
        self,
        task_name: str,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        models: Dict[str, Any],
        train_split: float = 0.8,
        epochs: int = 100,
        dt: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """Run experiment on a single task with multiple models."""
        print(f"\nRunning experiment on {task_name}...")
        
        # Split data
        n_train = int(len(inputs) * train_split)
        
        # Ensure inputs are 2D for training
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
            
        # For sequence data, we need to handle differently
        if inputs.ndim == 3:  # (n_sequences, seq_len, features)
            # Use first sequence for now (can be extended)
            train_inputs = inputs[0]  # First sequence
            train_targets = targets[0]
            val_inputs = inputs[min(1, len(inputs)-1)]
            val_targets = targets[min(1, len(targets)-1)]
        else:
            train_inputs = inputs[:n_train]
            train_targets = targets[:n_train]
            val_inputs = inputs[n_train:]
            val_targets = targets[n_train:]
        
        results = {}
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            start_time = time.time()
            
            # Create trainer
            trainer = LiquidNetworkTrainer(
                model=model,
                learning_rate=1e-3,
                optimizer_name='adam',
                loss_fn='mse'
            )
            
            # Train model
            try:
                history = trainer.fit(
                    train_data=(train_inputs, train_targets),
                    val_data=(val_inputs, val_targets) if len(val_inputs) > 0 else None,
                    epochs=epochs,
                    dt=dt,
                    verbose=False
                )
                
                training_time = time.time() - start_time
                
                # Evaluate performance
                final_train_loss = history['train_loss'][-1]
                final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('nan')
                
                # Additional metrics
                predictions, _ = model.forward(val_inputs if len(val_inputs) > 0 else train_inputs, dt=dt)
                actual_targets = val_targets if len(val_targets) > 0 else train_targets
                
                mse = float(jnp.mean((predictions - actual_targets) ** 2))
                mae = float(jnp.mean(jnp.abs(predictions - actual_targets)))
                
                results[model_name] = {
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'mse': mse,
                    'mae': mae,
                    'training_time': training_time,
                    'history': history,
                    'converged': final_train_loss < 1.0,  # Simple convergence criterion
                    'n_parameters': sum(p.size for p in model.params.values())
                }
                
                print(f"    {model_name}: Train Loss = {final_train_loss:.6f}, Val Loss = {final_val_loss:.6f}")
                
            except Exception as e:
                print(f"    {model_name}: Training failed with error: {str(e)}")
                results[model_name] = {
                    'final_train_loss': float('inf'),
                    'final_val_loss': float('inf'),
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'training_time': 0.0,
                    'history': {},
                    'converged': False,
                    'error': str(e)
                }
        
        return results
    
    def run_ablation_study(
        self,
        task_name: str,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        base_config: Dict[str, Any],
        ablation_configs: Dict[str, Dict[str, Any]],
        epochs: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """Run ablation study varying specific components."""
        print(f"\nRunning ablation study on {task_name}...")
        
        results = {}
        
        for config_name, config_changes in ablation_configs.items():
            print(f"  Testing configuration: {config_name}")
            
            # Create model with modified config
            config = base_config.copy()
            config.update(config_changes)
            
            self.key, subkey = random.split(self.key)
            
            if config.get('model_type', 'liquid') == 'liquid':
                model = LiquidNeuralNetwork(key=subkey, **config)
            else:
                model = ContinuousTimeRNN(key=subkey, **config)
            
            # Run experiment
            models = {config_name: model}
            config_results = self.run_single_task_experiment(
                task_name, inputs, targets, models, epochs=epochs
            )
            
            results[config_name] = config_results[config_name]
        
        return results
    
    def benchmark_computational_efficiency(
        self,
        models: Dict[str, Any],
        input_sizes: List[int] = [10, 50, 100],
        sequence_lengths: List[int] = [50, 100, 200],
        n_trials: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark computational efficiency across different configurations."""
        print("\nBenchmarking computational efficiency...")
        
        efficiency_results = {}
        
        for model_name, base_model in models.items():
            print(f"  Benchmarking {model_name}...")
            efficiency_results[model_name] = {}
            
            for input_size in input_sizes:
                for seq_len in sequence_lengths:
                    config_name = f"input_{input_size}_seq_{seq_len}"
                    
                    # Create test data
                    self.key, subkey = random.split(self.key)
                    test_inputs = random.normal(subkey, (seq_len, input_size))
                    
                    # Time forward pass
                    forward_times = []
                    
                    for _ in range(n_trials):
                        start_time = time.time()
                        outputs, states = base_model.forward(test_inputs)
                        forward_time = time.time() - start_time
                        forward_times.append(forward_time)
                    
                    avg_forward_time = np.mean(forward_times)
                    std_forward_time = np.std(forward_times)
                    
                    efficiency_results[model_name][config_name] = {
                        'avg_forward_time': avg_forward_time,
                        'std_forward_time': std_forward_time,
                        'throughput': seq_len / avg_forward_time,  # sequences per second
                        'input_size': input_size,
                        'sequence_length': seq_len
                    }
        
        return efficiency_results
    
    def statistical_significance_test(
        self,
        results_a: List[float],
        results_b: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform statistical significance test between two sets of results."""
        from scipy import stats
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results_a) - 1) * np.var(results_a, ddof=1) + 
                             (len(results_b) - 1) * np.var(results_b, ddof=1)) / 
                            (len(results_a) + len(results_b) - 2))
        cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        }
    
    def run_comprehensive_benchmark(
        self,
        epochs: int = 50,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all tasks and models."""
        print("Starting comprehensive benchmark suite...")
        
        # Generate benchmark tasks
        benchmark_tasks = self.task_generator.create_benchmark_suite()
        
        comprehensive_results = {}
        
        for task_name, (inputs, targets) in benchmark_tasks.items():
            print(f"\nProcessing {task_name}...")
            
            # Determine input/output dimensions
            if inputs.ndim == 3:
                input_size = inputs.shape[2]
            elif inputs.ndim == 2:
                input_size = inputs.shape[1] if inputs.shape[1] > 1 else 1
            else:
                input_size = 1
                
            if targets.ndim == 3:
                output_size = targets.shape[2]
            elif targets.ndim == 2:
                output_size = targets.shape[1] if targets.shape[1] > 1 else 1
            else:
                output_size = 1
            
            task_results = []
            
            # Run multiple times for statistical significance
            for run in range(n_runs):
                print(f"  Run {run + 1}/{n_runs}")
                
                # Create fresh models for each run
                models = self.create_baseline_models(input_size, output_size)
                
                # Run experiment
                run_results = self.run_single_task_experiment(
                    task_name, inputs, targets, models, epochs=epochs
                )
                
                task_results.append(run_results)
            
            # Aggregate results
            aggregated_results = self._aggregate_multiple_runs(task_results)
            comprehensive_results[task_name] = aggregated_results
        
        # Generate summary statistics
        summary = self._generate_benchmark_summary(comprehensive_results)
        comprehensive_results['summary'] = summary
        
        self.benchmark_results = comprehensive_results
        return comprehensive_results
    
    def _aggregate_multiple_runs(self, run_results: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate results from multiple runs."""
        aggregated = {}
        
        # Get model names from first run
        model_names = list(run_results[0].keys())
        
        for model_name in model_names:
            model_results = []
            for run in run_results:
                if model_name in run and 'final_train_loss' in run[model_name]:
                    model_results.append(run[model_name])
            
            if model_results:
                # Compute statistics
                train_losses = [r['final_train_loss'] for r in model_results if not np.isinf(r['final_train_loss'])]
                val_losses = [r['final_val_loss'] for r in model_results if not np.isinf(r['final_val_loss']) and not np.isnan(r['final_val_loss'])]
                training_times = [r['training_time'] for r in model_results]
                converged_runs = [r['converged'] for r in model_results]
                
                aggregated[model_name] = {
                    'mean_train_loss': np.mean(train_losses) if train_losses else float('inf'),
                    'std_train_loss': np.std(train_losses) if train_losses else 0.0,
                    'mean_val_loss': np.mean(val_losses) if val_losses else float('inf'),
                    'std_val_loss': np.std(val_losses) if val_losses else 0.0,
                    'mean_training_time': np.mean(training_times),
                    'std_training_time': np.std(training_times),
                    'convergence_rate': np.mean(converged_runs),
                    'n_runs': len(model_results),
                    'successful_runs': len([r for r in model_results if r.get('converged', False)])
                }
        
        return aggregated
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all benchmarks."""
        summary = {
            'best_model_per_task': {},
            'overall_rankings': {},
            'average_performance': {}
        }
        
        model_names = set()
        for task_results in results.values():
            model_names.update(task_results.keys())
        
        # Find best model per task
        for task_name, task_results in results.items():
            best_model = None
            best_loss = float('inf')
            
            for model_name, model_results in task_results.items():
                if isinstance(model_results, dict) and 'mean_val_loss' in model_results:
                    val_loss = model_results['mean_val_loss']
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = model_name
            
            summary['best_model_per_task'][task_name] = {
                'model': best_model,
                'loss': best_loss
            }
        
        # Overall model rankings
        model_scores = {name: [] for name in model_names}
        
        for task_results in results.values():
            task_losses = [(name, res.get('mean_val_loss', float('inf'))) 
                          for name, res in task_results.items() 
                          if isinstance(res, dict)]
            
            # Sort by loss (lower is better)
            task_losses.sort(key=lambda x: x[1])
            
            # Assign scores (1st place = n_models points, last place = 1 point)
            for rank, (model_name, _) in enumerate(task_losses):
                score = len(task_losses) - rank
                model_scores[model_name].append(score)
        
        # Average scores
        avg_scores = {name: np.mean(scores) if scores else 0 
                     for name, scores in model_scores.items()}
        
        summary['overall_rankings'] = dict(sorted(avg_scores.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True))
        
        return summary
    
    def visualize_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize experiment results."""
        n_tasks = len([k for k in results.keys() if k != 'summary'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: Training loss comparison
        ax = axes[0]
        task_names = [k for k in results.keys() if k != 'summary']
        model_names = list(results[task_names[0]].keys())
        
        x = np.arange(len(task_names))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            train_losses = []
            train_stds = []
            
            for task_name in task_names:
                task_results = results[task_name]
                if model_name in task_results:
                    train_losses.append(task_results[model_name].get('mean_train_loss', float('inf')))
                    train_stds.append(task_results[model_name].get('std_train_loss', 0.0))
                else:
                    train_losses.append(float('inf'))
                    train_stds.append(0.0)
            
            # Filter out infinite values for plotting
            finite_indices = [j for j, loss in enumerate(train_losses) if not np.isinf(loss)]
            finite_losses = [train_losses[j] for j in finite_indices]
            finite_stds = [train_stds[j] for j in finite_indices]
            finite_x = [x[j] + i * width for j in finite_indices]
            
            if finite_losses:
                ax.bar(finite_x, finite_losses, width, 
                      yerr=finite_stds, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(task_names, rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot 2: Convergence rates
        ax = axes[1]
        convergence_rates = {}
        
        for model_name in model_names:
            rates = []
            for task_name in task_names:
                if model_name in results[task_name]:
                    rate = results[task_name][model_name].get('convergence_rate', 0.0)
                    rates.append(rate)
            convergence_rates[model_name] = np.mean(rates) if rates else 0.0
        
        models = list(convergence_rates.keys())
        rates = list(convergence_rates.values())
        
        ax.bar(models, rates, alpha=0.8)
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Model Convergence Rates')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Plot 3: Training time comparison
        ax = axes[2]
        training_times = {}
        
        for model_name in model_names:
            times = []
            for task_name in task_names:
                if model_name in results[task_name]:
                    time_val = results[task_name][model_name].get('mean_training_time', 0.0)
                    times.append(time_val)
            training_times[model_name] = np.mean(times) if times else 0.0
        
        models = list(training_times.keys())
        times = list(training_times.values())
        
        ax.bar(models, times, alpha=0.8)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Average Training Time')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Plot 4: Overall rankings
        if 'summary' in results and 'overall_rankings' in results['summary']:
            ax = axes[3]
            rankings = results['summary']['overall_rankings']
            
            models = list(rankings.keys())
            scores = list(rankings.values())
            
            bars = ax.bar(models, scores, alpha=0.8)
            ax.set_ylabel('Average Score')
            ax.set_title('Overall Model Rankings')
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Color bars by rank
            for i, bar in enumerate(bars):
                if i == 0:
                    bar.set_color('gold')
                elif i == 1:
                    bar.set_color('silver')
                elif i == 2:
                    bar.set_color('#CD7F32')  # Bronze
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive text report of results."""
        report = []
        report.append("LIQUID NEURAL NETWORK VALIDATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary section
        if 'summary' in results:
            summary = results['summary']
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 20)
            
            if 'overall_rankings' in summary:
                report.append("Overall Model Rankings:")
                for i, (model, score) in enumerate(summary['overall_rankings'].items()):
                    report.append(f"  {i+1}. {model}: {score:.3f}")
            
            report.append("")
            
            if 'best_model_per_task' in summary:
                report.append("Best Model Per Task:")
                for task, info in summary['best_model_per_task'].items():
                    report.append(f"  {task}: {info['model']} (loss: {info['loss']:.6f})")
            
            report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 20)
        
        task_names = [k for k in results.keys() if k != 'summary']
        
        for task_name in task_names:
            report.append(f"\n{task_name.upper()}")
            report.append("-" * len(task_name))
            
            task_results = results[task_name]
            
            for model_name, model_results in task_results.items():
                if isinstance(model_results, dict):
                    report.append(f"\n  {model_name}:")
                    report.append(f"    Train Loss: {model_results.get('mean_train_loss', 'N/A'):.6f} ± {model_results.get('std_train_loss', 0):.6f}")
                    report.append(f"    Val Loss:   {model_results.get('mean_val_loss', 'N/A'):.6f} ± {model_results.get('std_val_loss', 0):.6f}")
                    report.append(f"    Train Time: {model_results.get('mean_training_time', 0):.3f} ± {model_results.get('std_training_time', 0):.3f} seconds")
                    report.append(f"    Converged:  {model_results.get('convergence_rate', 0):.1%} of runs")
                    report.append(f"    Success:    {model_results.get('successful_runs', 0)}/{model_results.get('n_runs', 0)} runs")
        
        return "\n".join(report)