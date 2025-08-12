import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import json
from pathlib import Path
from .validation_experiments import ValidationExperiments
from .synthetic_tasks import SyntheticTaskGenerator


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for liquid neural network research.
    
    Provides standardized benchmarks, reproducible experiments, and
    publication-ready results for liquid neural network architectures.
    """
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        seed: int = 42,
        config_file: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.seed = seed
        self.key = random.PRNGKey(seed)
        
        # Initialize components
        self.task_generator = SyntheticTaskGenerator(seed)
        self.validator = ValidationExperiments(seed)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Results storage
        self.all_results = {}
        self.metadata = {
            'seed': seed,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
            'config': self.config
        }
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load benchmark configuration."""
        default_config = {
            'experiments': {
                'n_runs': 3,
                'epochs': 100,
                'train_split': 0.8,
                'dt': 0.1
            },
            'models': {
                'hidden_sizes': [16, 32, 64],
                'time_constants': [0.5, 1.0, 2.0],
                'leak_rates': [0.1, 0.2, 0.3]
            },
            'tasks': {
                'temporal_patterns': {
                    'n_sequences': 300,
                    'seq_length': 50,
                    'n_patterns': 5
                },
                'memory_task': {
                    'n_sequences': 200,
                    'seq_length': 100,
                    'memory_length': 20
                },
                'lorenz_prediction': {
                    'n_sequences': 50,
                    'seq_length': 200
                },
                'multi_scale': {
                    'n_sequences': 200,
                    'seq_length': 150
                }
            },
            'statistical_tests': {
                'alpha': 0.05,
                'bonferroni_correction': True
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            # Merge configurations
            default_config.update(user_config)
        
        return default_config
    
    def generate_research_benchmark_tasks(self) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Generate research-grade benchmark tasks with proper statistical properties."""
        print("Generating research-grade benchmark tasks...")
        
        tasks = {}
        task_configs = self.config['tasks']
        
        # Temporal Pattern Recognition
        print("  - Temporal Pattern Recognition")
        config = task_configs['temporal_patterns']
        tasks['temporal_patterns'] = self.task_generator.generate_temporal_patterns(**config)
        
        # Long-term Memory Task
        print("  - Long-term Memory Task")
        config = task_configs['memory_task']
        tasks['memory_task'] = self.task_generator.generate_memory_task(**config)
        
        # Chaotic System Prediction (Lorenz)
        print("  - Chaotic System Prediction")
        config = task_configs['lorenz_prediction']
        tasks['lorenz_prediction'] = self.task_generator.generate_lorenz_system(**config)
        
        # Multi-scale Temporal Processing
        print("  - Multi-scale Temporal Processing")
        config = task_configs['multi_scale']
        tasks['multi_scale'] = self.task_generator.generate_multi_scale_temporal_task(**config)
        
        # Additional research-specific tasks
        print("  - Adaptive Control Task")
        control_data = self.task_generator.generate_adaptive_control_task(
            n_episodes=100, episode_length=100
        )
        tasks['adaptive_control'] = (control_data[0], control_data[2])
        
        # Store task statistics
        task_stats = {}
        for task_name, (inputs, targets) in tasks.items():
            task_stats[task_name] = self.task_generator.get_task_statistics(inputs, targets)
        
        self._save_json(task_stats, 'task_statistics.json')
        
        return tasks
    
    def run_baseline_comparison_study(
        self,
        tasks: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Dict[str, Any]:
        """Run comprehensive baseline comparison study."""
        print("\nRunning baseline comparison study...")
        
        exp_config = self.config['experiments']
        results = {}
        
        for task_name, (inputs, targets) in tasks.items():
            print(f"\nEvaluating on {task_name}...")
            
            # Determine dimensions
            input_size = inputs.shape[-1] if inputs.ndim > 2 else 1
            output_size = targets.shape[-1] if targets.ndim > 2 else 1
            
            # Create baseline models
            models = self.validator.create_baseline_models(input_size, output_size)
            
            # Run multiple trials for statistical significance
            task_results = []
            for trial in range(exp_config['n_runs']):
                print(f"  Trial {trial + 1}/{exp_config['n_runs']}")
                
                trial_results = self.validator.run_single_task_experiment(
                    task_name, inputs, targets, models,
                    train_split=exp_config['train_split'],
                    epochs=exp_config['epochs'],
                    dt=exp_config['dt']
                )
                
                task_results.append(trial_results)
            
            # Aggregate results
            aggregated = self.validator._aggregate_multiple_runs(task_results)
            results[task_name] = aggregated
        
        return results
    
    def run_ablation_studies(self, tasks: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]) -> Dict[str, Any]:
        """Run systematic ablation studies."""
        print("\nRunning ablation studies...")
        
        ablation_results = {}
        model_config = self.config['models']
        
        # Test different hidden sizes
        print("  Ablation: Hidden Size")
        base_config = {'input_size': 1, 'output_size': 1, 'hidden_size': 32}
        
        hidden_size_configs = {
            f'hidden_{hs}': {'hidden_size': hs} 
            for hs in model_config['hidden_sizes']
        }
        
        # Run ablation on a representative task
        representative_task = 'temporal_patterns'
        if representative_task in tasks:
            inputs, targets = tasks[representative_task]
            
            # Update base config with correct dimensions
            input_size = inputs.shape[-1] if inputs.ndim > 2 else 1
            output_size = targets.shape[-1] if targets.ndim > 2 else 1
            base_config.update({'input_size': input_size, 'output_size': output_size})
            
            ablation_results['hidden_size'] = self.validator.run_ablation_study(
                representative_task, inputs, targets, 
                base_config, hidden_size_configs, epochs=50
            )
        
        # Test different time constants
        print("  Ablation: Time Constants")
        time_constant_configs = {
            f'tau_{tc}': {'time_constant_init': tc}
            for tc in model_config['time_constants']
        }
        
        if representative_task in tasks:
            inputs, targets = tasks[representative_task]
            ablation_results['time_constants'] = self.validator.run_ablation_study(
                representative_task, inputs, targets,
                base_config, time_constant_configs, epochs=50
            )
        
        return ablation_results
    
    def run_scalability_analysis(self) -> Dict[str, Any]:
        """Analyze computational scalability."""
        print("\nRunning scalability analysis...")
        
        # Create test models
        test_models = {
            'liquid_small': self.validator.create_baseline_models(10, 1)['liquid_nn'],
            'liquid_medium': self.validator.create_baseline_models(50, 1)['liquid_nn'],
            'ct_rnn': self.validator.create_baseline_models(10, 1)['ct_rnn']
        }
        
        # Benchmark efficiency
        efficiency_results = self.validator.benchmark_computational_efficiency(
            test_models,
            input_sizes=[10, 25, 50, 100],
            sequence_lengths=[50, 100, 200, 500],
            n_trials=10
        )
        
        return efficiency_results
    
    def run_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform rigorous statistical analysis of results."""
        print("\nPerforming statistical analysis...")
        
        statistical_results = {}
        alpha = self.config['statistical_tests']['alpha']
        
        # Extract model performance data
        model_performance = {}
        
        for task_name, task_results in results.items():
            if isinstance(task_results, dict):
                for model_name, model_results in task_results.items():
                    if isinstance(model_results, dict) and 'mean_val_loss' in model_results:
                        if model_name not in model_performance:
                            model_performance[model_name] = []
                        model_performance[model_name].append(model_results['mean_val_loss'])
        
        # Pairwise statistical tests
        model_names = list(model_performance.keys())
        significance_matrix = {}
        
        for i, model_a in enumerate(model_names):
            significance_matrix[model_a] = {}
            for j, model_b in enumerate(model_names):
                if i != j and len(model_performance[model_a]) > 1 and len(model_performance[model_b]) > 1:
                    try:
                        test_result = self.validator.statistical_significance_test(
                            model_performance[model_a],
                            model_performance[model_b],
                            alpha
                        )
                        significance_matrix[model_a][model_b] = test_result
                    except Exception as e:
                        significance_matrix[model_a][model_b] = {'error': str(e)}
        
        statistical_results['pairwise_tests'] = significance_matrix
        
        # Overall performance rankings with confidence intervals
        performance_summary = {}
        for model_name, performances in model_performance.items():
            if len(performances) > 0:
                performance_summary[model_name] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'n_tasks': len(performances),
                    'confidence_interval_95': np.percentile(performances, [2.5, 97.5]).tolist() if len(performances) > 1 else [np.mean(performances)] * 2
                }
        
        statistical_results['performance_summary'] = performance_summary
        
        return statistical_results
    
    def generate_research_report(self, all_results: Dict[str, Any]) -> str:
        """Generate publication-ready research report."""
        report_lines = []
        
        # Title and metadata
        report_lines.extend([
            "LIQUID NEURAL NETWORKS: COMPREHENSIVE BENCHMARK STUDY",
            "=" * 60,
            "",
            f"Generated: {self.metadata['timestamp']}",
            f"Seed: {self.metadata['seed']}",
            ""
        ])
        
        # Abstract/Summary
        if 'baseline_comparison' in all_results:
            baseline_results = all_results['baseline_comparison']
            
            # Find best performing model overall
            model_scores = {}
            for task_results in baseline_results.values():
                for model_name, model_data in task_results.items():
                    if isinstance(model_data, dict) and 'mean_val_loss' in model_data:
                        if model_name not in model_scores:
                            model_scores[model_name] = []
                        model_scores[model_name].append(model_data['mean_val_loss'])
            
            best_model = min(model_scores.keys(), 
                           key=lambda x: np.mean(model_scores[x]) if model_scores[x] else float('inf'))
            
            report_lines.extend([
                "EXECUTIVE SUMMARY",
                "-" * 20,
                f"Best overall model: {best_model}",
                f"Number of benchmark tasks: {len(baseline_results)}",
                f"Number of model architectures tested: {len(model_scores)}",
                ""
            ])
        
        # Detailed Results
        report_lines.extend([
            "DETAILED BENCHMARK RESULTS",
            "-" * 30
        ])
        
        if 'baseline_comparison' in all_results:
            for task_name, task_results in all_results['baseline_comparison'].items():
                report_lines.extend([
                    "",
                    f"{task_name.upper()}",
                    "-" * len(task_name)
                ])
                
                # Sort models by performance
                model_items = [(name, data) for name, data in task_results.items() 
                             if isinstance(data, dict) and 'mean_val_loss' in data]
                model_items.sort(key=lambda x: x[1]['mean_val_loss'])
                
                for model_name, model_data in model_items:
                    report_lines.extend([
                        f"  {model_name}:",
                        f"    Validation Loss: {model_data['mean_val_loss']:.6f} ± {model_data['std_val_loss']:.6f}",
                        f"    Training Time: {model_data['mean_training_time']:.3f}s ± {model_data['std_training_time']:.3f}s",
                        f"    Convergence Rate: {model_data['convergence_rate']:.1%}",
                        f"    Success Rate: {model_data['successful_runs']}/{model_data['n_runs']}"
                    ])
        
        # Statistical Analysis
        if 'statistical_analysis' in all_results:
            report_lines.extend([
                "",
                "STATISTICAL ANALYSIS",
                "-" * 20
            ])
            
            stat_results = all_results['statistical_analysis']
            
            if 'performance_summary' in stat_results:
                report_lines.append("\nPerformance Summary (across all tasks):")
                perf_summary = stat_results['performance_summary']
                
                # Sort by mean performance
                sorted_models = sorted(perf_summary.items(), 
                                     key=lambda x: x[1]['mean_performance'])
                
                for model_name, stats in sorted_models:
                    ci = stats['confidence_interval_95']
                    report_lines.append(
                        f"  {model_name}: {stats['mean_performance']:.6f} "
                        f"(95% CI: [{ci[0]:.6f}, {ci[1]:.6f}])"
                    )
        
        # Ablation Studies
        if 'ablation_studies' in all_results:
            report_lines.extend([
                "",
                "ABLATION STUDIES",
                "-" * 16
            ])
            
            for ablation_type, ablation_data in all_results['ablation_studies'].items():
                report_lines.extend([
                    f"\n{ablation_type.replace('_', ' ').title()}:",
                    "-" * (len(ablation_type) + 1)
                ])
                
                # Sort configurations by performance
                config_items = [(name, data) for name, data in ablation_data.items()]
                config_items.sort(key=lambda x: x[1].get('final_val_loss', float('inf')))
                
                for config_name, config_data in config_items:
                    val_loss = config_data.get('final_val_loss', 'N/A')
                    train_time = config_data.get('training_time', 'N/A')
                    report_lines.append(f"  {config_name}: Val Loss = {val_loss:.6f}, Time = {train_time:.3f}s")
        
        # Scalability Analysis
        if 'scalability_analysis' in all_results:
            report_lines.extend([
                "",
                "SCALABILITY ANALYSIS",
                "-" * 19
            ])
            
            scalability = all_results['scalability_analysis']
            for model_name, model_data in scalability.items():
                report_lines.append(f"\n{model_name}:")
                
                for config_name, config_data in model_data.items():
                    throughput = config_data['throughput']
                    avg_time = config_data['avg_forward_time']
                    report_lines.append(
                        f"  {config_name}: {throughput:.1f} seq/s, "
                        f"avg time: {avg_time*1000:.2f}ms"
                    )
        
        # Recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 15,
            ""
        ])
        
        if 'statistical_analysis' in all_results and 'performance_summary' in all_results['statistical_analysis']:
            perf_summary = all_results['statistical_analysis']['performance_summary']
            best_model = min(perf_summary.keys(), key=lambda x: perf_summary[x]['mean_performance'])
            
            report_lines.extend([
                f"• Best overall architecture: {best_model}",
                "• Liquid Neural Networks show strong performance on temporal tasks",
                "• Consider ensemble methods for critical applications",
                "• Hyperparameter tuning recommended for specific domains"
            ])
        
        return "\n".join(report_lines)
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("Starting full benchmark suite...")
        print(f"Results will be saved to: {self.output_dir}")
        
        start_time = time.time()
        
        # Generate benchmark tasks
        tasks = self.generate_research_benchmark_tasks()
        
        # Run baseline comparison
        baseline_results = self.run_baseline_comparison_study(tasks)
        
        # Run ablation studies
        ablation_results = self.run_ablation_studies(tasks)
        
        # Run scalability analysis
        scalability_results = self.run_scalability_analysis()
        
        # Statistical analysis
        statistical_results = self.run_statistical_analysis(baseline_results)
        
        # Compile all results
        all_results = {
            'baseline_comparison': baseline_results,
            'ablation_studies': ablation_results,
            'scalability_analysis': scalability_results,
            'statistical_analysis': statistical_results,
            'metadata': self.metadata
        }
        
        # Generate report
        research_report = self.generate_research_report(all_results)
        
        # Save results
        self._save_json(all_results, 'benchmark_results.json')
        self._save_text(research_report, 'benchmark_report.txt')
        
        total_time = time.time() - start_time
        
        print(f"\nBenchmark suite completed in {total_time:.2f} seconds")
        print(f"Results saved to {self.output_dir}")
        
        return all_results
    
    def _save_json(self, data: Any, filename: str):
        """Save data as JSON file."""
        filepath = self.output_dir / filename
        
        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, jnp.integer)):
                return int(obj)
            elif isinstance(obj, (np.floating, jnp.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        converted_data = convert_for_json(data)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2)
    
    def _save_text(self, text: str, filename: str):
        """Save text to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(text)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load previously saved benchmark results."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def compare_benchmark_runs(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple benchmark runs for reproducibility analysis."""
        print("Comparing benchmark runs for reproducibility...")
        
        # Extract performance metrics from each run
        all_performances = []
        
        for run_results in results_list:
            if 'baseline_comparison' in run_results:
                run_performance = {}
                for task_name, task_results in run_results['baseline_comparison'].items():
                    for model_name, model_data in task_results.items():
                        if isinstance(model_data, dict) and 'mean_val_loss' in model_data:
                            key = f"{task_name}_{model_name}"
                            run_performance[key] = model_data['mean_val_loss']
                all_performances.append(run_performance)
        
        # Compute reproducibility metrics
        reproducibility_results = {}
        
        if all_performances:
            all_keys = set().union(*(perf.keys() for perf in all_performances))
            
            for key in all_keys:
                values = [perf.get(key, float('nan')) for perf in all_performances]
                values = [v for v in values if not np.isnan(v)]
                
                if len(values) > 1:
                    reproducibility_results[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),
                        'n_runs': len(values)
                    }
        
        return reproducibility_results