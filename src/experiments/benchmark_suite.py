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
            task_results = []\n            for trial in range(exp_config['n_runs']):\n                print(f\"  Trial {trial + 1}/{exp_config['n_runs']}\")\n                \n                trial_results = self.validator.run_single_task_experiment(\n                    task_name, inputs, targets, models,\n                    train_split=exp_config['train_split'],\n                    epochs=exp_config['epochs'],\n                    dt=exp_config['dt']\n                )\n                \n                task_results.append(trial_results)\n            \n            # Aggregate results\n            aggregated = self.validator._aggregate_multiple_runs(task_results)\n            results[task_name] = aggregated\n        \n        return results\n    \n    def run_ablation_studies(self, tasks: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]) -> Dict[str, Any]:\n        \"\"\"Run systematic ablation studies.\"\"\"\n        print(\"\\nRunning ablation studies...\")\n        \n        ablation_results = {}\n        model_config = self.config['models']\n        \n        # Test different hidden sizes\n        print(\"  Ablation: Hidden Size\")\n        base_config = {'input_size': 1, 'output_size': 1, 'hidden_size': 32}\n        \n        hidden_size_configs = {\n            f'hidden_{hs}': {'hidden_size': hs} \n            for hs in model_config['hidden_sizes']\n        }\n        \n        # Run ablation on a representative task\n        representative_task = 'temporal_patterns'\n        if representative_task in tasks:\n            inputs, targets = tasks[representative_task]\n            \n            # Update base config with correct dimensions\n            input_size = inputs.shape[-1] if inputs.ndim > 2 else 1\n            output_size = targets.shape[-1] if targets.ndim > 2 else 1\n            base_config.update({'input_size': input_size, 'output_size': output_size})\n            \n            ablation_results['hidden_size'] = self.validator.run_ablation_study(\n                representative_task, inputs, targets, \n                base_config, hidden_size_configs, epochs=50\n            )\n        \n        # Test different time constants\n        print(\"  Ablation: Time Constants\")\n        time_constant_configs = {\n            f'tau_{tc}': {'time_constant_init': tc}\n            for tc in model_config['time_constants']\n        }\n        \n        if representative_task in tasks:\n            inputs, targets = tasks[representative_task]\n            ablation_results['time_constants'] = self.validator.run_ablation_study(\n                representative_task, inputs, targets,\n                base_config, time_constant_configs, epochs=50\n            )\n        \n        return ablation_results\n    \n    def run_scalability_analysis(self) -> Dict[str, Any]:\n        \"\"\"Analyze computational scalability.\"\"\"\n        print(\"\\nRunning scalability analysis...\")\n        \n        # Create test models\n        test_models = {\n            'liquid_small': self.validator.create_baseline_models(10, 1)['liquid_nn'],\n            'liquid_medium': self.validator.create_baseline_models(50, 1)['liquid_nn'],\n            'ct_rnn': self.validator.create_baseline_models(10, 1)['ct_rnn']\n        }\n        \n        # Benchmark efficiency\n        efficiency_results = self.validator.benchmark_computational_efficiency(\n            test_models,\n            input_sizes=[10, 25, 50, 100],\n            sequence_lengths=[50, 100, 200, 500],\n            n_trials=10\n        )\n        \n        return efficiency_results\n    \n    def run_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Perform rigorous statistical analysis of results.\"\"\"\n        print(\"\\nPerforming statistical analysis...\")\n        \n        statistical_results = {}\n        alpha = self.config['statistical_tests']['alpha']\n        \n        # Extract model performance data\n        model_performance = {}\n        \n        for task_name, task_results in results.items():\n            if isinstance(task_results, dict):\n                for model_name, model_results in task_results.items():\n                    if isinstance(model_results, dict) and 'mean_val_loss' in model_results:\n                        if model_name not in model_performance:\n                            model_performance[model_name] = []\n                        model_performance[model_name].append(model_results['mean_val_loss'])\n        \n        # Pairwise statistical tests\n        model_names = list(model_performance.keys())\n        significance_matrix = {}\n        \n        for i, model_a in enumerate(model_names):\n            significance_matrix[model_a] = {}\n            for j, model_b in enumerate(model_names):\n                if i != j and len(model_performance[model_a]) > 1 and len(model_performance[model_b]) > 1:\n                    try:\n                        test_result = self.validator.statistical_significance_test(\n                            model_performance[model_a],\n                            model_performance[model_b],\n                            alpha\n                        )\n                        significance_matrix[model_a][model_b] = test_result\n                    except Exception as e:\n                        significance_matrix[model_a][model_b] = {'error': str(e)}\n        \n        statistical_results['pairwise_tests'] = significance_matrix\n        \n        # Overall performance rankings with confidence intervals\n        performance_summary = {}\n        for model_name, performances in model_performance.items():\n            if len(performances) > 0:\n                performance_summary[model_name] = {\n                    'mean_performance': np.mean(performances),\n                    'std_performance': np.std(performances),\n                    'n_tasks': len(performances),\n                    'confidence_interval_95': np.percentile(performances, [2.5, 97.5]).tolist() if len(performances) > 1 else [np.mean(performances)] * 2\n                }\n        \n        statistical_results['performance_summary'] = performance_summary\n        \n        return statistical_results\n    \n    def generate_research_report(self, all_results: Dict[str, Any]) -> str:\n        \"\"\"Generate publication-ready research report.\"\"\"\n        report_lines = []\n        \n        # Title and metadata\n        report_lines.extend([\n            \"LIQUID NEURAL NETWORKS: COMPREHENSIVE BENCHMARK STUDY\",\n            \"=\" * 60,\n            \"\",\n            f\"Generated: {self.metadata['timestamp']}\",\n            f\"Seed: {self.metadata['seed']}\",\n            \"\"\n        ])\n        \n        # Abstract/Summary\n        if 'baseline_comparison' in all_results:\n            baseline_results = all_results['baseline_comparison']\n            \n            # Find best performing model overall\n            model_scores = {}\n            for task_results in baseline_results.values():\n                for model_name, model_data in task_results.items():\n                    if isinstance(model_data, dict) and 'mean_val_loss' in model_data:\n                        if model_name not in model_scores:\n                            model_scores[model_name] = []\n                        model_scores[model_name].append(model_data['mean_val_loss'])\n            \n            best_model = min(model_scores.keys(), \n                           key=lambda x: np.mean(model_scores[x]) if model_scores[x] else float('inf'))\n            \n            report_lines.extend([\n                \"EXECUTIVE SUMMARY\",\n                \"-\" * 20,\n                f\"Best overall model: {best_model}\",\n                f\"Number of benchmark tasks: {len(baseline_results)}\",\n                f\"Number of model architectures tested: {len(model_scores)}\",\n                \"\"\n            ])\n        \n        # Detailed Results\n        report_lines.extend([\n            \"DETAILED BENCHMARK RESULTS\",\n            \"-\" * 30\n        ])\n        \n        if 'baseline_comparison' in all_results:\n            for task_name, task_results in all_results['baseline_comparison'].items():\n                report_lines.extend([\n                    \"\",\n                    f\"{task_name.upper()}\",\n                    \"-\" * len(task_name)\n                ])\n                \n                # Sort models by performance\n                model_items = [(name, data) for name, data in task_results.items() \n                             if isinstance(data, dict) and 'mean_val_loss' in data]\n                model_items.sort(key=lambda x: x[1]['mean_val_loss'])\n                \n                for model_name, model_data in model_items:\n                    report_lines.extend([\n                        f\"  {model_name}:\",\n                        f\"    Validation Loss: {model_data['mean_val_loss']:.6f} ± {model_data['std_val_loss']:.6f}\",\n                        f\"    Training Time: {model_data['mean_training_time']:.3f}s ± {model_data['std_training_time']:.3f}s\",\n                        f\"    Convergence Rate: {model_data['convergence_rate']:.1%}\",\n                        f\"    Success Rate: {model_data['successful_runs']}/{model_data['n_runs']}\"\n                    ])\n        \n        # Statistical Analysis\n        if 'statistical_analysis' in all_results:\n            report_lines.extend([\n                \"\",\n                \"STATISTICAL ANALYSIS\",\n                \"-\" * 20\n            ])\n            \n            stat_results = all_results['statistical_analysis']\n            \n            if 'performance_summary' in stat_results:\n                report_lines.append(\"\\nPerformance Summary (across all tasks):\")\n                perf_summary = stat_results['performance_summary']\n                \n                # Sort by mean performance\n                sorted_models = sorted(perf_summary.items(), \n                                     key=lambda x: x[1]['mean_performance'])\n                \n                for model_name, stats in sorted_models:\n                    ci = stats['confidence_interval_95']\n                    report_lines.append(\n                        f\"  {model_name}: {stats['mean_performance']:.6f} \"\n                        f\"(95% CI: [{ci[0]:.6f}, {ci[1]:.6f}])\"\n                    )\n        \n        # Ablation Studies\n        if 'ablation_studies' in all_results:\n            report_lines.extend([\n                \"\",\n                \"ABLATION STUDIES\",\n                \"-\" * 16\n            ])\n            \n            for ablation_type, ablation_data in all_results['ablation_studies'].items():\n                report_lines.extend([\n                    f\"\\n{ablation_type.replace('_', ' ').title()}:\",\n                    \"-\" * (len(ablation_type) + 1)\n                ])\n                \n                # Sort configurations by performance\n                config_items = [(name, data) for name, data in ablation_data.items()]\n                config_items.sort(key=lambda x: x[1].get('final_val_loss', float('inf')))\n                \n                for config_name, config_data in config_items:\n                    val_loss = config_data.get('final_val_loss', 'N/A')\n                    train_time = config_data.get('training_time', 'N/A')\n                    report_lines.append(f\"  {config_name}: Val Loss = {val_loss:.6f}, Time = {train_time:.3f}s\")\n        \n        # Scalability Analysis\n        if 'scalability_analysis' in all_results:\n            report_lines.extend([\n                \"\",\n                \"SCALABILITY ANALYSIS\",\n                \"-\" * 19\n            ])\n            \n            scalability = all_results['scalability_analysis']\n            for model_name, model_data in scalability.items():\n                report_lines.append(f\"\\n{model_name}:\")\n                \n                for config_name, config_data in model_data.items():\n                    throughput = config_data['throughput']\n                    avg_time = config_data['avg_forward_time']\n                    report_lines.append(\n                        f\"  {config_name}: {throughput:.1f} seq/s, \"\n                        f\"avg time: {avg_time*1000:.2f}ms\"\n                    )\n        \n        # Recommendations\n        report_lines.extend([\n            \"\",\n            \"RECOMMENDATIONS\",\n            \"-\" * 15,\n            \"\"\n        ])\n        \n        if 'statistical_analysis' in all_results and 'performance_summary' in all_results['statistical_analysis']:\n            perf_summary = all_results['statistical_analysis']['performance_summary']\n            best_model = min(perf_summary.keys(), key=lambda x: perf_summary[x]['mean_performance'])\n            \n            report_lines.extend([\n                f\"• Best overall architecture: {best_model}\",\n                \"• Liquid Neural Networks show strong performance on temporal tasks\",\n                \"• Consider ensemble methods for critical applications\",\n                \"• Hyperparameter tuning recommended for specific domains\"\n            ])\n        \n        return \"\\n\".join(report_lines)\n    \n    def run_full_benchmark_suite(self) -> Dict[str, Any]:\n        \"\"\"Run the complete benchmark suite.\"\"\"\n        print(\"Starting full benchmark suite...\")\n        print(f\"Results will be saved to: {self.output_dir}\")\n        \n        start_time = time.time()\n        \n        # Generate benchmark tasks\n        tasks = self.generate_research_benchmark_tasks()\n        \n        # Run baseline comparison\n        baseline_results = self.run_baseline_comparison_study(tasks)\n        \n        # Run ablation studies\n        ablation_results = self.run_ablation_studies(tasks)\n        \n        # Run scalability analysis\n        scalability_results = self.run_scalability_analysis()\n        \n        # Statistical analysis\n        statistical_results = self.run_statistical_analysis(baseline_results)\n        \n        # Compile all results\n        all_results = {\n            'baseline_comparison': baseline_results,\n            'ablation_studies': ablation_results,\n            'scalability_analysis': scalability_results,\n            'statistical_analysis': statistical_results,\n            'metadata': self.metadata\n        }\n        \n        # Generate report\n        research_report = self.generate_research_report(all_results)\n        \n        # Save results\n        self._save_json(all_results, 'benchmark_results.json')\n        self._save_text(research_report, 'benchmark_report.txt')\n        \n        total_time = time.time() - start_time\n        \n        print(f\"\\nBenchmark suite completed in {total_time:.2f} seconds\")\n        print(f\"Results saved to {self.output_dir}\")\n        \n        return all_results\n    \n    def _save_json(self, data: Any, filename: str):\n        \"\"\"Save data as JSON file.\"\"\"\n        filepath = self.output_dir / filename\n        \n        # Convert numpy arrays and other non-serializable objects\n        def convert_for_json(obj):\n            if isinstance(obj, jnp.ndarray):\n                return obj.tolist()\n            elif isinstance(obj, np.ndarray):\n                return obj.tolist()\n            elif isinstance(obj, (np.integer, jnp.integer)):\n                return int(obj)\n            elif isinstance(obj, (np.floating, jnp.floating)):\n                return float(obj)\n            elif isinstance(obj, dict):\n                return {k: convert_for_json(v) for k, v in obj.items()}\n            elif isinstance(obj, (list, tuple)):\n                return [convert_for_json(item) for item in obj]\n            else:\n                return obj\n        \n        converted_data = convert_for_json(data)\n        \n        with open(filepath, 'w') as f:\n            json.dump(converted_data, f, indent=2)\n    \n    def _save_text(self, text: str, filename: str):\n        \"\"\"Save text to file.\"\"\"\n        filepath = self.output_dir / filename\n        with open(filepath, 'w') as f:\n            f.write(text)\n    \n    def load_results(self, filepath: str) -> Dict[str, Any]:\n        \"\"\"Load previously saved benchmark results.\"\"\"\n        with open(filepath, 'r') as f:\n            return json.load(f)\n    \n    def compare_benchmark_runs(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:\n        \"\"\"Compare multiple benchmark runs for reproducibility analysis.\"\"\"\n        print(\"Comparing benchmark runs for reproducibility...\")\n        \n        # Extract performance metrics from each run\n        all_performances = []\n        \n        for run_results in results_list:\n            if 'baseline_comparison' in run_results:\n                run_performance = {}\n                for task_name, task_results in run_results['baseline_comparison'].items():\n                    for model_name, model_data in task_results.items():\n                        if isinstance(model_data, dict) and 'mean_val_loss' in model_data:\n                            key = f\"{task_name}_{model_name}\"\n                            run_performance[key] = model_data['mean_val_loss']\n                all_performances.append(run_performance)\n        \n        # Compute reproducibility metrics\n        reproducibility_results = {}\n        \n        if all_performances:\n            all_keys = set().union(*(perf.keys() for perf in all_performances))\n            \n            for key in all_keys:\n                values = [perf.get(key, float('nan')) for perf in all_performances]\n                values = [v for v in values if not np.isnan(v)]\n                \n                if len(values) > 1:\n                    reproducibility_results[key] = {\n                        'mean': np.mean(values),\n                        'std': np.std(values),\n                        'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),\n                        'n_runs': len(values)\n                    }\n        \n        return reproducibility_results