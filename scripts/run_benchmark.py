#!/usr/bin/env python3
"""
Benchmark execution script for liquid neural networks.

This script provides a command-line interface for running comprehensive
benchmarks with customizable configurations and output options.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import framework components
try:
    from experiments.benchmark_suite import BenchmarkSuite
    from experiments.validation_experiments import ValidationExperiments
    from experiments.synthetic_tasks import SyntheticTaskGenerator
    from utils.visualization import create_comprehensive_visualization_report
    from utils.metrics import create_performance_report
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class BenchmarkRunner:
    """
    High-level benchmark runner with performance optimizations.
    """
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        config_file: Optional[str] = None,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize components
        if IMPORTS_AVAILABLE:
            self.benchmark_suite = BenchmarkSuite(
                output_dir=str(self.output_dir),
                seed=self.config.get('seed', 42),
                config_file=config_file
            )
            
            self.validation_experiments = ValidationExperiments(
                seed=self.config.get('seed', 42)
            )
            
            self.task_generator = SyntheticTaskGenerator(
                seed=self.config.get('seed', 42)
            )
        else:
            if self.verbose:
                print(f"Warning: Framework imports not available: {IMPORT_ERROR}")
            
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load benchmark configuration."""
        default_config = {
            'seed': 42,
            'experiments': {
                'run_baseline_comparison': True,
                'run_ablation_studies': True,
                'run_scalability_analysis': True,
                'run_statistical_analysis': True,
                'n_runs': 3,
                'epochs': 100
            },
            'visualization': {
                'create_plots': True,
                'save_animations': False,
                'plot_format': 'png'
            },
            'output': {
                'save_detailed_results': True,
                'save_summary_only': False,
                'create_visualization_report': True,
                'create_performance_report': True
            },
            'performance': {
                'parallel_execution': False,
                'memory_limit_gb': 8,
                'timeout_minutes': 60
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            
            # Merge configurations
            def merge_configs(default, user):
                for key, value in user.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        merge_configs(default[key], value)
                    else:
                        default[key] = value
            
            merge_configs(default_config, user_config)
        
        return default_config
    
    def run_quick_benchmark(self) -> Dict[str, Any]:
        """Run a quick benchmark for testing purposes."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Framework imports not available"}
        
        if self.verbose:
            print("Running quick benchmark...")
        
        start_time = time.time()
        
        # Generate minimal task set
        tasks = {
            'simple_sine': self._generate_simple_sine_task(),
            'memory_test': self._generate_simple_memory_task()
        }
        
        # Run basic comparison
        results = {}
        for task_name, (inputs, targets) in tasks.items():
            if self.verbose:
                print(f"  Processing {task_name}...")
            
            # Determine dimensions
            input_size = inputs.shape[-1] if inputs.ndim > 1 else 1
            output_size = targets.shape[-1] if targets.ndim > 1 else 1
            
            # Create simple models
            models = self.validation_experiments.create_baseline_models(
                input_size, output_size
            )
            
            # Quick training (reduced epochs)
            task_results = self.validation_experiments.run_single_task_experiment(
                task_name, inputs, targets, models,
                epochs=20,  # Reduced for speed
                train_split=0.8
            )
            
            results[task_name] = task_results
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"Quick benchmark completed in {elapsed_time:.2f} seconds")
        
        return {
            'results': results,
            'runtime_seconds': elapsed_time,
            'benchmark_type': 'quick'
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Framework imports not available"}
        
        if self.verbose:
            print("Running full benchmark suite...")
        
        start_time = time.time()
        
        # Run complete benchmark
        results = self.benchmark_suite.run_full_benchmark_suite()
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"Full benchmark completed in {elapsed_time:.2f} seconds")
        
        return {
            'results': results,
            'runtime_seconds': elapsed_time,
            'benchmark_type': 'full'
        }
    
    def run_custom_benchmark(
        self, 
        tasks: Optional[Dict[str, Any]] = None,
        models: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run custom benchmark with user-specified tasks and models."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Framework imports not available"}
        
        if self.verbose:
            print("Running custom benchmark...")
        
        # Implementation would go here
        # For now, fall back to quick benchmark
        return self.run_quick_benchmark()
    
    def _generate_simple_sine_task(self):
        """Generate simple sine wave prediction task."""
        import numpy as np
        
        t = np.linspace(0, 4*np.pi, 100)
        inputs = np.sin(t).reshape(-1, 1)
        targets = np.sin(t + 0.1).reshape(-1, 1)
        
        return inputs, targets
    
    def _generate_simple_memory_task(self):
        """Generate simple memory task."""
        import numpy as np
        
        np.random.seed(self.config['seed'])
        
        # Simple sequence where output depends on input from 5 steps ago
        seq_len = 50
        inputs = np.random.randn(seq_len, 1)
        targets = np.zeros((seq_len, 1))
        
        for i in range(5, seq_len):
            targets[i] = inputs[i-5]
        
        return inputs, targets
    
    def create_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization report."""
        if not IMPORTS_AVAILABLE:
            return {}
        
        if self.verbose:
            print("Creating visualizations...")
        
        viz_output_dir = self.output_dir / "visualizations"
        viz_output_dir.mkdir(exist_ok=True)
        
        try:
            saved_plots = create_comprehensive_visualization_report(
                results, str(viz_output_dir)
            )
            
            if self.verbose:
                print(f"Created {len(saved_plots)} visualization plots")
            
            return saved_plots
            
        except Exception as e:
            if self.verbose:
                print(f"Visualization creation failed: {e}")
            return {}
    
    def create_performance_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed performance analysis."""
        if not IMPORTS_AVAILABLE:
            return {}
        
        if self.verbose:
            print("Creating performance analysis...")
        
        perf_output_dir = self.output_dir / "performance_analysis"
        perf_output_dir.mkdir(exist_ok=True)
        
        try:
            # Extract predictions and targets from results
            # This would need to be adapted based on actual result structure
            
            # For now, return placeholder
            analysis = {
                'message': 'Performance analysis placeholder',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save analysis
            with open(perf_output_dir / "analysis_summary.json", 'w') as f:
                json.dump(analysis, f, indent=2)
            
            return analysis
            
        except Exception as e:
            if self.verbose:
                print(f"Performance analysis creation failed: {e}")
            return {}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive text report."""
        if not IMPORTS_AVAILABLE:
            return "Framework imports not available - cannot generate report"
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "LIQUID NEURAL NETWORK BENCHMARK REPORT",
            "=" * 50,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output Directory: {self.output_dir}",
            f"Configuration: {self.config.get('seed', 'default')}",
            ""
        ])
        
        # Results summary
        if 'results' in results:
            benchmark_results = results['results']
            
            if isinstance(benchmark_results, dict):
                report_lines.extend([
                    "BENCHMARK RESULTS SUMMARY",
                    "-" * 30,
                    f"Number of experiments: {len(benchmark_results)}",
                    f"Benchmark type: {results.get('benchmark_type', 'unknown')}",
                    f"Total runtime: {results.get('runtime_seconds', 0):.2f} seconds",
                    ""
                ])
                
                # Individual experiment results
                for experiment_name, experiment_results in benchmark_results.items():
                    report_lines.extend([
                        f"Experiment: {experiment_name}",
                        "-" * (len(experiment_name) + 12)
                    ])
                    
                    if isinstance(experiment_results, dict):
                        for model_name, model_results in experiment_results.items():
                            if isinstance(model_results, dict):
                                final_loss = model_results.get('final_train_loss', 'N/A')
                                val_loss = model_results.get('final_val_loss', 'N/A')
                                train_time = model_results.get('training_time', 'N/A')
                                
                                report_lines.extend([
                                    f"  {model_name}:",
                                    f"    Training Loss: {final_loss}",
                                    f"    Validation Loss: {val_loss}",
                                    f"    Training Time: {train_time}s",
                                ])
                    
                    report_lines.append("")
        
        # Configuration details
        report_lines.extend([
            "CONFIGURATION DETAILS",
            "-" * 25
        ])
        
        def format_config(config, indent=0):
            lines = []
            for key, value in config.items():
                if isinstance(value, dict):
                    lines.append("  " * indent + f"{key}:")
                    lines.extend(format_config(value, indent + 1))
                else:
                    lines.append("  " * indent + f"{key}: {value}")
            return lines
        
        report_lines.extend(format_config(self.config))
        
        # Footer
        report_lines.extend([
            "",
            "=" * 50,
            "Report generated by Liquid Neural Framework",
            "For more details, see individual result files in the output directory."
        ])
        
        return "\\n".join(report_lines)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run liquid neural network benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --mode quick
  python run_benchmark.py --mode full --config my_config.json
  python run_benchmark.py --mode custom --output results_dir
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['quick', 'full', 'custom'],
        default='quick',
        help='Benchmark mode to run'
    )
    
    parser.add_argument(
        '--output',
        default='benchmark_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate report only (skip benchmarks)'
    )
    
    args = parser.parse_args()
    
    # Check framework availability
    if not IMPORTS_AVAILABLE:
        print(f"Error: Framework components not available: {IMPORT_ERROR}")
        print("Please ensure all dependencies are installed:")
        print("  pip install jax jaxlib numpy matplotlib seaborn scipy")
        sys.exit(1)
    
    # Initialize runner
    runner = BenchmarkRunner(
        output_dir=args.output,
        config_file=args.config,
        verbose=args.verbose
    )
    
    # Override seed if provided
    if args.seed != 42:
        runner.config['seed'] = args.seed
    
    if args.verbose:
        print(f"Liquid Neural Network Benchmark Runner")
        print(f"Mode: {args.mode}")
        print(f"Output directory: {args.output}")
        print(f"Random seed: {runner.config['seed']}")
        print()
    
    if not args.report_only:
        # Run benchmark
        if args.mode == 'quick':
            results = runner.run_quick_benchmark()
        elif args.mode == 'full':
            results = runner.run_full_benchmark()
        elif args.mode == 'custom':
            results = runner.run_custom_benchmark()
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Save raw results
        results_file = runner.output_dir / f"raw_results_{args.mode}.json"
        with open(results_file, 'w') as f:
            def convert_for_json(obj):
                if hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif hasattr(obj, '__float__'):  # numpy scalar
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_for_json(results), f, indent=2)
        
        if args.verbose:
            print(f"Raw results saved to {results_file}")
        
        # Create visualizations
        if not args.no_visualizations:
            try:
                visualizations = runner.create_visualizations(results)
                if args.verbose and visualizations:
                    print(f"Visualizations saved: {list(visualizations.keys())}")
            except Exception as e:
                if args.verbose:
                    print(f"Visualization creation failed: {e}")
        
        # Create performance analysis
        try:
            performance_analysis = runner.create_performance_analysis(results)
            if args.verbose:
                print("Performance analysis created")
        except Exception as e:
            if args.verbose:
                print(f"Performance analysis failed: {e}")
    
    else:
        # Load existing results for report generation
        results_file = runner.output_dir / f"raw_results_{args.mode}.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            print(f"No existing results found at {results_file}")
            sys.exit(1)
    
    # Generate comprehensive report
    report = runner.generate_report(results)
    
    # Save report
    report_file = runner.output_dir / f"benchmark_report_{args.mode}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    if args.verbose:
        print(f"\\nBenchmark report saved to {report_file}")
        print("\\nReport Summary:")
        print("-" * 20)
        
        # Print first few lines of report
        report_lines = report.split('\\n')
        for line in report_lines[:15]:
            print(line)
        
        if len(report_lines) > 15:
            print("...")
            print(f"[Report continues for {len(report_lines) - 15} more lines]")
    
    print("\\nBenchmark execution completed successfully!")


if __name__ == "__main__":
    main()