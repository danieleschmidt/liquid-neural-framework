#!/usr/bin/env python3
"""
Performance profiling and optimization tools for liquid neural networks.

This script provides comprehensive performance analysis including:
- Memory usage profiling
- Computational bottleneck detection
- Scalability analysis
- Optimization recommendations
"""

import sys
import os
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from collections import defaultdict
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class PerformanceProfiler:
    """
    Comprehensive performance profiling for liquid neural networks.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.profile_data = {
            'timing': defaultdict(list),
            'memory': defaultdict(list),
            'operations': defaultdict(int),
            'bottlenecks': [],
            'recommendations': []
        }
        
        # System info
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def profile_function(
        self, 
        func: Callable,
        name: str,
        *args, 
        **kwargs
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Profile a single function execution.
        
        Returns:
            Tuple of (function_result, profile_metrics)
        """
        # Memory before
        if self.enable_memory_tracking:
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Force garbage collection for cleaner measurement
        gc.collect()
        
        # Time the function
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
        
        # Memory after
        if self.enable_memory_tracking:
            memory_after = process.memory_info().rss / (1024**2)  # MB
            memory_delta = memory_after - memory_before
        else:
            memory_before = memory_after = memory_delta = 0
        
        # Compile metrics
        metrics = {
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu_time - start_cpu_time,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta
        }
        
        # Store in profile data
        self.profile_data['timing'][name].append(metrics['wall_time'])
        self.profile_data['memory'][name].append(memory_delta)
        self.profile_data['operations'][name] += 1
        
        return result, metrics
    
    def profile_model_operations(self, model, input_data) -> Dict[str, Any]:
        """
        Profile key model operations.
        """
        if not HAS_NUMPY:
            return {"error": "NumPy required for profiling"}
        
        profiles = {}
        
        # Test if model has required methods
        if not hasattr(model, 'forward'):
            return {"error": "Model must have 'forward' method"}
        
        # Profile forward pass
        try:
            _, forward_metrics = self.profile_function(
                model.forward, 'forward_pass', input_data
            )
            profiles['forward_pass'] = forward_metrics
        except Exception as e:
            profiles['forward_pass'] = {"error": str(e)}
        
        # Profile parameter access
        if hasattr(model, 'params'):
            try:
                def get_params():
                    return model.params
                
                _, param_metrics = self.profile_function(
                    get_params, 'parameter_access'
                )
                profiles['parameter_access'] = param_metrics
            except Exception as e:
                profiles['parameter_access'] = {"error": str(e)}
        
        return profiles
    
    def profile_data_operations(self, data_size_range: List[int]) -> Dict[str, Any]:
        """
        Profile data-related operations across different sizes.
        """
        if not HAS_NUMPY:
            return {"error": "NumPy required for data profiling"}
        
        data_profiles = {}
        
        for size in data_size_range:
            size_profiles = {}
            
            # Array creation
            def create_array():
                return np.random.randn(size, 10)
            
            _, create_metrics = self.profile_function(
                create_array, f'create_array_{size}'
            )
            size_profiles['array_creation'] = create_metrics
            
            # Array operations
            test_array = np.random.randn(size, 10)
            
            def array_multiply():
                return test_array * 2.0
            
            _, multiply_metrics = self.profile_function(
                array_multiply, f'array_multiply_{size}'
            )
            size_profiles['array_multiply'] = multiply_metrics
            
            # Matrix operations
            if size <= 1000:  # Avoid memory issues
                test_matrix = np.random.randn(size, size)
                
                def matrix_multiply():
                    return test_matrix @ test_matrix.T
                
                try:
                    _, matmul_metrics = self.profile_function(
                        matrix_multiply, f'matrix_multiply_{size}'
                    )
                    size_profiles['matrix_multiply'] = matmul_metrics
                except Exception as e:
                    size_profiles['matrix_multiply'] = {"error": str(e)}
            
            data_profiles[f'size_{size}'] = size_profiles
        
        return data_profiles
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Analyze performance data to identify bottlenecks.
        """
        bottlenecks = []
        
        # Timing analysis
        for operation, times in self.profile_data['timing'].items():
            if len(times) > 1:
                avg_time = np.mean(times) if HAS_NUMPY else sum(times) / len(times)
                max_time = max(times)
                
                # Flag slow operations
                if avg_time > 1.0:  # > 1 second average
                    bottlenecks.append({
                        'type': 'slow_operation',
                        'operation': operation,
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'severity': 'high' if avg_time > 5.0 else 'medium'
                    })
                
                # Flag inconsistent performance
                if HAS_NUMPY:
                    std_time = np.std(times)
                    cv = std_time / avg_time if avg_time > 0 else 0
                    
                    if cv > 0.5:  # High coefficient of variation
                        bottlenecks.append({
                            'type': 'inconsistent_performance',
                            'operation': operation,
                            'coefficient_variation': cv,
                            'avg_time': avg_time,
                            'std_time': std_time,
                            'severity': 'medium'
                        })
        
        # Memory analysis
        for operation, memory_deltas in self.profile_data['memory'].items():
            if len(memory_deltas) > 1 and HAS_NUMPY:
                avg_memory = np.mean(memory_deltas)
                max_memory = max(memory_deltas)
                
                # Flag high memory operations
                if avg_memory > 100:  # > 100 MB average
                    bottlenecks.append({
                        'type': 'high_memory_usage',
                        'operation': operation,
                        'avg_memory_mb': avg_memory,
                        'max_memory_mb': max_memory,
                        'severity': 'high' if avg_memory > 500 else 'medium'
                    })
        
        self.profile_data['bottlenecks'] = bottlenecks
        return bottlenecks
    
    def generate_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate optimization recommendations based on profiling data.
        """
        recommendations = []
        
        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_operation':
                recommendations.append({
                    'category': 'performance',
                    'priority': bottleneck['severity'],
                    'issue': f"Slow operation: {bottleneck['operation']}",
                    'recommendation': "Consider optimizing computation or using vectorized operations",
                    'technical_details': f"Average time: {bottleneck['avg_time']:.3f}s"
                })
            
            elif bottleneck['type'] == 'high_memory_usage':
                recommendations.append({
                    'category': 'memory',
                    'priority': bottleneck['severity'],
                    'issue': f"High memory usage: {bottleneck['operation']}",
                    'recommendation': "Consider batch processing or memory-efficient algorithms",
                    'technical_details': f"Average memory: {bottleneck['avg_memory_mb']:.1f} MB"
                })
            
            elif bottleneck['type'] == 'inconsistent_performance':
                recommendations.append({
                    'category': 'stability',
                    'priority': bottleneck['severity'],
                    'issue': f"Inconsistent performance: {bottleneck['operation']}",
                    'recommendation': "Investigate external factors affecting performance",
                    'technical_details': f"Coefficient of variation: {bottleneck['coefficient_variation']:.3f}"
                })
        
        # General recommendations based on system info
        available_memory_gb = self.system_info['memory_available_gb']
        
        if available_memory_gb < 2:
            recommendations.append({
                'category': 'system',
                'priority': 'high',
                'issue': "Low available memory",
                'recommendation': "Consider reducing batch sizes or using memory-efficient models",
                'technical_details': f"Available memory: {available_memory_gb:.1f} GB"
            })
        
        cpu_count = self.system_info['cpu_count']
        if cpu_count > 1:
            recommendations.append({
                'category': 'parallelization',
                'priority': 'medium',
                'issue': "Multi-core system detected",
                'recommendation': "Consider using parallel processing for independent operations",
                'technical_details': f"Available cores: {cpu_count}"
            })
        
        self.profile_data['recommendations'] = recommendations
        return recommendations
    
    def create_performance_report(self) -> str:
        """
        Create comprehensive performance report.
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "LIQUID NEURAL NETWORK PERFORMANCE ANALYSIS",
            "=" * 50,
            f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # System Information
        report_lines.extend([
            "SYSTEM INFORMATION",
            "-" * 20,
            f"CPU Cores: {self.system_info['cpu_count']}",
            f"Total Memory: {self.system_info['memory_total_gb']:.1f} GB",
            f"Available Memory: {self.system_info['memory_available_gb']:.1f} GB",
            f"Platform: {self.system_info['platform']}",
            ""
        ])
        
        # Performance Summary
        if self.profile_data['timing']:
            report_lines.extend([
                "PERFORMANCE SUMMARY",
                "-" * 20
            ])
            
            for operation, times in self.profile_data['timing'].items():
                if times:
                    avg_time = np.mean(times) if HAS_NUMPY else sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    
                    report_lines.extend([
                        f"{operation}:",
                        f"  Average: {avg_time:.4f}s",
                        f"  Range: {min_time:.4f}s - {max_time:.4f}s",
                        f"  Executions: {len(times)}"
                    ])
            
            report_lines.append("")
        
        # Memory Usage
        if self.profile_data['memory']:
            report_lines.extend([
                "MEMORY USAGE ANALYSIS",
                "-" * 25
            ])
            
            for operation, memory_deltas in self.profile_data['memory'].items():
                if memory_deltas and HAS_NUMPY:
                    avg_memory = np.mean(memory_deltas)
                    max_memory = max(memory_deltas)
                    
                    report_lines.extend([
                        f"{operation}:",
                        f"  Average delta: {avg_memory:.1f} MB",
                        f"  Maximum delta: {max_memory:.1f} MB"
                    ])
            
            report_lines.append("")
        
        # Bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        if bottlenecks:
            report_lines.extend([
                "IDENTIFIED BOTTLENECKS",
                "-" * 22
            ])
            
            for i, bottleneck in enumerate(bottlenecks, 1):
                report_lines.extend([
                    f"{i}. {bottleneck['type'].replace('_', ' ').title()}",
                    f"   Operation: {bottleneck['operation']}",
                    f"   Severity: {bottleneck['severity']}",
                ])
                
                # Add specific details
                if 'avg_time' in bottleneck:
                    report_lines.append(f"   Average time: {bottleneck['avg_time']:.4f}s")
                if 'avg_memory_mb' in bottleneck:
                    report_lines.append(f"   Average memory: {bottleneck['avg_memory_mb']:.1f} MB")
                
                report_lines.append("")
        
        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            report_lines.extend([
                "OPTIMIZATION RECOMMENDATIONS",
                "-" * 30
            ])
            
            # Group by priority
            high_priority = [r for r in recommendations if r['priority'] == 'high']
            medium_priority = [r for r in recommendations if r['priority'] == 'medium']
            
            if high_priority:
                report_lines.extend(["HIGH PRIORITY:", ""])
                for i, rec in enumerate(high_priority, 1):
                    report_lines.extend([
                        f"{i}. {rec['issue']}",
                        f"   Recommendation: {rec['recommendation']}",
                        f"   Details: {rec['technical_details']}",
                        ""
                    ])
            
            if medium_priority:
                report_lines.extend(["MEDIUM PRIORITY:", ""])
                for i, rec in enumerate(medium_priority, 1):
                    report_lines.extend([
                        f"{i}. {rec['issue']}",
                        f"   Recommendation: {rec['recommendation']}",
                        f"   Details: {rec['technical_details']}",
                        ""
                    ])
        
        # Footer
        report_lines.extend([
            "=" * 50,
            "Performance analysis complete.",
            "For detailed optimization, consider profiling specific bottlenecks."
        ])
        
        return "\\n".join(report_lines)
    
    def export_profile_data(self, filepath: str):
        """Export profile data to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if HAS_NUMPY and isinstance(obj, np.ndarray):
                return obj.tolist()
            elif HAS_NUMPY and isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        export_data = {
            'system_info': self.system_info,
            'profile_data': convert_for_json(dict(self.profile_data)),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


class ScalabilityTester:
    """
    Test scalability across different problem sizes.
    """
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
    
    def test_sequence_length_scaling(
        self,
        model_factory: Callable,
        sequence_lengths: List[int],
        input_size: int = 10,
        n_trials: int = 3
    ) -> Dict[str, List[float]]:
        """
        Test how performance scales with sequence length.
        """
        results = {
            'sequence_lengths': sequence_lengths,
            'avg_times': [],
            'memory_usage': []
        }
        
        for seq_len in sequence_lengths:
            times = []
            memories = []
            
            for trial in range(n_trials):
                # Create model and data
                model = model_factory()
                
                if HAS_NUMPY:
                    test_data = np.random.randn(seq_len, input_size)
                else:
                    # Fallback without numpy
                    test_data = [[0.0] * input_size for _ in range(seq_len)]
                
                # Profile forward pass
                try:
                    _, metrics = self.profiler.profile_function(
                        model.forward, f'scaling_test_seq_{seq_len}_trial_{trial}',
                        test_data
                    )
                    
                    times.append(metrics['wall_time'])
                    memories.append(metrics['memory_delta_mb'])
                    
                except Exception as e:
                    print(f"Scaling test failed for seq_len {seq_len}, trial {trial}: {e}")
                    continue
            
            if times:
                avg_time = np.mean(times) if HAS_NUMPY else sum(times) / len(times)
                avg_memory = np.mean(memories) if HAS_NUMPY else sum(memories) / len(memories)
                
                results['avg_times'].append(avg_time)
                results['memory_usage'].append(avg_memory)
            else:
                results['avg_times'].append(0.0)
                results['memory_usage'].append(0.0)
        
        return results
    
    def test_batch_size_scaling(
        self,
        model_factory: Callable,
        batch_sizes: List[int],
        sequence_length: int = 100,
        input_size: int = 10,
        n_trials: int = 3
    ) -> Dict[str, List[float]]:
        """
        Test how performance scales with batch size.
        """
        results = {
            'batch_sizes': batch_sizes,
            'avg_times': [],
            'memory_usage': []
        }
        
        for batch_size in batch_sizes:
            times = []
            memories = []
            
            for trial in range(n_trials):
                # Create model
                model = model_factory()
                
                # Create batch data
                if HAS_NUMPY:
                    # For now, simulate batch by running multiple sequences
                    test_data = np.random.randn(sequence_length, input_size)
                else:
                    test_data = [[0.0] * input_size for _ in range(sequence_length)]
                
                # Profile batch processing (simulate)
                def batch_forward():
                    results = []
                    for _ in range(batch_size):
                        result = model.forward(test_data)
                        results.append(result)
                    return results
                
                try:
                    _, metrics = self.profiler.profile_function(
                        batch_forward, f'batch_test_size_{batch_size}_trial_{trial}'
                    )
                    
                    # Normalize by batch size
                    times.append(metrics['wall_time'] / batch_size)
                    memories.append(metrics['memory_delta_mb'])
                    
                except Exception as e:
                    print(f"Batch scaling test failed for size {batch_size}, trial {trial}: {e}")
                    continue
            
            if times:
                avg_time = np.mean(times) if HAS_NUMPY else sum(times) / len(times)
                avg_memory = np.mean(memories) if HAS_NUMPY else sum(memories) / len(memories)
                
                results['avg_times'].append(avg_time)
                results['memory_usage'].append(avg_memory)
            else:
                results['avg_times'].append(0.0)
                results['memory_usage'].append(0.0)
        
        return results


def create_demo_model():
    """Create a simple demo model for testing when real models aren't available."""
    class DemoModel:
        def __init__(self):
            self.params = {'weights': [1.0, 2.0, 3.0]}
        
        def forward(self, inputs):
            # Simple computation for demo
            if HAS_NUMPY:
                if isinstance(inputs, np.ndarray):
                    return np.mean(inputs, axis=1)
                else:
                    return np.array([sum(row)/len(row) for row in inputs])
            else:
                # Without numpy
                if isinstance(inputs[0], (list, tuple)):
                    return [sum(row)/len(row) for row in inputs]
                else:
                    return sum(inputs) / len(inputs)
    
    return DemoModel()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile liquid neural network performance")
    parser.add_argument('--output', default='performance_analysis', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run demo profiling')
    parser.add_argument('--scalability', action='store_true', help='Run scalability tests')
    parser.add_argument('--memory-tracking', action='store_true', default=True, help='Enable memory tracking')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize profiler
    profiler = PerformanceProfiler(enable_memory_tracking=args.memory_tracking)
    
    print("Liquid Neural Network Performance Profiler")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Memory tracking: {'Enabled' if args.memory_tracking else 'Disabled'}")
    print()
    
    if args.demo:
        print("Running demonstration profiling...")
        
        # Create demo model
        model = create_demo_model()
        
        # Generate test data
        if HAS_NUMPY:
            test_data = np.random.randn(100, 10)
            small_data = np.random.randn(10, 10)
            large_data = np.random.randn(1000, 10)
        else:
            test_data = [[0.1] * 10 for _ in range(100)]
            small_data = [[0.1] * 10 for _ in range(10)]
            large_data = [[0.1] * 10 for _ in range(1000)]
        
        # Profile model operations
        print("  Profiling model operations...")
        model_profiles = profiler.profile_model_operations(model, test_data)
        
        # Profile different data sizes
        print("  Profiling different data sizes...")
        _, small_metrics = profiler.profile_function(
            model.forward, 'small_forward', small_data
        )
        
        _, large_metrics = profiler.profile_function(
            model.forward, 'large_forward', large_data
        )
        
        # Profile data operations
        print("  Profiling data operations...")
        data_profiles = profiler.profile_data_operations([10, 100, 500])
    
    if args.scalability:
        print("Running scalability tests...")
        
        # Initialize scalability tester
        tester = ScalabilityTester(profiler)
        
        # Test sequence length scaling
        def model_factory():
            return create_demo_model()
        
        print("  Testing sequence length scaling...")
        seq_scaling = tester.test_sequence_length_scaling(
            model_factory, 
            sequence_lengths=[10, 50, 100, 200],
            n_trials=2  # Reduced for demo
        )
        
        print("  Testing batch size scaling...")
        batch_scaling = tester.test_batch_size_scaling(
            model_factory,
            batch_sizes=[1, 5, 10],
            n_trials=2
        )
    
    # Generate comprehensive report
    print("Generating performance report...")
    report = profiler.create_performance_report()
    
    # Save report
    report_file = output_dir / "performance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Export raw data
    data_file = output_dir / "profile_data.json"
    profiler.export_profile_data(str(data_file))
    
    print(f"\\nPerformance analysis complete!")
    print(f"Report saved to: {report_file}")
    print(f"Raw data saved to: {data_file}")
    print()
    
    # Show summary
    print("PERFORMANCE SUMMARY")
    print("-" * 20)
    
    # Show top timing results
    timing_data = profiler.profile_data['timing']
    if timing_data:
        for operation, times in list(timing_data.items())[:3]:
            if times:
                avg_time = np.mean(times) if HAS_NUMPY else sum(times) / len(times)
                print(f"{operation}: {avg_time:.4f}s average")
    
    # Show bottlenecks
    bottlenecks = profiler.analyze_bottlenecks()
    if bottlenecks:
        print(f"\\nIdentified {len(bottlenecks)} performance bottlenecks")
        for bottleneck in bottlenecks[:2]:  # Show top 2
            print(f"  - {bottleneck['type']}: {bottleneck['operation']}")
    
    # Show recommendations
    recommendations = profiler.generate_recommendations()
    high_priority = [r for r in recommendations if r['priority'] == 'high']
    if high_priority:
        print(f"\\n{len(high_priority)} high-priority optimization opportunities identified")
    
    print("\\nFor detailed analysis, see the generated report.")


if __name__ == "__main__":
    main()