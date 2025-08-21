"""
Enhanced logging system for liquid neural networks.

This module provides comprehensive logging with performance monitoring,
debugging capabilities, and integration with external monitoring systems.
"""

import logging
import time
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import numpy as np
from contextlib import contextmanager


class NetworkLogger:
    """
    Advanced logger for neural network operations with performance tracking.
    """
    
    def __init__(self, name: str = "liquid_neural_framework", 
                 log_level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 enable_performance_tracking: bool = True,
                 enable_memory_tracking: bool = True):
        """
        Initialize enhanced logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional log file path
            enable_performance_tracking: Enable performance metrics
            enable_memory_tracking: Enable memory usage tracking
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with formatter
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_memory_tracking = enable_memory_tracking
        
        # Performance metrics storage
        self.performance_metrics = {}
        self.operation_counts = {}
        self.error_counts = {}
        
        # Memory tracking
        self.peak_memory_usage = 0
        
        self.logger.info(f"Enhanced logger initialized for {name}")
    
    def log_network_creation(self, network_type: str, input_size: int, 
                           hidden_size: int, output_size: int, **kwargs):
        """Log network creation with architecture details."""
        self.logger.info(
            f"Creating {network_type} network: "
            f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}"
        )
        
        for key, value in kwargs.items():
            self.logger.debug(f"Network parameter {key}: {value}")
    
    def log_training_start(self, epochs: int, batch_size: int, learning_rate: float):
        """Log training start with parameters."""
        self.logger.info(
            f"Starting training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}"
        )
        self.performance_metrics['training_start_time'] = time.time()
    
    def log_epoch(self, epoch: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log epoch results."""
        msg = f"Epoch {epoch}: loss={loss:.6f}"
        
        if metrics:
            metric_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            msg += f", {metric_str}"
        
        self.logger.info(msg)
    
    def log_gradient_info(self, gradients: Dict[str, np.ndarray]):
        """Log gradient statistics for debugging."""
        for name, grad in gradients.items():
            grad_norm = np.linalg.norm(grad)
            grad_max = np.max(np.abs(grad))
            grad_mean = np.mean(np.abs(grad))
            
            self.logger.debug(
                f"Gradient {name}: norm={grad_norm:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}"
            )
            
            # Warn about potential issues
            if grad_norm > 100:
                self.logger.warning(f"Large gradient norm in {name}: {grad_norm:.2f}")
            elif grad_norm < 1e-8:
                self.logger.warning(f"Very small gradient norm in {name}: {grad_norm:.2e}")
    
    def log_weight_statistics(self, weights: Dict[str, np.ndarray]):
        """Log weight statistics for monitoring."""
        for name, weight in weights.items():
            weight_norm = np.linalg.norm(weight)
            weight_max = np.max(np.abs(weight))
            weight_mean = np.mean(np.abs(weight))
            weight_std = np.std(weight)
            
            self.logger.debug(
                f"Weight {name}: norm={weight_norm:.6f}, max={weight_max:.6f}, "
                f"mean={weight_mean:.6f}, std={weight_std:.6f}"
            )
    
    def log_activation_statistics(self, activations: Dict[str, np.ndarray]):
        """Log activation statistics for monitoring."""
        for name, activation in activations.items():
            act_mean = np.mean(activation)
            act_std = np.std(activation)
            act_max = np.max(activation)
            act_min = np.min(activation)
            
            # Check for dead neurons
            dead_neurons = np.sum(np.abs(activation) < 1e-6) / activation.size
            
            self.logger.debug(
                f"Activation {name}: mean={act_mean:.6f}, std={act_std:.6f}, "
                f"range=[{act_min:.6f}, {act_max:.6f}], dead_neurons={dead_neurons:.2%}"
            )
            
            if dead_neurons > 0.5:
                self.logger.warning(f"High dead neuron ratio in {name}: {dead_neurons:.2%}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context information."""
        self.error_counts[type(error).__name__] = self.error_counts.get(type(error).__name__, 0) + 1
        
        error_msg = f"Error in {context}: {type(error).__name__}: {str(error)}"
        self.logger.error(error_msg)
        
        # Log error frequency
        total_errors = sum(self.error_counts.values())
        self.logger.debug(f"Total errors so far: {total_errors}")
    
    def log_performance_metric(self, operation: str, duration: float, 
                              additional_metrics: Optional[Dict[str, Any]] = None):
        """Log performance metrics for operations."""
        if not self.enable_performance_tracking:
            return
        
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        metric_entry = {
            'duration': duration,
            'timestamp': time.time()
        }
        
        if additional_metrics:
            metric_entry.update(additional_metrics)
        
        self.performance_metrics[operation].append(metric_entry)
        
        # Update operation counts
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        
        # Log if operation is slow
        if duration > 1.0:  # More than 1 second
            self.logger.warning(f"Slow operation {operation}: {duration:.3f}s")
        else:
            self.logger.debug(f"Operation {operation}: {duration:.3f}s")
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        if not self.enable_memory_tracking:
            return
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)
            
            if context:
                self.logger.debug(f"Memory usage {context}: {memory_mb:.2f} MB")
            else:
                self.logger.debug(f"Memory usage: {memory_mb:.2f} MB")
            
            # Warn about high memory usage
            if memory_mb > 1000:  # More than 1GB
                self.logger.warning(f"High memory usage: {memory_mb:.2f} MB")
                
        except ImportError:
            # psutil not available
            pass
    
    @contextmanager
    def log_operation(self, operation_name: str, log_memory: bool = False):
        """Context manager for logging operations with timing."""
        start_time = time.time()
        
        if log_memory:
            self.log_memory_usage(f"before {operation_name}")
        
        self.logger.debug(f"Starting operation: {operation_name}")
        
        try:
            yield
            duration = time.time() - start_time
            self.log_performance_metric(operation_name, duration)
            self.logger.debug(f"Completed operation: {operation_name} in {duration:.3f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_error(e, operation_name)
            self.logger.error(f"Failed operation: {operation_name} after {duration:.3f}s")
            raise
        
        finally:
            if log_memory:
                self.log_memory_usage(f"after {operation_name}")
    
    def log_research_experiment(self, experiment_name: str, parameters: Dict[str, Any],
                              results: Dict[str, float]):
        """Log research experiment results."""
        self.logger.info(f"Research Experiment: {experiment_name}")
        
        # Log parameters
        param_str = json.dumps(parameters, indent=2)
        self.logger.info(f"Parameters:\n{param_str}")
        
        # Log results
        results_str = ", ".join([f"{k}={v:.6f}" for k, v in results.items()])
        self.logger.info(f"Results: {results_str}")
        
        # Save to file for later analysis
        experiment_log = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'parameters': parameters,
            'results': results
        }
        
        self._save_experiment_log(experiment_log)
    
    def _save_experiment_log(self, experiment_log: Dict[str, Any]):
        """Save experiment log to JSON file."""
        log_dir = Path("logs/experiments")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"experiment_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        summary = {
            'operation_counts': self.operation_counts.copy(),
            'error_counts': self.error_counts.copy(),
            'peak_memory_mb': self.peak_memory_usage,
            'performance_stats': {}
        }
        
        # Calculate performance statistics
        for operation, metrics in self.performance_metrics.items():
            durations = [m['duration'] for m in metrics]
            summary['performance_stats'][operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'mean_time': np.mean(durations),
                'std_time': np.std(durations),
                'min_time': min(durations),
                'max_time': max(durations)
            }
        
        return summary
    
    def save_performance_report(self, filename: str = None):
        """Save detailed performance report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = self.get_performance_summary()
        report['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {filename}")


class DebugLogger(NetworkLogger):
    """
    Extended logger with advanced debugging capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_checkpoints = []
        self.tensor_history = {}
    
    def log_tensor_checkpoint(self, name: str, tensor: np.ndarray, 
                            step: int = None, save_tensor: bool = False):
        """Log tensor state at checkpoint for debugging."""
        stats = {
            'name': name,
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'mean': float(np.mean(tensor)),
            'std': float(np.std(tensor)),
            'min': float(np.min(tensor)),
            'max': float(np.max(tensor)),
            'norm': float(np.linalg.norm(tensor)),
            'step': step,
            'timestamp': time.time()
        }
        
        if save_tensor:
            stats['tensor_data'] = tensor.tolist()
        
        checkpoint = {
            'checkpoint_id': len(self.debug_checkpoints),
            'tensor_stats': stats
        }
        
        self.debug_checkpoints.append(checkpoint)
        
        # Maintain tensor history
        if name not in self.tensor_history:
            self.tensor_history[name] = []
        self.tensor_history[name].append(stats)
        
        self.logger.debug(
            f"Checkpoint {checkpoint['checkpoint_id']}: {name} "
            f"shape={tensor.shape}, mean={stats['mean']:.6f}, "
            f"std={stats['std']:.6f}, norm={stats['norm']:.6f}"
        )
    
    def analyze_tensor_evolution(self, tensor_name: str) -> Dict[str, Any]:
        """Analyze how a tensor evolved over time."""
        if tensor_name not in self.tensor_history:
            return {'error': f'No history found for tensor {tensor_name}'}
        
        history = self.tensor_history[tensor_name]
        
        analysis = {
            'tensor_name': tensor_name,
            'num_checkpoints': len(history),
            'evolution': {
                'mean_trend': [h['mean'] for h in history],
                'std_trend': [h['std'] for h in history],
                'norm_trend': [h['norm'] for h in history]
            }
        }
        
        # Detect trends
        if len(history) > 1:
            mean_values = np.array([h['mean'] for h in history])
            norm_values = np.array([h['norm'] for h in history])
            
            # Check for explosion/vanishing
            if np.any(norm_values > 1000):
                analysis['warnings'] = analysis.get('warnings', [])
                analysis['warnings'].append('Potential gradient explosion detected')
            
            if np.any(norm_values < 1e-8):
                analysis['warnings'] = analysis.get('warnings', [])
                analysis['warnings'].append('Potential vanishing gradients detected')
            
            # Check for NaN
            if np.any(np.isnan(mean_values)):
                analysis['warnings'] = analysis.get('warnings', [])
                analysis['warnings'].append('NaN values detected')
        
        return analysis
    
    def save_debug_report(self, filename: str = None):
        """Save comprehensive debug report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'checkpoints': self.debug_checkpoints,
            'tensor_history': self.tensor_history,
            'tensor_analyses': {}
        }
        
        # Add analyses for all tracked tensors
        for tensor_name in self.tensor_history.keys():
            report['tensor_analyses'][tensor_name] = self.analyze_tensor_evolution(tensor_name)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Debug report saved to {filename}")


# Global logger instance
_global_logger = None

def get_logger(name: str = "liquid_neural_framework") -> NetworkLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = NetworkLogger(name)
    return _global_logger


def setup_logging(log_level: int = logging.INFO, log_file: str = None, 
                 enable_debug: bool = False) -> Union[NetworkLogger, DebugLogger]:
    """Setup global logging configuration."""
    global _global_logger
    
    if enable_debug:
        _global_logger = DebugLogger(
            "liquid_neural_framework", 
            log_level=log_level, 
            log_file=log_file
        )
    else:
        _global_logger = NetworkLogger(
            "liquid_neural_framework", 
            log_level=log_level, 
            log_file=log_file
        )
    
    return _global_logger