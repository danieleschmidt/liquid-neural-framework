"""
Logging and monitoring utilities for liquid neural networks.

This module provides structured logging, metrics collection, and monitoring
capabilities for training and inference.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from datetime import datetime


class LiquidNetworkLogger:
    """
    Structured logger for liquid neural network experiments.
    
    Provides hierarchical logging with experiment tracking, metrics collection,
    and automatic file management.
    """
    
    def __init__(
        self,
        name: str = "liquid_network",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        experiment_id: Optional[str] = None
    ):
        self.name = name
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up logger
        self.logger = logging.getLogger(f"{name}_{self.experiment_id}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.metrics_history = []
        self.timers = {}
        
        self.info(f"Initialized logger for experiment {self.experiment_id}")
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            msg = f"{msg} | {json.dumps(kwargs, default=str)}"
        self.logger.debug(msg)
    
    def info(self, msg: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            msg = f"{msg} | {json.dumps(kwargs, default=str)}"
        self.logger.info(msg)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            msg = f"{msg} | {json.dumps(kwargs, default=str)}"
        self.logger.warning(msg)
    
    def error(self, msg: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            msg = f"{msg} | {json.dumps(kwargs, default=str)}"
        self.logger.error(msg)
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = time.time()
        self.debug(f"Started timer: {name}")
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name not in self.timers:
            self.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        self.debug(f"Timer '{name}' finished", elapsed_time=elapsed)
        return elapsed
    
    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "train",
        step: Optional[int] = None
    ):
        """
        Log training/validation metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric name -> value
            phase: Training phase ("train", "val", "test")
            step: Optional step number within epoch
        """
        timestamp = datetime.now().isoformat()
        
        metrics_entry = {
            "timestamp": timestamp,
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics
        }
        
        if step is not None:
            metrics_entry["step"] = step
        
        self.metrics_history.append(metrics_entry)
        
        # Log key metrics
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch} ({phase}): {metrics_str}")
    
    def log_model_info(
        self,
        model_name: str,
        architecture: Dict[str, Any],
        num_parameters: int
    ):
        """Log model architecture information."""
        self.info(
            f"Model initialized: {model_name}",
            architecture=architecture,
            num_parameters=num_parameters
        )
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.info("Training configuration", config=config)
    
    def log_stability_check(
        self,
        epoch: int,
        stability_results: Dict[str, bool],
        outputs_stats: Optional[Dict[str, float]] = None,
        states_stats: Optional[Dict[str, float]] = None
    ):
        """
        Log numerical stability check results.
        
        Args:
            epoch: Current epoch
            stability_results: Results from stability checking
            outputs_stats: Statistics about model outputs
            states_stats: Statistics about hidden states
        """
        if not stability_results.get("overall_stable", True):
            self.warning(
                f"Stability issues detected at epoch {epoch}",
                stability=stability_results,
                outputs_stats=outputs_stats,
                states_stats=states_stats
            )
        else:
            self.debug(
                f"Stability check passed at epoch {epoch}",
                stability=stability_results
            )
    
    def log_adaptation_info(
        self,
        epoch: int,
        adaptation_stats: Dict[str, Any]
    ):
        """Log adaptive parameter statistics."""
        self.debug(
            f"Adaptation info at epoch {epoch}",
            adaptation=adaptation_stats
        )
    
    def save_metrics(self, filepath: str):
        """Save metrics history to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                "experiment_id": self.experiment_id,
                "metrics_history": self.metrics_history
            }, f, indent=2, default=str)
        
        self.info(f"Metrics saved to {filepath}")
    
    def get_latest_metrics(self, phase: str = "train") -> Optional[Dict[str, float]]:
        """Get the latest metrics for a given phase."""
        for entry in reversed(self.metrics_history):
            if entry["phase"] == phase:
                return entry["metrics"]
        return None
    
    def get_metric_history(self, metric_name: str, phase: str = "train") -> List[float]:
        """Get history of a specific metric."""
        values = []
        for entry in self.metrics_history:
            if entry["phase"] == phase and metric_name in entry["metrics"]:
                values.append(entry["metrics"][metric_name])
        return values


class PerformanceMonitor:
    """
    Monitor performance metrics and resource usage.
    """
    
    def __init__(self, logger: Optional[LiquidNetworkLogger] = None):
        self.logger = logger
        self.start_time = None
        self.peak_memory = 0
        self.operation_times = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        if self.logger:
            self.logger.debug("Performance monitoring started")
    
    def record_operation_time(self, operation: str, duration: float):
        """Record time for a specific operation."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)
        
        if self.logger:
            self.logger.debug(f"Operation '{operation}' took {duration:.4f}s")
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        if operation not in self.operation_times:
            return {}
        
        times = self.operation_times[operation]
        return {
            "count": len(times),
            "total_time": sum(times),
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": min(times),
            "max_time": max(times)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "total_runtime": total_time,
            "peak_memory_mb": self.peak_memory,
            "operation_stats": {}
        }
        
        for operation in self.operation_times:
            summary["operation_stats"][operation] = self.get_operation_stats(operation)
        
        return summary


class MetricsCollector:
    """
    Collect and compute various metrics for liquid neural networks.
    """
    
    @staticmethod
    def compute_array_stats(array: jnp.ndarray, name: str = "array") -> Dict[str, float]:
        """Compute comprehensive statistics for an array."""
        return {
            f"{name}_mean": float(jnp.mean(array)),
            f"{name}_std": float(jnp.std(array)),
            f"{name}_min": float(jnp.min(array)),
            f"{name}_max": float(jnp.max(array)),
            f"{name}_norm": float(jnp.linalg.norm(array)),
            f"{name}_nan_count": int(jnp.sum(jnp.isnan(array))),
            f"{name}_inf_count": int(jnp.sum(jnp.isinf(array)))
        }
    
    @staticmethod
    def compute_prediction_metrics(
        predictions: jnp.ndarray,
        targets: jnp.ndarray
    ) -> Dict[str, float]:
        """Compute prediction quality metrics."""
        mse = float(jnp.mean((predictions - targets) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - targets)))
        
        # R-squared
        ss_res = jnp.sum((targets - predictions) ** 2)
        ss_tot = jnp.sum((targets - jnp.mean(targets)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        # Correlation
        corr = float(jnp.corrcoef(predictions.flatten(), targets.flatten())[0, 1])
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": float(jnp.sqrt(mse)),
            "r2_score": r2,
            "correlation": corr if not jnp.isnan(corr) else 0.0
        }
    
    @staticmethod
    def compute_stability_metrics(
        hidden_states: jnp.ndarray,
        outputs: jnp.ndarray
    ) -> Dict[str, float]:
        """Compute numerical stability metrics."""
        metrics = {}
        
        # Hidden state metrics
        metrics.update(MetricsCollector.compute_array_stats(hidden_states, "hidden"))
        
        # Output metrics
        metrics.update(MetricsCollector.compute_array_stats(outputs, "output"))
        
        # Gradient estimates (finite differences)
        if hidden_states.shape[0] > 1:
            hidden_grad_norm = float(jnp.mean(jnp.linalg.norm(
                jnp.diff(hidden_states, axis=0), axis=-1
            )))
            metrics["hidden_gradient_norm"] = hidden_grad_norm
        
        if outputs.shape[0] > 1:
            output_grad_norm = float(jnp.mean(jnp.linalg.norm(
                jnp.diff(outputs, axis=0), axis=-1
            )))
            metrics["output_gradient_norm"] = output_grad_norm
        
        return metrics
    
    @staticmethod
    def compute_adaptation_metrics(
        adaptation_info: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """Compute metrics for adaptive parameters."""
        metrics = {}
        
        for param_name, values in adaptation_info.items():
            if isinstance(values, jnp.ndarray):
                metrics.update(MetricsCollector.compute_array_stats(values, param_name))
        
        return metrics


# Global logger instance (can be overridden)
_global_logger = None

def get_logger() -> LiquidNetworkLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = LiquidNetworkLogger()
    return _global_logger

def set_global_logger(logger: LiquidNetworkLogger):
    """Set global logger instance."""
    global _global_logger
    _global_logger = logger