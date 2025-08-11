"""
Enhanced logging utilities for liquid neural framework models.
"""

import logging
import sys
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class ModelPerformanceLogger:
    """Performance and training logger for liquid neural networks."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "liquid_experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
        # Set up main logger
        self.logger = self._setup_logger()
        
        # Metrics storage
        self.metrics_history = []
        self.current_epoch_metrics = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger(f"liquid_neural.{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def log_model_init(self, model_type: str, config: Dict[str, Any]):
        """Log model initialization."""
        self.logger.info(f"Initializing {model_type}")
        self.logger.info(f"Model configuration: {json.dumps(config, indent=2)}")
        
    def log_training_start(self, total_epochs: int, dataset_size: int):
        """Log training start."""
        self.logger.info(f"Starting training for {total_epochs} epochs")
        self.logger.info(f"Dataset size: {dataset_size} samples")
        
    def log_epoch_start(self, epoch: int):
        """Log epoch start."""
        self.current_epoch_metrics = {
            'epoch': epoch,
            'start_time': time.time()
        }
        self.logger.info(f"Starting epoch {epoch}")
        
    def log_batch_metrics(self, batch_idx: int, metrics: Dict[str, float]):
        """Log batch-level metrics."""
        if batch_idx % 100 == 0:  # Log every 100 batches
            metric_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            self.logger.info(f"Batch {batch_idx} - {metric_str}")\n            \n    def log_epoch_end(self, metrics: Dict[str, float]):\n        \"\"\"Log epoch end with metrics.\"\"\"\n        epoch_time = time.time() - self.current_epoch_metrics['start_time']\n        \n        self.current_epoch_metrics.update({\n            'end_time': time.time(),\n            'duration': epoch_time,\n            **metrics\n        })\n        \n        self.metrics_history.append(self.current_epoch_metrics.copy())\n        \n        metric_str = \", \".join([f\"{k}: {v:.6f}\" for k, v in metrics.items()])\n        self.logger.info(\n            f\"Epoch {self.current_epoch_metrics['epoch']} completed in {epoch_time:.2f}s - {metric_str}\"\n        )\n        \n    def log_stability_check(self, spectral_radius: float, tau_stats: Dict[str, float]):\n        \"\"\"Log network stability metrics.\"\"\"\n        self.logger.info(f\"Network stability - Spectral radius: {spectral_radius:.6f}\")\n        self.logger.info(f\"Time constants - Min: {tau_stats.get('min', 0):.6f}, \"\n                        f\"Max: {tau_stats.get('max', 0):.6f}, \"\n                        f\"Mean: {tau_stats.get('mean', 0):.6f}\")\n        \n        if spectral_radius > 1.0:\n            self.logger.warning(\"Network may be unstable (spectral radius > 1.0)\")\n            \n    def log_numerical_issues(self, issue_type: str, details: str):\n        \"\"\"Log numerical stability issues.\"\"\"\n        self.logger.warning(f\"Numerical issue detected - {issue_type}: {details}\")\n        \n    def log_performance_metrics(self, \n                               forward_time: float, \n                               backward_time: float, \n                               memory_usage: Optional[float] = None):\n        \"\"\"Log performance metrics.\"\"\"\n        self.logger.info(f\"Performance - Forward: {forward_time:.6f}s, \"\n                        f\"Backward: {backward_time:.6f}s\")\n        if memory_usage:\n            self.logger.info(f\"Memory usage: {memory_usage:.2f} MB\")\n            \n    def log_validation_results(self, val_metrics: Dict[str, float]):\n        \"\"\"Log validation results.\"\"\"\n        metric_str = \", \".join([f\"{k}: {v:.6f}\" for k, v in val_metrics.items()])\n        self.logger.info(f\"Validation results - {metric_str}\")\n        \n    def log_experiment_summary(self, final_metrics: Dict[str, float]):\n        \"\"\"Log experiment summary.\"\"\"\n        total_time = time.time() - self.start_time\n        \n        self.logger.info(\"=\" * 50)\n        self.logger.info(\"EXPERIMENT SUMMARY\")\n        self.logger.info(f\"Total time: {total_time:.2f}s ({total_time/60:.2f} min)\")\n        \n        for metric, value in final_metrics.items():\n            self.logger.info(f\"Final {metric}: {value:.6f}\")\n            \n        self.logger.info(\"=\" * 50)\n        \n    def save_metrics_history(self):\n        \"\"\"Save metrics history to JSON file.\"\"\"\n        metrics_file = self.log_dir / f\"{self.experiment_name}_metrics.json\"\n        with open(metrics_file, 'w') as f:\n            json.dump(self.metrics_history, f, indent=2)\n            \n        self.logger.info(f\"Metrics saved to {metrics_file}\")\n        \n    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min'):\n        \"\"\"Get epoch with best performance.\"\"\"\n        if not self.metrics_history:\n            return None\n            \n        if mode == 'min':\n            best_epoch = min(self.metrics_history, \n                           key=lambda x: x.get(metric, float('inf')))\n        else:\n            best_epoch = max(self.metrics_history, \n                           key=lambda x: x.get(metric, float('-inf')))\n                           \n        return best_epoch\n        \n    def log_error(self, error: Exception, context: str = \"\"):\n        \"\"\"Log errors with context.\"\"\"\n        self.logger.error(f\"Error in {context}: {type(error).__name__}: {str(error)}\")\n        \n    def log_warning(self, message: str):\n        \"\"\"Log warning message.\"\"\"\n        self.logger.warning(message)\n        \n    def log_info(self, message: str):\n        \"\"\"Log info message.\"\"\"\n        self.logger.info(message)\n\n\nclass LiquidNetworkDebugger:\n    \"\"\"Debugging utilities for liquid neural networks.\"\"\"\n    \n    def __init__(self, logger: ModelPerformanceLogger):\n        self.logger = logger\n        \n    def check_gradient_flow(self, gradients: Dict[str, Any]):\n        \"\"\"Check gradient flow and log issues.\"\"\"\n        try:\n            import jax.numpy as jnp\n            \n            for name, grad in gradients.items():\n                if grad is not None:\n                    grad_norm = float(jnp.linalg.norm(grad))\n                    \n                    if grad_norm == 0:\n                        self.logger.log_warning(f\"Zero gradients detected in {name}\")\n                    elif grad_norm > 10.0:\n                        self.logger.log_warning(f\"Large gradients detected in {name}: {grad_norm:.6f}\")\n                    elif jnp.any(jnp.isnan(grad)):\n                        self.logger.log_numerical_issues(\"NaN gradients\", f\"Parameter: {name}\")\n                    elif jnp.any(jnp.isinf(grad)):\n                        self.logger.log_numerical_issues(\"Inf gradients\", f\"Parameter: {name}\")\n                        \n        except ImportError:\n            self.logger.log_warning(\"JAX not available for gradient checking\")\n            \n    def log_network_state(self, network, inputs, hidden_state=None):\n        \"\"\"Log detailed network state information.\"\"\"\n        try:\n            import jax.numpy as jnp\n            \n            # Log input statistics\n            self.logger.log_info(f\"Input shape: {inputs.shape}\")\n            self.logger.log_info(f\"Input stats - Min: {float(jnp.min(inputs)):.6f}, \"\n                               f\"Max: {float(jnp.max(inputs)):.6f}, \"\n                               f\"Mean: {float(jnp.mean(inputs)):.6f}\")\n            \n            if hidden_state is not None:\n                self.logger.log_info(f\"Hidden state shape: {hidden_state.shape}\")\n                self.logger.log_info(f\"Hidden stats - Min: {float(jnp.min(hidden_state)):.6f}, \"\n                                   f\"Max: {float(jnp.max(hidden_state)):.6f}, \"\n                                   f\"Mean: {float(jnp.mean(hidden_state)):.6f}\")\n                                   \n        except Exception as e:\n            self.logger.log_error(e, \"network state logging\")"