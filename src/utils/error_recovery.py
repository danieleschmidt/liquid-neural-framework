"""
Error recovery and resilience mechanisms for liquid neural networks.
Includes automatic checkpoint saving, model restoration, and graceful degradation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pickle
import time
import os
from typing import Any, Dict, Optional, Callable, List
from pathlib import Path
import warnings
import logging


class ModelCheckpoint:
    """Automatic model checkpointing system."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 max_checkpoints: int = 5, save_frequency: int = 100):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_frequency = save_frequency
        self.step_counter = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, model: eqx.Module, step: int, 
                       metadata: Optional[Dict] = None) -> str:
        """Save model checkpoint with metadata."""
        timestamp = int(time.time())
        checkpoint_name = f"checkpoint_step_{step}_time_{timestamp}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            "model": model,
            "step": step,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
        
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoints found")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def auto_save(self, model: eqx.Module, step: int, 
                  metadata: Optional[Dict] = None) -> Optional[str]:
        """Automatically save checkpoint based on frequency."""
        self.step_counter += 1
        
        if self.step_counter % self.save_frequency == 0:
            return self.save_checkpoint(model, step, metadata)
        return None
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time and remove oldest
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            try:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")


class CircuitBreaker:
    """Circuit breaker pattern for model operations."""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise RuntimeError("Circuit breaker is OPEN - operation blocked")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker reset to CLOSED")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
            
            raise e


class GracefulDegradation:
    """Graceful degradation strategies for model failures."""
    
    def __init__(self):
        self.fallback_models = []
        self.logger = logging.getLogger(__name__)
    
    def add_fallback(self, model: eqx.Module, priority: int = 0):
        """Add fallback model with priority (lower = higher priority)."""
        self.fallback_models.append((priority, model))
        self.fallback_models.sort(key=lambda x: x[0])
    
    def safe_inference(self, primary_model: eqx.Module, 
                      inputs: Any, *args, **kwargs) -> Any:
        """Perform inference with fallback on failures."""
        # Try primary model
        try:
            result = primary_model(inputs, *args, **kwargs)
            return result
        except Exception as e:
            self.logger.warning(f"Primary model failed: {e}")
        
        # Try fallback models
        for priority, fallback_model in self.fallback_models:
            try:
                self.logger.info(f"Trying fallback model (priority {priority})")
                result = fallback_model(inputs, *args, **kwargs)
                self.logger.info("Fallback successful")
                return result
            except Exception as e:
                self.logger.warning(f"Fallback model failed: {e}")
                continue
        
        # All models failed
        raise RuntimeError("All models failed - no fallback available")


class ErrorRecoveryManager:
    """Comprehensive error recovery management."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpointer = ModelCheckpoint(checkpoint_dir)
        self.circuit_breaker = CircuitBreaker()
        self.graceful_degradation = GracefulDegradation()
        self.error_history = []
        self.recovery_strategies = []
        
        self.logger = logging.getLogger(__name__)
    
    def register_recovery_strategy(self, strategy: Callable):
        """Register custom recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    def handle_error(self, error: Exception, model: eqx.Module, 
                    step: int) -> Optional[eqx.Module]:
        """Handle error with recovery strategies."""
        error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "step": step,
            "timestamp": time.time()
        }
        self.error_history.append(error_info)
        
        self.logger.error(f"Error at step {step}: {error}")
        
        # Try recovery strategies
        for strategy in self.recovery_strategies:
            try:
                recovered_model = strategy(error, model, step)
                if recovered_model is not None:
                    self.logger.info(f"Recovery successful with strategy: {strategy.__name__}")
                    return recovered_model
            except Exception as strategy_error:
                self.logger.warning(f"Recovery strategy failed: {strategy_error}")
        
        # Default recovery: load latest checkpoint
        try:
            checkpoint = self.checkpointer.load_checkpoint()
            self.logger.info("Recovered from checkpoint")
            return checkpoint["model"]
        except Exception as checkpoint_error:
            self.logger.error(f"Checkpoint recovery failed: {checkpoint_error}")
        
        return None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [
            e for e in self.error_history 
            if time.time() - e["timestamp"] < 3600  # Last hour
        ]
        
        error_types = {}
        for error in self.error_history:
            error_type = error["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_types": error_types,
            "last_error": self.error_history[-1] if self.error_history else None
        }


def auto_recovery(func):
    """Decorator for automatic error recovery."""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_recovery_manager'):
            self._recovery_manager = ErrorRecoveryManager()
        
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'model'):
                recovered_model = self._recovery_manager.handle_error(
                    e, self.model, getattr(self, 'step', 0)
                )
                if recovered_model is not None:
                    self.model = recovered_model
                    return func(self, *args, **kwargs)
            raise e
    
    return wrapper


class RobustModelWrapper:
    """Complete robust wrapper with all recovery mechanisms."""
    
    def __init__(self, model: eqx.Module, checkpoint_dir: str = "./checkpoints"):
        self.model = model
        self.recovery_manager = ErrorRecoveryManager(checkpoint_dir)
        self.step = 0
        
        # Add simple fallback model
        try:
            # Create a simple linear fallback
            if hasattr(model, 'output_layer'):
                input_size = model.output_layer.in_features
                output_size = model.output_layer.out_features
                fallback = eqx.nn.Linear(input_size, output_size, key=jax.random.PRNGKey(42))
                self.recovery_manager.graceful_degradation.add_fallback(fallback, priority=1)
        except Exception:
            pass
    
    @auto_recovery
    def __call__(self, *args, **kwargs):
        """Robust forward pass with error recovery."""
        result = self.recovery_manager.circuit_breaker.call(
            self.model, *args, **kwargs
        )
        return result
    
    def save_checkpoint(self, metadata: Optional[Dict] = None):
        """Save checkpoint."""
        return self.recovery_manager.checkpointer.save_checkpoint(
            self.model, self.step, metadata
        )
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "step": self.step,
            "circuit_breaker_state": self.recovery_manager.circuit_breaker.state,
            "error_stats": self.recovery_manager.get_error_stats()
        }