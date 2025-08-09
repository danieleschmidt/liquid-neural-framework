"""
Security utilities for liquid neural networks.

This module provides security features including input sanitization,
resource monitoring, and protection against adversarial inputs.
"""

import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Optional, Tuple, List
import warnings
from ..utils.logging import get_logger


class SecurityMonitor:
    """
    Monitor for security-related issues in neural network operations.
    """
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.logger = get_logger()
        self.suspicious_patterns = []
        
    def check_input_anomalies(
        self,
        inputs: jnp.ndarray,
        threshold_std_devs: float = 5.0
    ) -> Dict[str, Any]:
        """
        Check for anomalous input patterns that might indicate adversarial attacks.
        
        Args:
            inputs: Input data to check
            threshold_std_devs: Number of standard deviations for anomaly detection
            
        Returns:
            Dictionary of anomaly detection results
        """
        if not self.enable_monitoring:
            return {"anomalies_detected": False}
        
        results = {"anomalies_detected": False, "anomaly_types": []}
        
        # Check for extreme values
        input_mean = jnp.mean(inputs)
        input_std = jnp.std(inputs)
        extreme_values = jnp.abs(inputs - input_mean) > (threshold_std_devs * input_std)
        
        if jnp.any(extreme_values):
            results["anomalies_detected"] = True
            results["anomaly_types"].append("extreme_values")
            results["extreme_value_count"] = int(jnp.sum(extreme_values))
        
        # Check for suspicious patterns (e.g., all zeros, all same value)
        if jnp.all(inputs == inputs.flat[0]):
            results["anomalies_detected"] = True
            results["anomaly_types"].append("constant_input")
        
        # Check for gradient-like patterns (potential adversarial perturbations)
        if inputs.ndim >= 2:
            gradients = jnp.diff(inputs, axis=0)
            if jnp.std(gradients) > 10 * jnp.std(inputs):
                results["anomalies_detected"] = True
                results["anomaly_types"].append("high_frequency_noise")
        
        # Log if anomalies detected
        if results["anomalies_detected"]:
            self.logger.warning("Input anomalies detected", anomalies=results)
            self.suspicious_patterns.append({
                "input_shape": inputs.shape,
                "anomalies": results,
                "input_stats": {
                    "mean": float(input_mean),
                    "std": float(input_std),
                    "min": float(jnp.min(inputs)),
                    "max": float(jnp.max(inputs))
                }
            })
        
        return results
    
    def sanitize_inputs(
        self,
        inputs: jnp.ndarray,
        clip_range: Optional[Tuple[float, float]] = None,
        noise_scale: float = 0.0
    ) -> jnp.ndarray:
        """
        Sanitize inputs by clipping and optionally adding noise.
        
        Args:
            inputs: Input data to sanitize
            clip_range: Range to clip inputs (min, max)
            noise_scale: Scale of noise to add for robustness
            
        Returns:
            Sanitized inputs
        """
        sanitized = inputs
        
        # Clip extreme values
        if clip_range is not None:
            min_val, max_val = clip_range
            sanitized = jnp.clip(sanitized, min_val, max_val)
            
            if jnp.any((inputs < min_val) | (inputs > max_val)):
                self.logger.debug("Input values clipped for security")
        
        # Add defensive noise
        if noise_scale > 0.0:
            key = random.PRNGKey(42)  # Fixed seed for reproducibility
            noise = random.normal(key, sanitized.shape) * noise_scale
            sanitized = sanitized + noise
            self.logger.debug(f"Added defensive noise with scale {noise_scale}")
        
        # Replace any remaining NaN/Inf values
        if jnp.any(~jnp.isfinite(sanitized)):
            finite_mean = jnp.mean(jnp.where(jnp.isfinite(sanitized), sanitized, 0.0))
            sanitized = jnp.where(jnp.isfinite(sanitized), sanitized, finite_mean)
            self.logger.warning("Replaced non-finite input values")
        
        return sanitized
    
    def check_output_integrity(
        self,
        outputs: jnp.ndarray,
        expected_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Check output integrity for unexpected behaviors.
        
        Args:
            outputs: Model outputs to check
            expected_range: Expected range of outputs
            
        Returns:
            Integrity check results
        """
        results = {"integrity_ok": True, "issues": []}
        
        # Check for non-finite values
        if jnp.any(~jnp.isfinite(outputs)):
            results["integrity_ok"] = False
            results["issues"].append("non_finite_outputs")
            results["nan_count"] = int(jnp.sum(jnp.isnan(outputs)))
            results["inf_count"] = int(jnp.sum(jnp.isinf(outputs)))
        
        # Check expected range
        if expected_range is not None:
            min_val, max_val = expected_range
            out_of_range = (outputs < min_val) | (outputs > max_val)
            if jnp.any(out_of_range):
                results["integrity_ok"] = False
                results["issues"].append("out_of_range_outputs")
                results["out_of_range_count"] = int(jnp.sum(out_of_range))
        
        # Check for suspicious patterns in outputs
        if jnp.all(outputs == outputs.flat[0]):
            results["integrity_ok"] = False
            results["issues"].append("constant_outputs")
        
        if not results["integrity_ok"]:
            self.logger.warning("Output integrity issues detected", issues=results)
        
        return results
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security monitoring report."""
        return {
            "monitoring_enabled": self.enable_monitoring,
            "suspicious_patterns_detected": len(self.suspicious_patterns),
            "recent_patterns": self.suspicious_patterns[-5:] if self.suspicious_patterns else []
        }


class ResourceMonitor:
    """
    Monitor resource usage to prevent resource exhaustion attacks.
    """
    
    def __init__(self, max_sequence_length: int = 10000, max_batch_size: int = 1000):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.logger = get_logger()
    
    def check_resource_limits(
        self,
        inputs: jnp.ndarray,
        operation: str = "forward_pass"
    ) -> Dict[str, Any]:
        """
        Check if operation is within resource limits.
        
        Args:
            inputs: Input data to process
            operation: Type of operation being performed
            
        Returns:
            Resource check results
        """
        results = {"within_limits": True, "issues": []}
        
        # Check sequence length
        if inputs.ndim >= 2:
            seq_length = inputs.shape[0] if inputs.ndim == 2 else inputs.shape[1]
            if seq_length > self.max_sequence_length:
                results["within_limits"] = False
                results["issues"].append(f"sequence_length_exceeded_{seq_length}")
        
        # Check batch size  
        if inputs.ndim == 3:
            batch_size = inputs.shape[0]
            if batch_size > self.max_batch_size:
                results["within_limits"] = False
                results["issues"].append(f"batch_size_exceeded_{batch_size}")
        
        # Estimate memory usage
        estimated_memory_mb = inputs.nbytes / (1024 * 1024)
        if estimated_memory_mb > 1000:  # 1GB limit
            results["within_limits"] = False
            results["issues"].append(f"memory_usage_high_{estimated_memory_mb:.1f}MB")
        
        if not results["within_limits"]:
            self.logger.error(
                f"Resource limits exceeded for {operation}",
                resource_check=results,
                input_shape=inputs.shape
            )
        
        return results
    
    def safe_process_large_sequence(
        self,
        inputs: jnp.ndarray,
        process_fn: callable,
        chunk_size: int = 1000
    ) -> jnp.ndarray:
        """
        Safely process large sequences by chunking.
        
        Args:
            inputs: Large input sequence
            process_fn: Function to process each chunk
            chunk_size: Size of each chunk
            
        Returns:
            Processed outputs
        """
        if inputs.shape[0] <= chunk_size:
            return process_fn(inputs)
        
        self.logger.info(f"Processing large sequence in chunks of {chunk_size}")
        
        chunks = []
        for i in range(0, inputs.shape[0], chunk_size):
            end_idx = min(i + chunk_size, inputs.shape[0])
            chunk = inputs[i:end_idx]
            
            chunk_output = process_fn(chunk)
            chunks.append(chunk_output)
        
        return jnp.concatenate(chunks, axis=0)


def create_secure_model_wrapper(model_class):
    """
    Create a secure wrapper around a model class that adds security monitoring.
    
    Args:
        model_class: The model class to wrap
        
    Returns:
        Wrapped model class with security features
    """
    
    class SecureModelWrapper:
        def __init__(self, *args, enable_security=True, **kwargs):
            self.model = model_class(*args, **kwargs)
            self.security_monitor = SecurityMonitor(enable_monitoring=enable_security)
            self.resource_monitor = ResourceMonitor()
            
        def __call__(self, inputs, *args, **kwargs):
            # Security checks
            anomaly_check = self.security_monitor.check_input_anomalies(inputs)
            if anomaly_check["anomalies_detected"]:
                warnings.warn("Input anomalies detected - proceeding with caution")
            
            # Resource checks
            resource_check = self.resource_monitor.check_resource_limits(inputs)
            if not resource_check["within_limits"]:
                raise RuntimeError(f"Resource limits exceeded: {resource_check['issues']}")
            
            # Sanitize inputs
            sanitized_inputs = self.security_monitor.sanitize_inputs(
                inputs, clip_range=(-100.0, 100.0), noise_scale=0.001
            )
            
            # Call original model
            outputs = self.model(sanitized_inputs, *args, **kwargs)
            
            # Check output integrity
            if isinstance(outputs, tuple):
                primary_output = outputs[0]
            else:
                primary_output = outputs
                
            integrity_check = self.security_monitor.check_output_integrity(primary_output)
            if not integrity_check["integrity_ok"]:
                warnings.warn("Output integrity issues detected")
            
            return outputs
        
        def __getattr__(self, name):
            # Delegate other attributes to the wrapped model
            return getattr(self.model, name)
        
        def get_security_report(self):
            """Get security monitoring report."""
            return self.security_monitor.get_security_report()
    
    return SecureModelWrapper


def validate_model_checkpoints(checkpoint_path: str) -> Dict[str, Any]:
    """
    Validate model checkpoints for security issues.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Validation results
    """
    logger = get_logger()
    results = {"valid": True, "issues": []}
    
    try:
        # Basic file checks would go here
        # For now, just log the validation attempt
        logger.info(f"Validating checkpoint: {checkpoint_path}")
        
        # In a real implementation, you would:
        # 1. Check file integrity (checksums)
        # 2. Scan for suspicious patterns in weights
        # 3. Validate parameter ranges
        # 4. Check for potential backdoors
        
        results["validation_timestamp"] = "placeholder"
        
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"validation_error: {str(e)}")
        logger.error(f"Checkpoint validation failed: {str(e)}")
    
    return results