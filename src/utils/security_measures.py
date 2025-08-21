"""
Security measures and input sanitization for liquid neural networks.

This module provides security utilities to prevent malicious inputs,
ensure data privacy, and implement secure training practices.
"""

import numpy as np
import hashlib
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when security violations are detected."""
    pass


class InputSanitizer:
    """
    Sanitizes and validates inputs to prevent security vulnerabilities.
    """
    
    def __init__(self, max_input_size: int = 1000000, 
                 max_sequence_length: int = 10000,
                 allowed_dtypes: List[str] = None):
        """
        Initialize input sanitizer.
        
        Args:
            max_input_size: Maximum allowed input tensor size
            max_sequence_length: Maximum sequence length
            allowed_dtypes: List of allowed data types
        """
        self.max_input_size = max_input_size
        self.max_sequence_length = max_sequence_length
        
        if allowed_dtypes is None:
            self.allowed_dtypes = ['float32', 'float64', 'int32', 'int64']
        else:
            self.allowed_dtypes = allowed_dtypes
    
    def sanitize_tensor(self, tensor: np.ndarray, name: str = "input") -> np.ndarray:
        """
        Sanitize input tensor for security and stability.
        
        Args:
            tensor: Input tensor to sanitize
            name: Name of tensor for error reporting
            
        Returns:
            Sanitized tensor
            
        Raises:
            SecurityError: If tensor violates security constraints
        """
        # Check data type
        if str(tensor.dtype) not in self.allowed_dtypes:
            raise SecurityError(
                f"{name} has disallowed dtype {tensor.dtype}. "
                f"Allowed: {self.allowed_dtypes}"
            )
        
        # Check tensor size to prevent memory attacks
        if tensor.size > self.max_input_size:
            raise SecurityError(
                f"{name} size {tensor.size} exceeds maximum {self.max_input_size}"
            )
        
        # Check for malicious patterns
        self._check_malicious_patterns(tensor, name)
        
        # Sanitize values
        sanitized = self._sanitize_values(tensor)
        
        logger.debug(f"Successfully sanitized tensor {name} with shape {tensor.shape}")
        return sanitized
    
    def _check_malicious_patterns(self, tensor: np.ndarray, name: str):
        """Check for patterns that might indicate malicious inputs."""
        
        # Check for extremely large values that could cause overflow
        max_abs_value = np.max(np.abs(tensor))
        if max_abs_value > 1e10:
            warnings.warn(
                f"{name} contains very large values (max: {max_abs_value:.2e}). "
                "This might indicate a potential attack vector."
            )
        
        # Check for patterns that could exploit numerical instabilities
        if np.any(np.isnan(tensor)):
            raise SecurityError(f"{name} contains NaN values which could exploit vulnerabilities")
        
        if np.any(np.isinf(tensor)):
            raise SecurityError(f"{name} contains infinite values which could exploit vulnerabilities")
        
        # Check for adversarial pattern indicators
        gradient_magnitude = np.mean(np.abs(np.gradient(tensor.flatten())))
        if gradient_magnitude > 1000:
            warnings.warn(
                f"{name} has high gradient magnitude ({gradient_magnitude:.2f}), "
                "which might indicate adversarial perturbations"
            )
    
    def _sanitize_values(self, tensor: np.ndarray) -> np.ndarray:
        """Sanitize tensor values for security."""
        # Clip extreme values
        sanitized = np.clip(tensor, -1e6, 1e6)
        
        # Replace any remaining NaN/inf with safe values
        sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return sanitized
    
    def validate_sequence_length(self, sequence_length: int, name: str = "sequence"):
        """Validate sequence length to prevent DoS attacks."""
        if sequence_length > self.max_sequence_length:
            raise SecurityError(
                f"{name} length {sequence_length} exceeds maximum {self.max_sequence_length}"
            )


class DataPrivacyManager:
    """
    Manages data privacy and implements differential privacy measures.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize privacy manager.
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Probability of privacy failure
        """
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
    
    def add_gaussian_noise(self, tensor: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Add calibrated Gaussian noise for differential privacy.
        
        Args:
            tensor: Input tensor
            sensitivity: L2 sensitivity of the computation
            
        Returns:
            Tensor with added noise
        """
        # Calculate noise scale based on differential privacy theory
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        # Add Gaussian noise
        noise = np.random.normal(0, sigma, tensor.shape)
        noisy_tensor = tensor + noise
        
        # Update privacy budget
        self.privacy_budget_used += self.epsilon
        
        logger.debug(f"Added Gaussian noise with σ={sigma:.6f}, budget used: {self.privacy_budget_used:.6f}")
        
        return noisy_tensor
    
    def clip_gradients_for_privacy(self, gradients: Dict[str, np.ndarray], 
                                  clip_norm: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Clip gradients to bounded sensitivity for differential privacy.
        
        Args:
            gradients: Dictionary of parameter gradients
            clip_norm: Maximum allowed gradient norm
            
        Returns:
            Clipped gradients
        """
        clipped_gradients = {}
        
        for name, grad in gradients.items():
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm > clip_norm:
                # Clip gradient
                clipped_grad = grad * (clip_norm / grad_norm)
                logger.debug(f"Clipped gradient {name}: {grad_norm:.6f} -> {clip_norm}")
            else:
                clipped_grad = grad.copy()
            
            clipped_gradients[name] = clipped_grad
        
        return clipped_gradients
    
    def check_privacy_budget(self) -> Dict[str, float]:
        """Check remaining privacy budget."""
        remaining_budget = max(0, self.epsilon - self.privacy_budget_used)
        
        return {
            'total_budget': self.epsilon,
            'used_budget': self.privacy_budget_used,
            'remaining_budget': remaining_budget,
            'budget_exhausted': remaining_budget <= 0
        }


class SecureModelManager:
    """
    Manages secure model loading, saving, and integrity verification.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize secure model manager.
        
        Args:
            encryption_key: Optional encryption key for model files
        """
        self.encryption_key = encryption_key
        self.model_hashes = {}
    
    def compute_model_hash(self, model_dict: Dict[str, Any]) -> str:
        """
        Compute cryptographic hash of model parameters.
        
        Args:
            model_dict: Dictionary containing model parameters
            
        Returns:
            SHA-256 hash of model
        """
        # Serialize model parameters deterministically
        serialized = json.dumps(self._serialize_for_hash(model_dict), sort_keys=True)
        
        # Compute hash
        hash_obj = hashlib.sha256(serialized.encode('utf-8'))
        model_hash = hash_obj.hexdigest()
        
        return model_hash
    
    def _serialize_for_hash(self, obj: Any) -> Any:
        """Serialize object for consistent hashing."""
        if isinstance(obj, np.ndarray):
            return {
                'type': 'ndarray',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        elif isinstance(obj, dict):
            return {k: self._serialize_for_hash(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_hash(item) for item in obj]
        else:
            return obj
    
    def save_model_securely(self, model_dict: Dict[str, Any], filepath: str, 
                           include_hash: bool = True) -> str:
        """
        Save model with integrity verification.
        
        Args:
            model_dict: Model parameters to save
            filepath: Path to save model
            include_hash: Whether to include integrity hash
            
        Returns:
            Model hash for verification
        """
        # Compute model hash
        model_hash = self.compute_model_hash(model_dict)
        
        # Prepare save data
        save_data = {
            'model': model_dict,
            'timestamp': time.time(),
            'version': '1.0'
        }
        
        if include_hash:
            save_data['integrity_hash'] = model_hash
        
        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=self._json_serializer)
        
        # Store hash for later verification
        self.model_hashes[filepath] = model_hash
        
        logger.info(f"Model saved securely to {filepath} with hash {model_hash[:16]}...")
        
        return model_hash
    
    def load_model_securely(self, filepath: str, verify_hash: bool = True) -> Dict[str, Any]:
        """
        Load model with integrity verification.
        
        Args:
            filepath: Path to model file
            verify_hash: Whether to verify integrity hash
            
        Returns:
            Model parameters
            
        Raises:
            SecurityError: If integrity verification fails
        """
        if not os.path.exists(filepath):
            raise SecurityError(f"Model file not found: {filepath}")
        
        # Load data
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        model_dict = save_data.get('model', {})
        stored_hash = save_data.get('integrity_hash')
        
        if verify_hash and stored_hash:
            # Verify integrity
            computed_hash = self.compute_model_hash(model_dict)
            
            if computed_hash != stored_hash:
                raise SecurityError(
                    f"Model integrity verification failed for {filepath}. "
                    f"Expected: {stored_hash[:16]}..., Got: {computed_hash[:16]}..."
                )
            
            logger.info(f"Model integrity verified for {filepath}")
        
        return model_dict
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class AccessController:
    """
    Controls access to sensitive operations and resources.
    """
    
    def __init__(self, max_operations_per_hour: int = 1000,
                 sensitive_operations: List[str] = None):
        """
        Initialize access controller.
        
        Args:
            max_operations_per_hour: Rate limit for operations
            sensitive_operations: List of operations requiring special access
        """
        self.max_operations_per_hour = max_operations_per_hour
        
        if sensitive_operations is None:
            self.sensitive_operations = [
                'model_save', 'model_load', 'gradient_update', 'parameter_access'
            ]
        else:
            self.sensitive_operations = sensitive_operations
        
        self.operation_history = []
        self.blocked_operations = set()
    
    def check_access(self, operation: str, user_id: str = "default") -> bool:
        """
        Check if operation is allowed.
        
        Args:
            operation: Operation to check
            user_id: User identifier
            
        Returns:
            True if operation is allowed
            
        Raises:
            SecurityError: If operation is blocked
        """
        current_time = time.time()
        
        # Clean old entries (older than 1 hour)
        hour_ago = current_time - 3600
        self.operation_history = [
            entry for entry in self.operation_history 
            if entry['timestamp'] > hour_ago
        ]
        
        # Check rate limiting
        recent_operations = len([
            entry for entry in self.operation_history
            if entry['user_id'] == user_id
        ])
        
        if recent_operations >= self.max_operations_per_hour:
            raise SecurityError(
                f"Rate limit exceeded for user {user_id}: "
                f"{recent_operations} operations in the last hour"
            )
        
        # Check if operation is blocked
        operation_key = f"{user_id}:{operation}"
        if operation_key in self.blocked_operations:
            raise SecurityError(f"Operation {operation} is blocked for user {user_id}")
        
        # Log operation
        self.operation_history.append({
            'operation': operation,
            'user_id': user_id,
            'timestamp': current_time
        })
        
        if operation in self.sensitive_operations:
            logger.warning(f"Sensitive operation {operation} accessed by {user_id}")
        
        return True
    
    def block_operation(self, operation: str, user_id: str = "default"):
        """Block specific operation for a user."""
        operation_key = f"{user_id}:{operation}"
        self.blocked_operations.add(operation_key)
        logger.warning(f"Blocked operation {operation} for user {user_id}")
    
    def unblock_operation(self, operation: str, user_id: str = "default"):
        """Unblock specific operation for a user."""
        operation_key = f"{user_id}:{operation}"
        self.blocked_operations.discard(operation_key)
        logger.info(f"Unblocked operation {operation} for user {user_id}")


class SecurityAuditor:
    """
    Audits security events and maintains security logs.
    """
    
    def __init__(self, log_file: str = "security_audit.log"):
        """
        Initialize security auditor.
        
        Args:
            log_file: Path to security log file
        """
        self.log_file = log_file
        self.security_events = []
        
        # Setup security logger
        self.security_logger = logging.getLogger("security_audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
        self.security_logger.setLevel(logging.INFO)
    
    def log_security_event(self, event_type: str, description: str, 
                          severity: str = "INFO", user_id: str = "unknown",
                          additional_data: Optional[Dict[str, Any]] = None):
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            description: Description of the event
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            user_id: User associated with event
            additional_data: Additional event data
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'user_id': user_id,
            'additional_data': additional_data or {}
        }
        
        self.security_events.append(event)
        
        # Log to file
        log_message = f"{event_type} - {description} (User: {user_id})"
        
        if severity == "INFO":
            self.security_logger.info(log_message)
        elif severity == "WARNING":
            self.security_logger.warning(log_message)
        elif severity == "ERROR":
            self.security_logger.error(log_message)
        elif severity == "CRITICAL":
            self.security_logger.critical(log_message)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of security events in the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [
            event for event in self.security_events
            if event['timestamp'] > cutoff_time
        ]
        
        # Count events by type and severity
        event_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            event_type = event['event_type']
            severity = event['severity']
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'time_window_hours': hours,
            'total_events': len(recent_events),
            'event_type_counts': event_counts,
            'severity_counts': severity_counts,
            'critical_events': [
                event for event in recent_events 
                if event['severity'] == 'CRITICAL'
            ]
        }


# Global security components
_sanitizer = None
_privacy_manager = None
_access_controller = None
_security_auditor = None


def get_sanitizer() -> InputSanitizer:
    """Get global input sanitizer instance."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer()
    return _sanitizer


def get_privacy_manager() -> DataPrivacyManager:
    """Get global privacy manager instance."""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = DataPrivacyManager()
    return _privacy_manager


def get_access_controller() -> AccessController:
    """Get global access controller instance."""
    global _access_controller
    if _access_controller is None:
        _access_controller = AccessController()
    return _access_controller


def get_security_auditor() -> SecurityAuditor:
    """Get global security auditor instance."""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = SecurityAuditor()
    return _security_auditor


def secure_operation(operation_name: str):
    """Decorator for securing sensitive operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check access
            access_controller = get_access_controller()
            auditor = get_security_auditor()
            
            try:
                access_controller.check_access(operation_name)
                auditor.log_security_event(
                    "OPERATION_ACCESS", 
                    f"Accessed operation: {operation_name}",
                    "INFO"
                )
                
                result = func(*args, **kwargs)
                
                auditor.log_security_event(
                    "OPERATION_SUCCESS", 
                    f"Successfully completed: {operation_name}",
                    "INFO"
                )
                
                return result
                
            except SecurityError as e:
                auditor.log_security_event(
                    "SECURITY_VIOLATION", 
                    f"Security violation in {operation_name}: {str(e)}",
                    "ERROR"
                )
                raise
            except Exception as e:
                auditor.log_security_event(
                    "OPERATION_ERROR", 
                    f"Error in {operation_name}: {str(e)}",
                    "WARNING"
                )
                raise
        
        return wrapper
    return decorator