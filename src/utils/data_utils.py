import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Union, Callable
import warnings
from pathlib import Path


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class DataGenerator:
    """
    Robust data generation utilities with comprehensive error handling.
    
    Provides safe data generation with validation, error recovery,
    and detailed logging of data characteristics.
    """
    
    def __init__(self, seed: int = 42, validate_data: bool = True):
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.validate_data = validate_data
        
        # Data validation thresholds
        self.validation_config = {
            'max_sequence_length': 10000,
            'max_feature_dimension': 1000,
            'max_batch_size': 10000,
            'min_sequence_length': 1,
            'finite_check': True,
            'nan_tolerance': 0.0,  # Fraction of NaN values allowed
            'inf_tolerance': 0.0   # Fraction of Inf values allowed
        }
    
    def validate_array(
        self, 
        array: jnp.ndarray, 
        name: str = "array",
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[jnp.dtype] = None
    ) -> jnp.ndarray:
        """
        Comprehensive array validation with detailed error reporting.
        """
        if not isinstance(array, (jnp.ndarray, np.ndarray)):
            raise DataValidationError(f"{name} must be a JAX or NumPy array, got {type(array)}")
        
        # Convert to JAX array if needed
        if isinstance(array, np.ndarray):
            array = jnp.array(array)
        
        # Check for empty array
        if array.size == 0:
            raise DataValidationError(f"{name} is empty")
        
        # Shape validation
        if expected_shape is not None and array.shape != expected_shape:
            raise DataValidationError(
                f"{name} has shape {array.shape}, expected {expected_shape}"
            )
        
        # Dimension constraints
        if array.ndim > 0 and array.shape[0] > self.validation_config['max_batch_size']:
            warnings.warn(
                f"{name} batch size {array.shape[0]} exceeds recommended maximum "
                f"{self.validation_config['max_batch_size']}"
            )
        
        if array.ndim > 1 and array.shape[1] > self.validation_config['max_sequence_length']:
            warnings.warn(
                f"{name} sequence length {array.shape[1]} exceeds recommended maximum "
                f"{self.validation_config['max_sequence_length']}"
            )
        
        # Finite value checks
        if self.validation_config['finite_check']:
            nan_count = jnp.sum(jnp.isnan(array))
            inf_count = jnp.sum(jnp.isinf(array))
            total_elements = array.size
            
            nan_fraction = float(nan_count / total_elements)
            inf_fraction = float(inf_count / total_elements)
            
            if nan_fraction > self.validation_config['nan_tolerance']:
                raise DataValidationError(
                    f"{name} contains {nan_fraction:.2%} NaN values "
                    f"(threshold: {self.validation_config['nan_tolerance']:.2%})"
                )
            
            if inf_fraction > self.validation_config['inf_tolerance']:
                raise DataValidationError(
                    f"{name} contains {inf_fraction:.2%} Inf values "
                    f"(threshold: {self.validation_config['inf_tolerance']:.2%})"
                )
        
        # Data type validation
        if expected_dtype is not None and array.dtype != expected_dtype:
            warnings.warn(
                f"{name} has dtype {array.dtype}, expected {expected_dtype}. "
                "Automatic conversion will be attempted."
            )
            try:
                array = array.astype(expected_dtype)
            except (ValueError, TypeError) as e:
                raise DataValidationError(f"Cannot convert {name} to {expected_dtype}: {e}")
        
        return array
    
    def safe_sequence_generation(
        self,
        generation_func: Callable,
        fallback_func: Optional[Callable] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Safely generate sequences with error recovery and retries.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self.key, subkey = random.split(self.key)
                result = generation_func(key=subkey, **kwargs)
                
                if isinstance(result, tuple) and len(result) == 2:
                    inputs, targets = result
                else:
                    raise DataValidationError("Generation function must return (inputs, targets) tuple")
                
                # Validate generated data
                inputs = self.validate_array(inputs, "generated_inputs")
                targets = self.validate_array(targets, "generated_targets")
                
                # Check alignment
                if inputs.shape[0] != targets.shape[0]:
                    raise DataValidationError(
                        f"Inputs batch size {inputs.shape[0]} doesn't match "
                        f"targets batch size {targets.shape[0]}"
                    )
                
                return inputs, targets
                
            except Exception as e:
                last_error = e
                warnings.warn(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Modify parameters for retry
                    if 'noise_level' in kwargs:
                        kwargs['noise_level'] *= 0.5  # Reduce noise
                    if 'seq_length' in kwargs and kwargs['seq_length'] > 50:
                        kwargs['seq_length'] = max(50, kwargs['seq_length'] // 2)  # Shorter sequences
        
        # All retries failed, try fallback
        if fallback_func is not None:
            try:
                warnings.warn("Using fallback generation method")
                return self.safe_sequence_generation(fallback_func, max_retries=1, **kwargs)
            except Exception as fallback_error:
                raise DataValidationError(
                    f"Both primary and fallback generation failed. "
                    f"Last error: {last_error}, Fallback error: {fallback_error}"
                )
        
        raise DataValidationError(f"Data generation failed after {max_retries} attempts: {last_error}")
    
    def robust_data_split(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        train_split: float = 0.8,
        validation_split: float = 0.1,
        shuffle: bool = True,
        stratify: bool = False
    ) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Robust data splitting with validation and error handling.
        """
        # Validate inputs
        inputs = self.validate_array(inputs, "inputs")
        targets = self.validate_array(targets, "targets")
        
        # Check split parameters
        if not 0 < train_split < 1:
            raise ConfigurationError(f"train_split must be between 0 and 1, got {train_split}")
        
        if not 0 <= validation_split < 1:
            raise ConfigurationError(f"validation_split must be between 0 and 1, got {validation_split}")
        
        if train_split + validation_split >= 1:
            raise ConfigurationError(
                f"train_split ({train_split}) + validation_split ({validation_split}) "
                "must be less than 1"
            )
        
        n_samples = inputs.shape[0]
        if n_samples < 3:
            raise DataValidationError(f"Need at least 3 samples for splitting, got {n_samples}")
        
        # Calculate split sizes
        n_train = max(1, int(n_samples * train_split))
        n_val = max(0, int(n_samples * validation_split))
        n_test = n_samples - n_train - n_val
        
        if n_test < 0:
            # Adjust splits to ensure non-negative test set
            n_test = max(1, n_samples - n_train - n_val)
            n_val = max(0, n_samples - n_train - n_test)
        
        # Generate indices
        if shuffle:
            self.key, subkey = random.split(self.key)
            indices = random.permutation(subkey, n_samples)
        else:
            indices = jnp.arange(n_samples)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val] if n_val > 0 else jnp.array([], dtype=jnp.int32)
        test_indices = indices[n_train + n_val:] if n_test > 0 else jnp.array([], dtype=jnp.int32)
        
        # Create splits
        splits = {}
        
        if len(train_indices) > 0:
            splits['train'] = (inputs[train_indices], targets[train_indices])
        
        if len(val_indices) > 0:
            splits['val'] = (inputs[val_indices], targets[val_indices])
        
        if len(test_indices) > 0:
            splits['test'] = (inputs[test_indices], targets[test_indices])
        
        # Validation of splits
        total_split_samples = sum(split[0].shape[0] for split in splits.values())
        if total_split_samples != n_samples:
            raise DataValidationError(
                f"Split samples ({total_split_samples}) don't match input samples ({n_samples})"
            )
        
        return splits
    
    def sanitize_data(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        clip_values: bool = True,
        fill_nan: Optional[float] = 0.0,
        fill_inf: Optional[float] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sanitize data by handling NaN/Inf values and clipping extreme values.
        """
        inputs = self.validate_array(inputs, "inputs")
        targets = self.validate_array(targets, "targets")
        
        # Handle NaN values
        if fill_nan is not None:
            nan_mask_inputs = jnp.isnan(inputs)
            nan_mask_targets = jnp.isnan(targets)
            
            if jnp.any(nan_mask_inputs):
                inputs = jnp.where(nan_mask_inputs, fill_nan, inputs)
                warnings.warn(f"Filled {jnp.sum(nan_mask_inputs)} NaN values in inputs with {fill_nan}")
            
            if jnp.any(nan_mask_targets):
                targets = jnp.where(nan_mask_targets, fill_nan, targets)
                warnings.warn(f"Filled {jnp.sum(nan_mask_targets)} NaN values in targets with {fill_nan}")
        
        # Handle Inf values
        if fill_inf is not None:
            inf_mask_inputs = jnp.isinf(inputs)
            inf_mask_targets = jnp.isinf(targets)
            
            if jnp.any(inf_mask_inputs):
                inputs = jnp.where(inf_mask_inputs, fill_inf, inputs)
                warnings.warn(f"Filled {jnp.sum(inf_mask_inputs)} Inf values in inputs with {fill_inf}")
            
            if jnp.any(inf_mask_targets):
                targets = jnp.where(inf_mask_targets, fill_inf, targets)
                warnings.warn(f"Filled {jnp.sum(inf_mask_targets)} Inf values in targets with {fill_inf}")
        
        # Clip extreme values
        if clip_values:
            # Determine reasonable clipping bounds based on data statistics
            input_percentiles = jnp.percentile(inputs[jnp.isfinite(inputs)], [1, 99])
            target_percentiles = jnp.percentile(targets[jnp.isfinite(targets)], [1, 99])
            
            # Expand bounds to avoid over-clipping
            input_range = input_percentiles[1] - input_percentiles[0]
            target_range = target_percentiles[1] - target_percentiles[0]
            
            input_bounds = [
                input_percentiles[0] - 0.5 * input_range,
                input_percentiles[1] + 0.5 * input_range
            ]
            target_bounds = [
                target_percentiles[0] - 0.5 * target_range,
                target_percentiles[1] + 0.5 * target_range
            ]
            
            inputs = jnp.clip(inputs, input_bounds[0], input_bounds[1])
            targets = jnp.clip(targets, target_bounds[0], target_bounds[1])
        
        return inputs, targets
    
    def get_data_statistics(self, data: jnp.ndarray, name: str = "data") -> Dict[str, Any]:
        """
        Comprehensive data statistics for monitoring and debugging.
        """
        data = self.validate_array(data, name)
        
        # Basic statistics
        finite_data = data[jnp.isfinite(data)]
        
        stats = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'total_elements': data.size,
            'finite_elements': finite_data.size,
            'nan_count': int(jnp.sum(jnp.isnan(data))),
            'inf_count': int(jnp.sum(jnp.isinf(data))),
            'mean': float(jnp.mean(finite_data)) if finite_data.size > 0 else float('nan'),
            'std': float(jnp.std(finite_data)) if finite_data.size > 0 else float('nan'),
            'min': float(jnp.min(finite_data)) if finite_data.size > 0 else float('nan'),
            'max': float(jnp.max(finite_data)) if finite_data.size > 0 else float('nan'),
            'median': float(jnp.median(finite_data)) if finite_data.size > 0 else float('nan')
        }
        
        # Percentiles
        if finite_data.size > 0:
            percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
            stats['percentiles'] = {
                f'p{p}': float(jnp.percentile(finite_data, p))
                for p in percentiles
            }
        
        # Data quality metrics
        stats['quality'] = {
            'completeness': float(finite_data.size / data.size),
            'has_negatives': bool(jnp.any(finite_data < 0)) if finite_data.size > 0 else False,
            'has_zeros': bool(jnp.any(finite_data == 0)) if finite_data.size > 0 else False,
            'dynamic_range': float(jnp.max(finite_data) - jnp.min(finite_data)) if finite_data.size > 0 else 0.0
        }
        
        return stats


class DataPreprocessor:
    """
    Robust data preprocessing with error handling and recovery.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.fitted_params = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default preprocessing configuration."""
        return {
            'normalization': {
                'method': 'standard',  # 'standard', 'minmax', 'robust'
                'feature_range': (-1, 1),
                'clip_outliers': True,
                'outlier_threshold': 3.0
            },
            'sequence_processing': {
                'max_length': 1000,
                'padding_value': 0.0,
                'truncation': 'post'  # 'pre', 'post'
            },
            'validation': {
                'check_finite': True,
                'handle_missing': 'interpolate',  # 'drop', 'fill', 'interpolate'
                'missing_fill_value': 0.0
            }
        }
    
    def robust_normalize(
        self,
        data: jnp.ndarray,
        method: Optional[str] = None,
        fit: bool = True,
        feature_axis: int = -1
    ) -> jnp.ndarray:
        """
        Robust normalization with error handling and parameter storage.
        """
        method = method or self.config['normalization']['method']
        
        try:
            if method == 'standard':
                return self._standard_normalize(data, fit, feature_axis)
            elif method == 'minmax':
                return self._minmax_normalize(data, fit, feature_axis)
            elif method == 'robust':
                return self._robust_normalize(data, fit, feature_axis)
            else:
                raise ConfigurationError(f"Unknown normalization method: {method}")
                
        except Exception as e:
            warnings.warn(f"Normalization failed, returning original data: {e}")
            return data
    
    def _standard_normalize(
        self, 
        data: jnp.ndarray, 
        fit: bool, 
        feature_axis: int
    ) -> jnp.ndarray:
        """Standard (z-score) normalization."""
        if fit:
            # Compute statistics on finite values only
            finite_mask = jnp.isfinite(data)
            if not jnp.any(finite_mask):
                warnings.warn("No finite values found for normalization")
                return data
            
            finite_data = data[finite_mask]
            mean = jnp.mean(finite_data, axis=feature_axis, keepdims=True)
            std = jnp.std(finite_data, axis=feature_axis, keepdims=True)
            
            # Prevent division by zero
            std = jnp.maximum(std, 1e-8)
            
            self.fitted_params['mean'] = mean
            self.fitted_params['std'] = std
        else:
            if 'mean' not in self.fitted_params or 'std' not in self.fitted_params:
                raise ConfigurationError("Must fit before transform or provide fitted parameters")
            mean = self.fitted_params['mean']
            std = self.fitted_params['std']
        
        normalized = (data - mean) / std
        
        # Handle any remaining non-finite values
        normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
        
        return normalized
    
    def _minmax_normalize(
        self, 
        data: jnp.ndarray, 
        fit: bool, 
        feature_axis: int
    ) -> jnp.ndarray:
        """Min-max normalization to specified range."""
        feature_range = self.config['normalization']['feature_range']
        
        if fit:
            finite_mask = jnp.isfinite(data)
            if not jnp.any(finite_mask):
                warnings.warn("No finite values found for normalization")
                return data
            
            finite_data = data[finite_mask]
            data_min = jnp.min(finite_data, axis=feature_axis, keepdims=True)
            data_max = jnp.max(finite_data, axis=feature_axis, keepdims=True)
            
            # Handle constant features
            data_range = data_max - data_min
            data_range = jnp.maximum(data_range, 1e-8)
            
            self.fitted_params['data_min'] = data_min
            self.fitted_params['data_range'] = data_range
        else:
            if 'data_min' not in self.fitted_params or 'data_range' not in self.fitted_params:
                raise ConfigurationError("Must fit before transform")
            data_min = self.fitted_params['data_min']
            data_range = self.fitted_params['data_range']
        
        # Scale to [0, 1] then to desired range
        normalized = (data - data_min) / data_range
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        # Handle non-finite values
        normalized = jnp.where(jnp.isfinite(normalized), normalized, feature_range[0])
        
        return normalized
    
    def _robust_normalize(
        self, 
        data: jnp.ndarray, 
        fit: bool, 
        feature_axis: int
    ) -> jnp.ndarray:
        """Robust normalization using median and IQR."""
        if fit:
            finite_mask = jnp.isfinite(data)
            if not jnp.any(finite_mask):
                warnings.warn("No finite values found for normalization")
                return data
            
            finite_data = data[finite_mask]
            median = jnp.median(finite_data, axis=feature_axis, keepdims=True)
            q75 = jnp.percentile(finite_data, 75, axis=feature_axis, keepdims=True)
            q25 = jnp.percentile(finite_data, 25, axis=feature_axis, keepdims=True)
            iqr = jnp.maximum(q75 - q25, 1e-8)
            
            self.fitted_params['median'] = median
            self.fitted_params['iqr'] = iqr
        else:
            if 'median' not in self.fitted_params or 'iqr' not in self.fitted_params:
                raise ConfigurationError("Must fit before transform")
            median = self.fitted_params['median']
            iqr = self.fitted_params['iqr']
        
        normalized = (data - median) / iqr
        normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
        
        return normalized
    
    def safe_sequence_padding(
        self,
        sequences: List[jnp.ndarray],
        max_length: Optional[int] = None,
        padding_value: float = 0.0
    ) -> jnp.ndarray:
        """
        Safely pad sequences to uniform length with error handling.
        """
        if not sequences:
            raise DataValidationError("Empty sequence list provided")
        
        # Validate all sequences
        for i, seq in enumerate(sequences):
            if not isinstance(seq, (jnp.ndarray, np.ndarray)):
                raise DataValidationError(f"Sequence {i} is not an array")
            if seq.ndim == 0:
                raise DataValidationError(f"Sequence {i} is a scalar")
        
        # Determine maximum length
        lengths = [seq.shape[0] for seq in sequences]
        if max_length is None:
            max_length = max(lengths)
        
        max_length = min(max_length, self.config['sequence_processing']['max_length'])
        
        # Determine feature dimension
        feature_dims = [seq.shape[1:] for seq in sequences if seq.ndim > 1]
        if feature_dims:
            # Check consistency
            if not all(dim == feature_dims[0] for dim in feature_dims):
                raise DataValidationError("Inconsistent feature dimensions across sequences")
            feature_shape = feature_dims[0]
        else:
            feature_shape = ()
        
        # Create padded array
        padded_shape = (len(sequences), max_length) + feature_shape
        padded_sequences = jnp.full(padded_shape, padding_value)
        
        # Fill in sequences
        for i, seq in enumerate(sequences):
            seq_len = min(seq.shape[0], max_length)
            if seq.ndim == 1 and feature_shape == ():
                padded_sequences = padded_sequences.at[i, :seq_len].set(seq[:seq_len])
            else:
                padded_sequences = padded_sequences.at[i, :seq_len].set(seq[:seq_len])
        
        return padded_sequences
    
    def handle_missing_values(
        self,
        data: jnp.ndarray,
        method: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Robust missing value handling with multiple strategies.
        """
        method = method or self.config['validation']['handle_missing']
        
        missing_mask = ~jnp.isfinite(data)
        if not jnp.any(missing_mask):
            return data  # No missing values
        
        try:
            if method == 'fill':
                fill_value = self.config['validation']['missing_fill_value']
                return jnp.where(missing_mask, fill_value, data)
            
            elif method == 'interpolate':
                return self._interpolate_missing(data, missing_mask)
            
            elif method == 'drop':
                # For sequences, this is more complex - just fill for now
                warnings.warn("Drop method not implemented for sequences, using fill instead")
                fill_value = self.config['validation']['missing_fill_value']
                return jnp.where(missing_mask, fill_value, data)
            
            else:
                raise ConfigurationError(f"Unknown missing value method: {method}")
                
        except Exception as e:
            warnings.warn(f"Missing value handling failed, using fill: {e}")
            fill_value = self.config['validation']['missing_fill_value']
            return jnp.where(missing_mask, fill_value, data)
    
    def _interpolate_missing(
        self,
        data: jnp.ndarray,
        missing_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Linear interpolation for missing values."""
        if data.ndim == 1:
            return self._interpolate_1d(data, missing_mask)
        else:
            # Interpolate along the first axis (time dimension)
            result = data.copy()
            for i in range(data.shape[1]):
                result = result.at[:, i].set(
                    self._interpolate_1d(data[:, i], missing_mask[:, i])
                )
            return result
    
    def _interpolate_1d(self, data: jnp.ndarray, missing_mask: jnp.ndarray) -> jnp.ndarray:
        """1D linear interpolation."""
        if jnp.all(missing_mask):
            # All values missing, fill with zero
            return jnp.zeros_like(data)
        
        if not jnp.any(missing_mask):
            # No missing values
            return data
        
        # Find valid indices
        valid_indices = jnp.where(~missing_mask)[0]
        
        if len(valid_indices) == 1:
            # Only one valid value, use it for all
            return jnp.full_like(data, data[valid_indices[0]])
        
        # Simple linear interpolation (can be improved)
        result = data.copy()
        
        # Forward fill for leading missing values
        first_valid = valid_indices[0]
        result = result.at[:first_valid].set(data[first_valid])
        
        # Backward fill for trailing missing values
        last_valid = valid_indices[-1]
        result = result.at[last_valid+1:].set(data[last_valid])
        
        # Linear interpolation for middle missing values
        for i in range(len(data)):
            if missing_mask[i]:
                # Find surrounding valid values
                left_idx = jnp.max(jnp.where(valid_indices < i, valid_indices, -1))
                right_idx = jnp.min(jnp.where(valid_indices > i, valid_indices, len(data)))
                
                if left_idx >= 0 and right_idx < len(data):
                    # Interpolate
                    left_val = data[left_idx]
                    right_val = data[right_idx]
                    alpha = (i - left_idx) / (right_idx - left_idx)
                    result = result.at[i].set(left_val + alpha * (right_val - left_val))
        
        return result