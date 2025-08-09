"""
Performance optimization utilities for liquid neural networks.

This module provides optimization features including caching, JIT compilation,
memory management, and performance profiling.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import functools
import time
from typing import Dict, Any, Optional, Callable, Tuple, List
import threading
from collections import OrderedDict
import weakref
from ..utils.logging import get_logger, PerformanceMonitor


class LRUCache:
    """
    Least Recently Used cache implementation for JAX arrays.
    """
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add LRU caching to a function."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = self._make_key(args, kwargs)
            
            with self.lock:
                if key in self.cache:
                    # Cache hit - move to end (most recently used)
                    result = self.cache.pop(key)
                    self.cache[key] = result
                    self.hits += 1
                    return result
                else:
                    # Cache miss - compute result
                    result = func(*args, **kwargs)
                    self.cache[key] = result
                    self.misses += 1
                    
                    # Remove oldest items if cache is full
                    while len(self.cache) > self.maxsize:
                        self.cache.popitem(last=False)
                    
                    return result
        
        wrapper.cache_info = lambda: {
            'hits': self.hits,
            'misses': self.misses,
            'maxsize': self.maxsize,
            'currsize': len(self.cache)
        }
        wrapper.cache_clear = self.clear
        
        return wrapper
    
    def _make_key(self, args: tuple, kwargs: dict) -> tuple:
        """Create a hashable key from function arguments."""
        key_parts = []
        
        for arg in args:
            if isinstance(arg, jnp.ndarray):
                # Use shape and a hash of the first few elements for arrays
                shape_key = arg.shape
                if arg.size > 0:
                    flat_arg = arg.flatten()
                    sample_size = min(10, flat_arg.size)
                    sample_hash = hash(tuple(flat_arg[:sample_size].tolist()))
                    key_parts.append(('array', shape_key, sample_hash))
                else:
                    key_parts.append(('array', shape_key, 0))
            else:
                key_parts.append(arg)
        
        # Add kwargs
        if kwargs:
            key_parts.append(tuple(sorted(kwargs.items())))
        
        return tuple(key_parts)
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class ComputationGraph:
    """
    Optimized computation graph for liquid neural networks.
    """
    
    def __init__(self):
        self.compiled_functions = {}
        self.performance_monitor = PerformanceMonitor()
        self.logger = get_logger()
    
    def compile_forward_pass(self, model_class: type) -> Callable:
        """
        Create an optimized, JIT-compiled forward pass function.
        
        Args:
            model_class: The model class to optimize
            
        Returns:
            Compiled forward pass function
        """
        model_name = model_class.__name__
        
        if model_name in self.compiled_functions:
            return self.compiled_functions[model_name]
        
        @jit
        def optimized_forward(params, inputs, hidden_state, dt):
            """Optimized forward pass implementation."""
            
            # Vectorized operations for better performance
            def scan_step(carry, x):
                h, _ = carry
                
                # Compute dynamics efficiently
                if hasattr(params, 'W_in'):
                    # Liquid neural network dynamics
                    tau_actual = jnp.exp(params.tau)
                    input_term = jnp.dot(params.W_in, x)
                    recurrent_term = jnp.dot(params.W_rec, jnp.tanh(h))
                    dhdt = (1.0 / tau_actual) * (-h + jnp.tanh(input_term + recurrent_term + params.bias))
                else:
                    # Continuous-time RNN dynamics
                    activation_input = jnp.dot(params.W_rec, h) + jnp.dot(params.W_in, x) + params.bias
                    dhdt = -h + jnp.tanh(activation_input)
                
                # Integration step
                h_next = h + dt * dhdt
                
                # Output computation
                output = jnp.dot(params.W_out, jnp.tanh(h_next))
                
                return (h_next, output), (output, h_next)
            
            # Use lax.scan for optimal performance
            _, (outputs, states) = lax.scan(scan_step, (hidden_state, None), inputs)
            
            return outputs, states
        
        self.compiled_functions[model_name] = optimized_forward
        self.logger.debug(f"Compiled optimized forward pass for {model_name}")
        
        return optimized_forward
    
    def compile_batch_forward(self, model_class: type) -> Callable:
        """
        Create a batch-optimized forward pass using vmap.
        
        Args:
            model_class: The model class to optimize
            
        Returns:
            Batch-compiled forward pass function
        """
        single_forward = self.compile_forward_pass(model_class)
        
        # Vectorize over batch dimension
        batch_forward = vmap(single_forward, in_axes=(None, 0, 0, None))
        
        return jit(batch_forward)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "compiled_functions": list(self.compiled_functions.keys()),
            "performance_summary": self.performance_monitor.get_summary()
        }


class MemoryPool:
    """
    Memory pool for efficient tensor allocation and reuse.
    """
    
    def __init__(self):
        self.pools = {}  # shape -> list of arrays
        self.pool_sizes = {}
        self.allocations = 0
        self.reuses = 0
        self.logger = get_logger()
    
    def get_array(self, shape: Tuple[int, ...], dtype=jnp.float32) -> jnp.ndarray:
        """
        Get an array from the pool or allocate a new one.
        
        Args:
            shape: Required array shape
            dtype: Array data type
            
        Returns:
            Array with the requested shape
        """
        key = (shape, dtype)
        
        if key in self.pools and self.pools[key]:
            # Reuse from pool
            array = self.pools[key].pop()
            self.reuses += 1
            return jnp.zeros_like(array)  # Reset to zeros
        else:
            # Allocate new array
            array = jnp.zeros(shape, dtype=dtype)
            self.allocations += 1
            return array
    
    def return_array(self, array: jnp.ndarray):
        """
        Return an array to the pool for reuse.
        
        Args:
            array: Array to return to pool
        """
        key = (array.shape, array.dtype)
        
        if key not in self.pools:
            self.pools[key] = []
            self.pool_sizes[key] = 0
        
        # Limit pool size to prevent memory bloat
        max_pool_size = 10
        if self.pool_sizes[key] < max_pool_size:
            self.pools[key].append(array)
            self.pool_sizes[key] += 1
    
    def clear_pools(self):
        """Clear all memory pools."""
        self.pools.clear()
        self.pool_sizes.clear()
        self.logger.debug("Memory pools cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        total_pooled = sum(len(pool) for pool in self.pools.values())
        
        return {
            "total_allocations": self.allocations,
            "total_reuses": self.reuses,
            "reuse_ratio": self.reuses / max(1, self.allocations + self.reuses),
            "active_pools": len(self.pools),
            "total_pooled_arrays": total_pooled
        }


class AdaptiveOptimizer:
    """
    Adaptive optimization strategies for liquid neural networks.
    """
    
    def __init__(self):
        self.optimization_history = []
        self.current_strategy = "balanced"
        self.performance_metrics = {}
        self.logger = get_logger()
    
    def optimize_integration_step(
        self,
        current_dt: float,
        stability_measure: float,
        accuracy_error: float,
        target_accuracy: float = 1e-4
    ) -> float:
        """
        Adaptively optimize the integration time step.
        
        Args:
            current_dt: Current time step
            stability_measure: Stability measure of the system
            accuracy_error: Current accuracy error
            target_accuracy: Target accuracy threshold
            
        Returns:
            Optimized time step
        """
        # Adaptive step size control
        if accuracy_error > target_accuracy:
            # Reduce step size for better accuracy
            new_dt = current_dt * 0.8
        elif accuracy_error < target_accuracy * 0.1:
            # Increase step size for efficiency
            new_dt = current_dt * 1.2
        else:
            new_dt = current_dt
        
        # Stability constraint
        max_dt = 1.0 / (stability_measure + 1e-8)  # Prevent division by zero
        new_dt = min(new_dt, max_dt * 0.5)  # Safety margin
        
        # Bounds
        new_dt = jnp.clip(new_dt, 1e-6, 1.0)
        
        if abs(new_dt - current_dt) > current_dt * 0.1:
            self.logger.debug(
                f"Adapted integration step: {current_dt:.6f} -> {new_dt:.6f}",
                stability=stability_measure,
                accuracy_error=accuracy_error
            )
        
        return float(new_dt)
    
    def select_computation_strategy(
        self,
        sequence_length: int,
        batch_size: int,
        model_complexity: int
    ) -> Dict[str, Any]:
        """
        Select optimal computation strategy based on problem characteristics.
        
        Args:
            sequence_length: Length of input sequences
            batch_size: Batch size
            model_complexity: Measure of model complexity
            
        Returns:
            Optimization strategy configuration
        """
        total_operations = sequence_length * batch_size * model_complexity
        
        if total_operations < 1000:
            strategy = {
                "use_jit": True,
                "use_vmap": False,
                "chunk_size": None,
                "parallel_sequences": False
            }
        elif total_operations < 100000:
            strategy = {
                "use_jit": True,
                "use_vmap": True,
                "chunk_size": None,
                "parallel_sequences": False
            }
        else:
            strategy = {
                "use_jit": True,
                "use_vmap": True,
                "chunk_size": min(1000, sequence_length // 4),
                "parallel_sequences": True
            }
        
        self.logger.debug(
            "Selected computation strategy",
            strategy=strategy,
            total_ops=total_operations
        )
        
        return strategy


class PerformanceProfiler:
    """
    Detailed performance profiling for liquid neural networks.
    """
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
        self.logger = get_logger()
    
    def start_profile(self, operation_name: str):
        """Start profiling an operation."""
        self.active_profiles[operation_name] = {
            "start_time": time.time(),
            "operation": operation_name
        }
    
    def end_profile(self, operation_name: str) -> float:
        """End profiling and return duration."""
        if operation_name not in self.active_profiles:
            self.logger.warning(f"No active profile for {operation_name}")
            return 0.0
        
        start_info = self.active_profiles.pop(operation_name)
        duration = time.time() - start_info["start_time"]
        
        # Store profile data
        if operation_name not in self.profiles:
            self.profiles[operation_name] = []
        
        self.profiles[operation_name].append({
            "duration": duration,
            "timestamp": time.time()
        })
        
        # Keep only recent profiles (last 100)
        if len(self.profiles[operation_name]) > 100:
            self.profiles[operation_name] = self.profiles[operation_name][-100:]
        
        return duration
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_profile(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_profile(operation_name)
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for operation, profiles in self.profiles.items():
            if profiles:
                durations = [p["duration"] for p in profiles]
                summary[operation] = {
                    "count": len(profiles),
                    "total_time": sum(durations),
                    "mean_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                    "recent_time": profiles[-1]["duration"]
                }
        
        return summary


# Global instances
_global_memory_pool = None
_global_computation_graph = None
_global_profiler = None

def get_memory_pool() -> MemoryPool:
    """Get or create global memory pool."""
    global _global_memory_pool
    if _global_memory_pool is None:
        _global_memory_pool = MemoryPool()
    return _global_memory_pool

def get_computation_graph() -> ComputationGraph:
    """Get or create global computation graph."""
    global _global_computation_graph
    if _global_computation_graph is None:
        _global_computation_graph = ComputationGraph()
    return _global_computation_graph

def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def optimize_model(model):
    """
    Apply comprehensive optimizations to a model.
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model wrapper
    """
    
    class OptimizedModelWrapper:
        def __init__(self, base_model):
            self.model = base_model
            self.computation_graph = get_computation_graph()
            self.memory_pool = get_memory_pool()
            self.profiler = get_profiler()
            self.cache = LRUCache(maxsize=64)
            
            # Compile optimized functions
            self.optimized_forward = self.computation_graph.compile_forward_pass(type(base_model))
            
        @property
        def __class__(self):
            # Make isinstance work correctly
            return type(self.model)
        
        def __call__(self, *args, **kwargs):
            """Optimized forward pass."""
            return self.optimized_forward(self.model, *args, **kwargs)
        
        def __getattr__(self, name):
            """Delegate attribute access to base model."""
            return getattr(self.model, name)
        
        def get_optimization_stats(self):
            """Get optimization statistics."""
            return {
                "cache_stats": self.cache.cache_info(),
                "memory_stats": self.memory_pool.get_stats(),
                "performance_stats": self.profiler.get_performance_summary()
            }
    
    return OptimizedModelWrapper(model)