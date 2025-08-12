"""
Performance optimization utilities for liquid neural networks.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any, Callable, Optional, Tuple
import time
import numpy as np
from functools import partial
import threading
import queue
from concurrent.futures import ThreadPoolExecutor


class JITCache:
    """Cache for JIT-compiled functions."""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get_or_compile(self, func, key: str, static_argnums=None):
        """Get cached JIT function or compile and cache."""
        with self._lock:
            if key not in self._cache:
                if static_argnums is not None:
                    self._cache[key] = jax.jit(func, static_argnums=static_argnums)
                else:
                    self._cache[key] = jax.jit(func)
            return self._cache[key]
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


# Global JIT cache
jit_cache = JITCache()


def optimize_model_for_inference(model):
    """Optimize model for faster inference."""
    # Pre-JIT compile common operations
    
    @jax.jit
    def fast_forward(model, inputs):
        return model(inputs)
    
    @jax.jit 
    def fast_liquid_states(model, inputs):
        if hasattr(model, 'get_liquid_states'):
            return model.get_liquid_states(inputs)
        return None
    
    # Create optimized model wrapper
    class OptimizedModel:
        def __init__(self, original_model):
            self.original_model = original_model
            self.fast_forward = fast_forward
            self.fast_liquid_states = fast_liquid_states
        
        def __call__(self, inputs):
            return self.fast_forward(self.original_model, inputs)
        
        def get_liquid_states(self, inputs):
            return self.fast_liquid_states(self.original_model, inputs)
        
        def __getattr__(self, name):
            return getattr(self.original_model, name)
    
    return OptimizedModel(model)


def batch_inference(model, data: jnp.ndarray, batch_size: int = 32, 
                   show_progress: bool = False) -> jnp.ndarray:
    """Perform batched inference for large datasets."""
    
    if data.shape[0] <= batch_size:
        return model(data)
    
    # Optimize model for inference
    optimized_model = optimize_model_for_inference(model)
    
    results = []
    total_batches = (data.shape[0] + batch_size - 1) // batch_size
    
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        batch_result = optimized_model(batch)
        results.append(batch_result)
        
        if show_progress:
            batch_num = i // batch_size + 1
            print(f"Processed batch {batch_num}/{total_batches}")
    
    return jnp.concatenate(results, axis=0)


def parallel_model_evaluation(models: Dict[str, Any], inputs: jnp.ndarray, 
                             max_workers: int = 4) -> Dict[str, jnp.ndarray]:
    """Evaluate multiple models in parallel."""
    
    def evaluate_model(model_item):
        name, model = model_item
        try:
            result = model(inputs)
            return name, result
        except Exception as e:
            return name, f"Error: {str(e)}"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_model, item) for item in models.items()]
        results = {}
        
        for future in futures:
            name, result = future.result()
            results[name] = result
    
    return results


class AdaptiveComputationOptimizer:
    """Optimize computation based on input characteristics."""
    
    def __init__(self):
        self.computation_stats = {}
        self.threshold_cache = {}
    
    def should_use_fast_path(self, inputs: jnp.ndarray, model_name: str) -> bool:
        """Decide whether to use optimized computation path."""
        batch_size, seq_len = inputs.shape[:2]
        
        # Simple heuristics for fast path
        if batch_size * seq_len < 1000:
            return True
        
        if seq_len < 50:
            return True
            
        return False
    
    def optimize_sequence_processing(self, model, inputs: jnp.ndarray):
        """Optimize sequence processing based on input characteristics."""
        
        if self.should_use_fast_path(inputs, "default"):
            # Use standard processing for small inputs
            return model(inputs)
        else:
            # Use chunked processing for large inputs
            return self._chunked_sequence_processing(model, inputs)
    
    def _chunked_sequence_processing(self, model, inputs: jnp.ndarray, 
                                   chunk_size: int = 100):
        """Process long sequences in chunks."""
        batch_size, seq_len, features = inputs.shape
        
        if seq_len <= chunk_size:
            return model(inputs)
        
        outputs = []
        for i in range(0, seq_len, chunk_size):
            chunk = inputs[:, i:i + chunk_size, :]
            chunk_output = model(chunk)
            outputs.append(chunk_output)
        
        return jnp.concatenate(outputs, axis=1)


class MemoryOptimizer:
    """Optimize memory usage during training and inference."""
    
    def __init__(self):
        self.gradient_checkpointing = False
        self.mixed_precision = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        self.gradient_checkpointing = True
    
    def enable_mixed_precision(self):
        """Enable mixed precision training."""
        self.mixed_precision = True
    
    def optimize_training_step(self, loss_fn, model, inputs, targets):
        """Memory-optimized training step."""
        
        if self.gradient_checkpointing:
            # Use gradient checkpointing
            return self._checkpointed_training_step(loss_fn, model, inputs, targets)
        else:
            return loss_fn(model, inputs, targets)
    
    def _checkpointed_training_step(self, loss_fn, model, inputs, targets):
        """Training step with gradient checkpointing."""
        # Simple checkpointing implementation
        @jax.checkpoint
        def forward_pass(model, inputs):
            return model(inputs)
        
        predictions = forward_pass(model, inputs)
        loss = jnp.mean((predictions - targets) ** 2)
        return loss


class CachingOptimizer:
    """Cache frequently computed values."""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.max_cache_size = max_cache_size
    
    def cached_computation(self, key: str, computation_fn, *args, **kwargs):
        """Cache computation results."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        
        # Compute result
        result = computation_fn(*args, **kwargs)
        
        # Add to cache
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self.cache[key] = result
        self.access_times[key] = time.time()
        
        return result
    
    def _evict_oldest(self):
        """Evict least recently used item."""
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear_cache(self):
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()


class DynamicBatchSizer:
    """Dynamically adjust batch size based on memory and performance."""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 256):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 1
        self.performance_history = []
        self.memory_usage_history = []
    
    def update_batch_size(self, execution_time: float, memory_usage: float, 
                         success: bool) -> int:
        """Update batch size based on performance feedback."""
        
        if not success:
            # Reduce batch size if execution failed
            self.current_batch_size = max(
                self.min_batch_size, 
                self.current_batch_size // 2
            )
        else:
            # Record performance
            self.performance_history.append(execution_time)
            self.memory_usage_history.append(memory_usage)
            
            # Keep only recent history
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
                self.memory_usage_history = self.memory_usage_history[-10:]
            
            # Adjust based on performance trend
            if len(self.performance_history) >= 3:
                recent_times = self.performance_history[-3:]
                if all(recent_times[i] <= recent_times[i-1] for i in range(1, len(recent_times))):
                    # Performance improving, try larger batch
                    self.current_batch_size = min(
                        self.max_batch_size,
                        int(self.current_batch_size * 1.2)
                    )
        
        return self.current_batch_size


def create_optimized_training_loop(model, optimizer, loss_fn):
    """Create optimized training loop with JIT compilation."""
    
    @jax.jit
    def training_step(diff_params, static_params, opt_state, inputs, targets):
        """JIT-compiled training step."""
        
        def loss_computation(diff_p):
            model_combined = eqx.combine(diff_p, static_params)
            predictions = model_combined(inputs)
            return loss_fn(predictions, targets)
        
        loss, grads = jax.value_and_grad(loss_computation)(diff_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, diff_params)
        new_diff_params = eqx.apply_updates(diff_params, updates)
        
        return new_diff_params, new_opt_state, loss
    
    return training_step


class StreamingDataProcessor:
    """Process data streams efficiently."""
    
    def __init__(self, model, buffer_size: int = 1000):
        self.model = model
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.results_queue = queue.Queue()
        self.processing = False
    
    def start_processing(self):
        """Start background processing thread."""
        self.processing = True
        self.worker_thread = threading.Thread(target=self._process_stream)
        self.worker_thread.start()
    
    def stop_processing(self):
        """Stop background processing."""
        self.processing = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()
    
    def add_data(self, data: jnp.ndarray):
        """Add data to processing queue."""
        self.buffer.put(data)
    
    def get_result(self, timeout: float = 1.0):
        """Get processing result."""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _process_stream(self):
        """Background processing loop."""
        while self.processing:
            try:
                data = self.buffer.get(timeout=0.1)
                result = self.model(data)
                self.results_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                self.results_queue.put(f"Error: {str(e)}")


def benchmark_model_performance(model, input_shapes: list, num_iterations: int = 100):
    """Benchmark model performance across different input shapes."""
    
    results = {}
    
    for shape in input_shapes:
        shape_key = f"{'x'.join(map(str, shape))}"
        times = []
        
        # Generate test data
        test_input = jax.random.normal(jax.random.PRNGKey(42), shape, dtype=jnp.float32)
        
        # Warm up
        for _ in range(5):
            _ = model(test_input)
        
        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        results[shape_key] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': shape[0] / np.mean(times),  # samples per second
            'shape': shape
        }
    
    return results


def optimize_for_tpu():
    """Optimize settings for TPU execution."""
    # Set XLA flags for TPU optimization
    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
    
    # Enable faster compilation
    jax.config.update('jax_disable_jit', False)
    jax.config.update('jax_enable_x64', False)


def optimize_for_gpu():
    """Optimize settings for GPU execution."""
    # Enable memory preallocation
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    
    # Enable faster compilation
    jax.config.update('jax_disable_jit', False)


class ModelEnsemble:
    """Ensemble of models for improved performance and robustness."""
    
    def __init__(self, models: list, voting_strategy: str = 'average'):
        self.models = models
        self.voting_strategy = voting_strategy
        self.weights = jnp.ones(len(models)) / len(models)
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Ensemble prediction."""
        predictions = []
        
        for model in self.models:
            pred = model(inputs)
            predictions.append(pred)
        
        predictions = jnp.stack(predictions, axis=0)
        
        if self.voting_strategy == 'average':
            return jnp.mean(predictions, axis=0)
        elif self.voting_strategy == 'weighted':
            return jnp.average(predictions, axis=0, weights=self.weights)
        elif self.voting_strategy == 'median':
            return jnp.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def set_weights(self, weights: jnp.ndarray):
        """Set ensemble weights."""
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        self.weights = weights / jnp.sum(weights)  # Normalize


# Global optimizers
memory_optimizer = MemoryOptimizer()
caching_optimizer = CachingOptimizer()
adaptive_optimizer = AdaptiveComputationOptimizer()
dynamic_batch_sizer = DynamicBatchSizer()


def apply_all_optimizations(model, enable_jit: bool = True, 
                          enable_caching: bool = True,
                          enable_memory_opt: bool = True):
    """Apply all available optimizations to a model."""
    
    if enable_jit:
        model = optimize_model_for_inference(model)
    
    if enable_memory_opt:
        memory_optimizer.enable_gradient_checkpointing()
    
    # Add more optimizations as needed
    
    return model