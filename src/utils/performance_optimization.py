"""
Performance optimization utilities for liquid neural networks.
"""

import functools
import time
from typing import Dict, Any, Optional, Callable, Tuple
import warnings

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        self.compilation_cache = {}
        self.timing_stats = {}
        
    def compile_and_cache(self, func: Callable, cache_key: str) -> Callable:
        """JIT compile function and cache result."""
        if not HAS_JAX:
            return func
            
        if cache_key not in self.compilation_cache:\
            compiled_func = jit(func)
            self.compilation_cache[cache_key] = compiled_func
            
        return self.compilation_cache[cache_key]
    
    def vectorize_batch_operations(self, func: Callable, in_axes: int = 0) -> Callable:
        """Vectorize operations across batch dimension."""
        if not HAS_JAX:
            return func
            
        return vmap(func, in_axes=in_axes)
    
    def parallelize_multi_device(self, func: Callable, axis_name: str = 'batch') -> Callable:
        """Parallelize across multiple devices."""
        if not HAS_JAX:
            return func
            
        try:
            return pmap(func, axis_name=axis_name)
        except Exception:
            warnings.warn("Multi-device parallelization not available")
            return func
    
    def time_function(self, func: Callable, name: str):
        """Decorator to time function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            
            # For JAX, block until computation is complete
            if HAS_JAX and hasattr(result, 'block_until_ready'):
                result.block_until_ready()
                
            elapsed = time.time() - start_time
            
            if name not in self.timing_stats:
                self.timing_stats[name] = []
            self.timing_stats[name].append(elapsed)
            
            return result
        return wrapper
    
    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all measured functions."""
        stats = {}
        for name, times in self.timing_stats.items():
            stats[name] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        return stats


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def gradient_checkpointing(forward_fn: Callable) -> Callable:
        """Apply gradient checkpointing to reduce memory usage."""
        if not HAS_JAX:
            return forward_fn
            
        try:
            from jax import checkpoint
            return checkpoint(forward_fn)
        except ImportError:
            warnings.warn("Gradient checkpointing not available")
            return forward_fn
    
    @staticmethod
    def chunked_processing(inputs: jnp.ndarray, 
                          process_fn: Callable, 
                          chunk_size: int = 32) -> jnp.ndarray:
        """Process large inputs in chunks to reduce memory usage."""
        if len(inputs) <= chunk_size:
            return process_fn(inputs)
            
        results = []
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            chunk_result = process_fn(chunk)
            results.append(chunk_result)
            
        return jnp.concatenate(results, axis=0)
    
    @staticmethod
    def clear_cache():
        """Clear JAX compilation cache."""
        if HAS_JAX:
            try:
                from jax._src.lib import xla_bridge
                xla_bridge.clear_backends()
            except Exception:
                pass


class ConcurrentProcessor:
    """Concurrent processing utilities."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        
    def parallel_sequence_processing(self, 
                                   sequences: list, 
                                   process_fn: Callable,
                                   use_multiprocessing: bool = False) -> list:
        """Process multiple sequences in parallel."""
        if use_multiprocessing:
            try:
                from multiprocessing import Pool
                with Pool(self.max_workers) as pool:
                    return pool.map(process_fn, sequences)
            except Exception:
                warnings.warn("Multiprocessing not available, falling back to serial")
                
        # Fallback to serial processing
        return [process_fn(seq) for seq in sequences]
    
    def async_data_loading(self, data_loader):
        """Asynchronous data loading (placeholder for future implementation)."""
        # This would integrate with JAX's asynchronous data loading
        # For now, return the data loader as-is
        return data_loader


class ModelOptimizer:
    """High-level model optimization coordinator."""
    
    def __init__(self):
        self.perf_optimizer = PerformanceOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.concurrent_processor = ConcurrentProcessor()
        self.optimized_models = {}
        
    def optimize_liquid_network(self, network_class, **kwargs):
        """Apply comprehensive optimizations to liquid neural network."""
        
        class OptimizedLiquidNetwork(network_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # JIT compile forward pass
                self._compiled_forward = None
                self._compiled_sequence_forward = None
                
            def __call__(self, inputs, hidden_state, dt=0.1):
                if self._compiled_forward is None:
                    self._compiled_forward = jit(super().__call__)
                    
                return self._compiled_forward(inputs, hidden_state, dt)
                
            def forward_sequence(self, input_sequence, dt=0.1):
                if self._compiled_sequence_forward is None:
                    self._compiled_sequence_forward = jit(super().forward_sequence)
                    
                return self._compiled_sequence_forward(input_sequence, dt)
                
            def parallel_batch_forward(self, input_batches):
                """Process multiple batches in parallel."""
                batch_fn = vmap(self.__call__, in_axes=(0, 0, None))
                return batch_fn(input_batches[0], input_batches[1], 0.1)
                
        return OptimizedLiquidNetwork
    
    def optimize_training_loop(self, train_fn: Callable) -> Callable:
        """Optimize training loop with JIT compilation and memory management."""
        
        @jit
        def optimized_train_step(params, batch, opt_state):
            return train_fn(params, batch, opt_state)
            
        def optimized_train_loop(params, train_data, optimizer, epochs):
            opt_state = optimizer.init(params)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch in train_data:
                    # Memory-efficient batch processing
                    if batch[0].shape[0] > 128:  # Large batch
                        loss, opt_state = self.memory_optimizer.chunked_processing(
                            batch, 
                            lambda b: optimized_train_step(params, b, opt_state),
                            chunk_size=64
                        )
                    else:
                        loss, opt_state = optimized_train_step(params, batch, opt_state)
                    
                    epoch_loss += loss
                    batch_count += 1
                
                avg_loss = epoch_loss / batch_count
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
                
                # Periodic memory cleanup
                if epoch % 10 == 0:
                    self.memory_optimizer.clear_cache()
                    
            return params, opt_state
            
        return optimized_train_loop
    
    def benchmark_model(self, model, test_inputs, iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance."""
        if not HAS_JAX:
            return {"error": "JAX not available for benchmarking"}
            
        # Warm up
        for _ in range(5):
            model(test_inputs[0], test_inputs[1])
            
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            result = model(test_inputs[0], test_inputs[1])
            if hasattr(result[0], 'block_until_ready'):
                result[0].block_until_ready()
            times.append(time.time() - start)
            
        return {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'throughput': len(test_inputs[0]) / (sum(times) / len(times))  # samples/second
        }
    
    def auto_tune_batch_size(self, model, sample_input, max_batch_size: int = 512):
        """Automatically find optimal batch size."""
        if not HAS_JAX:
            return 32  # Default fallback
            
        best_batch_size = 32
        best_throughput = 0
        
        for batch_size in [16, 32, 64, 128, 256, 512]:
            if batch_size > max_batch_size:
                break
                
            try:
                # Create test batch
                test_batch = jnp.tile(sample_input[0][:1], (batch_size, 1))
                test_hidden = jnp.tile(sample_input[1][:1], (batch_size, 1))
                
                # Benchmark
                benchmark_results = self.benchmark_model(
                    model, 
                    (test_batch, test_hidden),
                    iterations=20
                )
                
                throughput = benchmark_results['throughput']
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    
            except Exception as e:
                warnings.warn(f"Batch size {batch_size} failed: {e}")
                break
                
        return best_batch_size


# Global optimizer instance
global_optimizer = ModelOptimizer()