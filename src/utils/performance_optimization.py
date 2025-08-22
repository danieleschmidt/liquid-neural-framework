"""
Performance optimization utilities for liquid neural networks.
"""

import functools
import time
from typing import Dict, Any, Optional, Callable, Tuple, List
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
            
        if cache_key not in self.compilation_cache:
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


# Additional optimizations for Generation 3: Scaling

class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 8, 
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3):
        """
        Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: CPU utilization threshold to scale up
            scale_down_threshold: CPU utilization threshold to scale down
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.utilization_history = []
        self.scaling_decisions = []
        
    def monitor_performance(self, current_load: float, response_time: float) -> int:
        """
        Monitor performance and make scaling decisions.
        
        Args:
            current_load: Current system load (0.0-1.0)
            response_time: Current response time in seconds
            
        Returns:
            Recommended number of workers
        """
        # Record utilization
        self.utilization_history.append(current_load)
        
        # Keep only recent history
        if len(self.utilization_history) > 100:
            self.utilization_history = self.utilization_history[-100:]
        
        # Calculate moving average
        if len(self.utilization_history) >= 5:
            avg_utilization = sum(self.utilization_history[-5:]) / 5
        else:
            avg_utilization = current_load
        
        # Scaling decisions
        new_workers = self.current_workers
        
        # Scale up if high utilization or slow response time
        if (avg_utilization > self.scale_up_threshold or response_time > 1.0) and \
           self.current_workers < self.max_workers:
            new_workers = min(self.max_workers, self.current_workers + 1)
            decision = "scale_up"
        
        # Scale down if low utilization and fast response time
        elif avg_utilization < self.scale_down_threshold and response_time < 0.1 and \
             self.current_workers > self.min_workers:
            new_workers = max(self.min_workers, self.current_workers - 1)
            decision = "scale_down"
        else:
            decision = "no_change"
        
        if new_workers != self.current_workers:
            self.scaling_decisions.append({
                'timestamp': time.time(),
                'old_workers': self.current_workers,
                'new_workers': new_workers,
                'utilization': avg_utilization,
                'response_time': response_time,
                'decision': decision
            })
            
            self.current_workers = new_workers
        
        return self.current_workers
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'scaling_decisions': len(self.scaling_decisions),
            'avg_utilization': sum(self.utilization_history) / max(len(self.utilization_history), 1)
        }


class LoadBalancer:
    """Simple load balancer for distributing work across workers."""
    
    def __init__(self, strategy: str = "round_robin"):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy ("round_robin", "least_loaded")
        """
        self.strategy = strategy
        self.workers = []
        self.worker_loads = {}
        self.current_worker_index = 0
        
    def add_worker(self, worker_id: str):
        """Add a worker to the pool."""
        if worker_id not in self.workers:
            self.workers.append(worker_id)
            self.worker_loads[worker_id] = 0
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from the pool."""
        if worker_id in self.workers:
            self.workers.remove(worker_id)
            self.worker_loads.pop(worker_id, None)
    
    def get_next_worker(self) -> str:
        """Get next worker based on load balancing strategy."""
        if not self.workers:
            raise ValueError("No workers available")
        
        if self.strategy == "round_robin":
            worker = self.workers[self.current_worker_index]
            self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
            return worker
        
        elif self.strategy == "least_loaded":
            # Find worker with minimum load
            min_load_worker = min(self.workers, key=lambda w: self.worker_loads[w])
            return min_load_worker
        
        else:
            # Default to round robin
            return self.workers[0]
    
    def update_worker_load(self, worker_id: str, load: float):
        """Update worker load for load balancing decisions."""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = load
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'total_workers': len(self.workers),
            'worker_loads': self.worker_loads.copy(),
            'avg_load': sum(self.worker_loads.values()) / max(len(self.worker_loads), 1),
            'strategy': self.strategy
        }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger circuit break
            recovery_timeout: Time to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        current_time = time.time()
        
        if self.state == "OPEN":
            # Check if we should try recovery
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time
        }


class ResourceMonitor:
    """Monitor system resources and provide optimization recommendations."""
    
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.recommendations = []
        
    def record_metrics(self, cpu_percent: float, memory_percent: float):
        """Record resource metrics."""
        current_time = time.time()
        
        self.cpu_history.append({'timestamp': current_time, 'value': cpu_percent})
        self.memory_history.append({'timestamp': current_time, 'value': memory_percent})
        
        # Keep only recent history (last hour)
        cutoff_time = current_time - 3600
        self.cpu_history = [m for m in self.cpu_history if m['timestamp'] > cutoff_time]
        self.memory_history = [m for m in self.memory_history if m['timestamp'] > cutoff_time]
        
        # Generate recommendations
        self._generate_recommendations(cpu_percent, memory_percent)
    
    def _generate_recommendations(self, cpu_percent: float, memory_percent: float):
        """Generate optimization recommendations based on resource usage."""
        recommendations = []
        
        # CPU recommendations
        if cpu_percent > 90:
            recommendations.append({
                'type': 'CPU_HIGH',
                'message': 'High CPU usage detected. Consider scaling up or optimizing computation.',
                'severity': 'HIGH',
                'timestamp': time.time()
            })
        elif cpu_percent < 20:
            recommendations.append({
                'type': 'CPU_LOW',
                'message': 'Low CPU usage detected. Consider scaling down to save resources.',
                'severity': 'LOW',
                'timestamp': time.time()
            })
        
        # Memory recommendations
        if memory_percent > 85:
            recommendations.append({
                'type': 'MEMORY_HIGH',
                'message': 'High memory usage detected. Consider memory optimization or scaling.',
                'severity': 'HIGH',
                'timestamp': time.time()
            })
        
        # Add to recommendations list
        self.recommendations.extend(recommendations)
        
        # Keep only recent recommendations (last hour)
        cutoff_time = time.time() - 3600
        self.recommendations = [r for r in self.recommendations if r['timestamp'] > cutoff_time]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.cpu_history or not self.memory_history:
            return {'error': 'No resource data available'}
        
        recent_cpu = [m['value'] for m in self.cpu_history[-10:]]  # Last 10 measurements
        recent_memory = [m['value'] for m in self.memory_history[-10:]]
        
        return {
            'cpu_stats': {
                'current': recent_cpu[-1] if recent_cpu else 0,
                'avg': sum(recent_cpu) / len(recent_cpu),
                'max': max(recent_cpu),
                'min': min(recent_cpu)
            },
            'memory_stats': {
                'current': recent_memory[-1] if recent_memory else 0,
                'avg': sum(recent_memory) / len(recent_memory),
                'max': max(recent_memory),
                'min': min(recent_memory)
            },
            'recommendations': self.recommendations[-5:],  # Last 5 recommendations
            'data_points': len(self.cpu_history)
        }


class PerformanceOptimizationSuite:
    """Comprehensive performance optimization suite for Generation 3."""
    
    def __init__(self):
        self.model_optimizer = ModelOptimizer()
        self.auto_scaler = AutoScaler()
        self.load_balancer = LoadBalancer()
        self.circuit_breaker = CircuitBreaker()
        self.resource_monitor = ResourceMonitor()
        
        # Global performance metrics
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
    def optimize_system_performance(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive system performance optimization.
        
        Args:
            current_metrics: Current system metrics (cpu_percent, memory_percent, response_time, etc.)
            
        Returns:
            Optimization recommendations and actions
        """
        cpu_percent = current_metrics.get('cpu_percent', 0)
        memory_percent = current_metrics.get('memory_percent', 0)
        response_time = current_metrics.get('response_time', 0)
        
        # Record metrics
        self.resource_monitor.record_metrics(cpu_percent, memory_percent)
        
        # Auto-scaling decision
        load = max(cpu_percent, memory_percent) / 100.0
        recommended_workers = self.auto_scaler.monitor_performance(load, response_time)
        
        # Update performance stats
        self.request_count += 1
        self.total_response_time += response_time
        
        # Get resource summary
        resource_summary = self.resource_monitor.get_resource_summary()
        
        optimization_report = {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'recommended_workers': recommended_workers,
            'scaling_stats': self.auto_scaler.get_scaling_stats(),
            'load_balancer_stats': self.load_balancer.get_load_stats(),
            'circuit_breaker_state': self.circuit_breaker.get_state(),
            'resource_summary': resource_summary,
            'performance_stats': {
                'total_requests': self.request_count,
                'avg_response_time': self.total_response_time / max(self.request_count, 1),
                'error_rate': self.error_count / max(self.request_count, 1)
            }
        }
        
        return optimization_report
    
    def handle_system_error(self, error: Exception, context: str = ""):
        """Handle system errors with circuit breaker pattern."""
        self.error_count += 1
        
        # Circuit breaker logic would be applied here
        # For now, just log the error context
        error_info = {
            'timestamp': time.time(),
            'error': str(error),
            'context': context,
            'circuit_breaker_state': self.circuit_breaker.get_state()
        }
        
        return error_info
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        recommendations = []
        
        # Get resource recommendations
        resource_summary = self.resource_monitor.get_resource_summary()
        if 'recommendations' in resource_summary:
            recommendations.extend(resource_summary['recommendations'])
        
        # Performance-based recommendations
        if self.request_count > 100:
            avg_response_time = self.total_response_time / self.request_count
            error_rate = self.error_count / self.request_count
            
            if avg_response_time > 1.0:
                recommendations.append({
                    'type': 'PERFORMANCE',
                    'message': f'High average response time ({avg_response_time:.3f}s). Consider optimization.',
                    'severity': 'MEDIUM',
                    'timestamp': time.time()
                })
            
            if error_rate > 0.05:  # 5% error rate
                recommendations.append({
                    'type': 'RELIABILITY',
                    'message': f'High error rate ({error_rate:.2%}). Check system stability.',
                    'severity': 'HIGH',
                    'timestamp': time.time()
                })
        
        return recommendations


# Global optimizer instances
global_optimizer = ModelOptimizer()
global_optimization_suite = PerformanceOptimizationSuite()