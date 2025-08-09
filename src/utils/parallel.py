"""
Parallel processing utilities for liquid neural networks.

This module provides concurrent processing, distributed computing,
and auto-scaling capabilities.
"""

import jax
import jax.numpy as jnp
from jax import pmap, device_put, devices
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Callable, Optional, Tuple
import threading
import time
from dataclasses import dataclass
from ..utils.logging import get_logger


@dataclass
class ProcessingTask:
    """Represents a processing task."""
    task_id: str
    inputs: jnp.ndarray
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class ParallelProcessor:
    """
    Parallel processor for liquid neural networks.
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, len(devices()))
        self.logger = get_logger()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.device_count = len(devices())
        
        # Initialize device mesh for distributed computing
        if self.device_count > 1:
            self.mesh = Mesh(mesh_utils.create_device_mesh((self.device_count,)), ('devices',))
        else:
            self.mesh = None
        
        self.logger.info(f"Initialized parallel processor with {self.max_workers} workers on {self.device_count} devices")
    
    def parallel_forward_pass(
        self,
        model_fn: Callable,
        batch_inputs: jnp.ndarray,
        model_params: Any,
        **kwargs
    ) -> jnp.ndarray:
        """
        Perform parallel forward pass across multiple devices.
        
        Args:
            model_fn: Model forward function
            batch_inputs: Batch of input sequences [batch_size, seq_len, input_dim]
            model_params: Model parameters
            **kwargs: Additional arguments for the model
            
        Returns:
            Batch of outputs
        """
        if self.device_count == 1:
            # Single device - use regular computation
            return model_fn(model_params, batch_inputs, **kwargs)
        
        # Multi-device parallel processing
        batch_size = batch_inputs.shape[0]
        
        # Split batch across devices
        device_batch_size = batch_size // self.device_count
        remainder = batch_size % self.device_count
        
        # Create device-specific batches
        device_batches = []
        start_idx = 0
        
        for i in range(self.device_count):
            # Handle uneven batch sizes
            current_batch_size = device_batch_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            if current_batch_size > 0:
                device_batch = batch_inputs[start_idx:end_idx]
                device_batches.append(device_batch)
            else:
                device_batches.append(jnp.empty((0, *batch_inputs.shape[1:])))
            
            start_idx = end_idx
        
        # Distribute batches to devices
        device_batches = [device_put(batch, device) for batch, device in zip(device_batches, devices())]
        device_params = [device_put(model_params, device) for device in devices()]
        
        # Create parallel mapped function
        parallel_fn = pmap(model_fn, in_axes=(0, 0))
        
        # Execute in parallel
        device_outputs = parallel_fn(jnp.stack(device_params), jnp.stack(device_batches))
        
        # Concatenate results
        outputs = jnp.concatenate([out for out in device_outputs if out.shape[0] > 0], axis=0)
        
        self.logger.debug(f"Parallel forward pass: {batch_size} samples across {self.device_count} devices")
        
        return outputs
    
    def async_process_sequences(
        self,
        model_fn: Callable,
        sequences: List[jnp.ndarray],
        model_params: Any,
        callback: Optional[Callable] = None
    ) -> List[asyncio.Future]:
        """
        Asynchronously process multiple sequences.
        
        Args:
            model_fn: Model forward function
            sequences: List of input sequences
            model_params: Model parameters
            callback: Optional callback for completed tasks
            
        Returns:
            List of futures for the processing tasks
        """
        futures = []
        
        for i, sequence in enumerate(sequences):
            future = self.executor.submit(
                self._process_single_sequence,
                model_fn, sequence, model_params, i, callback
            )
            futures.append(future)
        
        self.logger.debug(f"Started async processing of {len(sequences)} sequences")
        return futures
    
    def _process_single_sequence(
        self,
        model_fn: Callable,
        sequence: jnp.ndarray,
        model_params: Any,
        sequence_id: int,
        callback: Optional[Callable] = None
    ):
        """Process a single sequence."""
        try:
            start_time = time.time()
            result = model_fn(model_params, sequence)
            duration = time.time() - start_time
            
            if callback:
                callback(sequence_id, result, duration)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing sequence {sequence_id}: {str(e)}")
            raise


class TaskQueue:
    """
    Priority-based task queue for processing liquid neural network tasks.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.tasks = []  # Priority queue
        self.completed_tasks = {}
        self.lock = threading.Lock()
        self.logger = get_logger()
    
    def add_task(self, task: ProcessingTask) -> bool:
        """
        Add a task to the queue.
        
        Args:
            task: Task to add
            
        Returns:
            True if task was added, False if queue is full
        """
        with self.lock:
            if len(self.tasks) >= self.max_size:
                self.logger.warning("Task queue is full, dropping task")
                return False
            
            # Insert task based on priority (higher priority first)
            import bisect
            bisect.insort(self.tasks, (-task.priority, task.created_at, task))
            
            self.logger.debug(f"Added task {task.task_id} with priority {task.priority}")
            return True
    
    def get_next_task(self) -> Optional[ProcessingTask]:
        """Get the next task from the queue."""
        with self.lock:
            if not self.tasks:
                return None
            
            _, _, task = self.tasks.pop(0)
            return task
    
    def complete_task(self, task_id: str, result: Any, duration: float):
        """Mark a task as completed."""
        with self.lock:
            self.completed_tasks[task_id] = {
                "result": result,
                "duration": duration,
                "completed_at": time.time()
            }
            
            # Clean up old completed tasks
            if len(self.completed_tasks) > 100:
                oldest_tasks = sorted(self.completed_tasks.items(), 
                                    key=lambda x: x[1]["completed_at"])
                for task_id, _ in oldest_tasks[:50]:
                    del self.completed_tasks[task_id]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "pending_tasks": len(self.tasks),
                "completed_tasks": len(self.completed_tasks),
                "queue_capacity": self.max_size,
                "queue_utilization": len(self.tasks) / self.max_size
            }


class AutoScaler:
    """
    Auto-scaling system for liquid neural network processing.
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        monitoring_interval: float = 10.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitoring_interval = monitoring_interval
        
        self.current_workers = min_workers
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=min_workers)
        self.monitoring_thread = None
        self.is_monitoring = False
        
        self.performance_history = []
        self.scaling_events = []
        
        self.logger = get_logger()
    
    def start_monitoring(self, task_queue: TaskQueue):
        """Start auto-scaling monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_and_scale,
            args=(task_queue,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Started auto-scaling monitoring")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_and_scale(self, task_queue: TaskQueue):
        """Monitor performance and scale workers."""
        while self.is_monitoring:
            try:
                # Get queue statistics
                stats = task_queue.get_queue_stats()
                utilization = stats["queue_utilization"]
                
                # Record performance metrics
                self.performance_history.append({
                    "timestamp": time.time(),
                    "utilization": utilization,
                    "workers": self.current_workers,
                    "pending_tasks": stats["pending_tasks"]
                })
                
                # Keep only recent history
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
                
                # Scaling decisions
                if utilization > self.scale_up_threshold and self.current_workers < self.max_workers:
                    self._scale_up()
                elif utilization < self.scale_down_threshold and self.current_workers > self.min_workers:
                    self._scale_down()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling monitor: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _scale_up(self):
        """Scale up the number of workers."""
        old_workers = self.current_workers
        self.current_workers = min(self.current_workers + 1, self.max_workers)
        
        if self.current_workers > old_workers:
            # Recreate thread pool with more workers
            self.worker_pool.shutdown(wait=False)
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers)
            
            self.scaling_events.append({
                "timestamp": time.time(),
                "action": "scale_up",
                "old_workers": old_workers,
                "new_workers": self.current_workers
            })
            
            self.logger.info(f"Scaled up: {old_workers} -> {self.current_workers} workers")
    
    def _scale_down(self):
        """Scale down the number of workers."""
        old_workers = self.current_workers
        self.current_workers = max(self.current_workers - 1, self.min_workers)
        
        if self.current_workers < old_workers:
            # Recreate thread pool with fewer workers
            self.worker_pool.shutdown(wait=False)
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers)
            
            self.scaling_events.append({
                "timestamp": time.time(),
                "action": "scale_down",
                "old_workers": old_workers,
                "new_workers": self.current_workers
            })
            
            self.logger.info(f"Scaled down: {old_workers} -> {self.current_workers} workers")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scaling_events": len(self.scaling_events),
            "recent_events": self.scaling_events[-5:] if self.scaling_events else [],
            "performance_history_size": len(self.performance_history)
        }


class DistributedProcessor:
    """
    Distributed processing system for large-scale liquid neural network computations.
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.device_count = len(devices())
        
        # Set up device mesh if multiple devices available
        if self.device_count > 1:
            self.setup_device_mesh()
        else:
            self.mesh = None
            self.logger.info("Single device mode - no distributed processing")
    
    def setup_device_mesh(self):
        """Set up device mesh for distributed computing."""
        try:
            # Create 1D mesh for simplicity
            device_mesh = mesh_utils.create_device_mesh((self.device_count,))
            self.mesh = Mesh(device_mesh, ('batch',))
            
            self.logger.info(f"Set up device mesh with {self.device_count} devices")
        except Exception as e:
            self.logger.error(f"Failed to set up device mesh: {str(e)}")
            self.mesh = None
    
    def distribute_computation(
        self,
        computation_fn: Callable,
        data: jnp.ndarray,
        partition_spec: PartitionSpec = None
    ) -> jnp.ndarray:
        """
        Distribute computation across devices.
        
        Args:
            computation_fn: Function to compute
            data: Input data
            partition_spec: How to partition the data
            
        Returns:
            Computed results
        """
        if self.mesh is None:
            return computation_fn(data)
        
        # Default partitioning along batch dimension
        if partition_spec is None:
            partition_spec = P('batch')
        
        # Shard data across devices
        with self.mesh:
            sharded_data = jax.device_put(data, partition_spec)
            result = computation_fn(sharded_data)
            
            # Gather results
            return jax.device_get(result)


# Global instances
_global_parallel_processor = None
_global_task_queue = None
_global_auto_scaler = None

def get_parallel_processor() -> ParallelProcessor:
    """Get or create global parallel processor."""
    global _global_parallel_processor
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelProcessor()
    return _global_parallel_processor

def get_task_queue() -> TaskQueue:
    """Get or create global task queue."""
    global _global_task_queue
    if _global_task_queue is None:
        _global_task_queue = TaskQueue()
    return _global_task_queue

def get_auto_scaler() -> AutoScaler:
    """Get or create global auto-scaler."""
    global _global_auto_scaler
    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler()
    return _global_auto_scaler