"""
Scaling infrastructure for liquid neural networks.
Includes auto-scaling, load balancing, and distributed computing capabilities.
"""

import jax
import jax.numpy as jnp
import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import psutil


@dataclass
class WorkloadMetrics:
    """Metrics for workload monitoring."""
    requests_per_second: float
    average_latency: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    queue_length: int
    timestamp: float


class LoadBalancer:
    """Load balancer for distributing requests across model instances."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances = []
        self.instance_stats = {}
        self.current_index = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_instance(self, instance_id: str, model_callable: Callable):
        """Add a model instance to the load balancer."""
        with self.lock:
            self.instances.append({
                "id": instance_id,
                "model": model_callable,
                "active": True
            })
            self.instance_stats[instance_id] = {
                "requests": 0,
                "errors": 0,
                "total_latency": 0.0,
                "last_used": time.time()
            }
        self.logger.info(f"Added instance {instance_id}")
    
    def remove_instance(self, instance_id: str):
        """Remove a model instance from the load balancer."""
        with self.lock:
            self.instances = [
                inst for inst in self.instances 
                if inst["id"] != instance_id
            ]
            if instance_id in self.instance_stats:
                del self.instance_stats[instance_id]
        self.logger.info(f"Removed instance {instance_id}")
    
    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get next instance based on load balancing strategy."""
        with self.lock:
            active_instances = [inst for inst in self.instances if inst["active"]]
            
            if not active_instances:
                return None
            
            if self.strategy == "round_robin":
                instance = active_instances[self.current_index % len(active_instances)]
                self.current_index += 1
                return instance
            
            elif self.strategy == "least_connections":
                # Use least recently used as proxy for least connections
                return min(
                    active_instances,
                    key=lambda x: self.instance_stats[x["id"]]["last_used"]
                )
            
            elif self.strategy == "least_latency":
                # Choose instance with lowest average latency
                return min(
                    active_instances,
                    key=lambda x: self._get_average_latency(x["id"])
                )
            
            else:
                return active_instances[0]
    
    def _get_average_latency(self, instance_id: str) -> float:
        """Get average latency for an instance."""
        stats = self.instance_stats[instance_id]
        if stats["requests"] == 0:
            return 0.0
        return stats["total_latency"] / stats["requests"]
    
    def process_request(self, *args, **kwargs) -> Any:
        """Process a request using load balancing."""
        instance = self.get_next_instance()
        if instance is None:
            raise RuntimeError("No active instances available")
        
        instance_id = instance["id"]
        start_time = time.perf_counter()
        
        try:
            result = instance["model"](*args, **kwargs)
            
            # Update stats
            latency = time.perf_counter() - start_time
            with self.lock:
                stats = self.instance_stats[instance_id]
                stats["requests"] += 1
                stats["total_latency"] += latency
                stats["last_used"] = time.time()
            
            return result
            
        except Exception as e:
            # Update error stats
            with self.lock:
                self.instance_stats[instance_id]["errors"] += 1
            raise e
    
    def get_instance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all instances."""
        with self.lock:
            stats = {}
            for instance_id, raw_stats in self.instance_stats.items():
                stats[instance_id] = {
                    "requests": raw_stats["requests"],
                    "errors": raw_stats["errors"],
                    "error_rate": (
                        raw_stats["errors"] / max(raw_stats["requests"], 1)
                    ),
                    "average_latency": self._get_average_latency(instance_id),
                    "last_used": raw_stats["last_used"]
                }
            return stats


class AutoScaler:
    """Auto-scaling system for model instances."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3,
                 scale_up_cooldown: float = 60.0, scale_down_cooldown: float = 120.0):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.metrics_history = []
        self.scaling_decisions = []
        
        self.logger = logging.getLogger(__name__)
    
    def should_scale_up(self, metrics: WorkloadMetrics) -> bool:
        """Determine if we should scale up."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_up < self.scale_up_cooldown:
            return False
        
        # Check metrics
        conditions = [
            metrics.cpu_usage > self.scale_up_threshold,
            metrics.memory_usage > self.scale_up_threshold,
            metrics.average_latency > 1.0,  # 1 second threshold
            metrics.queue_length > 10
        ]
        
        # Scale up if any critical condition is met
        return any(conditions)
    
    def should_scale_down(self, metrics: WorkloadMetrics) -> bool:
        """Determine if we should scale down."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_down < self.scale_down_cooldown:
            return False
        
        # Check metrics (all must be low)
        conditions = [
            metrics.cpu_usage < self.scale_down_threshold,
            metrics.memory_usage < self.scale_down_threshold,
            metrics.average_latency < 0.1,  # 100ms threshold
            metrics.queue_length < 2
        ]
        
        # Scale down only if all conditions are met
        return all(conditions)
    
    def make_scaling_decision(self, metrics: WorkloadMetrics, 
                            current_instances: int) -> str:
        """Make scaling decision based on metrics."""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        decision = "maintain"
        
        if current_instances < self.max_instances and self.should_scale_up(metrics):
            decision = "scale_up"
            self.last_scale_up = time.time()
            
        elif current_instances > self.min_instances and self.should_scale_down(metrics):
            decision = "scale_down"
            self.last_scale_down = time.time()
        
        # Log decision
        decision_record = {
            "timestamp": time.time(),
            "decision": decision,
            "current_instances": current_instances,
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "latency": metrics.average_latency,
                "queue_length": metrics.queue_length
            }
        }
        self.scaling_decisions.append(decision_record)
        
        if decision != "maintain":
            self.logger.info(f"Scaling decision: {decision} (instances: {current_instances})")
        
        return decision


class DistributedModelManager:
    """Manage distributed model instances with auto-scaling."""
    
    def __init__(self, model_factory: Callable[[], Any], 
                 initial_instances: int = 2):
        self.model_factory = model_factory
        self.load_balancer = LoadBalancer(strategy="least_latency")
        self.auto_scaler = AutoScaler()
        self.request_queue = queue.Queue()
        
        self.running = False
        self.worker_threads = []
        self.metrics_thread = None
        self.scaling_thread = None
        
        self.current_metrics = WorkloadMetrics(
            requests_per_second=0.0,
            average_latency=0.0,
            error_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            queue_length=0,
            timestamp=time.time()
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize instances
        self._initialize_instances(initial_instances)
    
    def _initialize_instances(self, count: int):
        """Initialize model instances."""
        for i in range(count):
            self._add_instance(f"instance_{i}")
    
    def _add_instance(self, instance_id: str):
        """Add a new model instance."""
        try:
            model = self.model_factory()
            self.load_balancer.add_instance(instance_id, model)
            self.logger.info(f"Successfully added instance {instance_id}")
        except Exception as e:
            self.logger.error(f"Failed to add instance {instance_id}: {e}")
    
    def _remove_instance(self, instance_id: str):
        """Remove a model instance."""
        self.load_balancer.remove_instance(instance_id)
    
    def start(self):
        """Start the distributed model manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(4):  # 4 worker threads
            thread = threading.Thread(target=self._worker_loop)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        # Start auto-scaling thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()
        
        self.logger.info("Distributed model manager started")
    
    def stop(self):
        """Stop the distributed model manager."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5.0)
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        self.logger.info("Distributed model manager stopped")
    
    def _worker_loop(self):
        """Main worker loop for processing requests."""
        while self.running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1.0)
                
                # Process request
                start_time = time.perf_counter()
                try:
                    result = self.load_balancer.process_request(
                        *request["args"], **request["kwargs"]
                    )
                    request["future"].set_result(result)
                except Exception as e:
                    request["future"].set_exception(e)
                finally:
                    latency = time.perf_counter() - start_time
                    self._update_request_metrics(latency, success=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def _metrics_loop(self):
        """Metrics collection loop."""
        request_counts = []
        latencies = []
        
        while self.running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_percent = psutil.virtual_memory().percent
                
                # Calculate request metrics
                current_time = time.time()
                recent_requests = [
                    count for count, timestamp in request_counts
                    if current_time - timestamp < 60.0  # Last minute
                ]
                
                recent_latencies = [
                    lat for lat, timestamp in latencies
                    if current_time - timestamp < 60.0  # Last minute
                ]
                
                rps = len(recent_requests) / 60.0 if recent_requests else 0.0
                avg_latency = (
                    sum(recent_latencies) / len(recent_latencies)
                    if recent_latencies else 0.0
                )
                
                # Update current metrics
                self.current_metrics = WorkloadMetrics(
                    requests_per_second=rps,
                    average_latency=avg_latency,
                    error_rate=0.0,  # TODO: Track errors
                    cpu_usage=cpu_percent / 100.0,
                    memory_usage=memory_percent / 100.0,
                    queue_length=self.request_queue.qsize(),
                    timestamp=current_time
                )
                
                # Clean old data
                request_counts = [
                    item for item in request_counts
                    if current_time - item[1] < 300.0  # Keep 5 minutes
                ]
                latencies = [
                    item for item in latencies
                    if current_time - item[1] < 300.0  # Keep 5 minutes
                ]
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
            
            time.sleep(10.0)  # Collect metrics every 10 seconds
    
    def _scaling_loop(self):
        """Auto-scaling loop."""
        while self.running:
            try:
                current_instances = len(self.load_balancer.instances)
                decision = self.auto_scaler.make_scaling_decision(
                    self.current_metrics, current_instances
                )
                
                if decision == "scale_up":
                    new_instance_id = f"instance_{current_instances}"
                    self._add_instance(new_instance_id)
                    
                elif decision == "scale_down" and current_instances > 1:
                    # Remove the least used instance
                    instance_stats = self.load_balancer.get_instance_stats()
                    if instance_stats:
                        least_used = min(
                            instance_stats.items(),
                            key=lambda x: x[1]["requests"]
                        )
                        self._remove_instance(least_used[0])
                
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
            
            time.sleep(30.0)  # Check scaling every 30 seconds
    
    def _update_request_metrics(self, latency: float, success: bool):
        """Update request metrics."""
        # This would be implemented to track request metrics
        pass
    
    def predict(self, *args, **kwargs) -> Future:
        """Submit prediction request."""
        future = Future()
        request = {
            "args": args,
            "kwargs": kwargs,
            "future": future
        }
        
        self.request_queue.put(request)
        return future
    
    def predict_sync(self, *args, **kwargs) -> Any:
        """Synchronous prediction."""
        future = self.predict(*args, **kwargs)
        return future.result(timeout=30.0)  # 30 second timeout
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "running": self.running,
            "instances": len(self.load_balancer.instances),
            "queue_length": self.request_queue.qsize(),
            "current_metrics": {
                "rps": self.current_metrics.requests_per_second,
                "latency": self.current_metrics.average_latency,
                "cpu_usage": self.current_metrics.cpu_usage,
                "memory_usage": self.current_metrics.memory_usage
            },
            "instance_stats": self.load_balancer.get_instance_stats()
        }


class ResourceMonitor:
    """Monitor system resources for scaling decisions."""
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.resource_history = []
        self.alerts = []
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect resource metrics
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=1.0),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent,
                    "network_io": psutil.net_io_counters()._asdict(),
                    "process_count": len(psutil.pids())
                }
                
                self.resource_history.append(metrics)
                
                # Keep only recent history
                if len(self.resource_history) > 1440:  # 24 hours at 1 minute intervals
                    self.resource_history.pop(0)
                
                # Check for alerts
                self._check_resource_alerts(metrics)
                
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_resource_alerts(self, metrics: Dict[str, Any]):
        """Check for resource-based alerts."""
        current_time = time.time()
        
        # CPU alert
        if metrics["cpu_percent"] > 90:
            alert = {
                "type": "high_cpu",
                "value": metrics["cpu_percent"],
                "threshold": 90,
                "timestamp": current_time
            }
            self.alerts.append(alert)
        
        # Memory alert
        if metrics["memory_percent"] > 85:
            alert = {
                "type": "high_memory",
                "value": metrics["memory_percent"],
                "threshold": 85,
                "timestamp": current_time
            }
            self.alerts.append(alert)
        
        # Disk alert
        if metrics["disk_usage"] > 90:
            alert = {
                "type": "high_disk",
                "value": metrics["disk_usage"],
                "threshold": 90,
                "timestamp": current_time
            }
            self.alerts.append(alert)
        
        # Keep only recent alerts
        self.alerts = [
            alert for alert in self.alerts
            if current_time - alert["timestamp"] < 3600  # 1 hour
        ]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        if not self.resource_history:
            return {}
        return self.resource_history[-1]
    
    def get_resource_trends(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get resource trends over time."""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.resource_history
            if m["timestamp"] > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            "cpu_percent": [m["cpu_percent"] for m in recent_metrics],
            "memory_percent": [m["memory_percent"] for m in recent_metrics],
            "disk_usage": [m["disk_usage"] for m in recent_metrics],
            "timestamps": [m["timestamp"] for m in recent_metrics]
        }