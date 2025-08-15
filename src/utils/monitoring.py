"""
Comprehensive monitoring and telemetry for liquid neural networks.
Includes performance metrics, health checks, and real-time monitoring.
"""

import jax
import jax.numpy as jnp
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from pathlib import Path
import threading
import queue
import datetime


class PerformanceMetrics:
    """Track performance metrics for model operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.start_times = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        if operation not in self.start_times:
            self.logger.warning(f"Timer for {operation} was not started")
            return 0.0
        
        duration = time.perf_counter() - self.start_times[operation]
        self.metrics[f"{operation}_duration"].append(duration)
        del self.start_times[operation]
        return duration
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric value."""
        timestamp = time.time()
        metric_data = {
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {}
        }
        self.metrics[name].append(metric_data)
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter."""
        self.counters[name] += amount
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics:
            return {}
        
        values = [
            m["value"] if isinstance(m, dict) else m 
            for m in self.metrics[metric_name]
        ]
        
        if not values:
            return {}
        
        values_array = jnp.array(values)
        return {
            "count": len(values),
            "mean": float(jnp.mean(values_array)),
            "std": float(jnp.std(values_array)),
            "min": float(jnp.min(values_array)),
            "max": float(jnp.max(values_array)),
            "p50": float(jnp.percentile(values_array, 50)),
            "p95": float(jnp.percentile(values_array, 95)),
            "p99": float(jnp.percentile(values_array, 99))
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_statistics(name) for name in self.metrics.keys()}


class HealthChecker:
    """Health monitoring for model systems."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_results = {}
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      description: str = ""):
        """Register a health check function."""
        self.checks[name] = {
            "func": check_func,
            "description": description
        }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {"status": "error", "message": f"Check {name} not found"}
        
        try:
            start_time = time.perf_counter()
            result = self.checks[name]["func"]()
            duration = time.perf_counter() - start_time
            
            check_result = {
                "status": "healthy" if result else "unhealthy",
                "duration": duration,
                "timestamp": time.time(),
                "description": self.checks[name]["description"]
            }
            
            self.last_check_results[name] = check_result
            return check_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
                "description": self.checks[name]["description"]
            }
            self.last_check_results[name] = error_result
            return error_result
    
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks.keys():
            results[name] = self.run_check(name)
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if not results:
            return "unknown"
        
        statuses = [r["status"] for r in results.values()]
        
        if all(s == "healthy" for s in statuses):
            return "healthy"
        elif any(s == "error" for s in statuses):
            return "critical"
        else:
            return "degraded"


class ModelMonitor:
    """Comprehensive model monitoring."""
    
    def __init__(self, model_name: str = "liquid_neural_network"):
        self.model_name = model_name
        self.metrics = PerformanceMetrics()
        self.health_checker = HealthChecker()
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        def memory_check():
            """Basic memory usage check."""
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90
            except ImportError:
                return True  # Skip if psutil not available
        
        def jax_device_check():
            """Check JAX device availability."""
            try:
                devices = jax.devices()
                return len(devices) > 0
            except Exception:
                return False
        
        self.health_checker.register_check(
            "memory", memory_check, "System memory usage"
        )
        self.health_checker.register_check(
            "jax_devices", jax_device_check, "JAX device availability"
        )
    
    def monitor_forward_pass(self, func):
        """Decorator to monitor forward pass operations."""
        def wrapper(*args, **kwargs):
            self.metrics.start_timer("forward_pass")
            self.metrics.increment_counter("forward_pass_calls")
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                self.metrics.increment_counter("forward_pass_success")
                
                # Monitor output health
                if isinstance(result, tuple):
                    for i, output in enumerate(result):
                        if isinstance(output, jnp.ndarray):
                            self._monitor_tensor_health(output, f"output_{i}")
                elif isinstance(result, jnp.ndarray):
                    self._monitor_tensor_health(result, "output")
                
                return result
                
            except Exception as e:
                self.metrics.increment_counter("forward_pass_errors")
                self.logger.error(f"Forward pass error: {e}")
                raise
            finally:
                self.metrics.end_timer("forward_pass")
        
        return wrapper
    
    def _monitor_tensor_health(self, tensor: jnp.ndarray, name: str):
        """Monitor tensor for health metrics."""
        # Check for NaN/Inf
        nan_count = jnp.sum(jnp.isnan(tensor))
        inf_count = jnp.sum(jnp.isinf(tensor))
        
        self.metrics.record_metric(f"{name}_nan_count", float(nan_count))
        self.metrics.record_metric(f"{name}_inf_count", float(inf_count))
        
        # Record statistics
        if tensor.size > 0:
            self.metrics.record_metric(f"{name}_mean", float(jnp.mean(tensor)))
            self.metrics.record_metric(f"{name}_std", float(jnp.std(tensor)))
            self.metrics.record_metric(f"{name}_min", float(jnp.min(tensor)))
            self.metrics.record_metric(f"{name}_max", float(jnp.max(tensor)))
    
    def monitor_training_step(self, loss: float, gradients: Any):
        """Monitor training step metrics."""
        self.metrics.record_metric("loss", loss)
        
        # Monitor gradient health
        if gradients is not None:
            grad_norm = jnp.sqrt(sum(
                jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gradients)
                if isinstance(g, jnp.ndarray)
            ))
            self.metrics.record_metric("gradient_norm", float(grad_norm))
            
            # Check for gradient issues
            nan_grads = sum(
                jnp.sum(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(gradients)
                if isinstance(g, jnp.ndarray)
            )
            self.metrics.record_metric("gradient_nan_count", float(nan_grads))
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        return {
            "model_name": self.model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "health": self.health_checker.get_overall_health(),
            "health_checks": self.health_checker.last_check_results,
            "metrics": self.metrics.get_all_statistics(),
            "counters": dict(self.metrics.counters)
        }
    
    def save_report(self, filepath: Optional[str] = None):
        """Save monitoring report to file."""
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"monitoring_report_{timestamp}.json"
        
        report = self.get_monitoring_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Monitoring report saved to {filepath}")


class RealTimeMonitor:
    """Real-time monitoring with background thread."""
    
    def __init__(self, monitor: ModelMonitor, check_interval: float = 30.0):
        self.monitor = monitor
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.alerts = queue.Queue()
        self.alert_thresholds = {}
        
    def set_alert_threshold(self, metric: str, threshold: float, 
                          condition: str = "greater"):
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric] = {
            "threshold": threshold,
            "condition": condition
        }
    
    def start(self):
        """Start real-time monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        self.monitor.logger.info("Real-time monitoring started")
    
    def stop(self):
        """Stop real-time monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.monitor.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Run health checks
                health_results = self.monitor.health_checker.run_all_checks()
                
                # Check for alerts
                self._check_alerts(health_results)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.monitor.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.check_interval)
    
    def _check_alerts(self, health_results: Dict[str, Dict[str, Any]]):
        """Check for alert conditions."""
        # Health-based alerts
        for check_name, result in health_results.items():
            if result["status"] != "healthy":
                alert = {
                    "type": "health_check_failed",
                    "check": check_name,
                    "status": result["status"],
                    "timestamp": time.time()
                }
                self.alerts.put(alert)
        
        # Metric-based alerts
        for metric, config in self.alert_thresholds.items():
            stats = self.monitor.metrics.get_statistics(metric)
            if not stats:
                continue
            
            current_value = stats.get("mean", 0)
            threshold = config["threshold"]
            condition = config["condition"]
            
            triggered = False
            if condition == "greater" and current_value > threshold:
                triggered = True
            elif condition == "less" and current_value < threshold:
                triggered = True
            
            if triggered:
                alert = {
                    "type": "metric_threshold",
                    "metric": metric,
                    "value": current_value,
                    "threshold": threshold,
                    "condition": condition,
                    "timestamp": time.time()
                }
                self.alerts.put(alert)
    
    def get_alerts(self, max_alerts: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        alerts = []
        count = 0
        
        while not self.alerts.empty() and count < max_alerts:
            alerts.append(self.alerts.get())
            count += 1
        
        return alerts