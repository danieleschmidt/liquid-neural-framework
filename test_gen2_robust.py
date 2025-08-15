#!/usr/bin/env python3
"""
Test script for Generation 2 robust implementation.
Tests error handling, validation, monitoring, and recovery mechanisms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import tempfile
import time
from models.liquid_neural_network import LiquidNeuralNetwork
from utils.model_validation import ModelValidator, ValidationError, safe_forward_pass
from utils.error_recovery import RobustModelWrapper, ModelCheckpoint, CircuitBreaker
from utils.monitoring import ModelMonitor, PerformanceMetrics, HealthChecker


def test_model_validation():
    """Test model validation functionality."""
    print("🧪 Testing ModelValidator...")
    
    validator = ModelValidator()
    
    # Test input validation
    try:
        x = jnp.array([1.0, 2.0, 3.0])
        validated = validator.validate_input_tensor(x, (3,), "test_input")
        assert validated.shape == (3,), "Input validation failed"
    except ValidationError:
        print("❌ Input validation test failed")
        return False
    
    # Test NaN detection
    try:
        x_nan = jnp.array([1.0, jnp.nan, 3.0])
        validator.validate_input_tensor(x_nan, (3,), "nan_input")
        print("❌ NaN detection failed")
        return False
    except ValidationError:
        pass  # Expected behavior
    
    # Test time step validation
    try:
        dt = validator.validate_time_step(0.01)
        assert dt == 0.01, "Time step validation failed"
    except ValidationError:
        print("❌ Time step validation failed")
        return False
    
    print("✅ ModelValidator tests passed")
    return True


def test_safe_forward_pass():
    """Test safe forward pass decorator."""
    print("🧪 Testing safe_forward_pass decorator...")
    
    @safe_forward_pass
    def dummy_forward(x):
        return x * 2
    
    @safe_forward_pass
    def failing_forward(x):
        return jnp.array([jnp.nan, 1.0])
    
    # Test successful forward pass
    try:
        result = dummy_forward(jnp.array([1.0, 2.0]))
        assert jnp.allclose(result, jnp.array([2.0, 4.0])), "Safe forward pass failed"
    except ValidationError:
        print("❌ Safe forward pass test failed")
        return False
    
    # Test failing forward pass (should raise ValidationError)
    try:
        failing_forward(jnp.array([1.0]))
        print("❌ Safe forward pass should have detected NaN")
        return False
    except ValidationError:
        pass  # Expected behavior
    
    print("✅ Safe forward pass tests passed")
    return True


def test_model_checkpoint():
    """Test model checkpointing functionality."""
    print("🧪 Testing ModelCheckpoint...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpointer = ModelCheckpoint(temp_dir, max_checkpoints=2)
        
        # Create a simple model
        key = jax.random.PRNGKey(42)
        model = LiquidNeuralNetwork(5, 10, 3, num_layers=1, key=key)
        
        # Save checkpoint
        try:
            checkpoint_path = checkpointer.save_checkpoint(model, step=100, 
                                                         metadata={"loss": 0.5})
            assert os.path.exists(checkpoint_path), "Checkpoint file not created"
        except Exception as e:
            print(f"❌ Checkpoint saving failed: {e}")
            return False
        
        # Load checkpoint
        try:
            loaded = checkpointer.load_checkpoint(checkpoint_path)
            assert loaded["step"] == 100, "Checkpoint step mismatch"
            assert loaded["metadata"]["loss"] == 0.5, "Checkpoint metadata mismatch"
        except Exception as e:
            print(f"❌ Checkpoint loading failed: {e}")
            return False
    
    print("✅ ModelCheckpoint tests passed")
    return True


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("🧪 Testing CircuitBreaker...")
    
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    
    def failing_func():
        raise RuntimeError("Test failure")
    
    def success_func():
        return "success"
    
    # Test failure accumulation
    for i in range(2):
        try:
            breaker.call(failing_func)
        except RuntimeError:
            pass
    
    # Circuit should be open now
    try:
        breaker.call(success_func)
        print("❌ Circuit breaker should be OPEN")
        return False
    except RuntimeError as e:
        if "Circuit breaker is OPEN" not in str(e):
            print(f"❌ Unexpected error: {e}")
            return False
    
    # Wait for recovery timeout
    time.sleep(0.2)
    
    # Should allow one attempt in HALF_OPEN
    try:
        result = breaker.call(success_func)
        assert result == "success", "Circuit breaker recovery failed"
    except Exception as e:
        print(f"❌ Circuit breaker recovery failed: {e}")
        return False
    
    print("✅ CircuitBreaker tests passed")
    return True


def test_robust_model_wrapper():
    """Test robust model wrapper."""
    print("🧪 Testing RobustModelWrapper...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model
        key = jax.random.PRNGKey(42)
        model = LiquidNeuralNetwork(5, 10, 3, num_layers=1, key=key)
        
        # Wrap with robust wrapper
        wrapper = RobustModelWrapper(model, checkpoint_dir=temp_dir)
        
        # Test normal operation
        try:
            x = jax.random.normal(key, (5,))
            hidden = wrapper.model.init_hidden()
            result, _ = wrapper(x, hidden)
            assert result.shape == (3,), "Robust wrapper normal operation failed"
        except Exception as e:
            print(f"❌ Robust wrapper normal operation failed: {e}")
            return False
        
        # Test checkpoint saving
        try:
            wrapper.save_checkpoint({"test": "metadata"})
        except Exception as e:
            print(f"❌ Robust wrapper checkpoint failed: {e}")
            return False
        
        # Test health report
        try:
            report = wrapper.get_health_report()
            assert "step" in report, "Health report missing step"
            assert "circuit_breaker_state" in report, "Health report missing circuit breaker state"
        except Exception as e:
            print(f"❌ Health report failed: {e}")
            return False
    
    print("✅ RobustModelWrapper tests passed")
    return True


def test_performance_metrics():
    """Test performance metrics tracking."""
    print("🧪 Testing PerformanceMetrics...")
    
    metrics = PerformanceMetrics(max_history=10)
    
    # Test timer functionality
    metrics.start_timer("test_operation")
    time.sleep(0.01)
    duration = metrics.end_timer("test_operation")
    
    assert duration > 0, "Timer duration should be positive"
    
    # Test metric recording
    metrics.record_metric("test_metric", 1.5)
    metrics.record_metric("test_metric", 2.5)
    metrics.record_metric("test_metric", 3.5)
    
    # Test statistics
    stats = metrics.get_statistics("test_metric")
    assert stats["count"] == 3, f"Expected count 3, got {stats['count']}"
    assert abs(stats["mean"] - 2.5) < 1e-6, f"Expected mean 2.5, got {stats['mean']}"
    
    # Test counter
    metrics.increment_counter("test_counter", 5)
    metrics.increment_counter("test_counter", 3)
    assert metrics.counters["test_counter"] == 8, "Counter increment failed"
    
    print("✅ PerformanceMetrics tests passed")
    return True


def test_model_monitor():
    """Test model monitoring functionality."""
    print("🧪 Testing ModelMonitor...")
    
    monitor = ModelMonitor("test_model")
    
    # Test health checks
    health = monitor.health_checker.get_overall_health()
    assert health in ["healthy", "degraded", "critical", "unknown"], f"Invalid health status: {health}"
    
    # Test forward pass monitoring
    key = jax.random.PRNGKey(42)
    model = LiquidNeuralNetwork(5, 10, 3, num_layers=1, key=key)
    
    @monitor.monitor_forward_pass
    def test_forward(x, hidden):
        return model(x, hidden)
    
    x = jax.random.normal(key, (5,))
    hidden = model.init_hidden()
    
    try:
        result, _ = test_forward(x, hidden)
        assert result.shape == (3,), "Monitored forward pass failed"
    except Exception as e:
        print(f"❌ Monitored forward pass failed: {e}")
        return False
    
    # Test monitoring report
    try:
        report = monitor.get_monitoring_report()
        assert "model_name" in report, "Report missing model name"
        assert "health" in report, "Report missing health"
        assert "metrics" in report, "Report missing metrics"
    except Exception as e:
        print(f"❌ Monitoring report failed: {e}")
        return False
    
    print("✅ ModelMonitor tests passed")
    return True


def test_all_robust_features():
    """Run all robustness tests."""
    print("🚀 Starting Generation 2 Robustness Tests\n")
    
    tests = [
        test_model_validation,
        test_safe_forward_pass,
        test_model_checkpoint,
        test_circuit_breaker,
        test_robust_model_wrapper,
        test_performance_metrics,
        test_model_monitor
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL GENERATION 2 ROBUSTNESS FEATURES WORKING!")
        return True
    else:
        print("⚠️  Some robustness tests failed")
        return False


if __name__ == "__main__":
    success = test_all_robust_features()
    sys.exit(0 if success else 1)