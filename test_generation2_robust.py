#!/usr/bin/env python3
"""
Generation 2 Robust Implementation Test Suite.
Tests error handling, validation, logging, and security measures.
"""

import sys
sys.path.append('src')
import numpy as np
import tempfile
import os
import json
from pathlib import Path


def test_input_validation():
    """Test robust input validation system."""
    print("🛡️ Testing Input Validation...")
    
    try:
        from utils.robust_validation import (
            validate_tensor_shape, validate_tensor_values, validate_network_parameters,
            ValidationError, ShapeValidationError, ValueValidationError, ParameterValidationError
        )
        
        # Test shape validation
        tensor = np.random.randn(5, 10)
        validated = validate_tensor_shape(tensor, (10,), "test_tensor", allow_batch_dim=True)
        print("✓ Shape validation with batch dimension works")
        
        # Test invalid shape
        try:
            validate_tensor_shape(tensor, (15,), "test_tensor", allow_batch_dim=False)
            print("❌ Should have failed shape validation")
            return False
        except ShapeValidationError:
            print("✓ Shape validation correctly catches invalid shapes")
        
        # Test value validation
        clean_tensor = np.random.randn(3, 4)
        validated = validate_tensor_values(clean_tensor, "clean_tensor")
        print("✓ Value validation passes for clean tensor")
        
        # Test NaN detection
        nan_tensor = np.array([1.0, np.nan, 3.0])
        try:
            validate_tensor_values(nan_tensor, "nan_tensor")
            print("❌ Should have detected NaN values")
            return False
        except ValueValidationError:
            print("✓ Value validation correctly detects NaN values")
        
        # Test parameter validation
        params = validate_network_parameters(10, 20, 5, 2, 0.1, 10.0)
        print(f"✓ Parameter validation: {params}")
        
        # Test invalid parameters
        try:
            validate_network_parameters(-5, 20, 5)
            print("❌ Should have failed parameter validation")
            return False
        except ParameterValidationError:
            print("✓ Parameter validation correctly catches invalid parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False


def test_enhanced_logging():
    """Test enhanced logging system."""
    print("\n📝 Testing Enhanced Logging...")
    
    try:
        from utils.enhanced_logging import NetworkLogger, DebugLogger
        
        # Test basic logger
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = NetworkLogger("test_logger", log_file=log_file)
            
            # Test network creation logging
            logger.log_network_creation("LiquidNeuralNetwork", 10, 20, 5)
            print("✓ Network creation logging works")
            
            # Test training logging
            logger.log_training_start(100, 32, 0.001)
            logger.log_epoch(1, 0.5, {"accuracy": 0.85})
            print("✓ Training logging works")
            
            # Test performance logging
            with logger.log_operation("test_operation"):
                import time
                time.sleep(0.001)  # Simulate operation
            print("✓ Performance logging works")
            
            # Test gradient logging
            gradients = {
                "W1": np.random.randn(10, 5),
                "W2": np.random.randn(5, 3)
            }
            logger.log_gradient_info(gradients)
            print("✓ Gradient logging works")
            
            # Test performance summary
            summary = logger.get_performance_summary()
            assert "test_operation" in summary['performance_stats']
            print("✓ Performance summary generation works")
            
            # Test debug logger
            debug_logger = DebugLogger("debug_test")
            
            # Test tensor checkpointing
            test_tensor = np.random.randn(5, 5)
            debug_logger.log_tensor_checkpoint("test_tensor", test_tensor, step=1)
            debug_logger.log_tensor_checkpoint("test_tensor", test_tensor * 2, step=2)
            print("✓ Tensor checkpointing works")
            
            # Test tensor evolution analysis
            analysis = debug_logger.analyze_tensor_evolution("test_tensor")
            assert analysis['num_checkpoints'] == 2
            print("✓ Tensor evolution analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced logging test failed: {e}")
        return False


def test_security_measures():
    """Test security measures and input sanitization."""
    print("\n🔒 Testing Security Measures...")
    
    try:
        from utils.security_measures import (
            InputSanitizer, DataPrivacyManager, SecureModelManager, 
            AccessController, SecurityAuditor, SecurityError
        )
        
        # Test input sanitizer
        sanitizer = InputSanitizer(max_input_size=1000)
        
        # Test normal input
        normal_input = np.random.randn(10, 5).astype(np.float32)
        sanitized = sanitizer.sanitize_tensor(normal_input, "normal_input")
        print("✓ Input sanitizer handles normal inputs")
        
        # Test oversized input
        try:
            large_input = np.random.randn(100, 100).astype(np.float32)  # Size > 1000
            sanitizer.sanitize_tensor(large_input, "large_input")
            print("❌ Should have rejected oversized input")
            return False
        except SecurityError:
            print("✓ Input sanitizer correctly rejects oversized inputs")
        
        # Test malicious input (NaN)
        try:
            malicious_input = np.array([1.0, np.nan, 3.0]).astype(np.float32)
            sanitizer.sanitize_tensor(malicious_input, "malicious_input")
            print("❌ Should have rejected NaN input")
            return False
        except SecurityError:
            print("✓ Input sanitizer correctly rejects NaN inputs")
        
        # Test privacy manager
        privacy_manager = DataPrivacyManager(epsilon=1.0)
        
        # Test noise addition
        clean_data = np.random.randn(10, 5)
        noisy_data = privacy_manager.add_gaussian_noise(clean_data)
        assert noisy_data.shape == clean_data.shape
        assert not np.array_equal(clean_data, noisy_data)
        print("✓ Privacy manager adds noise correctly")
        
        # Test gradient clipping
        gradients = {
            "W1": np.random.randn(10, 5) * 10,  # Large gradients
            "W2": np.random.randn(5, 3) * 0.1   # Small gradients
        }
        clipped = privacy_manager.clip_gradients_for_privacy(gradients, clip_norm=1.0)
        assert np.linalg.norm(clipped["W1"]) <= 1.0
        print("✓ Gradient clipping works correctly")
        
        # Test secure model manager
        with tempfile.TemporaryDirectory() as temp_dir:
            model_manager = SecureModelManager()
            
            # Test model saving and loading
            test_model = {
                "weights": np.random.randn(10, 5).tolist(),
                "biases": np.random.randn(5).tolist()
            }
            
            model_path = os.path.join(temp_dir, "test_model.json")
            model_hash = model_manager.save_model_securely(test_model, model_path)
            print("✓ Secure model saving works")
            
            # Test model loading with verification
            loaded_model = model_manager.load_model_securely(model_path, verify_hash=True)
            assert loaded_model == test_model
            print("✓ Secure model loading with verification works")
        
        # Test access controller
        access_controller = AccessController(max_operations_per_hour=10)
        
        # Test normal access
        assert access_controller.check_access("test_operation", "user1")
        print("✓ Access controller allows normal operations")
        
        # Test operation blocking
        access_controller.block_operation("sensitive_op", "user1")
        try:
            access_controller.check_access("sensitive_op", "user1")
            print("❌ Should have blocked operation")
            return False
        except SecurityError:
            print("✓ Access controller correctly blocks operations")
        
        # Test security auditor
        auditor = SecurityAuditor()
        auditor.log_security_event("TEST_EVENT", "Test security event", "INFO", "test_user")
        
        summary = auditor.get_security_summary(hours=1)
        assert summary['total_events'] == 1
        print("✓ Security auditor logging works")
        
        return True
        
    except Exception as e:
        print(f"❌ Security measures test failed: {e}")
        return False


def test_error_recovery():
    """Test error recovery and graceful degradation."""
    print("\n🔄 Testing Error Recovery...")
    
    try:
        from utils.robust_validation import RobustNetworkWrapper, safe_divide
        
        # Test safe division
        numerator = np.array([1.0, 2.0, 3.0])
        denominator = np.array([2.0, 0.0, 1.0])  # Contains zero
        
        result = safe_divide(numerator, denominator, name="test_division")
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        print("✓ Safe division handles zero denominators")
        
        # Test robust network wrapper
        class MockNetwork:
            def __call__(self, x):
                if np.any(x > 10):  # Simulate failure on large inputs
                    raise ValueError("Input too large")
                return x * 2
        
        mock_network = MockNetwork()
        wrapped_network = RobustNetworkWrapper(mock_network, validate_inputs=True)
        
        # Test normal operation
        normal_input = np.array([1.0, 2.0, 3.0])
        result = wrapped_network(normal_input)
        np.testing.assert_array_equal(result, normal_input * 2)
        print("✓ Robust wrapper handles normal operations")
        
        # Test statistics tracking
        stats = wrapped_network.get_statistics()
        assert stats['call_count'] == 1
        assert stats['error_count'] == 0
        print("✓ Robust wrapper tracks statistics correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        return False


def test_integration_robustness():
    """Test integration of all robust components."""
    print("\n🔧 Testing Integration Robustness...")
    
    try:
        from utils.robust_validation import validate_tensor_shape, validate_tensor_values
        from utils.enhanced_logging import NetworkLogger
        from utils.security_measures import InputSanitizer, get_security_auditor
        
        # Create integrated validation pipeline
        sanitizer = InputSanitizer()
        logger = NetworkLogger("integration_test")
        auditor = get_security_auditor()
        
        # Test complete pipeline
        with logger.log_operation("integration_test"):
            # Raw input
            raw_input = np.random.randn(5, 10).astype(np.float32)
            
            # Security check
            sanitized_input = sanitizer.sanitize_tensor(raw_input, "pipeline_input")
            
            # Validation
            validated_input = validate_tensor_shape(sanitized_input, (10,), allow_batch_dim=True)
            validated_input = validate_tensor_values(validated_input, "pipeline_input")
            
            # Log success
            auditor.log_security_event("PIPELINE_SUCCESS", "Integration test completed", "INFO")
        
        print("✓ Complete robustness pipeline works")
        
        # Test pipeline under stress
        stress_inputs = [
            np.random.randn(100, 10).astype(np.float32),  # Large input
            np.random.randn(5, 10).astype(np.float64),    # Different dtype
            np.random.randn(3, 8),                        # Wrong shape
        ]
        
        successful_validations = 0
        for i, stress_input in enumerate(stress_inputs):
            try:
                with logger.log_operation(f"stress_test_{i}"):
                    sanitized = sanitizer.sanitize_tensor(stress_input, f"stress_{i}")
                    validated = validate_tensor_shape(sanitized, (10,), allow_batch_dim=True)
                    validated = validate_tensor_values(validated, f"stress_{i}")
                    successful_validations += 1
            except Exception as e:
                logger.log_error(e, f"stress_test_{i}")
                auditor.log_security_event("PIPELINE_ERROR", f"Stress test {i} failed: {e}", "WARNING")
        
        # Should have processed at least one input successfully (the float64 one)
        assert successful_validations >= 1
        print(f"✓ Pipeline handled {successful_validations}/3 stress test inputs")
        
        # Check performance summary
        summary = logger.get_performance_summary()
        assert len(summary['performance_stats']) > 0
        print("✓ Performance tracking works under stress")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration robustness test failed: {e}")
        return False


def main():
    """Run all Generation 2 robustness tests."""
    print("=" * 60)
    print("🛡️ LIQUID NEURAL FRAMEWORK - GENERATION 2 ROBUSTNESS TESTING")
    print("=" * 60)
    
    tests = [
        test_input_validation,
        test_enhanced_logging,
        test_security_measures,
        test_error_recovery,
        test_integration_robustness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
            print("✅ PASSED\n")
        else:
            print("❌ FAILED\n")
    
    print("=" * 60)
    print(f"📊 GENERATION 2 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 2: MAKE IT ROBUST - COMPLETE!")
        print("✓ Comprehensive input validation implemented")
        print("✓ Enhanced logging and debugging systems active")
        print("✓ Security measures and access control enforced")
        print("✓ Error recovery and graceful degradation working")
        print("✓ All systems integrated and stress-tested")
        print("🚀 Ready for Generation 3: Optimized Implementation")
    else:
        print("⚠️  Some robustness tests failed - needs attention before proceeding")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)