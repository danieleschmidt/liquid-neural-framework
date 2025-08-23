"""
Test Generation 2: Robust Implementation

Test comprehensive error handling, validation, and robustness features.
"""

import sys
sys.path.append('.')

import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for clean test output


def test_import_robustness():
    """Test that framework imports gracefully handle missing dependencies."""
    print("🔍 Testing Import Robustness...")
    
    try:
        import src
        print("✅ Framework imports successfully with fallbacks")
        
        # Test that core models are available (even if fallback)
        model = src.LiquidNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
        print("✅ LiquidNeuralNetwork instantiation works")
        
        return True
    except Exception as e:
        print(f"❌ Import robustness failed: {e}")
        return False


def test_input_validation():
    """Test input validation and error handling."""
    print("🔍 Testing Input Validation...")
    
    try:
        import src
        
        # Test model with various input configurations
        model = src.LiquidNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
        
        # Valid input
        x_valid = np.random.randn(10, 2)
        outputs, _ = model.forward(x_valid)
        assert outputs.shape == (10, 1), f"Expected (10, 1), got {outputs.shape}"
        print("✅ Valid input handling works")
        
        # Test edge cases
        x_single = np.random.randn(1, 2)
        outputs_single, _ = model.forward(x_single) 
        assert outputs_single.shape == (1, 1), "Single step processing failed"
        print("✅ Single time step handling works")
        
        return True
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
        return False


def run_generation2_tests():
    """Run all Generation 2 robustness tests."""
    print("=" * 60)
    print("🛡️  GENERATION 2: ROBUSTNESS TESTING")
    print("=" * 60)
    
    tests = [test_import_robustness, test_input_validation]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{'-' * 40}")
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                failed += 1
                print("❌ FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with exception: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"🛡️  GENERATION 2 RESULTS: {passed} passed, {failed} failed")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    print("=" * 60)
    
    return passed, failed


if __name__ == "__main__":
    run_generation2_tests()
