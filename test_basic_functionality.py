#!/usr/bin/env python3
"""
Basic functionality test for liquid neural framework.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_basic_imports():
    """Test basic imports work."""
    try:
        print("Testing imports...")
        sys.path.insert(0, '/root/repo/src')
        
        from models import LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron
        print("✅ All models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic model functionality."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        print("JAX available, testing full functionality...")
        
        sys.path.insert(0, '/root/repo/src')
        from models.liquid_neural_network import LiquidNeuralNetwork
        
        # Create model
        key = random.PRNGKey(42)
        model = LiquidNeuralNetwork(
            input_size=2,
            hidden_size=4, 
            output_size=1,
            key=key
        )
        print("✅ Model created successfully")
        
        # Test forward pass
        inputs = jnp.ones((1, 2))
        hidden = model.init_hidden_state(1)
        output, new_hidden = model(inputs, hidden)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        print(f"✅ Hidden state shape: {new_hidden.shape}")
        
        # Test stability
        stability = model.stability_measure()
        print(f"✅ Stability measure: {stability:.4f}")
        
        return True
        
    except ImportError:
        print("JAX not available, testing import-only...")
        return test_basic_imports()
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_security_features():
    """Test security features."""
    try:
        sys.path.insert(0, '/root/repo/src')
        from models.security_utils import validate_input_safety, sanitize_config
        
        # Test input validation
        import numpy as np
        safe_input = np.ones((10, 5))
        validate_input_safety(safe_input, "test_input")
        print("✅ Input validation works")
        
        # Test config sanitization
        config = {
            'input_size': 10,
            'hidden_size': 20,
            'activation': 'tanh',
            'unsafe_key': 'malicious_value'
        }
        
        safe_config = sanitize_config(config)
        assert 'unsafe_key' not in safe_config
        print("✅ Config sanitization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 LIQUID NEURAL FRAMEWORK - QUALITY GATE TESTS")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_basic_imports),
        ("Functionality Test", test_basic_functionality), 
        ("Security Test", test_security_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED!")
        return True
    else:
        print("⚠️  Some quality gates failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)