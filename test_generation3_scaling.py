"""
Test Generation 3: Scaling and Optimization

Test performance optimizations, batch processing, caching, and scaling capabilities.
"""

import sys
sys.path.append('.')
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


def test_optimized_models():
    """Test optimized model implementations."""
    print("🚀 Testing Optimized Models...")
    
    from src.models.optimized_models import OptimizedLiquidNeuralNetwork
    
    # Test OptimizedLiquidNeuralNetwork
    opt_lnn = OptimizedLiquidNeuralNetwork(
        input_size=4, hidden_size=8, output_size=2, 
        enable_caching=True, adaptive_dt=True, seed=42
    )
    
    x = np.random.randn(20, 4)
    outputs, hidden_states = opt_lnn.forward(x)
    
    assert outputs.shape == (20, 2), f"Optimized LNN output shape: {outputs.shape}"
    assert np.all(np.isfinite(outputs)), "Optimized LNN non-finite outputs"
    
    # Test memory usage reporting
    memory_stats = opt_lnn.get_memory_usage()
    assert 'cache_size' in memory_stats, "Missing memory statistics"
    
    print("✅ OptimizedLiquidNeuralNetwork works")
    return True


def test_batch_processing():
    """Test batch processing capabilities."""
    print("📦 Testing Batch Processing...")
    
    from src.models.optimized_models import OptimizedLiquidNeuralNetwork
    
    model = OptimizedLiquidNeuralNetwork(
        input_size=3, hidden_size=6, output_size=2, seed=42
    )
    
    # Test batch processing
    batch_size = 5
    seq_len = 12
    x_batch = np.random.randn(batch_size, seq_len, 3)
    
    outputs_batch, hidden_states_batch = model.forward_batch(x_batch)
    
    assert outputs_batch.shape == (batch_size, seq_len, 2), f"Batch output shape: {outputs_batch.shape}"
    assert hidden_states_batch.shape == (batch_size, seq_len, 6), f"Batch hidden shape: {hidden_states_batch.shape}"
    assert np.all(np.isfinite(outputs_batch)), "Batch processing non-finite outputs"
    
    print("✅ Batch processing works correctly")
    return True


def run_generation3_tests():
    """Run Generation 3 scaling and optimization tests."""
    print("=" * 80)
    print("🚀 GENERATION 3: SCALING & OPTIMIZATION TESTING")
    print("=" * 80)
    
    tests = [
        test_optimized_models,
        test_batch_processing
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{'-' * 60}")
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
    
    print(f"\n{'=' * 80}")
    print(f"🚀 GENERATION 3 RESULTS: {passed} passed, {failed} failed")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    print("=" * 80)
    
    return passed, failed


if __name__ == "__main__":
    run_generation3_tests()
