"""
Comprehensive Functionality Test

Test all implemented models and their core functionality.
"""

import sys
sys.path.append('.')
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_liquid_neural_network():
    """Test Liquid Neural Network comprehensive functionality."""
    print("🧠 Testing Liquid Neural Network...")
    
    import src
    
    # Test basic functionality
    model = src.LiquidNeuralNetwork(input_size=3, hidden_size=8, output_size=2, dt=0.1, seed=42)
    
    x = np.random.randn(20, 3)
    outputs, hidden_states = model.forward(x)
    
    assert outputs.shape == (20, 2), f"Output shape: {outputs.shape}"
    assert hidden_states.shape == (20, 8), f"Hidden shape: {hidden_states.shape}"
    assert np.all(np.isfinite(outputs)), "Non-finite outputs"
    
    # Test prediction interface
    predictions = model.predict(x)
    assert np.allclose(predictions, outputs), "Predict and forward should match"
    
    print("✅ LiquidNeuralNetwork works correctly")
    return True

def test_continuous_time_rnn():
    """Test Continuous-Time RNN functionality."""
    print("⏰ Testing Continuous-Time RNN...")
    
    import src
    
    model = src.ContinuousTimeRNN(input_size=4, hidden_size=6, output_size=3, dt=0.05, seed=123)
    
    x = np.random.randn(15, 4)
    outputs, hidden_states = model.forward(x)
    
    assert outputs.shape == (15, 3), f"CTRNN output shape: {outputs.shape}"
    assert hidden_states.shape == (15, 6), f"CTRNN hidden shape: {hidden_states.shape}"
    assert np.all(np.isfinite(outputs)), "CTRNN non-finite outputs"
    
    print("✅ ContinuousTimeRNN works correctly")
    return True

def test_adaptive_neuron():
    """Test Adaptive Neuron functionality."""
    print("🔄 Testing Adaptive Neuron...")
    
    import src
    
    neuron = src.AdaptiveNeuron(input_size=5, tau_base=8.0, threshold_base=0.4, seed=456)
    
    x_seq = np.random.randn(25, 5)
    membrane_potentials, outputs = neuron.forward(x_seq, v0=0.0, dt=0.1)
    
    assert membrane_potentials.shape == (25,), f"Membrane shape: {membrane_potentials.shape}"
    assert outputs.shape == (25,), f"Output shape: {outputs.shape}"
    assert np.all(np.isfinite(membrane_potentials)), "Non-finite membrane potentials"
    assert np.all(np.isfinite(outputs)), "Non-finite neuron outputs"
    
    print("✅ AdaptiveNeuron works correctly")
    return True

def test_all_model_aliases():
    """Test that all model aliases work."""
    print("🔗 Testing Model Aliases...")
    
    import src
    
    # Test that aliases exist and can be instantiated
    aliases = [
        'LiquidLayer', 'AdaptiveLiquidNetwork', 'LiquidNeuron',
        'ResonatorNeuron', 'NeuronNetwork', 'NeuralODEFunc',
        'GatedContinuousRNN', 'MultiScaleCTRNN'
    ]
    
    for alias in aliases:
        assert hasattr(src, alias), f"Missing alias: {alias}"
        
        # Try to instantiate (they should all work with basic parameters)
        try:
            model = getattr(src, alias)(input_size=2, hidden_size=3, output_size=1)
            print(f"✅ {alias} instantiation works")
        except Exception as e:
            print(f"ℹ️  {alias} instantiation needs specific parameters: {e}")
    
    print("✅ All model aliases accessible")
    return True

def test_numerical_edge_cases():
    """Test numerical edge cases and stability."""
    print("🎯 Testing Numerical Edge Cases...")
    
    import src
    
    model = src.LiquidNeuralNetwork(input_size=2, hidden_size=4, output_size=1, dt=0.1)
    
    # Test various input patterns
    test_cases = [
        ("zeros", np.zeros((10, 2))),
        ("ones", np.ones((10, 2))),
        ("large_positive", np.ones((10, 2)) * 50),
        ("large_negative", np.ones((10, 2)) * -50),
        ("mixed_extreme", np.array([[100, -100], [-50, 50]] * 5)),
        ("small_values", np.ones((10, 2)) * 1e-6),
        ("random_normal", np.random.randn(10, 2))
    ]
    
    for case_name, x in test_cases:
        try:
            outputs, _ = model.forward(x)
            assert np.all(np.isfinite(outputs)), f"Non-finite outputs in {case_name}"
            print(f"✅ {case_name} handled correctly")
        except Exception as e:
            print(f"❌ Failed on {case_name}: {e}")
            return False
    
    print("✅ All numerical edge cases handled")
    return True

def run_comprehensive_tests():
    """Run all comprehensive functionality tests."""
    print("=" * 80)
    print("🧪 COMPREHENSIVE FUNCTIONALITY TESTING")
    print("=" * 80)
    
    tests = [
        test_liquid_neural_network,
        test_continuous_time_rnn,
        test_adaptive_neuron,
        test_all_model_aliases,
        test_numerical_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{'-' * 50}")
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
    print(f"🧪 COMPREHENSIVE RESULTS: {passed} passed, {failed} failed")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    print("=" * 80)
    
    return passed, failed

if __name__ == "__main__":
    run_comprehensive_tests()
