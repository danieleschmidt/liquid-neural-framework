"""
Comprehensive test suite for the Liquid Neural Framework.

Tests all core components, models, and research algorithms
to ensure robust functionality and research-grade quality.
"""

import numpy as np
import sys
import os
sys.path.insert(0, '.')

def test_core_framework():
    """Test core framework imports and basic functionality."""
    print("🧪 Testing Core Framework...")
    
    try:
        import src
        from src.models import LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron
        print("✅ Core imports successful")
        
        # Test basic model creation
        model = LiquidNeuralNetwork(input_size=5, liquid_size=32, output_size=1)
        print("✅ Model instantiation successful")
        
        return True
    except Exception as e:
        print(f"❌ Core framework test failed: {e}")
        return False

def test_liquid_neural_network():
    """Test liquid neural network functionality."""
    print("🌊 Testing Liquid Neural Network...")
    
    try:
        from src.models import LiquidNeuralNetwork
        
        # Test different configurations
        configs = [
            {"input_size": 5, "liquid_size": 32, "output_size": 1},
            {"input_size": 10, "liquid_size": 64, "output_size": 3},
            {"input_size": 20, "liquid_size": 128, "output_size": 5}
        ]
        
        for i, config in enumerate(configs):
            model = LiquidNeuralNetwork(**config)
            hidden = model.init_hidden_state(2)  # Batch size 2
            inputs = np.random.random((2, config["input_size"]))
            
            # Forward pass
            output, new_hidden = model(inputs, hidden)
            
            # Verify shapes
            assert output.shape == (2, config["output_size"]), f"Output shape mismatch: {output.shape}"
            assert new_hidden.shape == (2, config["liquid_size"]), f"Hidden shape mismatch: {new_hidden.shape}"
            
            # Test statistics
            stats = model.get_liquid_state_statistics(new_hidden)
            assert isinstance(stats, dict), "Statistics should be a dictionary"
            assert "mean_activity" in stats, "Statistics missing mean_activity"
            
            print(f"✅ Config {i+1}: {config} - PASSED")
        
        print("✅ Liquid Neural Network tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Liquid Neural Network test failed: {e}")
        return False

def test_continuous_time_rnn():
    """Test continuous-time RNN functionality."""
    print("⏰ Testing Continuous-Time RNN...")
    
    try:
        from src.models import ContinuousTimeRNN
        
        # Test basic CT-RNN
        model = ContinuousTimeRNN(input_size=8, hidden_size=32, output_size=2)
        hidden = model.init_hidden_state(1)
        inputs = np.random.random((1, 8))
        
        # Forward pass
        output, new_hidden = model(inputs, hidden, t=0.0)
        
        assert output.shape == (1, 2), f"Output shape mismatch: {output.shape}"
        assert new_hidden.shape == (1, 32), f"Hidden shape mismatch: {new_hidden.shape}"
        
        # Test temporal dynamics
        prev_hidden = hidden.copy()
        for t in range(5):
            output, hidden = model(inputs, hidden, t=float(t))
            
            # Verify state is changing (dynamics are working)
            diff = np.mean(np.abs(hidden - prev_hidden))
            assert diff > 1e-6, f"Hidden state not changing at step {t}"
            prev_hidden = hidden.copy()
        
        print("✅ Continuous-Time RNN tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Continuous-Time RNN test failed: {e}")
        return False

def test_adaptive_neuron():
    """Test adaptive neuron functionality."""
    print("🧠 Testing Adaptive Neurons...")
    
    try:
        from src.models import AdaptiveNeuron
        
        # Test liquid neuron
        liquid_neuron = AdaptiveNeuron(model_type="liquid", neuron_type="regular_spiking")
        state = liquid_neuron.init_state()
        
        # Test neuron dynamics
        for current in [0.0, 5.0, 10.0, 20.0]:
            new_state, diagnostics = liquid_neuron(state, current)
            
            assert isinstance(diagnostics, dict), "Diagnostics should be a dictionary"
            assert "output" in diagnostics, "Missing output in diagnostics"
            assert "voltage" in diagnostics, "Missing voltage in diagnostics"
            
            state = new_state
        
        # Test resonator neuron
        resonator = AdaptiveNeuron(model_type="resonator", natural_frequency=2.0)
        res_state = resonator.init_state()
        
        for signal in [0.0, 1.0, -1.0, 0.5]:
            new_state, diagnostics = resonator(res_state, signal)
            
            assert isinstance(diagnostics, dict), "Diagnostics should be a dictionary"
            assert "output" in diagnostics, "Missing output in diagnostics"
            
            res_state = new_state
        
        print("✅ Adaptive Neuron tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive Neuron test failed: {e}")
        return False

def test_temporal_dynamics():
    """Test temporal dynamics and memory properties."""
    print("📈 Testing Temporal Dynamics...")
    
    try:
        from src.models import LiquidNeuralNetwork
        
        model = LiquidNeuralNetwork(input_size=3, liquid_size=50, output_size=1)
        hidden = model.init_hidden_state(1)
        
        # Create a simple temporal pattern
        sequence_length = 20
        outputs = []
        
        for t in range(sequence_length):
            # Sine wave input
            inputs = np.array([[np.sin(0.1 * t), np.cos(0.1 * t), 0.5 * np.sin(0.05 * t)]])
            output, hidden = model(inputs, hidden)
            outputs.append(output[0, 0])
        
        outputs = np.array(outputs)
        
        # Check that outputs vary significantly (not stuck)
        output_variance = np.var(outputs)
        assert output_variance > 1e-4, f"Output variance too low: {output_variance}"
        
        # Check that the system has memory (autocorrelation)
        if len(outputs) > 10:
            autocorr = np.corrcoef(outputs[:-5], outputs[5:])[0, 1]
            assert abs(autocorr) > 0.1, f"Weak temporal memory: {autocorr}"
        
        print("✅ Temporal dynamics tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal dynamics test failed: {e}")
        return False

def test_network_stability():
    """Test network stability under various conditions."""
    print("⚖️ Testing Network Stability...")
    
    try:
        from src.models import LiquidNeuralNetwork
        
        model = LiquidNeuralNetwork(input_size=5, liquid_size=40, output_size=2)
        
        # Test with different input magnitudes
        test_cases = [
            {"scale": 0.1, "description": "small inputs"},
            {"scale": 1.0, "description": "normal inputs"},
            {"scale": 10.0, "description": "large inputs"},
        ]
        
        for case in test_cases:
            hidden = model.init_hidden_state(1)
            scale = case["scale"]
            
            stable = True
            for t in range(50):  # Longer sequence for stability testing
                inputs = np.random.random((1, 5)) * scale
                output, hidden = model(inputs, hidden)
                
                # Check for NaN or infinite values
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    stable = False
                    break
                if np.any(np.isnan(hidden)) or np.any(np.isinf(hidden)):
                    stable = False
                    break
                
                # Check for explosion (values too large)
                if np.max(np.abs(output)) > 1e6 or np.max(np.abs(hidden)) > 1e6:
                    stable = False
                    break
            
            assert stable, f"Network unstable with {case['description']}"
            print(f"✅ Stable with {case['description']}")
        
        print("✅ Network stability tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Network stability test failed: {e}")
        return False

def test_research_algorithms():
    """Test research algorithms and novel components."""
    print("🔬 Testing Research Algorithms...")
    
    try:
        # Test imports only (since JAX may not be available)
        from src.research.novel_algorithms import (
            compare_novel_algorithms,
            MetaAdaptiveLiquidNetwork,
            MultiScaleTemporalNetwork,
            QuantumInspiredContinuousComputation
        )
        
        print("✅ Research algorithm imports successful")
        
        # Test comparison function with dummy data
        dummy_data = np.random.random((10, 5))
        result = compare_novel_algorithms(dummy_data, sequence_length=5)
        
        assert isinstance(result, dict), "Comparison should return a dictionary"
        print("✅ Algorithm comparison function working")
        
        print("✅ Research algorithms tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Research algorithms test failed: {e}")
        return False

def test_benchmark_performance():
    """Benchmark framework performance."""
    print("⚡ Benchmarking Performance...")
    
    try:
        import time
        from src.models import LiquidNeuralNetwork
        
        # Performance test
        model = LiquidNeuralNetwork(input_size=20, liquid_size=100, output_size=5)
        hidden = model.init_hidden_state(10)  # Batch of 10
        
        # Warm up
        for _ in range(5):
            inputs = np.random.random((10, 20))
            output, hidden = model(inputs, hidden)
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            inputs = np.random.random((10, 20))
            output, hidden = model(inputs, hidden)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_iteration = total_time / num_iterations
        
        print(f"✅ Average time per iteration: {avg_time_per_iteration*1000:.2f} ms")
        print(f"✅ Throughput: {num_iterations/total_time:.1f} iterations/sec")
        
        # Performance should be reasonable (< 10ms per iteration for this size)
        assert avg_time_per_iteration < 0.01, f"Performance too slow: {avg_time_per_iteration*1000:.2f} ms"
        
        print("✅ Performance benchmark passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Running Comprehensive Liquid Neural Framework Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all test modules
    test_functions = [
        test_core_framework,
        test_liquid_neural_network,
        test_continuous_time_rnn,
        test_adaptive_neuron,
        test_temporal_dynamics,
        test_network_stability,
        test_research_algorithms,
        test_benchmark_performance,
    ]
    
    for test_func in test_functions:
        print(f"\n{'-' * 40}")
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            test_results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (test_func, result) in enumerate(zip(test_functions, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_func.__name__}")
    
    print(f"\nResults: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - FRAMEWORK IS ROBUST!")
        return True
    else:
        print(f"⚠️  {total-passed} tests failed - review needed")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)