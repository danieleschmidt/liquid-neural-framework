"""
Test Generation 3 Scaling Features - Simplified Version
"""

import jax
import jax.numpy as jnp
import sys
sys.path.append('src')

from models.liquid_neural_network import LiquidNeuralNetwork
from models.performance_optimizers import JITOptimizer, PerformanceProfiler, MemoryOptimizer
from models.scaling_models import resource_monitor

def test_generation3_scaling():
    """Test Generation 3 scaling and performance features."""
    
    print('🚀 Testing Generation 3 Scaling and Performance...')
    
    # Test 1: JIT Optimization
    print('\n1. Testing JIT compilation...')
    key = jax.random.PRNGKey(42)
    optimizer = JITOptimizer()
    
    # Create base model
    base_model = LiquidNeuralNetwork(input_size=10, hidden_sizes=[16], output_size=5, key=key)
    example_inputs = jax.random.normal(key, (10, 10))
    
    # JIT compile
    compiled_model = optimizer.optimize_model(base_model, example_inputs)
    
    # Test compiled model
    outputs, states = compiled_model(example_inputs)
    print(f'✅ JIT compilation successful: {example_inputs.shape} -> {outputs.shape}')
    
    # Test 2: Performance Profiling
    print('\n2. Testing performance profiling...')
    profiler = PerformanceProfiler()
    
    # Profile base model
    base_profile = profiler.profile_model(base_model, example_inputs, 'base_model')
    print(f'✅ Base model: {base_profile["avg_time"]:.4f}s avg, {base_profile["throughput_hz"]:.1f} Hz')
    
    # Profile compiled model
    compiled_profile = profiler.profile_model(compiled_model, example_inputs, 'compiled_model')
    print(f'✅ Compiled model: {compiled_profile["avg_time"]:.4f}s avg, {compiled_profile["throughput_hz"]:.1f} Hz')
    
    # Calculate speedup
    speedup = base_profile["avg_time"] / compiled_profile["avg_time"]
    print(f'✅ JIT Speedup: {speedup:.2f}x faster')
    
    # Test 3: Memory Optimization
    print('\n3. Testing memory optimization...')
    memory_optimizer = MemoryOptimizer()
    
    # Monitor memory during inference
    memory_stats = memory_optimizer.monitor_memory("test_inference")
    print(f'✅ Memory usage: {memory_stats["rss"]:.1f} MB RSS, {memory_stats["percent"]:.1f}%')
    
    # Test 4: Resource Monitoring
    print('\n4. Testing resource monitoring...')
    metrics = resource_monitor.collect_metrics()
    print(f'✅ System resources:')
    print(f'   CPU: {metrics["cpu_usage"]:.1f}%')
    print(f'   Memory: {metrics["memory_usage"]:.1f}%')
    print(f'   Available Memory: {metrics["memory_available_gb"]:.1f} GB')
    
    # Test 5: Batch Processing (Manual)
    print('\n5. Testing batch processing...')
    
    # Create batch of sequences
    batch_size = 4
    seq_length = 15
    batch_inputs = jax.random.normal(key, (batch_size, seq_length, 10))
    
    # Process each sequence in batch manually
    batch_outputs = []
    for i in range(batch_size):
        seq_output, _ = compiled_model(batch_inputs[i])
        batch_outputs.append(seq_output)
    
    batch_outputs = jnp.stack(batch_outputs)
    print(f'✅ Batch processing: {batch_inputs.shape} -> {batch_outputs.shape}')
    
    # Test 6: Throughput Testing
    print('\n6. Testing throughput...')
    
    # Measure throughput over multiple runs
    import time
    num_runs = 50
    
    start_time = time.time()
    for _ in range(num_runs):
        _ = compiled_model(example_inputs)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = num_runs / total_time
    print(f'✅ Throughput: {throughput:.1f} inferences/second')
    
    # Test 7: Performance Comparison
    print('\n7. Testing performance comparison...')
    
    models = {
        'base_liquid': base_model,
        'jit_compiled': compiled_model
    }
    
    comparison = profiler.compare_models(models, example_inputs)
    best_model = comparison['best_model']
    print(f'✅ Best performing model: {best_model}')
    
    # Summary
    print('\n🎉 Generation 3 Scaling Features Summary:')
    print(f'   ✅ JIT Compilation: {speedup:.2f}x speedup')
    print(f'   ✅ Memory Monitoring: {memory_stats["rss"]:.1f} MB usage')
    print(f'   ✅ Resource Tracking: CPU {metrics["cpu_usage"]:.1f}%, Memory {metrics["memory_usage"]:.1f}%')
    print(f'   ✅ Batch Processing: {batch_size} sequences processed')
    print(f'   ✅ Throughput: {throughput:.1f} inferences/second')
    print(f'   ✅ Performance Optimization: {best_model} is optimal')
    
    print('\n🚀 Generation 3 Performance Optimization Complete!')
    
    return {
        'jit_speedup': speedup,
        'memory_usage_mb': memory_stats["rss"],
        'cpu_usage_percent': metrics["cpu_usage"],
        'throughput_hz': throughput,
        'best_model': best_model
    }

if __name__ == "__main__":
    results = test_generation3_scaling()
    print(f'\nFinal Results: {results}')