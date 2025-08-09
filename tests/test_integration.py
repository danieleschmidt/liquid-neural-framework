"""
Integration tests for liquid neural framework.
"""

import pytest
import jax.numpy as jnp
from jax import random
import time

from src.models.liquid_network import LiquidNeuralNetwork
from src.models.continuous_rnn import ContinuousTimeRNN
from src.models.adaptive_neuron import AdaptiveNeuron, AdaptiveNeuronLayer
from src.utils.optimization import optimize_model, get_profiler
from src.utils.parallel import get_parallel_processor, ProcessingTask, get_task_queue
from src.utils.security import SecurityMonitor, create_secure_model_wrapper
from src.utils.logging import LiquidNetworkLogger
from src.algorithms.training import LiquidNetworkTrainer


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
        self.logger = LiquidNetworkLogger("test_integration", experiment_id="integration_test")
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline with all features."""
        # Generate synthetic data
        key, data_key = random.split(self.key)
        train_size, val_size = 100, 20
        seq_length, input_dim, output_dim = 50, 2, 1
        
        # Training data
        train_inputs = random.normal(data_key, (train_size, seq_length, input_dim))
        train_targets = jnp.sin(jnp.sum(train_inputs, axis=-1, keepdims=True))
        
        # Validation data
        val_key = random.split(key)[0]
        val_inputs = random.normal(val_key, (val_size, seq_length, input_dim))
        val_targets = jnp.sin(jnp.sum(val_inputs, axis=-1, keepdims=True))
        
        # Create model
        model = LiquidNeuralNetwork(input_dim, 16, output_dim, key=key)
        
        # Create trainer
        trainer = LiquidNetworkTrainer(
            model=model,
            learning_rate=1e-3,
            optimizer_name='adam',
            loss_fn='mse'
        )
        
        # Train model
        history = trainer.fit(
            train_data=(train_inputs[0], train_targets[0]),  # Single sequence for test
            val_data=(val_inputs[0], val_targets[0]),
            epochs=5,  # Short training for test
            dt=0.01,
            verbose=False
        )
        
        # Verify training occurred
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
        
        # Test final performance
        final_output, _ = model(val_inputs[0], dt=0.01)
        final_loss = float(jnp.mean((final_output - val_targets[0]) ** 2))
        
        self.logger.log_metrics(
            epoch=5, 
            metrics={"final_test_loss": final_loss},
            phase="test"
        )
        
        # Basic sanity check
        assert 0.0 <= final_loss <= 10.0  # Reasonable loss range
    
    def test_optimization_integration(self):
        """Test integration with optimization features."""
        # Create model
        model = LiquidNeuralNetwork(1, 8, 1, key=self.key)
        
        # Apply optimizations
        optimized_model = optimize_model(model)
        
        # Test data
        test_input = random.normal(self.key, (20, 1))
        
        # Test optimized forward pass
        outputs, states = optimized_model(test_input, dt=0.01)
        
        # Verify outputs
        assert outputs.shape == (20, 1)
        assert states.shape == (20, 8)
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.all(jnp.isfinite(states))
        
        # Check optimization stats
        opt_stats = optimized_model.get_optimization_stats()
        assert 'cache_stats' in opt_stats
        assert 'memory_stats' in opt_stats
        assert 'performance_stats' in opt_stats
    
    def test_security_integration(self):
        """Test integration with security features."""
        # Create secure model wrapper
        SecureModel = create_secure_model_wrapper(LiquidNeuralNetwork)
        secure_model = SecureModel(1, 8, 1, key=self.key)
        
        # Test with normal input
        normal_input = jnp.sin(jnp.linspace(0, 2*jnp.pi, 30)).reshape(-1, 1)
        outputs, states = secure_model(normal_input, dt=0.01)
        
        assert outputs.shape == (30, 1)
        
        # Test with potentially malicious input (extreme values)
        malicious_input = jnp.ones((10, 1)) * 1000
        
        # Should handle gracefully with warnings
        outputs, states = secure_model(malicious_input, dt=0.01)
        assert jnp.all(jnp.isfinite(outputs))
        
        # Check security report
        security_report = secure_model.get_security_report()
        assert 'monitoring_enabled' in security_report
    
    def test_parallel_processing_integration(self):
        """Test integration with parallel processing."""
        processor = get_parallel_processor()
        
        # Create model and data
        model = LiquidNeuralNetwork(1, 4, 1, key=self.key)
        
        # Create batch data
        batch_inputs = random.normal(self.key, (8, 20, 1))  # 8 sequences
        
        # Define model function
        def model_fn(params, inputs, dt=0.01):
            return model(inputs, dt=dt)[0]  # Just outputs
        
        # Test parallel processing
        if processor.device_count > 1:
            parallel_outputs = processor.parallel_forward_pass(
                model_fn, batch_inputs, model
            )
            assert parallel_outputs.shape == (8, 20, 1)
        else:
            # Single device fallback
            outputs = model_fn(model, batch_inputs[0])
            assert outputs.shape == (20, 1)
    
    def test_adaptive_layer_evolution(self):
        """Test adaptive neuron layer evolution over time."""
        layer = AdaptiveNeuronLayer(1, num_neurons=5, key=self.key)
        
        # Simulate long-term evolution
        states = jnp.zeros(5)
        n_steps = 100
        
        state_history = []
        adaptation_history = []
        
        for i in range(n_steps):
            # Time-varying input
            input_val = jnp.array([jnp.sin(0.1 * i)])
            adaptation_signal = 0.01 * jnp.sin(0.05 * i) * jnp.ones(5)
            
            states, layer = layer(states, input_val, adaptation_signals=adaptation_signal)
            
            state_history.append(states)
            adaptation_history.append(layer.get_layer_info())
        
        state_history = jnp.array(state_history)
        
        # Verify evolution
        assert state_history.shape == (n_steps, 5)
        assert jnp.all(jnp.isfinite(state_history))
        
        # Check parameter adaptation
        final_info = adaptation_history[-1]
        initial_info = adaptation_history[0]
        
        # Parameters should have evolved
        tau_change = jnp.mean(jnp.abs(final_info['tau'] - initial_info['tau']))
        assert tau_change > 1e-6  # Some adaptation occurred
    
    def test_multi_model_comparison(self):
        """Test comparison between different model architectures."""
        models = {
            'liquid_nn': LiquidNeuralNetwork(1, 8, 1, key=self.key),
            'continuous_rnn': ContinuousTimeRNN(1, 8, 1, key=self.key)
        }
        
        # Test data
        test_input = jnp.sin(jnp.linspace(0, 4*jnp.pi, 40)).reshape(-1, 1)
        
        results = {}
        
        for name, model in models.items():
            start_time = time.time()
            outputs, states = model(test_input, dt=0.01)
            duration = time.time() - start_time
            
            results[name] = {
                'outputs': outputs,
                'states': states,
                'duration': duration,
                'output_variance': float(jnp.var(outputs)),
                'state_variance': float(jnp.var(states))
            }
        
        # Verify all models produced valid outputs
        for name, result in results.items():
            assert result['outputs'].shape == (40, 1)
            assert result['states'].shape == (40, 8)
            assert jnp.all(jnp.isfinite(result['outputs']))
            assert jnp.all(jnp.isfinite(result['states']))
            assert result['duration'] > 0
        
        # Log comparison results
        for name, result in results.items():
            self.logger.log_metrics(
                epoch=0,
                metrics={
                    'duration': result['duration'],
                    'output_variance': result['output_variance'],
                    'state_variance': result['state_variance']
                },
                phase=f"comparison_{name}"
            )
    
    def test_task_queue_workflow(self):
        """Test task queue and processing workflow."""
        task_queue = get_task_queue()
        
        # Create processing tasks
        tasks = []
        for i in range(5):
            task = ProcessingTask(
                task_id=f"task_{i}",
                inputs=random.normal(self.key, (10, 1)),
                parameters={'dt': 0.01},
                priority=i  # Different priorities
            )
            tasks.append(task)
            success = task_queue.add_task(task)
            assert success
        
        # Process tasks
        processed_count = 0
        while processed_count < 5:
            task = task_queue.get_next_task()
            if task is not None:
                # Simulate processing
                time.sleep(0.01)  # Minimal processing time
                task_queue.complete_task(task.task_id, "result", 0.01)
                processed_count += 1
            else:
                break
        
        # Verify queue stats
        stats = task_queue.get_queue_stats()
        assert stats['pending_tasks'] == 0
        assert stats['completed_tasks'] == 5
    
    def test_robustness_edge_cases(self):
        """Test robustness with edge cases."""
        model = LiquidNeuralNetwork(2, 6, 1, key=self.key)
        
        # Test edge cases
        edge_cases = [
            jnp.zeros((5, 2)),  # All zeros
            jnp.ones((1, 2)),   # Single time step
            jnp.ones((1000, 2)) * 1e-8,  # Very small values
        ]
        
        for i, test_input in enumerate(edge_cases):
            try:
                outputs, states = model(test_input, dt=0.01)
                
                # Verify outputs are reasonable
                assert jnp.all(jnp.isfinite(outputs))
                assert jnp.all(jnp.isfinite(states))
                assert outputs.shape == (test_input.shape[0], 1)
                
                self.logger.debug(f"Edge case {i} passed")
                
            except Exception as e:
                self.logger.error(f"Edge case {i} failed: {str(e)}")
                raise
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large sequences."""
        model = LiquidNeuralNetwork(1, 4, 1, key=self.key)  # Small model
        
        # Test with progressively larger sequences
        for seq_len in [100, 500, 1000]:
            test_input = random.normal(self.key, (seq_len, 1))
            
            start_time = time.time()
            outputs, states = model(test_input, dt=0.01)
            duration = time.time() - start_time
            
            # Verify successful processing
            assert outputs.shape == (seq_len, 1)
            assert jnp.all(jnp.isfinite(outputs))
            
            # Log performance metrics
            self.logger.log_metrics(
                epoch=0,
                metrics={
                    'sequence_length': seq_len,
                    'processing_time': duration,
                    'time_per_step': duration / seq_len
                },
                phase="memory_efficiency"
            )
    
    def teardown_method(self):
        """Clean up after tests."""
        # Save test metrics
        try:
            self.logger.save_metrics("test_integration_metrics.json")
        except Exception as e:
            print(f"Warning: Could not save test metrics: {e}")


class TestStabilityAndReliability:
    """Test system stability and reliability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(999)
    
    def test_long_sequence_stability(self):
        """Test stability with very long sequences."""
        model = LiquidNeuralNetwork(1, 8, 1, tau_min=0.1, tau_max=2.0, key=self.key)
        
        # Very long sequence (2000 time steps)
        t = jnp.linspace(0, 20*jnp.pi, 2000)
        long_input = jnp.sin(0.1 * t + 0.3 * jnp.sin(0.05 * t)).reshape(-1, 1)
        
        outputs, states = model(long_input, dt=0.01)
        
        # Check for numerical stability
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.all(jnp.isfinite(states))
        assert jnp.all(jnp.abs(outputs) < 1000)  # Bounded outputs
        assert jnp.all(jnp.abs(states) < 1000)   # Bounded states
    
    def test_parameter_extreme_values(self):
        """Test behavior with extreme parameter values."""
        # Test with extreme tau values
        extreme_models = [
            LiquidNeuralNetwork(1, 4, 1, tau_min=1e-3, tau_max=1e-2, key=self.key),  # Very fast
            LiquidNeuralNetwork(1, 4, 1, tau_min=10.0, tau_max=50.0, key=self.key),  # Very slow
        ]
        
        test_input = jnp.sin(jnp.linspace(0, 2*jnp.pi, 50)).reshape(-1, 1)
        
        for i, model in enumerate(extreme_models):
            outputs, states = model(test_input, dt=0.01)
            
            # Should handle extreme parameters gracefully
            assert jnp.all(jnp.isfinite(outputs))
            assert jnp.all(jnp.isfinite(states))
    
    def test_concurrent_processing_safety(self):
        """Test thread safety of concurrent processing."""
        import threading
        
        model = LiquidNeuralNetwork(1, 6, 1, key=self.key)
        test_inputs = [random.normal(self.key, (20, 1)) for _ in range(10)]
        
        results = [None] * 10
        exceptions = [None] * 10
        
        def process_sequence(idx, inputs):
            try:
                outputs, states = model(inputs, dt=0.01)
                results[idx] = (outputs, states)
            except Exception as e:
                exceptions[idx] = e
        
        # Create threads
        threads = []
        for i, inputs in enumerate(test_inputs):
            thread = threading.Thread(target=process_sequence, args=(i, inputs))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        for i, exception in enumerate(exceptions):
            if exception is not None:
                raise exception
        
        # Verify all results
        for i, result in enumerate(results):
            assert result is not None
            outputs, states = result
            assert outputs.shape == (20, 1)
            assert jnp.all(jnp.isfinite(outputs))