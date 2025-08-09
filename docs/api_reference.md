# API Reference

## Core Models

### LiquidNeuralNetwork

The main liquid neural network implementation with adaptive time constants.

```python
from liquid_neural_framework import LiquidNeuralNetwork

model = LiquidNeuralNetwork(
    input_size=10,      # Number of input features
    hidden_size=32,     # Number of hidden neurons
    output_size=5,      # Number of output features
    tau_min=0.1,        # Minimum time constant
    tau_max=10.0,       # Maximum time constant
    key=jax.random.PRNGKey(42)
)

# Forward pass
outputs, hidden_states = model(input_sequence, dt=0.01)
```

**Parameters:**
- `input_size` (int): Dimensionality of input features
- `hidden_size` (int): Number of neurons in hidden layer
- `output_size` (int): Dimensionality of output features
- `tau_min` (float): Minimum time constant for adaptive neurons
- `tau_max` (float): Maximum time constant for adaptive neurons
- `key` (PRNGKey): JAX random key for initialization

**Methods:**
- `__call__(x_seq, h0=None, dt=0.01)`: Forward pass through network
- `get_tau()`: Get current time constants
- `stability_measure()`: Compute stability measure
- `get_eigenvalues()`: Get eigenvalues of recurrent matrix

### ContinuousTimeRNN

Continuous-time RNN with ODE integration.

```python
from liquid_neural_framework import ContinuousTimeRNN

model = ContinuousTimeRNN(
    input_size=10,
    hidden_size=32,
    output_size=5,
    key=jax.random.PRNGKey(42)
)

# Forward pass with different integration methods
outputs, states = model(input_sequence, dt=0.01, use_ode_solver=False)  # Euler
outputs, states = model(input_sequence, dt=0.01, use_ode_solver=True)   # Adaptive ODE
```

**Methods:**
- `get_fixed_points(input_vector, num_inits=10)`: Find fixed points
- `forward_step_euler(h, x, dt)`: Single Euler integration step
- `forward_step_ode(h, x, dt)`: Single ODE integration step

### AdaptiveNeuron

Individual adaptive neuron with parameter evolution.

```python
from liquid_neural_framework import AdaptiveNeuron

neuron = AdaptiveNeuron(
    input_size=5,
    tau_init=1.0,
    threshold_init=0.0,
    key=jax.random.PRNGKey(42)
)

# Evolve neuron state
new_state, updated_neuron = neuron.forward(
    current_state, 
    input_vector, 
    dt=0.01,
    adaptation_signal=0.1
)
```

**Adaptive Parameters:**
- Time constant (tau): Controls neuron dynamics speed
- Threshold: Activation threshold
- Sensitivity: Input sensitivity scaling

### AdaptiveNeuronLayer

Layer of interconnected adaptive neurons.

```python
from liquid_neural_framework import AdaptiveNeuronLayer

layer = AdaptiveNeuronLayer(
    input_size=10,
    num_neurons=20,
    key=jax.random.PRNGKey(42)
)

# Process through layer
new_states, updated_layer = layer(
    current_states,
    input_vector,
    dt=0.01,
    adaptation_signals=adaptation_array
)
```

## Training

### LiquidNetworkTrainer

Comprehensive training framework for liquid neural networks.

```python
from liquid_neural_framework import LiquidNetworkTrainer

trainer = LiquidNetworkTrainer(
    model=liquid_model,
    learning_rate=1e-3,
    optimizer_name='adam',
    loss_fn='mse',
    gradient_clip=1.0
)

# Train the model
history = trainer.fit(
    train_data=(train_inputs, train_targets),
    val_data=(val_inputs, val_targets),
    epochs=100,
    dt=0.01,
    verbose=True
)
```

**Supported Optimizers:**
- `adam`: Adam optimizer
- `adamw`: Adam with weight decay
- `sgd`: Stochastic gradient descent
- `rmsprop`: RMSprop optimizer

**Supported Loss Functions:**
- `mse`: Mean squared error
- `mae`: Mean absolute error
- `cross_entropy`: Cross entropy loss
- `huber`: Huber loss
- `temporal_consistency`: Temporal consistency loss

## Utilities

### Validation

Input validation and sanitization utilities.

```python
from liquid_neural_framework.utils.validation import (
    validate_model_parameters,
    validate_sequence_data,
    sanitize_weights,
    check_numerical_stability
)

# Validate model parameters
validate_model_parameters(input_size=10, hidden_size=32, output_size=5)

# Validate sequence data
validate_sequence_data(input_sequences, targets)

# Check numerical stability
stability = check_numerical_stability(outputs, hidden_states)
```

### Logging

Comprehensive logging and monitoring.

```python
from liquid_neural_framework.utils.logging import LiquidNetworkLogger

logger = LiquidNetworkLogger(
    name="experiment_1",
    log_level="INFO",
    log_file="experiment.log"
)

# Log metrics
logger.log_metrics(
    epoch=10,
    metrics={"loss": 0.123, "accuracy": 0.89},
    phase="train"
)

# Log model information
logger.log_model_info("LiquidNN", architecture_dict, num_params)
```

### Optimization

Performance optimization utilities.

```python
from liquid_neural_framework.utils.optimization import (
    optimize_model,
    LRUCache,
    get_computation_graph
)

# Apply optimizations
optimized_model = optimize_model(base_model)

# Use caching decorator
@LRUCache(maxsize=64)
def expensive_computation(x):
    return heavy_calculation(x)

# Get computation graph
graph = get_computation_graph()
compiled_fn = graph.compile_forward_pass(ModelClass)
```

### Parallel Processing

Concurrent and distributed processing.

```python
from liquid_neural_framework.utils.parallel import (
    get_parallel_processor,
    ProcessingTask,
    get_task_queue
)

# Parallel processing
processor = get_parallel_processor()
results = processor.parallel_forward_pass(model_fn, batch_inputs, model_params)

# Task queue
queue = get_task_queue()
task = ProcessingTask("task_1", input_data, {"dt": 0.01})
queue.add_task(task)
```

### Security

Security monitoring and protection.

```python
from liquid_neural_framework.utils.security import (
    SecurityMonitor,
    create_secure_model_wrapper
)

# Security monitoring
monitor = SecurityMonitor()
anomalies = monitor.check_input_anomalies(suspicious_input)
safe_input = monitor.sanitize_inputs(input_data)

# Secure model wrapper
SecureModel = create_secure_model_wrapper(LiquidNeuralNetwork)
secure_model = SecureModel(10, 32, 5, enable_security=True)
```

## Experiments

### BenchmarkSuite

Comprehensive benchmarking framework.

```python
from liquid_neural_framework import BenchmarkSuite

benchmark = BenchmarkSuite()

# Run standard benchmarks
results = benchmark.run_standard_benchmarks(model)

# Custom benchmark
results = benchmark.benchmark_model(
    model=model,
    dataset="custom_data",
    metrics=["mse", "mae", "r2_score"]
)
```

### ValidationExperiments

Research-grade validation experiments.

```python
from liquid_neural_framework import ValidationExperiments

validator = ValidationExperiments()

# Statistical significance testing
results = validator.statistical_significance_test(
    model_a, model_b, test_data
)

# Ablation study
ablation_results = validator.ablation_study(
    model_class, component_list, dataset
)
```

## Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `JAX_PLATFORM_NAME`: JAX platform (cpu, gpu, tpu)
- `LIQUID_NN_CACHE_SIZE`: Default cache size for optimizations
- `LIQUID_NN_MAX_WORKERS`: Maximum number of parallel workers

### Configuration Files

Create a `config.json` file:

```json
{
    "model": {
        "default_tau_min": 0.1,
        "default_tau_max": 10.0,
        "default_dt": 0.01
    },
    "training": {
        "default_lr": 1e-3,
        "default_epochs": 100,
        "default_batch_size": 32
    },
    "optimization": {
        "enable_jit": true,
        "cache_size": 128,
        "enable_parallel": true
    },
    "security": {
        "enable_monitoring": true,
        "input_clip_range": [-100, 100],
        "max_sequence_length": 10000
    }
}
```

## Error Handling

The framework provides comprehensive error handling:

```python
from liquid_neural_framework.utils.validation import ValidationError

try:
    model = LiquidNeuralNetwork(-1, 32, 5)  # Invalid input size
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    outputs = model(invalid_input)
except ValidationError as e:
    print(f"Input validation failed: {e}")
```

## Performance Tips

1. **Use JIT compilation**: Enable JIT for faster computation
2. **Batch processing**: Process multiple sequences together
3. **Optimize integration step**: Use adaptive step size control
4. **Memory pooling**: Reuse tensor allocations
5. **Parallel processing**: Utilize multiple devices when available

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic model usage and training
- `advanced_features.py`: Advanced features demonstration
- `research_experiments.py`: Research-grade experiments
- `performance_optimization.py`: Performance optimization techniques