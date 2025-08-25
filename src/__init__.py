"""
Liquid Neural Framework - Continuous-time adaptive neural networks with JAX.

A research framework for implementing and evaluating liquid neural networks,
continuous-time RNNs, and adaptive neural architectures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core models - with robust error handling
_core_models = {}
try:
    from .models import LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron
    from .models.liquid_neural_network import LiquidLayer, AdaptiveLiquidNetwork
    from .models.adaptive_neuron import LiquidNeuron, ResonatorNeuron, NeuronNetwork
    from .models.continuous_time_rnn import NeuralODEFunc, GatedContinuousRNN, MultiScaleCTRNN
    _core_models.update({
        'LiquidNeuralNetwork': LiquidNeuralNetwork,
        'ContinuousTimeRNN': ContinuousTimeRNN,
        'AdaptiveNeuron': AdaptiveNeuron,
        'LiquidLayer': LiquidLayer,
        'AdaptiveLiquidNetwork': AdaptiveLiquidNetwork,
        'LiquidNeuron': LiquidNeuron,
        'ResonatorNeuron': ResonatorNeuron,
        'NeuronNetwork': NeuronNetwork,
        'NeuralODEFunc': NeuralODEFunc,
        'GatedContinuousRNN': GatedContinuousRNN,
        'MultiScaleCTRNN': MultiScaleCTRNN
    })
except ImportError as e:
    import warnings
    warnings.warn(f"JAX-based models not available: {e}. Using NumPy fallback implementation.")
    # Use NumPy fallback implementation
    try:
        from .models.numpy_fallback import (
            LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron,
            LiquidLayer, AdaptiveLiquidNetwork, LiquidNeuron,
            ResonatorNeuron, NeuronNetwork, NeuralODEFunc,
            GatedContinuousRNN, MultiScaleCTRNN
        )
        _core_models.update({
            'LiquidNeuralNetwork': LiquidNeuralNetwork,
            'ContinuousTimeRNN': ContinuousTimeRNN,
            'AdaptiveNeuron': AdaptiveNeuron,
            'LiquidLayer': LiquidLayer,
            'AdaptiveLiquidNetwork': AdaptiveLiquidNetwork,
            'LiquidNeuron': LiquidNeuron,
            'ResonatorNeuron': ResonatorNeuron,
            'NeuronNetwork': NeuronNetwork,
            'NeuralODEFunc': NeuralODEFunc,
            'GatedContinuousRNN': GatedContinuousRNN,
            'MultiScaleCTRNN': MultiScaleCTRNN
        })
    except ImportError:
        # Ultimate fallback - placeholder classes  
        class _PlaceholderModel:
            def __init__(self, *args, **kwargs):
                raise ImportError("Dependencies not available. Install numpy or jax.")
        
        _core_models = {name: _PlaceholderModel for name in [
            'LiquidNeuralNetwork', 'ContinuousTimeRNN', 'AdaptiveNeuron',
            'LiquidLayer', 'AdaptiveLiquidNetwork', 'LiquidNeuron', 
            'ResonatorNeuron', 'NeuronNetwork', 'NeuralODEFunc',
            'GatedContinuousRNN', 'MultiScaleCTRNN'
        ]}

# Create placeholder for missing dependencies
class _PlaceholderModel:
    def __init__(self, *args, **kwargs):
        raise ImportError("Dependencies not available. Install required packages.")

# Ensure fallback availability
try:
    # Try to import JAX models first
    if not _core_models:
        raise ImportError("JAX models not available")
except ImportError:
    # Fallback to NumPy implementations
    try:
        from .models.numpy_fallback import (
            LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron,
            LiquidLayer, AdaptiveLiquidNetwork, LiquidNeuron,
            ResonatorNeuron, NeuronNetwork, NeuralODEFunc,
            GatedContinuousRNN, MultiScaleCTRNN
        )
        _core_models = {
            'LiquidNeuralNetwork': LiquidNeuralNetwork,
            'ContinuousTimeRNN': ContinuousTimeRNN,
            'AdaptiveNeuron': AdaptiveNeuron,
            'LiquidLayer': LiquidLayer,
            'AdaptiveLiquidNetwork': AdaptiveLiquidNetwork,
            'LiquidNeuron': LiquidNeuron,
            'ResonatorNeuron': ResonatorNeuron,
            'NeuronNetwork': NeuronNetwork,
            'NeuralODEFunc': NeuralODEFunc,
            'GatedContinuousRNN': GatedContinuousRNN,
            'MultiScaleCTRNN': MultiScaleCTRNN
        }
    except ImportError:
        pass

# Update exports with fallbacks
if _core_models:
    locals().update(_core_models)

# Training algorithms - with fallbacks
_algorithms = {}
try:
    from .algorithms import LiquidNetworkTrainer, AdaptiveOptimizer, ContinuousLearner
    _algorithms.update({
        'LiquidNetworkTrainer': LiquidNetworkTrainer,
        'AdaptiveOptimizer': AdaptiveOptimizer, 
        'ContinuousLearner': ContinuousLearner
    })
except ImportError as e:
    import warnings
    warnings.warn(f"Algorithm modules not fully available: {e}")
    _algorithms = {name: _PlaceholderModel for name in [
        'LiquidNetworkTrainer', 'AdaptiveOptimizer', 'ContinuousLearner'
    ]}

# Experimental tools - with fallbacks  
_experiments = {}
try:
    from .experiments import BenchmarkSuite, SyntheticTaskGenerator, ValidationExperiments
    _experiments.update({
        'BenchmarkSuite': BenchmarkSuite,
        'SyntheticTaskGenerator': SyntheticTaskGenerator,
        'ValidationExperiments': ValidationExperiments
    })
except ImportError as e:
    import warnings
    warnings.warn(f"Experimental modules not fully available: {e}")
    _experiments = {name: _PlaceholderModel for name in [
        'BenchmarkSuite', 'SyntheticTaskGenerator', 'ValidationExperiments'
    ]}

# Utilities - should be more robust
_utils = {}
try:
    from .utils import DataGenerator, ResultsVisualizer, PerformanceMetrics
    _utils.update({
        'DataGenerator': DataGenerator,
        'ResultsVisualizer': ResultsVisualizer,
        'PerformanceMetrics': PerformanceMetrics
    })
except ImportError as e:
    import warnings
    warnings.warn(f"Utility modules not fully available: {e}")
    _utils = {name: _PlaceholderModel for name in [
        'DataGenerator', 'ResultsVisualizer', 'PerformanceMetrics'
    ]}

# Export symbols
locals().update(_core_models)
locals().update(_algorithms) 
locals().update(_experiments)
locals().update(_utils)

__all__ = [
    # Core Models
    'LiquidNeuralNetwork',
    'ContinuousTimeRNN', 
    'AdaptiveNeuron',
    
    # Advanced Models
    'LiquidLayer',
    'AdaptiveLiquidNetwork',
    'LiquidNeuron',
    'ResonatorNeuron',
    'NeuronNetwork',
    'NeuralODEFunc',
    'GatedContinuousRNN',
    'MultiScaleCTRNN',
    
    # Algorithms
    'LiquidNetworkTrainer',
    'AdaptiveOptimizer',
    'ContinuousLearner',
    
    # Experiments
    'BenchmarkSuite',
    'SyntheticTaskGenerator',
    'ValidationExperiments',
    
    # Utils
    'DataGenerator',
    'ResultsVisualizer',
    'PerformanceMetrics'
]