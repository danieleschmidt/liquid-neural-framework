"""
Liquid Neural Framework - Continuous-time adaptive neural networks with JAX.

A research framework for implementing and evaluating liquid neural networks,
continuous-time RNNs, and adaptive neural architectures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core models
try:
    from .models import LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron
    from .models.liquid_neural_network import LiquidLayer, AdaptiveLiquidNetwork
    from .models.adaptive_neuron import LiquidNeuron, ResonatorNeuron, NeuronNetwork
    from .models.continuous_time_rnn import NeuralODEFunc, GatedContinuousRNN, MultiScaleCTRNN
except ImportError:
    # Fallback for when JAX is not available
    pass

# Training algorithms
from .algorithms import LiquidNetworkTrainer, AdaptiveOptimizer, ContinuousLearner

# Experimental tools
from .experiments import BenchmarkSuite, SyntheticTaskGenerator, ValidationExperiments

# Utilities
from .utils import DataGenerator, ResultsVisualizer, PerformanceMetrics

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