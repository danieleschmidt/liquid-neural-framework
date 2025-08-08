"""
Liquid Neural Framework - Continuous-time adaptive neural networks with JAX.

A research framework for implementing and evaluating liquid neural networks,
continuous-time RNNs, and adaptive neural architectures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core models
from .models import LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron

# Training algorithms
from .algorithms import LiquidNetworkTrainer, AdaptiveOptimizer, ContinuousLearner

# Experimental tools
from .experiments import BenchmarkSuite, SyntheticTaskGenerator, ValidationExperiments

# Utilities
from .utils import DataGenerator, ResultsVisualizer, PerformanceMetrics

__all__ = [
    # Models
    'LiquidNeuralNetwork',
    'ContinuousTimeRNN', 
    'AdaptiveNeuron',
    
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