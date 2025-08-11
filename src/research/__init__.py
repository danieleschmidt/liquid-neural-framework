"""
Research module for novel liquid neural network algorithms.
"""

from .novel_algorithms import (
    MetaAdaptiveLiquidNetwork,
    MultiScaleTemporalNetwork,
    QuantumInspiredContinuousComputation,
    compare_novel_algorithms
)

from .benchmarking_suite import (
    ComprehensiveBenchmarkSuite,
    PerformanceAnalyzer,
    StatisticalValidator
)

__all__ = [
    'MetaAdaptiveLiquidNetwork',
    'MultiScaleTemporalNetwork', 
    'QuantumInspiredContinuousComputation',
    'compare_novel_algorithms',
    'ComprehensiveBenchmarkSuite',
    'PerformanceAnalyzer',
    'StatisticalValidator'
]