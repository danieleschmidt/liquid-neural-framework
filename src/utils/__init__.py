# Import with fallback handling for missing JAX dependencies
try:
    from .data_utils import DataGenerator, DataPreprocessor
except ImportError:
    class DataGenerator:
        def __init__(self, *args, **kwargs): pass
    class DataPreprocessor:
        def __init__(self, *args, **kwargs): pass

try:
    from .visualization import ResultsVisualizer, NetworkVisualizer
except ImportError:
    class ResultsVisualizer:
        def __init__(self, *args, **kwargs): pass
    class NetworkVisualizer:
        def __init__(self, *args, **kwargs): pass

try:
    from .metrics import PerformanceMetrics, StatisticalAnalysis
except ImportError:
    class PerformanceMetrics:
        def __init__(self, *args, **kwargs): pass
    class StatisticalAnalysis:
        def __init__(self, *args, **kwargs): pass

__all__ = ['DataGenerator', 'DataPreprocessor', 'ResultsVisualizer', 'NetworkVisualizer', 'PerformanceMetrics', 'StatisticalAnalysis']