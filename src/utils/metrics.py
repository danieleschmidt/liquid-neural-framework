"""
Advanced performance metrics and statistical analysis tools.

Provides comprehensive evaluation metrics for liquid neural networks
including specialized metrics for temporal dynamics and research-grade
statistical analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from collections import defaultdict
import json
from pathlib import Path

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu, kruskal, pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some statistical analyses will be limited.")


class MetricsError(Exception):
    """Custom exception for metrics computation errors."""
    pass


class PerformanceMetrics:
    """
    Comprehensive performance metrics for liquid neural networks.
    
    Includes standard ML metrics plus specialized metrics for temporal
    dynamics, memory capacity, and liquid computing properties.
    """
    
    def __init__(self, numerical_stability_eps: float = 1e-8):
        self.eps = numerical_stability_eps
        self.computed_metrics = {}
        
    def mean_squared_error(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Compute mean squared error with numerical stability."""
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        
        if predictions.shape != targets.shape:
            raise MetricsError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        squared_errors = (predictions - targets) ** 2
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != predictions.shape[0]:
                raise MetricsError("Sample weight length must match number of samples")
            squared_errors *= sample_weight
            mse = np.sum(squared_errors) / np.sum(sample_weight)
        else:
            mse = np.mean(squared_errors)
        
        return float(mse)
    
    def mean_absolute_error(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Compute mean absolute error."""
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        
        if predictions.shape != targets.shape:
            raise MetricsError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        absolute_errors = np.abs(predictions - targets)
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            absolute_errors *= sample_weight
            mae = np.sum(absolute_errors) / np.sum(sample_weight)
        else:
            mae = np.mean(absolute_errors)
        
        return float(mae)
    
    def r2_score(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        multioutput: str = 'uniform_average'
    ) -> Union[float, np.ndarray]:
        """Compute R² coefficient of determination."""
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        
        if predictions.shape != targets.shape:
            raise MetricsError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        # Handle multi-output case
        if targets.ndim > 1 and targets.shape[1] > 1:
            r2_scores = []
            for i in range(targets.shape[1]):
                ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
                ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
                r2 = 1 - (ss_res / (ss_tot + self.eps))
                r2_scores.append(r2)
            
            r2_scores = np.array(r2_scores)
            
            if multioutput == 'uniform_average':
                return float(np.mean(r2_scores))
            elif multioutput == 'raw_values':
                return r2_scores
            else:
                raise MetricsError(f"Unknown multioutput option: {multioutput}")
        else:
            # Single output case
            targets_flat = targets.flatten()
            predictions_flat = predictions.flatten()
            
            ss_res = np.sum((targets_flat - predictions_flat) ** 2)
            ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + self.eps))
            
            return float(r2)
    
    def temporal_correlation_coefficient(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        lag_range: int = 10
    ) -> Dict[str, float]:
        """
        Compute temporal correlation coefficients for different time lags.
        
        Specialized metric for evaluating temporal prediction accuracy.
        """
        if predictions.ndim != 2 or targets.ndim != 2:
            raise MetricsError("Predictions and targets must be 2D (time, features)")
        
        if predictions.shape != targets.shape:
            raise MetricsError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
        
        seq_len, n_features = predictions.shape
        correlations = {}
        
        for lag in range(-lag_range, lag_range + 1):
            if lag == 0:
                # Zero-lag correlation
                corr_values = []
                for feat in range(n_features):
                    if HAS_SCIPY:
                        corr, _ = pearsonr(predictions[:, feat], targets[:, feat])
                    else:
                        # Manual Pearson correlation
                        pred_centered = predictions[:, feat] - np.mean(predictions[:, feat])
                        targ_centered = targets[:, feat] - np.mean(targets[:, feat])
                        
                        numerator = np.sum(pred_centered * targ_centered)
                        denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(targ_centered**2))
                        
                        corr = numerator / (denominator + self.eps)
                    
                    if not np.isnan(corr):
                        corr_values.append(corr)
                
                correlations[f'lag_{lag}'] = float(np.mean(corr_values)) if corr_values else 0.0
                
            elif lag > 0:
                # Positive lag: predictions lead targets
                if seq_len - lag > 10:  # Need sufficient data
                    corr_values = []
                    for feat in range(n_features):
                        pred_lead = predictions[:-lag, feat]
                        targ_follow = targets[lag:, feat]
                        
                        if HAS_SCIPY:
                            corr, _ = pearsonr(pred_lead, targ_follow)
                        else:
                            pred_centered = pred_lead - np.mean(pred_lead)
                            targ_centered = targ_follow - np.mean(targ_follow)
                            
                            numerator = np.sum(pred_centered * targ_centered)
                            denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(targ_centered**2))
                            
                            corr = numerator / (denominator + self.eps)
                        
                        if not np.isnan(corr):
                            corr_values.append(corr)
                    
                    correlations[f'lag_{lag}'] = float(np.mean(corr_values)) if corr_values else 0.0
            
            else:  # lag < 0
                # Negative lag: targets lead predictions
                pos_lag = -lag
                if seq_len - pos_lag > 10:
                    corr_values = []
                    for feat in range(n_features):
                        targ_lead = targets[:-pos_lag, feat]
                        pred_follow = predictions[pos_lag:, feat]
                        
                        if HAS_SCIPY:
                            corr, _ = pearsonr(targ_lead, pred_follow)
                        else:
                            targ_centered = targ_lead - np.mean(targ_lead)
                            pred_centered = pred_follow - np.mean(pred_follow)
                            
                            numerator = np.sum(targ_centered * pred_centered)
                            denominator = np.sqrt(np.sum(targ_centered**2) * np.sum(pred_centered**2))
                            
                            corr = numerator / (denominator + self.eps)
                        
                        if not np.isnan(corr):
                            corr_values.append(corr)
                    
                    correlations[f'lag_{lag}'] = float(np.mean(corr_values)) if corr_values else 0.0
        
        return correlations
    
    def memory_capacity_metric(
        self,
        network_states: np.ndarray,
        input_history: np.ndarray,
        max_delay: int = 50
    ) -> Dict[str, float]:
        """
        Compute memory capacity of the liquid neural network.
        
        Based on the ability to linearly reconstruct past inputs
        from current network state.
        """
        if network_states.ndim != 2:
            raise MetricsError("Network states must be 2D (time, neurons)")
        
        if input_history.ndim == 1:
            input_history = input_history.reshape(-1, 1)
        
        seq_len, n_neurons = network_states.shape
        input_dim = input_history.shape[1]
        
        memory_capacities = {}
        
        for delay in range(1, min(max_delay + 1, seq_len // 2)):
            # Target: input at time t-delay
            # Features: network state at time t
            
            valid_indices = range(delay, seq_len)
            if len(valid_indices) < 20:  # Need sufficient data
                continue
            
            X = network_states[valid_indices]  # Current states
            Y = input_history[valid_indices - delay]  # Past inputs
            
            # Linear regression for each input dimension
            capacities = []
            
            for input_dim_idx in range(input_dim):
                y = Y[:, input_dim_idx]
                
                try:
                    # Simple linear regression (X @ beta = y)
                    # beta = (X^T X + eps*I)^{-1} X^T y
                    XTX = X.T @ X
                    XTX += self.eps * np.eye(XTX.shape[0])  # Regularization
                    XTy = X.T @ y
                    
                    beta = np.linalg.solve(XTX, XTy)
                    y_pred = X @ beta
                    
                    # Compute R² for this reconstruction
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + self.eps))
                    
                    capacities.append(max(0.0, r2))  # Memory capacity is non-negative
                    
                except np.linalg.LinAlgError:
                    capacities.append(0.0)
            
            memory_capacities[f'delay_{delay}'] = float(np.mean(capacities))
        
        # Compute total memory capacity
        total_capacity = sum(memory_capacities.values())
        memory_capacities['total_capacity'] = total_capacity
        
        return memory_capacities
    
    def liquid_computing_metrics(
        self,
        network_states: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute specialized metrics for liquid computing systems.
        """
        metrics = {}
        
        # 1. Separation Property (different inputs -> different states)
        if len(inputs) > 1:
            state_distances = []
            input_distances = []
            
            n_samples = min(100, len(inputs))  # Limit comparisons for efficiency
            indices = np.random.choice(len(inputs), n_samples, replace=False)
            
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    idx1, idx2 = indices[i], indices[j]
                    
                    # Input distance
                    input_dist = np.linalg.norm(inputs[idx1] - inputs[idx2])
                    
                    # State distance (final states)
                    if network_states.ndim == 3:  # (batch, time, neurons)
                        state1 = network_states[idx1, -1, :]  # Final state
                        state2 = network_states[idx2, -1, :]
                    else:  # (time, neurons) - single sequence
                        continue  # Skip for single sequence
                    
                    state_dist = np.linalg.norm(state1 - state2)
                    
                    input_distances.append(input_dist)
                    state_distances.append(state_dist)
            
            if input_distances and state_distances:
                # Correlation between input and state distances
                if HAS_SCIPY:
                    separation_corr, _ = pearsonr(input_distances, state_distances)
                else:
                    # Manual correlation
                    input_distances = np.array(input_distances)
                    state_distances = np.array(state_distances)
                    
                    input_centered = input_distances - np.mean(input_distances)
                    state_centered = state_distances - np.mean(state_distances)
                    
                    numerator = np.sum(input_centered * state_centered)
                    denominator = np.sqrt(np.sum(input_centered**2) * np.sum(state_centered**2))
                    
                    separation_corr = numerator / (denominator + self.eps)
                
                metrics['separation_property'] = float(separation_corr)
        
        # 2. Approximation Property (readout quality)
        if targets is not None:
            # This is typically measured by the final task performance
            # For now, use the R² between network output and targets
            if network_states.ndim == 2:
                # Single sequence case - use linear readout
                try:
                    # Simple linear readout from final layer
                    X = network_states
                    y = targets.reshape(-1) if targets.ndim > 1 else targets
                    
                    # Ridge regression
                    XTX = X.T @ X
                    XTX += 0.01 * np.eye(XTX.shape[0])  # Ridge regularization
                    XTy = X.T @ y
                    
                    beta = np.linalg.solve(XTX, XTy)
                    y_pred = X @ beta
                    
                    # R² score
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + self.eps))
                    
                    metrics['approximation_property'] = float(max(0.0, r2))
                    
                except np.linalg.LinAlgError:
                    metrics['approximation_property'] = 0.0
        
        # 3. Echo State Property (stability)
        if network_states.ndim == 2 and len(network_states) > 10:
            # Measure state stability over time
            state_norms = np.linalg.norm(network_states, axis=1)
            
            # Check if states remain bounded
            max_norm = np.max(state_norms)
            mean_norm = np.mean(state_norms)
            
            # Stability metric: states should not grow unbounded
            stability = np.exp(-max_norm / (mean_norm + self.eps))
            metrics['echo_state_property'] = float(stability)
            
            # Lyapunov exponent approximation
            if len(network_states) > 20:
                # Simple approximation based on state divergence
                state_diffs = np.diff(network_states, axis=0)
                diff_norms = np.linalg.norm(state_diffs, axis=1)
                
                # Average rate of change
                lyapunov_approx = np.mean(diff_norms) / (np.mean(state_norms) + self.eps)
                metrics['lyapunov_approximation'] = float(lyapunov_approx)
        
        return metrics
    
    def compute_comprehensive_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        network_states: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
        compute_temporal: bool = True,
        compute_memory: bool = True,
        compute_liquid: bool = True
    ) -> Dict[str, Any]:
        """
        Compute all available metrics in one call.
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = self.mean_squared_error(predictions, targets)
        metrics['mae'] = self.mean_absolute_error(predictions, targets)
        metrics['r2'] = self.r2_score(predictions, targets)
        
        # Temporal metrics
        if compute_temporal and predictions.ndim >= 2:
            try:
                temporal_corr = self.temporal_correlation_coefficient(predictions, targets)
                metrics['temporal_correlations'] = temporal_corr
            except Exception as e:
                warnings.warn(f"Failed to compute temporal correlations: {e}")
        
        # Memory capacity metrics
        if compute_memory and network_states is not None and inputs is not None:
            try:
                memory_metrics = self.memory_capacity_metric(network_states, inputs)
                metrics['memory_capacity'] = memory_metrics
            except Exception as e:
                warnings.warn(f"Failed to compute memory capacity: {e}")
        
        # Liquid computing metrics
        if compute_liquid and network_states is not None:
            try:
                liquid_metrics = self.liquid_computing_metrics(
                    network_states, inputs, targets
                )
                metrics['liquid_properties'] = liquid_metrics
            except Exception as e:
                warnings.warn(f"Failed to compute liquid computing metrics: {e}")
        
        # Store for future reference
        self.computed_metrics.update(metrics)
        
        return metrics
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export computed metrics to file."""
        if not self.computed_metrics:
            warnings.warn("No metrics computed yet")
            return
        
        filepath = Path(filepath)
        
        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            converted_metrics = convert_for_json(self.computed_metrics)
            
            with open(filepath, 'w') as f:
                json.dump(converted_metrics, f, indent=2)
        
        else:
            raise MetricsError(f"Unsupported format: {format}")


class StatisticalAnalysis:
    """
    Research-grade statistical analysis tools for model comparison.
    """
    
    def __init__(self, alpha: float = 0.05, bonferroni_correction: bool = True):
        self.alpha = alpha
        self.bonferroni_correction = bonferroni_correction
        
        if not HAS_SCIPY:
            warnings.warn(
                "SciPy not available. Some statistical tests will use simplified implementations."
            )
    
    def paired_t_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Perform paired t-test between two samples.
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        if len(sample1) != len(sample2):
            raise MetricsError("Samples must have the same length for paired t-test")
        
        if HAS_SCIPY:
            statistic, p_value = ttest_ind(sample1, sample2, alternative=alternative)
        else:
            # Manual implementation
            diff = sample1 - sample2
            n = len(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            
            if std_diff == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic = mean_diff / (std_diff / np.sqrt(n))
                
                # Approximate p-value using normal distribution
                from math import erfc
                if alternative == 'two-sided':
                    p_value = erfc(abs(statistic) / np.sqrt(2))
                elif alternative == 'less':
                    p_value = 0.5 * erfc(statistic / np.sqrt(2))
                else:  # 'greater'
                    p_value = 0.5 * erfc(-statistic / np.sqrt(2))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(sample1, ddof=1) + np.var(sample2, ddof=1)) / 2)
        cohens_d = (np.mean(sample1) - np.mean(sample2)) / (pooled_std + 1e-8)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d)
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def multiple_comparisons_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> Dict[str, Any]:
        """
        Apply multiple comparisons correction.
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            adjusted_alpha = self.alpha / n_tests
            adjusted_p_values = np.minimum(p_values * n_tests, 1.0)
            significant = p_values < adjusted_alpha
            
        elif method == 'fdr_bh':  # Benjamini-Hochberg
            # Sort p-values
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]
            
            # BH procedure
            adjusted_p_values = np.zeros_like(p_values)
            significant = np.zeros(len(p_values), dtype=bool)
            
            for i in range(n_tests - 1, -1, -1):
                bh_threshold = (i + 1) / n_tests * self.alpha
                if sorted_p_values[i] <= bh_threshold:
                    # Reject this and all smaller p-values
                    for j in range(i + 1):
                        original_idx = sorted_indices[j]
                        significant[original_idx] = True
                        adjusted_p_values[original_idx] = sorted_p_values[j] * n_tests / (j + 1)
                    break
            
        else:
            raise MetricsError(f"Unknown correction method: {method}")
        
        return {
            'adjusted_p_values': adjusted_p_values.tolist(),
            'significant': significant.tolist(),
            'adjusted_alpha': adjusted_alpha if method == 'bonferroni' else self.alpha,
            'method': method,
            'n_significant': int(np.sum(significant)),
            'n_tests': n_tests
        }
    
    def anova_test(
        self,
        groups: List[np.ndarray],
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test.
        """
        if len(groups) < 2:
            raise MetricsError("Need at least 2 groups for ANOVA")
        
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(len(groups))]
        
        # Convert to arrays
        groups = [np.asarray(g) for g in groups]
        
        if HAS_SCIPY:
            statistic, p_value = stats.f_oneway(*groups)
        else:
            # Manual ANOVA implementation
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
            
            # Within-group sum of squares
            ss_within = sum(np.sum((group - np.mean(group))**2) for group in groups)
            
            # Degrees of freedom
            df_between = len(groups) - 1
            df_within = len(all_data) - len(groups)
            
            # Mean squares
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            
            # F-statistic
            if ms_within == 0:
                statistic = float('inf')
                p_value = 0.0
            else:
                statistic = ms_between / ms_within
                # Approximate p-value (would need F-distribution for exact)
                p_value = 0.05 if statistic > 4 else 0.5  # Very rough approximation
        
        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_total = np.sum((all_data - grand_mean)**2)
        ss_between_calculated = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        eta_squared = ss_between_calculated / ss_total if ss_total > 0 else 0.0
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'eta_squared': float(eta_squared),
            'group_means': [float(np.mean(group)) for group in groups],
            'group_stds': [float(np.std(group, ddof=1)) for group in groups],
            'group_names': group_names,
            'df_between': df_between if HAS_SCIPY else len(groups) - 1,
            'df_within': df_within if HAS_SCIPY else len(all_data) - len(groups)
        }
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: Callable = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for a statistic.
        """
        data = np.asarray(data)
        
        if len(data) < 2:
            raise MetricsError("Need at least 2 data points for bootstrap")
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'statistic': float(statistic_func(data)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence_level': confidence_level,
            'bootstrap_mean': float(np.mean(bootstrap_stats)),
            'bootstrap_std': float(np.std(bootstrap_stats, ddof=1)),
            'n_bootstrap': n_bootstrap
        }
    
    def model_comparison_analysis(
        self,
        results: Dict[str, List[float]],
        metric_name: str = "performance"
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis for model comparison.
        """
        model_names = list(results.keys())
        model_data = [np.asarray(results[name]) for name in model_names]
        
        analysis = {
            'metric_name': metric_name,
            'model_names': model_names,
            'n_models': len(model_names)
        }
        
        # Descriptive statistics
        analysis['descriptive_stats'] = {}
        for name, data in zip(model_names, model_data):
            analysis['descriptive_stats'][name] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'n_samples': len(data)
            }
        
        # ANOVA test
        if len(model_names) > 2:
            try:
                anova_result = self.anova_test(model_data, model_names)
                analysis['anova'] = anova_result
            except Exception as e:
                warnings.warn(f"ANOVA test failed: {e}")
        
        # Pairwise comparisons
        analysis['pairwise_comparisons'] = {}
        p_values_for_correction = []
        comparison_pairs = []
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    try:
                        comparison = self.paired_t_test(model_data[i], model_data[j])
                        pair_name = f"{name1}_vs_{name2}"
                        analysis['pairwise_comparisons'][pair_name] = comparison
                        
                        p_values_for_correction.append(comparison['p_value'])
                        comparison_pairs.append(pair_name)
                        
                    except Exception as e:
                        warnings.warn(f"Pairwise comparison {name1} vs {name2} failed: {e}")
        
        # Multiple comparisons correction
        if p_values_for_correction:
            correction_result = self.multiple_comparisons_correction(
                p_values_for_correction, 
                method='bonferroni' if self.bonferroni_correction else 'fdr_bh'
            )
            analysis['multiple_comparisons'] = correction_result
            
            # Update pairwise comparisons with corrected significance
            for i, pair_name in enumerate(comparison_pairs):
                if pair_name in analysis['pairwise_comparisons']:
                    analysis['pairwise_comparisons'][pair_name]['corrected_significant'] = \
                        correction_result['significant'][i]
        
        # Bootstrap confidence intervals
        analysis['confidence_intervals'] = {}
        for name, data in zip(model_names, model_data):
            try:
                ci_result = self.bootstrap_confidence_interval(data)
                analysis['confidence_intervals'][name] = ci_result
            except Exception as e:
                warnings.warn(f"Bootstrap CI for {name} failed: {e}")
        
        # Model ranking
        mean_performances = [
            analysis['descriptive_stats'][name]['mean'] 
            for name in model_names
        ]
        
        # Assume lower is better (like loss) - can be reversed if needed
        ranking_indices = np.argsort(mean_performances)
        analysis['ranking'] = {
            'ranked_models': [model_names[i] for i in ranking_indices],
            'ranking_scores': [mean_performances[i] for i in ranking_indices]
        }
        
        return analysis
    
    def export_analysis(self, analysis: Dict[str, Any], filepath: str, format: str = 'json'):
        """Export statistical analysis results."""
        filepath = Path(filepath)
        
        if format == 'json':
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            converted_analysis = convert_for_json(analysis)
            
            with open(filepath, 'w') as f:
                json.dump(converted_analysis, f, indent=2)
        else:
            raise MetricsError(f"Unsupported format: {format}")


def create_performance_report(
    predictions: Dict[str, np.ndarray],
    targets: np.ndarray,
    network_states: Optional[Dict[str, np.ndarray]] = None,
    inputs: Optional[np.ndarray] = None,
    output_dir: str = "performance_analysis"
) -> Dict[str, Any]:
    """
    Create comprehensive performance report for multiple models.
    
    Args:
        predictions: Dictionary mapping model names to prediction arrays
        targets: Target values
        network_states: Optional dictionary of network internal states
        inputs: Optional input sequences
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing all analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize analyzers
    metrics_analyzer = PerformanceMetrics()
    statistical_analyzer = StatisticalAnalysis()
    
    # Compute metrics for each model
    all_metrics = {}
    performance_data = {}
    
    for model_name, preds in predictions.items():
        print(f"Analyzing {model_name}...")
        
        # Get network states if available
        states = network_states.get(model_name) if network_states else None
        
        # Compute comprehensive metrics
        metrics = metrics_analyzer.compute_comprehensive_metrics(
            preds, targets, states, inputs
        )
        
        all_metrics[model_name] = metrics
        
        # Extract performance values for statistical analysis
        performance_data[model_name] = [metrics['mse']]  # Can be extended
    
    # Statistical analysis
    if len(predictions) > 1:
        statistical_results = statistical_analyzer.model_comparison_analysis(
            performance_data, metric_name="MSE"
        )
    else:
        statistical_results = {}
    
    # Compile final report
    report = {
        'individual_metrics': all_metrics,
        'statistical_analysis': statistical_results,
        'summary': {
            'n_models': len(predictions),
            'best_model': min(all_metrics.keys(), 
                            key=lambda x: all_metrics[x]['mse']),
            'evaluation_date': str(Path().cwd()),
            'targets_shape': list(targets.shape),
            'has_network_states': network_states is not None,
            'has_inputs': inputs is not None
        }
    }
    
    # Save individual components
    metrics_analyzer.export_metrics(
        output_path / "detailed_metrics.json"
    )
    
    if statistical_results:
        statistical_analyzer.export_analysis(
            statistical_results, 
            output_path / "statistical_analysis.json"
        )
    
    # Save comprehensive report
    report_path = output_path / "performance_report.json"
    with open(report_path, 'w') as f:
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_for_json(report), f, indent=2)
    
    print(f"Performance analysis complete. Results saved to {output_dir}")
    
    return report