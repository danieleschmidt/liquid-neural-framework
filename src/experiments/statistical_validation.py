"""
Statistical validation framework for research experiments.

This module provides comprehensive statistical testing, significance analysis,
and reproducibility verification for liquid neural network research.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap
    import equinox as eqx
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp

from typing import Dict, Any, Tuple, Optional, List, Callable
import time
import warnings
from scipy import stats
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Container for experiment results with metadata."""
    method_name: str
    performance_metrics: Dict[str, float]
    execution_time: float
    memory_usage: Optional[float]
    hyperparameters: Dict[str, Any]
    random_seed: int
    timestamp: float
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    is_significant: bool
    interpretation: str


class ReproducibilityValidator:
    """Validates reproducibility of experimental results."""
    
    def __init__(
        self,
        num_runs: int = 10,
        significance_level: float = 0.05,
        random_seed_base: int = 42
    ):
        self.num_runs = num_runs
        self.significance_level = significance_level
        self.random_seed_base = random_seed_base
    
    def run_multiple_experiments(
        self,
        experiment_func: Callable,
        experiment_kwargs: Dict[str, Any],
        method_name: str
    ) -> List[ExperimentResult]:
        """Run the same experiment multiple times with different seeds."""
        results = []
        
        for run_idx in range(self.num_runs):
            # Set unique random seed for this run
            current_seed = self.random_seed_base + run_idx
            experiment_kwargs['random_seed'] = current_seed
            
            # Record start time and run experiment
            start_time = time.time()
            try:
                performance_metrics = experiment_func(**experiment_kwargs)
                execution_time = time.time() - start_time
                
                result = ExperimentResult(
                    method_name=method_name,
                    performance_metrics=performance_metrics,
                    execution_time=execution_time,
                    memory_usage=None,  # Could add memory profiling
                    hyperparameters=experiment_kwargs.copy(),
                    random_seed=current_seed,
                    timestamp=time.time()
                )
                results.append(result)
                
            except Exception as e:
                warnings.warn(f"Experiment run {run_idx} failed: {str(e)}")
                continue
        
        if len(results) < self.num_runs // 2:
            raise ValueError(f"Too many failed runs. Only {len(results)}/{self.num_runs} succeeded.")
        
        return results
    
    def compute_reproducibility_metrics(
        self, 
        results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Compute reproducibility metrics across multiple runs."""
        if not results:
            raise ValueError("No results provided")
        
        # Extract performance metrics
        metric_names = list(results[0].performance_metrics.keys())
        metrics_by_name = {}
        
        for metric_name in metric_names:
            values = [r.performance_metrics[metric_name] for r in results]
            metrics_by_name[metric_name] = jnp.array(values)
        
        reproducibility_stats = {}
        
        for metric_name, values in metrics_by_name.items():
            mean_val = float(jnp.mean(values))
            std_val = float(jnp.std(values))
            cv = std_val / (abs(mean_val) + 1e-8)  # Coefficient of variation
            
            # Confidence interval
            n = len(values)
            sem = std_val / jnp.sqrt(n)  # Standard error of mean
            t_critical = stats.t.ppf(1 - self.significance_level/2, n-1)
            ci_lower = mean_val - t_critical * sem
            ci_upper = mean_val + t_critical * sem
            
            reproducibility_stats[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'coefficient_of_variation': cv,
                'min': float(jnp.min(values)),
                'max': float(jnp.max(values)),
                'confidence_interval_95': (ci_lower, ci_upper),
                'num_runs': n
            }
        
        # Overall reproducibility score (lower CV = more reproducible)
        overall_cv = jnp.mean(jnp.array([
            reproducibility_stats[name]['coefficient_of_variation'] 
            for name in metric_names
        ]))
        
        reproducibility_stats['overall_reproducibility_score'] = float(1.0 / (1.0 + overall_cv))
        reproducibility_stats['execution_times'] = {
            'mean': float(jnp.mean([r.execution_time for r in results])),
            'std': float(jnp.std([r.execution_time for r in results]))
        }
        
        return reproducibility_stats


class ComparativeStatisticalAnalyzer:
    """Performs comparative statistical analysis between methods."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_two_methods(
        self,
        results_a: List[ExperimentResult],
        results_b: List[ExperimentResult],
        metric_name: str = 'accuracy'
    ) -> StatisticalTestResult:
        """Compare two methods using appropriate statistical tests."""
        
        # Extract metric values
        values_a = jnp.array([r.performance_metrics[metric_name] for r in results_a])
        values_b = jnp.array([r.performance_metrics[metric_name] for r in results_b])
        
        # Check normality (simplified)
        n_a, n_b = len(values_a), len(values_b)
        
        # Use appropriate test based on sample sizes and distributions
        if n_a < 30 or n_b < 30:
            # Use t-test for smaller samples
            statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
            test_name = "Welch's t-test"
        else:
            # Use Mann-Whitney U test for larger samples (non-parametric)
            statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        # Compute effect size (Cohen's d)
        pooled_std = jnp.sqrt(((n_a - 1) * jnp.var(values_a) + (n_b - 1) * jnp.var(values_b)) / (n_a + n_b - 2))
        cohens_d = float((jnp.mean(values_a) - jnp.mean(values_b)) / (pooled_std + 1e-8))
        
        # Confidence interval for difference in means
        diff_mean = float(jnp.mean(values_a) - jnp.mean(values_b))
        se_diff = jnp.sqrt(jnp.var(values_a)/n_a + jnp.var(values_b)/n_b)
        
        # Use appropriate degrees of freedom for Welch's t-test
        df = ((jnp.var(values_a)/n_a + jnp.var(values_b)/n_b)**2 / 
              ((jnp.var(values_a)/n_a)**2/(n_a-1) + (jnp.var(values_b)/n_b)**2/(n_b-1)))
        
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        ci_lower = diff_mean - t_critical * float(se_diff)
        ci_upper = diff_mean + t_critical * float(se_diff)
        
        is_significant = p_value < self.significance_level
        
        # Interpretation
        if is_significant:
            if cohens_d > 0.8:
                interpretation = f"Method A significantly outperforms Method B with large effect size (d={cohens_d:.3f})"
            elif cohens_d > 0.5:
                interpretation = f"Method A significantly outperforms Method B with medium effect size (d={cohens_d:.3f})"
            elif cohens_d > 0.2:
                interpretation = f"Method A significantly outperforms Method B with small effect size (d={cohens_d:.3f})"
            else:
                interpretation = f"Significant difference detected but with negligible effect size (d={cohens_d:.3f})"
        else:
            interpretation = f"No significant difference detected (p={p_value:.4f}, d={cohens_d:.3f})"
        
        return StatisticalTestResult(
            test_name=test_name,
            test_statistic=float(statistic),
            p_value=float(p_value),
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def multiple_comparisons_analysis(
        self,
        all_results: Dict[str, List[ExperimentResult]],
        metric_name: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Perform multiple comparisons analysis with Bonferroni correction."""
        
        method_names = list(all_results.keys())
        n_methods = len(method_names)
        n_comparisons = n_methods * (n_methods - 1) // 2
        
        # Bonferroni correction
        corrected_alpha = self.significance_level / n_comparisons
        
        pairwise_results = {}
        significant_pairs = []
        
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method_a = method_names[i]
                method_b = method_names[j]
                
                # Temporarily adjust significance level
                original_alpha = self.significance_level
                self.significance_level = corrected_alpha
                
                comparison_result = self.compare_two_methods(
                    all_results[method_a], 
                    all_results[method_b], 
                    metric_name
                )
                
                # Restore original alpha
                self.significance_level = original_alpha
                
                pair_key = f"{method_a}_vs_{method_b}"
                pairwise_results[pair_key] = comparison_result
                
                if comparison_result.is_significant:
                    significant_pairs.append((method_a, method_b, comparison_result.effect_size))
        
        # Rank methods by mean performance
        method_means = {}
        for method_name, results in all_results.items():
            values = [r.performance_metrics[metric_name] for r in results]
            method_means[method_name] = float(jnp.mean(jnp.array(values)))
        
        ranked_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'pairwise_comparisons': pairwise_results,
            'significant_pairs': significant_pairs,
            'method_rankings': ranked_methods,
            'bonferroni_corrected_alpha': corrected_alpha,
            'total_comparisons': n_comparisons
        }


class AdvancedStatisticalValidator:
    """Advanced statistical validation with effect size analysis and power calculations."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        significance_level: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute statistical power for given effect size and sample size."""
        
        if significance_level is None:
            significance_level = self.significance_level
        
        # Simplified power calculation for t-test
        # In practice, would use more sophisticated power analysis
        
        from scipy.stats import norm
        
        # Critical value for two-tailed test
        z_alpha = norm.ppf(1 - significance_level / 2)
        
        # Standard error for effect size
        se = jnp.sqrt(2 / sample_size)
        
        # Non-centrality parameter
        ncp = abs(effect_size) / se
        
        # Power calculation
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        
        return {
            'statistical_power': float(power),
            'effect_size': effect_size,
            'sample_size': sample_size,
            'significance_level': significance_level,
            'critical_value': float(z_alpha),
            'non_centrality_parameter': float(ncp)
        }
    
    def sample_size_calculation(
        self,
        desired_power: float = 0.8,
        effect_size: float = 0.5,
        significance_level: Optional[float] = None
    ) -> int:
        """Calculate required sample size for desired power."""
        
        if significance_level is None:
            significance_level = self.significance_level
        
        from scipy.stats import norm
        
        # Critical values
        z_alpha = norm.ppf(1 - significance_level / 2)
        z_beta = norm.ppf(desired_power)
        
        # Required sample size per group
        n_per_group = 2 * ((z_alpha + z_beta) / abs(effect_size))**2
        
        return int(jnp.ceil(n_per_group))
    
    def bayesian_comparison(
        self,
        results_a: List[ExperimentResult],
        results_b: List[ExperimentResult],
        metric_name: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Perform Bayesian comparison of two methods."""
        
        values_a = jnp.array([r.performance_metrics[metric_name] for r in results_a])
        values_b = jnp.array([r.performance_metrics[metric_name] for r in results_b])
        
        # Simple Bayesian t-test approximation
        # In practice, would use more sophisticated Bayesian methods
        
        # Prior parameters (non-informative priors)
        prior_mean = 0.0
        prior_precision = 0.001
        
        # Posterior parameters for difference
        n_a, n_b = len(values_a), len(values_b)
        mean_a, mean_b = float(jnp.mean(values_a)), float(jnp.mean(values_b))
        var_a, var_b = float(jnp.var(values_a)), float(jnp.var(values_b))
        
        # Approximate posterior for difference in means
        diff_mean = mean_a - mean_b
        diff_var = var_a / n_a + var_b / n_b
        
        # Posterior parameters
        posterior_precision = prior_precision + 1.0 / diff_var
        posterior_mean = (prior_precision * prior_mean + diff_mean / diff_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        # Probability that method A is better than method B
        from scipy.stats import norm
        prob_a_better = 1 - norm.cdf(0, posterior_mean, jnp.sqrt(posterior_var))
        
        # Bayes factor (simplified)
        # This is a rough approximation - proper Bayes factors require more computation
        marginal_likelihood_ratio = jnp.exp(-0.5 * (posterior_mean**2) / posterior_var) / jnp.sqrt(2 * jnp.pi * posterior_var)
        bayes_factor = marginal_likelihood_ratio / (1 - marginal_likelihood_ratio + 1e-8)
        
        # Credible interval for difference
        credible_interval = (
            float(norm.ppf(0.025, posterior_mean, jnp.sqrt(posterior_var))),
            float(norm.ppf(0.975, posterior_mean, jnp.sqrt(posterior_var)))
        )
        
        # Interpretation
        if prob_a_better > 0.95:
            interpretation = "Strong evidence that Method A outperforms Method B"
        elif prob_a_better > 0.8:
            interpretation = "Moderate evidence that Method A outperforms Method B"
        elif prob_a_better < 0.05:
            interpretation = "Strong evidence that Method B outperforms Method A"
        elif prob_a_better < 0.2:
            interpretation = "Moderate evidence that Method B outperforms Method A"
        else:
            interpretation = "Inconclusive evidence for superiority of either method"
        
        return {
            'probability_a_better': float(prob_a_better),
            'posterior_mean_difference': float(posterior_mean),
            'posterior_variance': float(posterior_var),
            'credible_interval_95': credible_interval,
            'bayes_factor': float(bayes_factor),
            'interpretation': interpretation
        }


class ExperimentalDesignValidator:
    """Validates experimental design and suggests improvements."""
    
    def __init__(self):
        self.design_requirements = {
            'min_sample_size': 30,
            'min_num_runs': 10,
            'max_cv_threshold': 0.2,  # Coefficient of variation threshold
            'min_power': 0.8
        }
    
    def validate_experimental_design(
        self,
        experimental_setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate experimental design and provide recommendations."""
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'design_score': 0.0
        }
        
        # Check sample size
        sample_size = experimental_setup.get('sample_size', 0)
        if sample_size < self.design_requirements['min_sample_size']:
            validation_results['warnings'].append(
                f"Sample size ({sample_size}) is below recommended minimum ({self.design_requirements['min_sample_size']})"
            )
            validation_results['recommendations'].append(
                f"Consider increasing sample size to at least {self.design_requirements['min_sample_size']} for reliable results"
            )
            validation_results['is_valid'] = False
        
        # Check number of runs
        num_runs = experimental_setup.get('num_runs', 0)
        if num_runs < self.design_requirements['min_num_runs']:
            validation_results['warnings'].append(
                f"Number of runs ({num_runs}) is below recommended minimum ({self.design_requirements['min_num_runs']})"
            )
            validation_results['recommendations'].append(
                f"Consider running at least {self.design_requirements['min_num_runs']} independent experiments"
            )
            validation_results['is_valid'] = False
        
        # Check for proper controls
        has_baseline = experimental_setup.get('has_baseline', False)
        if not has_baseline:
            validation_results['warnings'].append("No baseline method specified for comparison")
            validation_results['recommendations'].append("Include established baseline methods for meaningful comparisons")
        
        # Check randomization
        has_randomization = experimental_setup.get('has_randomization', False)
        if not has_randomization:
            validation_results['warnings'].append("No randomization strategy specified")
            validation_results['recommendations'].append("Implement proper randomization of experimental conditions")
        
        # Calculate design score
        score_components = []
        score_components.append(min(1.0, sample_size / self.design_requirements['min_sample_size']))
        score_components.append(min(1.0, num_runs / self.design_requirements['min_num_runs']))
        score_components.append(1.0 if has_baseline else 0.0)
        score_components.append(1.0 if has_randomization else 0.0)
        
        validation_results['design_score'] = float(jnp.mean(jnp.array(score_components)))
        
        return validation_results
    
    def suggest_experimental_improvements(
        self,
        current_results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Analyze current results and suggest experimental improvements."""
        
        if not current_results:
            return {'error': 'No results provided for analysis'}
        
        # Analyze result variability
        metric_names = list(current_results[0].performance_metrics.keys())
        variability_analysis = {}
        
        for metric_name in metric_names:
            values = jnp.array([r.performance_metrics[metric_name] for r in current_results])
            cv = float(jnp.std(values) / (abs(jnp.mean(values)) + 1e-8))
            variability_analysis[metric_name] = cv
        
        suggestions = {
            'variability_analysis': variability_analysis,
            'high_variability_metrics': [],
            'suggestions': []
        }
        
        # Identify high variability metrics
        for metric_name, cv in variability_analysis.items():
            if cv > self.design_requirements['max_cv_threshold']:
                suggestions['high_variability_metrics'].append(metric_name)
                suggestions['suggestions'].append(
                    f"High variability in {metric_name} (CV={cv:.3f}). Consider: "
                    "1) Increasing sample size, 2) Better hyperparameter tuning, "
                    "3) More robust evaluation metrics"
                )
        
        # Analyze execution times
        exec_times = [r.execution_time for r in current_results]
        if len(set(exec_times)) > 1:  # Variable execution times
            time_cv = float(jnp.std(jnp.array(exec_times)) / (jnp.mean(jnp.array(exec_times)) + 1e-8))
            if time_cv > 0.3:
                suggestions['suggestions'].append(
                    f"High variability in execution times (CV={time_cv:.3f}). "
                    "Consider implementing more consistent timing measurements or "
                    "controlling for computational resources"
                )
        
        return suggestions