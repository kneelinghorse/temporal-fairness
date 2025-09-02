"""
BCa Bootstrap Confidence Intervals
Publication-quality bootstrap methods for temporal fairness analysis
Implements bias-corrected and accelerated bootstrap following Efron & Tibshirani (1993)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, List
from scipy import stats
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings


@dataclass
class BootstrapResult:
    """Container for bootstrap analysis results"""
    point_estimate: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    bias: float
    iterations: int
    method: str
    confidence_level: float
    z0: float  # Bias correction parameter
    a: float   # Acceleration parameter
    bootstrap_distribution: np.ndarray
    computation_time_ms: float


class BCaBootstrap:
    """
    Bias-Corrected and Accelerated (BCa) Bootstrap
    
    Implements the BCa bootstrap method for constructing confidence intervals
    with second-order accuracy, handling skewed distributions and transformations.
    
    References:
    - Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap
    - DiCiccio, T. J., & Efron, B. (1996). Bootstrap Confidence Intervals
    """
    
    def __init__(self, 
                 iterations: int = 10000,
                 confidence_level: float = 0.95,
                 parallel: bool = True,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        """
        Initialize BCa Bootstrap calculator
        
        Args:
            iterations: Number of bootstrap iterations (10000 for publication quality)
            confidence_level: Confidence level for intervals (default 0.95)
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.iterations = iterations
        self.confidence_level = confidence_level
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def calculate(self,
                  data: np.ndarray,
                  statistic: Callable,
                  stratify_by: Optional[np.ndarray] = None,
                  handle_temporal: bool = False,
                  block_size: Optional[int] = None) -> BootstrapResult:
        """
        Calculate BCa bootstrap confidence intervals
        
        Args:
            data: Input data array
            statistic: Function to calculate statistic of interest
            stratify_by: Optional array for stratified sampling
            handle_temporal: Whether to handle temporal correlation
            block_size: Block size for temporal bootstrap
        
        Returns:
            BootstrapResult with confidence intervals and diagnostics
        """
        start_time = time.perf_counter()
        
        # Calculate point estimate
        point_estimate = statistic(data)
        
        # Generate bootstrap samples
        if handle_temporal and block_size is None:
            block_size = int(np.sqrt(len(data)))
        
        bootstrap_stats = self._generate_bootstrap_samples(
            data, statistic, stratify_by, handle_temporal, block_size
        )
        
        # Calculate bias correction parameter (z0)
        z0 = self._calculate_bias_correction(point_estimate, bootstrap_stats)
        
        # Calculate acceleration parameter (a)
        a = self._calculate_acceleration(data, statistic)
        
        # Calculate BCa confidence intervals
        lower, upper = self._calculate_bca_intervals(
            bootstrap_stats, z0, a, self.confidence_level
        )
        
        # Calculate standard error
        standard_error = np.std(bootstrap_stats, ddof=1)
        
        # Calculate bias
        bias = np.mean(bootstrap_stats) - point_estimate
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        return BootstrapResult(
            point_estimate=point_estimate,
            confidence_interval=(lower, upper),
            standard_error=standard_error,
            bias=bias,
            iterations=self.iterations,
            method="BCa",
            confidence_level=self.confidence_level,
            z0=z0,
            a=a,
            bootstrap_distribution=bootstrap_stats,
            computation_time_ms=computation_time
        )
    
    def _generate_bootstrap_samples(self,
                                   data: np.ndarray,
                                   statistic: Callable,
                                   stratify_by: Optional[np.ndarray],
                                   handle_temporal: bool,
                                   block_size: Optional[int]) -> np.ndarray:
        """Generate bootstrap samples with optional stratification and temporal handling"""
        
        n = len(data)
        bootstrap_stats = np.zeros(self.iterations)
        
        if self.parallel and not handle_temporal:
            # Parallel processing for non-temporal data
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i in range(self.iterations):
                    future = executor.submit(
                        self._single_bootstrap,
                        data, statistic, stratify_by, None
                    )
                    futures.append(future)
                
                for i, future in enumerate(futures):
                    bootstrap_stats[i] = future.result()
        else:
            # Sequential processing or temporal data
            for i in range(self.iterations):
                if handle_temporal:
                    # Block bootstrap for temporal data
                    sample = self._block_bootstrap_sample(data, block_size)
                elif stratify_by is not None:
                    # Stratified bootstrap
                    sample = self._stratified_bootstrap_sample(data, stratify_by)
                else:
                    # Standard bootstrap
                    indices = np.random.choice(n, size=n, replace=True)
                    sample = data[indices]
                
                bootstrap_stats[i] = statistic(sample)
        
        return bootstrap_stats
    
    def _single_bootstrap(self,
                         data: np.ndarray,
                         statistic: Callable,
                         stratify_by: Optional[np.ndarray],
                         _: Any) -> float:
        """Single bootstrap iteration"""
        n = len(data)
        
        if stratify_by is not None:
            sample = self._stratified_bootstrap_sample(data, stratify_by)
        else:
            indices = np.random.choice(n, size=n, replace=True)
            sample = data[indices]
        
        return statistic(sample)
    
    def _stratified_bootstrap_sample(self,
                                    data: np.ndarray,
                                    stratify_by: np.ndarray) -> np.ndarray:
        """Generate stratified bootstrap sample maintaining group proportions"""
        
        unique_strata = np.unique(stratify_by)
        sample_indices = []
        
        for stratum in unique_strata:
            stratum_indices = np.where(stratify_by == stratum)[0]
            n_stratum = len(stratum_indices)
            
            # Maintain minimum of 30 per stratum if possible
            if n_stratum < 30:
                warnings.warn(f"Stratum {stratum} has only {n_stratum} samples")
            
            # Sample within stratum
            sampled = np.random.choice(stratum_indices, size=n_stratum, replace=True)
            sample_indices.extend(sampled)
        
        return data[sample_indices]
    
    def _block_bootstrap_sample(self,
                               data: np.ndarray,
                               block_size: int) -> np.ndarray:
        """Generate block bootstrap sample for temporal data"""
        
        n = len(data)
        n_blocks = n // block_size + (1 if n % block_size else 0)
        
        sample = []
        while len(sample) < n:
            # Randomly select a starting point
            start = np.random.randint(0, n - block_size + 1)
            block = data[start:start + block_size]
            sample.extend(block)
        
        return np.array(sample[:n])
    
    def _calculate_bias_correction(self,
                                  point_estimate: float,
                                  bootstrap_stats: np.ndarray) -> float:
        """Calculate bias correction parameter z0"""
        
        # Proportion of bootstrap statistics less than original estimate
        prop = np.mean(bootstrap_stats < point_estimate)
        
        # Avoid extreme values
        prop = np.clip(prop, 0.001, 0.999)
        
        # Calculate z0 using inverse normal CDF
        z0 = stats.norm.ppf(prop)
        
        return z0
    
    def _calculate_acceleration(self,
                               data: np.ndarray,
                               statistic: Callable) -> float:
        """Calculate acceleration parameter using jackknife"""
        
        n = len(data)
        jackknife_stats = np.zeros(n)
        
        # Calculate jackknife statistics
        for i in range(n):
            # Leave one out
            jack_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats[i] = statistic(jack_sample)
        
        # Calculate acceleration
        mean_jack = np.mean(jackknife_stats)
        
        numerator = np.sum((mean_jack - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((mean_jack - jackknife_stats) ** 2) ** 1.5)
        
        # Avoid division by zero
        if denominator == 0:
            return 0.0
        
        a = numerator / denominator
        
        return a
    
    def _calculate_bca_intervals(self,
                                bootstrap_stats: np.ndarray,
                                z0: float,
                                a: float,
                                confidence_level: float) -> Tuple[float, float]:
        """Calculate BCa confidence intervals"""
        
        # Calculate alpha levels
        alpha = 1 - confidence_level
        alpha_lower = alpha / 2
        alpha_upper = 1 - alpha / 2
        
        # Standard normal quantiles
        z_lower = stats.norm.ppf(alpha_lower)
        z_upper = stats.norm.ppf(alpha_upper)
        
        # BCa adjusted quantiles
        a1 = stats.norm.cdf(z0 + (z0 + z_lower) / (1 - a * (z0 + z_lower)))
        a2 = stats.norm.cdf(z0 + (z0 + z_upper) / (1 - a * (z0 + z_upper)))
        
        # Ensure valid percentiles
        a1 = np.clip(a1, 0.001, 0.999)
        a2 = np.clip(a2, 0.001, 0.999)
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_stats, a1 * 100)
        upper = np.percentile(bootstrap_stats, a2 * 100)
        
        return lower, upper
    
    def compare_methods(self,
                       data: np.ndarray,
                       statistic: Callable) -> Dict[str, BootstrapResult]:
        """Compare BCa with percentile and basic bootstrap methods"""
        
        results = {}
        
        # BCa method
        results['BCa'] = self.calculate(data, statistic)
        
        # Percentile method
        bootstrap_stats = results['BCa'].bootstrap_distribution
        alpha = 1 - self.confidence_level
        lower_p = np.percentile(bootstrap_stats, (alpha/2) * 100)
        upper_p = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        results['Percentile'] = BootstrapResult(
            point_estimate=results['BCa'].point_estimate,
            confidence_interval=(lower_p, upper_p),
            standard_error=results['BCa'].standard_error,
            bias=results['BCa'].bias,
            iterations=self.iterations,
            method="Percentile",
            confidence_level=self.confidence_level,
            z0=0,
            a=0,
            bootstrap_distribution=bootstrap_stats,
            computation_time_ms=0
        )
        
        # Basic method
        point_est = results['BCa'].point_estimate
        lower_b = 2 * point_est - upper_p
        upper_b = 2 * point_est - lower_p
        
        results['Basic'] = BootstrapResult(
            point_estimate=point_est,
            confidence_interval=(lower_b, upper_b),
            standard_error=results['BCa'].standard_error,
            bias=results['BCa'].bias,
            iterations=self.iterations,
            method="Basic",
            confidence_level=self.confidence_level,
            z0=0,
            a=0,
            bootstrap_distribution=bootstrap_stats,
            computation_time_ms=0
        )
        
        return results


def validate_53_percent_finding(data: np.ndarray,
                               protected_attr: np.ndarray,
                               outcome: np.ndarray) -> Dict[str, Any]:
    """
    Validate the 53% temporal bias finding with proper statistical power
    
    Args:
        data: Full dataset
        protected_attr: Protected attribute array
        outcome: Outcome array (0/1)
    
    Returns:
        Dictionary with validation results
    """
    
    bootstrap = BCaBootstrap(iterations=10000, confidence_level=0.95)
    
    # Calculate disparity for protected group
    def disparity_statistic(indices):
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            # Handle boolean mask
            subset_protected = protected_attr[indices]
            subset_outcome = outcome[indices]
        else:
            # Handle data directly
            subset_protected = protected_attr if len(indices) == len(protected_attr) else protected_attr[indices]
            subset_outcome = outcome if len(indices) == len(outcome) else outcome[indices]
        
        protected_positive = np.mean(subset_outcome[subset_protected == 1])
        non_protected_positive = np.mean(subset_outcome[subset_protected == 0])
        
        return protected_positive / non_protected_positive if non_protected_positive > 0 else 0
    
    # Run bootstrap analysis
    indices = np.arange(len(data))
    result = bootstrap.calculate(indices, disparity_statistic, stratify_by=protected_attr)
    
    # Check if 0.53 is within confidence interval
    validates = result.confidence_interval[0] <= 0.53 <= result.confidence_interval[1]
    
    # Calculate statistical power
    n = len(data)
    effect_size = 0.12  # Cohen's h from requirements
    power = calculate_power(n, effect_size, alpha=0.05)
    
    return {
        'validates': validates,
        'point_estimate': result.point_estimate,
        'confidence_interval': result.confidence_interval,
        'standard_error': result.standard_error,
        'bias': result.bias,
        'power': power,
        'sample_size': n,
        'computation_time_ms': result.computation_time_ms
    }


def calculate_power(n: int, effect_size: float, alpha: float = 0.05) -> float:
    """Calculate statistical power for given sample size and effect"""
    
    # Using approximation for binomial test power
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = (np.sqrt(n) * effect_size - z_alpha)
    power = stats.norm.cdf(z_beta)
    
    return power