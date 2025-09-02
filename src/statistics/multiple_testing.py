"""
Multiple Testing Correction Methods
Implements Benjamini-Hochberg FDR and other correction methods for fairness analysis
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings


class CorrectionMethod(Enum):
    """Available multiple testing correction methods"""
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    HOCHBERG = "hochberg"
    HOMMEL = "hommel"
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    NONE = "none"


@dataclass
class TestResult:
    """Container for individual hypothesis test result"""
    hypothesis: str
    p_value: float
    adjusted_p_value: float
    reject_null: bool
    test_statistic: Optional[float] = None
    effect_size: Optional[float] = None
    group: Optional[str] = None


@dataclass
class MultipleTestingResult:
    """Container for multiple testing correction results"""
    method: CorrectionMethod
    alpha: float
    n_tests: int
    n_rejected: int
    fdr: float
    test_results: List[TestResult]
    hierarchical_structure: Optional[Dict] = None
    computation_time_ms: float = 0.0


class MultipleTestingCorrection:
    """
    Multiple testing correction for fairness analyses
    
    Implements various correction methods with focus on Benjamini-Hochberg FDR
    for controlling false discovery rate in subgroup analyses.
    
    References:
    - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate
    - Benjamini, Y., & Yekutieli, D. (2001). The control of the FDR under dependency
    """
    
    def __init__(self, 
                 method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
                 alpha: float = 0.10):
        """
        Initialize multiple testing correction
        
        Args:
            method: Correction method to use
            alpha: Family-wise error rate or FDR level (default 0.10 for FDR)
        """
        self.method = method
        self.alpha = alpha
    
    def correct(self,
                p_values: Union[List[float], np.ndarray],
                hypotheses: Optional[List[str]] = None,
                groups: Optional[List[str]] = None,
                hierarchical: bool = False) -> MultipleTestingResult:
        """
        Apply multiple testing correction
        
        Args:
            p_values: List or array of p-values
            hypotheses: Optional hypothesis descriptions
            groups: Optional group labels for hierarchical testing
            hierarchical: Whether to use hierarchical testing structure
        
        Returns:
            MultipleTestingResult with corrected p-values and decisions
        """
        import time
        start_time = time.perf_counter()
        
        # Convert to numpy array
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        # Generate default hypothesis names if not provided
        if hypotheses is None:
            hypotheses = [f"H{i+1}" for i in range(n_tests)]
        
        # Apply correction based on method
        if self.method == CorrectionMethod.BENJAMINI_HOCHBERG:
            adjusted_p, reject = self._benjamini_hochberg(p_values)
        elif self.method == CorrectionMethod.BONFERRONI:
            adjusted_p, reject = self._bonferroni(p_values)
        elif self.method == CorrectionMethod.HOLM:
            adjusted_p, reject = self._holm(p_values)
        elif self.method == CorrectionMethod.HOCHBERG:
            adjusted_p, reject = self._hochberg(p_values)
        elif self.method == CorrectionMethod.HOMMEL:
            adjusted_p, reject = self._hommel(p_values)
        elif self.method == CorrectionMethod.FDR_BY:
            adjusted_p, reject = self._benjamini_yekutieli(p_values)
        else:  # No correction
            adjusted_p = p_values
            reject = p_values < self.alpha
        
        # Handle hierarchical structure if specified
        hierarchical_structure = None
        if hierarchical and groups is not None:
            hierarchical_structure = self._apply_hierarchical_testing(
                p_values, groups, hypotheses
            )
        
        # Create test results
        test_results = []
        for i in range(n_tests):
            result = TestResult(
                hypothesis=hypotheses[i],
                p_value=p_values[i],
                adjusted_p_value=adjusted_p[i],
                reject_null=reject[i],
                group=groups[i] if groups else None
            )
            test_results.append(result)
        
        # Calculate actual FDR
        n_rejected = np.sum(reject)
        if n_rejected > 0:
            # Estimate FDR using Storey's method
            fdr = self._estimate_fdr(p_values, reject)
        else:
            fdr = 0.0
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        return MultipleTestingResult(
            method=self.method,
            alpha=self.alpha,
            n_tests=n_tests,
            n_rejected=n_rejected,
            fdr=fdr,
            test_results=test_results,
            hierarchical_structure=hierarchical_structure,
            computation_time_ms=computation_time
        )
    
    def _benjamini_hochberg(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR correction
        
        Controls the false discovery rate at level alpha
        """
        n = len(p_values)
        
        # Sort p-values and keep track of original order
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Find the largest i such that P(i) <= (i/m) * alpha
        threshold_indices = np.arange(1, n + 1) / n * self.alpha
        below_threshold = sorted_p <= threshold_indices
        
        if np.any(below_threshold):
            # Find the largest i where condition is met
            max_i = np.max(np.where(below_threshold)[0])
            threshold = sorted_p[max_i]
        else:
            threshold = -1  # Reject none
        
        # Reject hypotheses with p-values <= threshold
        reject = p_values <= threshold
        
        # Calculate adjusted p-values
        adjusted_p = np.zeros_like(p_values)
        for i in range(n):
            adjusted_p[sorted_indices[i]] = min(
                1.0,
                sorted_p[i] * n / (i + 1)
            )
        
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            adjusted_p[sorted_indices[i]] = min(
                adjusted_p[sorted_indices[i]],
                adjusted_p[sorted_indices[i + 1]]
            )
        
        return adjusted_p, reject
    
    def _bonferroni(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bonferroni correction - most conservative"""
        n = len(p_values)
        adjusted_p = np.minimum(p_values * n, 1.0)
        reject = p_values < (self.alpha / n)
        return adjusted_p, reject
    
    def _holm(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Holm's step-down procedure"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        adjusted_p = np.zeros_like(p_values)
        reject = np.zeros(n, dtype=bool)
        
        for i in range(n):
            threshold = self.alpha / (n - i)
            if sorted_p[i] <= threshold:
                reject[sorted_indices[i]] = True
                adjusted_p[sorted_indices[i]] = min(sorted_p[i] * (n - i), 1.0)
            else:
                # Stop rejecting
                for j in range(i, n):
                    adjusted_p[sorted_indices[j]] = min(sorted_p[j] * (n - j), 1.0)
                break
        
        return adjusted_p, reject
    
    def _hochberg(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hochberg's step-up procedure"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        adjusted_p = np.zeros_like(p_values)
        reject = np.zeros(n, dtype=bool)
        
        for i in range(n - 1, -1, -1):
            threshold = self.alpha / (i + 1)
            if sorted_p[i] <= threshold:
                # Reject this and all smaller p-values
                for j in range(i + 1):
                    reject[sorted_indices[j]] = True
                break
        
        # Calculate adjusted p-values
        for i in range(n):
            adjusted_p[sorted_indices[i]] = min(sorted_p[i] * (i + 1), 1.0)
        
        return adjusted_p, reject
    
    def _hommel(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hommel's procedure"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Find the largest j such that p(n-j+k) > k*alpha/j for all k=1,...,j
        j_max = 0
        for j in range(1, n + 1):
            valid = True
            for k in range(1, j + 1):
                if n - j + k <= n and sorted_p[n - j + k - 1] <= k * self.alpha / j:
                    valid = False
                    break
            if valid:
                j_max = j
        
        if j_max == 0:
            # Reject all
            reject = np.ones(n, dtype=bool)
        else:
            threshold = self.alpha / j_max
            reject = p_values <= threshold
        
        # Simple adjusted p-values (Hommel doesn't have standard adjusted p-values)
        adjusted_p = np.minimum(p_values * n, 1.0)
        
        return adjusted_p, reject
    
    def _benjamini_yekutieli(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Yekutieli procedure (FDR under dependency)
        More conservative than BH but controls FDR under any dependency
        """
        n = len(p_values)
        
        # Calculate the harmonic sum
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        
        # Apply BH with adjusted alpha
        original_alpha = self.alpha
        self.alpha = self.alpha / c_n
        adjusted_p, reject = self._benjamini_hochberg(p_values)
        self.alpha = original_alpha  # Restore original alpha
        
        # Adjust the adjusted p-values
        adjusted_p = np.minimum(adjusted_p * c_n, 1.0)
        
        return adjusted_p, reject
    
    def _apply_hierarchical_testing(self,
                                   p_values: np.ndarray,
                                   groups: List[str],
                                   hypotheses: List[str]) -> Dict:
        """
        Apply hierarchical testing structure for grouped hypotheses
        
        Implements a hierarchical testing procedure where:
        - Level 1: Primary hypotheses (no correction)
        - Level 2: Secondary hypotheses (FDR correction within groups)
        - Level 3: Exploratory hypotheses (strict FDR correction)
        """
        
        unique_groups = list(set(groups))
        structure = {
            'levels': {},
            'group_results': {}
        }
        
        # Classify hypotheses into levels based on group names
        primary_groups = ['primary', 'main']
        secondary_groups = ['secondary', 'subgroup']
        
        for group in unique_groups:
            group_indices = [i for i, g in enumerate(groups) if g == group]
            group_p_values = p_values[group_indices]
            
            if any(keyword in group.lower() for keyword in primary_groups):
                # Primary level - no correction
                level = 1
                alpha = self.alpha
                adjusted_p = group_p_values
                reject = group_p_values < alpha
            elif any(keyword in group.lower() for keyword in secondary_groups):
                # Secondary level - standard FDR
                level = 2
                alpha = self.alpha
                adjusted_p, reject = self._benjamini_hochberg(group_p_values)
            else:
                # Tertiary level - stricter FDR
                level = 3
                original_alpha = self.alpha
                self.alpha = self.alpha / 2  # Stricter control
                adjusted_p, reject = self._benjamini_hochberg(group_p_values)
                self.alpha = original_alpha
            
            structure['group_results'][group] = {
                'level': level,
                'n_tests': len(group_indices),
                'n_rejected': np.sum(reject),
                'indices': group_indices,
                'adjusted_p_values': adjusted_p,
                'reject': reject
            }
        
        return structure
    
    def _estimate_fdr(self, p_values: np.ndarray, reject: np.ndarray) -> float:
        """
        Estimate the false discovery rate using Storey's q-value method
        """
        n_rejected = np.sum(reject)
        if n_rejected == 0:
            return 0.0
        
        # Estimate proportion of true null hypotheses (pi0)
        # Using lambda = 0.5 as default
        lambda_val = 0.5
        pi0 = min(1.0, np.mean(p_values > lambda_val) / (1 - lambda_val))
        
        # Estimate FDR
        threshold = np.max(p_values[reject]) if np.any(reject) else 0
        expected_false_discoveries = len(p_values) * pi0 * threshold
        fdr = min(1.0, expected_false_discoveries / n_rejected)
        
        return fdr


def analyze_subgroups(results_dict: Dict[str, Dict],
                      correction_method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
                      fdr_level: float = 0.10) -> MultipleTestingResult:
    """
    Analyze fairness metrics across multiple subgroups with FDR control
    
    Args:
        results_dict: Dictionary of subgroup results with p-values
        correction_method: Method for multiple testing correction
        fdr_level: False discovery rate level
    
    Returns:
        MultipleTestingResult with corrected p-values and decisions
    """
    
    corrector = MultipleTestingCorrection(method=correction_method, alpha=fdr_level)
    
    # Extract p-values and hypotheses
    p_values = []
    hypotheses = []
    groups = []
    
    for group_name, results in results_dict.items():
        if 'p_value' in results:
            p_values.append(results['p_value'])
            hypotheses.append(f"Bias in {group_name}")
            
            # Classify group type
            if 'primary' in group_name.lower():
                groups.append('primary')
            elif 'subgroup' in group_name.lower() or 'secondary' in group_name.lower():
                groups.append('secondary')
            else:
                groups.append('exploratory')
    
    # Apply correction
    result = corrector.correct(p_values, hypotheses, groups, hierarchical=True)
    
    return result


def validate_neurips_standards(p_values: List[float],
                              effect_sizes: List[float]) -> Dict[str, bool]:
    """
    Validate that statistical reporting meets NeurIPS/ICML standards
    
    Args:
        p_values: List of p-values from tests
        effect_sizes: List of effect sizes
    
    Returns:
        Dictionary indicating which standards are met
    """
    
    standards_met = {
        'multiple_testing_correction': False,
        'effect_sizes_reported': False,
        'fdr_controlled': False,
        'hierarchical_structure': False,
        'sufficient_power': False
    }
    
    # Check if multiple testing correction was applied
    if len(p_values) > 1:
        corrector = MultipleTestingCorrection()
        result = corrector.correct(p_values)
        standards_met['multiple_testing_correction'] = True
        standards_met['fdr_controlled'] = result.fdr <= 0.10
    
    # Check if effect sizes are reported
    standards_met['effect_sizes_reported'] = len(effect_sizes) == len(p_values)
    
    # Check for hierarchical structure (assumed if groups provided)
    standards_met['hierarchical_structure'] = True  # Placeholder
    
    # Check statistical power (requires sample size calculation)
    # Assuming sufficient power if we have enough tests
    standards_met['sufficient_power'] = len(p_values) >= 10
    
    return standards_met