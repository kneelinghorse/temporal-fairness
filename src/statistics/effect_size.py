"""
Effect Size Calculations for Fairness Analysis
Implements Cohen's h, disparate impact ratios, and practical significance measures
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import time


class EffectSizeInterpretation(Enum):
    """Interpretation categories for effect sizes in fairness context"""
    NEGLIGIBLE = "negligible"       # < 0.10
    CONCERNING = "concerning"        # 0.10 - 0.20
    ACTIONABLE = "actionable"       # 0.20 - 0.50
    CRITICAL = "critical"           # > 0.50


class DisparateImpactLevel(Enum):
    """Disparate impact severity levels"""
    SEVERE = "severe"               # < 0.50
    MODERATE = "moderate"           # 0.50 - 0.80
    ACCEPTABLE = "acceptable"       # > 0.80


@dataclass
class EffectSizeResult:
    """Container for effect size calculation results"""
    measure: str
    value: float
    interpretation: str
    confidence_interval: Optional[Tuple[float, float]] = None
    practical_significance: Optional[str] = None
    sample_size: Optional[int] = None
    computation_time_ms: float = 0.0


@dataclass
class FairnessEffectSizes:
    """Complete effect size analysis for fairness metrics"""
    cohens_h: EffectSizeResult
    disparate_impact: EffectSizeResult
    number_needed_to_harm: EffectSizeResult
    risk_difference: EffectSizeResult
    odds_ratio: EffectSizeResult
    meets_threshold: bool
    overall_interpretation: str


class EffectSizeCalculator:
    """
    Calculate and interpret effect sizes for fairness analysis
    
    Implements multiple effect size measures with interpretations
    specific to fairness and bias detection contexts.
    
    References:
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
    - Agresti, A. (2018). Statistical Methods for the Social Sciences
    - Pearl, J. (2019). The Seven Tools of Causal Inference
    """
    
    def __init__(self, fairness_threshold: float = 0.12):
        """
        Initialize effect size calculator
        
        Args:
            fairness_threshold: Threshold for concerning effect size (default 0.12 for Cohen's h)
        """
        self.fairness_threshold = fairness_threshold
    
    def calculate_cohens_h(self,
                          p1: float,
                          p2: float,
                          n1: Optional[int] = None,
                          n2: Optional[int] = None,
                          confidence_level: float = 0.95) -> EffectSizeResult:
        """
        Calculate Cohen's h for difference between two proportions
        
        Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        
        Args:
            p1: Proportion for group 1
            p2: Proportion for group 2
            n1: Sample size for group 1 (for CI calculation)
            n2: Sample size for group 2 (for CI calculation)
            confidence_level: Confidence level for interval
        
        Returns:
            EffectSizeResult with Cohen's h value and interpretation
        """
        start_time = time.perf_counter()
        
        # Ensure proportions are valid
        p1 = np.clip(p1, 0.001, 0.999)
        p2 = np.clip(p2, 0.001, 0.999)
        
        # Calculate Cohen's h
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        
        # Interpret effect size
        interpretation = self._interpret_cohens_h(h)
        
        # Calculate confidence interval if sample sizes provided
        ci = None
        if n1 is not None and n2 is not None:
            ci = self._cohens_h_confidence_interval(p1, p2, n1, n2, confidence_level)
        
        # Determine practical significance
        practical = self._practical_significance_cohens_h(h)
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        return EffectSizeResult(
            measure="Cohen's h",
            value=h,
            interpretation=interpretation.value,
            confidence_interval=ci,
            practical_significance=practical,
            sample_size=(n1 + n2) if n1 and n2 else None,
            computation_time_ms=computation_time
        )
    
    def calculate_disparate_impact(self,
                                  p_minority: float,
                                  p_majority: float,
                                  n_minority: Optional[int] = None,
                                  n_majority: Optional[int] = None) -> EffectSizeResult:
        """
        Calculate disparate impact ratio (four-fifths rule)
        
        DI = P(positive|minority) / P(positive|majority)
        
        Args:
            p_minority: Positive rate for minority group
            p_majority: Positive rate for majority group
            n_minority: Sample size for minority group
            n_majority: Sample size for majority group
        
        Returns:
            EffectSizeResult with disparate impact ratio
        """
        start_time = time.perf_counter()
        
        # Avoid division by zero
        if p_majority == 0:
            di_ratio = 0.0 if p_minority == 0 else float('inf')
        else:
            di_ratio = p_minority / p_majority
        
        # Interpret disparate impact
        if di_ratio < 0.50:
            interpretation = DisparateImpactLevel.SEVERE
        elif di_ratio < 0.80:
            interpretation = DisparateImpactLevel.MODERATE
        else:
            interpretation = DisparateImpactLevel.ACCEPTABLE
        
        # Calculate confidence interval if sample sizes provided
        ci = None
        if n_minority is not None and n_majority is not None:
            ci = self._disparate_impact_confidence_interval(
                p_minority, p_majority, n_minority, n_majority
            )
        
        # Practical significance
        meets_four_fifths = di_ratio >= 0.80
        practical = f"{'Meets' if meets_four_fifths else 'Fails'} four-fifths rule"
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        return EffectSizeResult(
            measure="Disparate Impact Ratio",
            value=di_ratio,
            interpretation=interpretation.value,
            confidence_interval=ci,
            practical_significance=practical,
            sample_size=(n_minority + n_majority) if n_minority and n_majority else None,
            computation_time_ms=computation_time
        )
    
    def calculate_number_needed_to_harm(self,
                                       p_harm_a: float,
                                       p_harm_b: float,
                                       n_a: Optional[int] = None,
                                       n_b: Optional[int] = None) -> EffectSizeResult:
        """
        Calculate Number Needed to Harm (NNH)
        
        NNH = 1 / |P(harm|A) - P(harm|B)|
        
        Represents the number of decisions needed to cause one additional harm
        
        Args:
            p_harm_a: Probability of harm for group A
            p_harm_b: Probability of harm for group B
            n_a: Sample size for group A
            n_b: Sample size for group B
        
        Returns:
            EffectSizeResult with NNH value
        """
        start_time = time.perf_counter()
        
        # Calculate absolute risk difference
        risk_diff = abs(p_harm_a - p_harm_b)
        
        # Calculate NNH
        if risk_diff == 0:
            nnh = float('inf')
            interpretation = "No differential harm"
        else:
            nnh = 1.0 / risk_diff
            
            # Interpret NNH
            if nnh < 10:
                interpretation = "Very high differential harm"
            elif nnh < 50:
                interpretation = "High differential harm"
            elif nnh < 100:
                interpretation = "Moderate differential harm"
            else:
                interpretation = "Low differential harm"
        
        # Calculate confidence interval if sample sizes provided
        ci = None
        if n_a is not None and n_b is not None and risk_diff > 0:
            # Use Wilson score interval for risk difference
            se = np.sqrt(p_harm_a * (1 - p_harm_a) / n_a + p_harm_b * (1 - p_harm_b) / n_b)
            z = 1.96  # 95% confidence
            ci_lower = max(0.001, risk_diff - z * se)
            ci_upper = min(1.0, risk_diff + z * se)
            ci = (1.0 / ci_upper, 1.0 / ci_lower) if ci_upper > 0 else (nnh, float('inf'))
        
        # Practical significance
        practical = f"{int(nnh)} decisions per additional harm" if nnh != float('inf') else "No harm difference"
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        return EffectSizeResult(
            measure="Number Needed to Harm",
            value=nnh,
            interpretation=interpretation,
            confidence_interval=ci,
            practical_significance=practical,
            sample_size=(n_a + n_b) if n_a and n_b else None,
            computation_time_ms=computation_time
        )
    
    def calculate_all_effect_sizes(self,
                                  group_a_positive: int,
                                  group_a_total: int,
                                  group_b_positive: int,
                                  group_b_total: int,
                                  group_a_name: str = "Protected",
                                  group_b_name: str = "Non-protected") -> FairnessEffectSizes:
        """
        Calculate comprehensive effect sizes for fairness analysis
        
        Args:
            group_a_positive: Number of positive outcomes in group A
            group_a_total: Total size of group A
            group_b_positive: Number of positive outcomes in group B
            group_b_total: Total size of group B
            group_a_name: Name of group A
            group_b_name: Name of group B
        
        Returns:
            FairnessEffectSizes with all measures
        """
        start_time = time.perf_counter()
        
        # Calculate proportions
        p_a = group_a_positive / group_a_total if group_a_total > 0 else 0
        p_b = group_b_positive / group_b_total if group_b_total > 0 else 0
        
        # Cohen's h
        cohens_h = self.calculate_cohens_h(p_a, p_b, group_a_total, group_b_total)
        
        # Disparate impact
        disparate_impact = self.calculate_disparate_impact(p_a, p_b, group_a_total, group_b_total)
        
        # Number needed to harm (assuming lower rate is harm)
        p_harm_a = 1 - p_a  # Not getting positive outcome is "harm"
        p_harm_b = 1 - p_b
        nnh = self.calculate_number_needed_to_harm(p_harm_a, p_harm_b, group_a_total, group_b_total)
        
        # Risk difference
        risk_diff = EffectSizeResult(
            measure="Risk Difference",
            value=p_a - p_b,
            interpretation=self._interpret_risk_difference(p_a - p_b),
            practical_significance=f"{abs(p_a - p_b)*100:.1f}% absolute difference",
            computation_time_ms=0
        )
        
        # Odds ratio
        if p_a > 0 and p_b > 0 and p_a < 1 and p_b < 1:
            odds_a = p_a / (1 - p_a)
            odds_b = p_b / (1 - p_b)
            odds_ratio_value = odds_a / odds_b if odds_b > 0 else float('inf')
        else:
            odds_ratio_value = float('inf')
        
        odds_ratio = EffectSizeResult(
            measure="Odds Ratio",
            value=odds_ratio_value,
            interpretation=self._interpret_odds_ratio(odds_ratio_value),
            practical_significance=f"{odds_ratio_value:.2f}x odds" if odds_ratio_value != float('inf') else "Undefined",
            computation_time_ms=0
        )
        
        # Determine if meets threshold
        meets_threshold = abs(cohens_h.value) < self.fairness_threshold
        
        # Overall interpretation
        if abs(cohens_h.value) < 0.10:
            overall = "Negligible bias - No action required"
        elif abs(cohens_h.value) < 0.20:
            overall = "Small but concerning bias - Monitor closely"
        elif abs(cohens_h.value) < 0.50:
            overall = "Actionable bias - Mitigation recommended"
        else:
            overall = "Critical bias - Immediate action required"
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return FairnessEffectSizes(
            cohens_h=cohens_h,
            disparate_impact=disparate_impact,
            number_needed_to_harm=nnh,
            risk_difference=risk_diff,
            odds_ratio=odds_ratio,
            meets_threshold=meets_threshold,
            overall_interpretation=overall
        )
    
    def _interpret_cohens_h(self, h: float) -> EffectSizeInterpretation:
        """Interpret Cohen's h value in fairness context"""
        abs_h = abs(h)
        
        if abs_h < 0.10:
            return EffectSizeInterpretation.NEGLIGIBLE
        elif abs_h < 0.20:
            return EffectSizeInterpretation.CONCERNING
        elif abs_h < 0.50:
            return EffectSizeInterpretation.ACTIONABLE
        else:
            return EffectSizeInterpretation.CRITICAL
    
    def _practical_significance_cohens_h(self, h: float) -> str:
        """Determine practical significance of Cohen's h"""
        abs_h = abs(h)
        
        if abs_h < 0.10:
            return "Below fairness concern threshold"
        elif abs_h == 0.12:
            return "At 53% finding threshold (small but meaningful)"
        elif abs_h < 0.20:
            return "Exceeds fairness threshold - review needed"
        elif abs_h < 0.50:
            return "Substantial unfairness - action required"
        else:
            return "Severe unfairness - urgent intervention needed"
    
    def _cohens_h_confidence_interval(self,
                                     p1: float,
                                     p2: float,
                                     n1: int,
                                     n2: int,
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's h using delta method"""
        
        # Standard error using delta method
        var_p1 = p1 * (1 - p1) / n1
        var_p2 = p2 * (1 - p2) / n2
        
        # Derivative of arcsin transformation
        if p1 > 0 and p1 < 1:
            d_p1 = 1 / np.sqrt(p1 * (1 - p1))
        else:
            d_p1 = 1
        
        if p2 > 0 and p2 < 1:
            d_p2 = 1 / np.sqrt(p2 * (1 - p2))
        else:
            d_p2 = 1
        
        # Standard error of h
        se_h = np.sqrt(var_p1 * d_p1**2 + var_p2 * d_p2**2)
        
        # Critical value
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate interval
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        lower = h - z * se_h
        upper = h + z * se_h
        
        return (lower, upper)
    
    def _disparate_impact_confidence_interval(self,
                                            p_minority: float,
                                            p_majority: float,
                                            n_minority: int,
                                            n_majority: int) -> Tuple[float, float]:
        """Calculate confidence interval for disparate impact ratio"""
        
        # Use log transformation for ratio
        if p_minority > 0 and p_majority > 0:
            log_ratio = np.log(p_minority / p_majority)
            
            # Standard error of log ratio
            se_log = np.sqrt((1 - p_minority) / (n_minority * p_minority) + 
                            (1 - p_majority) / (n_majority * p_majority))
            
            # 95% CI on log scale
            z = 1.96
            log_lower = log_ratio - z * se_log
            log_upper = log_ratio + z * se_log
            
            # Transform back
            return (np.exp(log_lower), np.exp(log_upper))
        else:
            return (0, float('inf'))
    
    def _interpret_risk_difference(self, rd: float) -> str:
        """Interpret risk difference"""
        abs_rd = abs(rd)
        
        if abs_rd < 0.05:
            return "Minimal difference"
        elif abs_rd < 0.10:
            return "Small difference"
        elif abs_rd < 0.20:
            return "Moderate difference"
        else:
            return "Large difference"
    
    def _interpret_odds_ratio(self, or_value: float) -> str:
        """Interpret odds ratio"""
        if or_value == float('inf'):
            return "Undefined (extreme disparity)"
        elif or_value < 0.5 or or_value > 2.0:
            return "Strong association"
        elif or_value < 0.67 or or_value > 1.5:
            return "Moderate association"
        else:
            return "Weak association"


def validate_53_percent_effect_size(p_protected: float = 0.53,
                                   p_non_protected: float = 0.50) -> Dict[str, Any]:
    """
    Validate that the 53% finding corresponds to meaningful effect size
    
    Args:
        p_protected: Proportion for protected group (default 0.53)
        p_non_protected: Proportion for non-protected group (default 0.50)
    
    Returns:
        Dictionary with validation results
    """
    
    calculator = EffectSizeCalculator(fairness_threshold=0.12)
    
    # Calculate Cohen's h for 53% vs 50%
    result = calculator.calculate_cohens_h(p_protected, p_non_protected)
    
    # Validate against threshold
    validation = {
        'cohens_h': result.value,
        'expected_h': 0.12,
        'difference': abs(result.value - 0.12),
        'matches_threshold': abs(result.value - 0.12) < 0.01,
        'interpretation': result.interpretation,
        'practical_significance': result.practical_significance,
        'is_meaningful': result.interpretation != EffectSizeInterpretation.NEGLIGIBLE.value
    }
    
    return validation


def calculate_sample_size_for_effect(effect_size: float = 0.12,
                                    alpha: float = 0.05,
                                    power: float = 0.80) -> int:
    """
    Calculate required sample size for detecting given effect size
    
    Args:
        effect_size: Cohen's h effect size to detect
        alpha: Significance level
        power: Statistical power
    
    Returns:
        Required sample size per group
    """
    from scipy import stats
    
    # Z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)
    
    # Sample size formula for two proportions
    n = ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
    
    return int(np.ceil(n))