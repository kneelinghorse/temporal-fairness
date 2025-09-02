"""
Enhanced bias detector incorporating research-based improvements.

Based on comprehensive research analysis, this module extends the base
BiasDetector with additional pattern detection capabilities and
production-ready optimizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from collections import deque
import warnings

from src.analysis.bias_detector import BiasDetector


class EnhancedBiasDetector(BiasDetector):
    """
    Enhanced bias detector with research-based improvements.
    
    Incorporates findings from temporal fairness research including:
    - Inspection paradox detection
    - Proxy variable discrimination analysis
    - Population Stability Index (PSI) for early warning
    - Quantile Demographic Drift (QDD) for label-free monitoring
    - Time-of-day bias patterns
    """
    
    def __init__(
        self,
        sensitivity: float = 0.95,
        min_pattern_length: int = 3,
        anomaly_threshold: float = 3.0,
        psi_threshold: float = 0.1,
        qdd_threshold: float = 0.15
    ):
        """
        Initialize enhanced bias detector.
        
        Args:
            sensitivity: Detection sensitivity
            min_pattern_length: Minimum time points for patterns
            anomaly_threshold: Z-score threshold for anomalies
            psi_threshold: Population Stability Index threshold
            qdd_threshold: Quantile Demographic Drift threshold
        """
        super().__init__(sensitivity, min_pattern_length, anomaly_threshold)
        self.psi_threshold = psi_threshold
        self.qdd_threshold = qdd_threshold
        
        # Circular buffer for O(1) window operations
        self.window_buffer = deque(maxlen=1000)
        
        # Historical baselines for drift detection
        self.baseline_distributions = {}
        
    def detect_inspection_paradox(
        self,
        scheduled_intervals: np.ndarray,
        actual_waits: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect inspection paradox where average experienced wait exceeds scheduled.
        
        Research shows bus riders experience ~20min wait for 10min scheduled buses.
        This pattern indicates systematic disadvantage for inflexible schedule users.
        
        Args:
            scheduled_intervals: Expected/scheduled wait times
            actual_waits: Actual experienced wait times
            groups: Optional group assignments
            
        Returns:
            Dictionary with inspection paradox detection results
        """
        # Calculate size-biased sampling effect
        mean_scheduled = np.mean(scheduled_intervals)
        mean_actual = np.mean(actual_waits)
        
        # Theoretical inspection paradox factor
        if mean_scheduled > 0:
            theoretical_factor = 1 + (np.var(scheduled_intervals) / (mean_scheduled ** 2))
            expected_actual = mean_scheduled * theoretical_factor
            
            # Detect significant paradox
            paradox_detected = mean_actual > expected_actual * 0.9  # 90% threshold
            
            result = {
                'detected': paradox_detected,
                'scheduled_mean': mean_scheduled,
                'actual_mean': mean_actual,
                'theoretical_expected': expected_actual,
                'paradox_factor': mean_actual / mean_scheduled if mean_scheduled > 0 else np.inf,
                'description': 'Inspection paradox detected - actual waits exceed theoretical expectation'
            }
            
            # Group-specific analysis if provided
            if groups is not None:
                group_paradox = {}
                for group in np.unique(groups):
                    mask = groups == group
                    if np.sum(mask) > self.min_samples:
                        group_actual = np.mean(actual_waits[mask])
                        group_factor = group_actual / mean_scheduled if mean_scheduled > 0 else np.inf
                        group_paradox[group] = {
                            'mean_wait': group_actual,
                            'paradox_factor': group_factor,
                            'excess_wait': group_actual - mean_scheduled
                        }
                
                result['group_analysis'] = group_paradox
                
                # Identify most affected group
                if group_paradox:
                    worst_group = max(group_paradox.items(), 
                                    key=lambda x: x[1]['paradox_factor'])
                    result['most_affected_group'] = worst_group[0]
                    result['max_paradox_factor'] = worst_group[1]['paradox_factor']
        else:
            result = {'detected': False, 'reason': 'invalid_scheduled_intervals'}
        
        return result
    
    def detect_proxy_discrimination(
        self,
        features: pd.DataFrame,
        outcomes: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        proxy_candidates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect proxy variable discrimination patterns.
        
        Research shows ZIP codes function as race proxies in 65% of redlined areas.
        This method identifies features acting as proxies for protected attributes.
        
        Args:
            features: Feature DataFrame
            outcomes: Decision outcomes
            protected_attributes: Dict of protected attribute arrays
            proxy_candidates: Optional list of suspected proxy columns
            
        Returns:
            Dictionary with proxy discrimination analysis
        """
        if proxy_candidates is None:
            # Common proxy patterns from research
            proxy_candidates = [
                col for col in features.columns
                if any(pattern in col.lower() for pattern in 
                      ['zip', 'postal', 'address', 'neighborhood', 'school',
                       'name', 'language', 'device', 'network', 'browser'])
            ]
        
        proxy_results = {}
        
        for proxy_col in proxy_candidates:
            if proxy_col not in features.columns:
                continue
            
            proxy_values = features[proxy_col].values
            
            # Check correlation with protected attributes
            for attr_name, attr_values in protected_attributes.items():
                # Mutual information for non-linear relationships
                try:
                    from sklearn.feature_selection import mutual_info_classif
                    mi_score = mutual_info_classif(
                        proxy_values.reshape(-1, 1),
                        attr_values,
                        random_state=42
                    )[0]
                    
                    # Cramér's V for categorical association
                    if len(np.unique(proxy_values)) < 100:  # Likely categorical
                        contingency = pd.crosstab(proxy_values, attr_values)
                        chi2 = stats.chi2_contingency(contingency)[0]
                        n = len(proxy_values)
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                    else:
                        cramers_v = 0
                    
                    # Check if proxy affects outcomes differently by group
                    outcome_disparity = self._calculate_proxy_outcome_disparity(
                        proxy_values, outcomes, attr_values
                    )
                    
                    proxy_results[f"{proxy_col}_{attr_name}"] = {
                        'mutual_information': mi_score,
                        'cramers_v': cramers_v,
                        'outcome_disparity': outcome_disparity,
                        'is_likely_proxy': mi_score > 0.1 or cramers_v > 0.3,
                        'proxy_strength': max(mi_score, cramers_v)
                    }
                    
                except Exception as e:
                    proxy_results[f"{proxy_col}_{attr_name}"] = {
                        'error': str(e),
                        'is_likely_proxy': False
                    }
        
        # Identify strongest proxies
        likely_proxies = [
            k for k, v in proxy_results.items()
            if v.get('is_likely_proxy', False)
        ]
        
        return {
            'detected': len(likely_proxies) > 0,
            'proxy_analysis': proxy_results,
            'likely_proxies': likely_proxies,
            'n_proxies_found': len(likely_proxies),
            'recommendation': 'Remove or adjust for proxy variables' if likely_proxies else 'No clear proxies detected'
        }
    
    def calculate_population_stability_index(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index for distribution shift detection.
        
        Research shows PSI can predict bias 6-12 months in advance.
        PSI > 0.1 indicates significant shift requiring intervention.
        
        Args:
            reference_data: Baseline/reference distribution
            current_data: Current distribution to compare
            n_bins: Number of bins for discretization
            
        Returns:
            PSI value and interpretation
        """
        # Create bins from reference data
        _, bin_edges = np.histogram(reference_data, bins=n_bins)
        
        # Calculate frequencies
        ref_freq, _ = np.histogram(reference_data, bins=bin_edges)
        curr_freq, _ = np.histogram(current_data, bins=bin_edges)
        
        # Normalize to probabilities
        ref_prob = (ref_freq + 1) / (len(reference_data) + n_bins)  # Laplace smoothing
        curr_prob = (curr_freq + 1) / (len(current_data) + n_bins)
        
        # Calculate PSI
        psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
        
        # Interpret PSI
        if psi < 0.1:
            stability = 'stable'
            risk = 'low'
        elif psi < 0.2:
            stability = 'moderate_shift'
            risk = 'medium'
        else:
            stability = 'significant_shift'
            risk = 'high'
        
        # Predict future bias risk (based on research findings)
        predicted_bias_timeline = None
        if psi > 0.15:
            predicted_bias_timeline = '3-6 months'
        elif psi > 0.1:
            predicted_bias_timeline = '6-12 months'
        
        return {
            'psi': psi,
            'stability': stability,
            'risk_level': risk,
            'predicted_bias_timeline': predicted_bias_timeline,
            'requires_intervention': psi > self.psi_threshold,
            'bin_details': {
                'reference_distribution': ref_prob.tolist(),
                'current_distribution': curr_prob.tolist(),
                'max_shift': np.max(np.abs(curr_prob - ref_prob))
            }
        }
    
    def calculate_quantile_demographic_drift(
        self,
        scores: np.ndarray,
        groups: np.ndarray,
        reference_scores: Optional[np.ndarray] = None,
        reference_groups: Optional[np.ndarray] = None,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ) -> Dict[str, Any]:
        """
        Calculate Quantile Demographic Drift for label-free monitoring.
        
        From FairCanary research - enables real-time monitoring without outcomes.
        Critical for production systems with delayed ground truth.
        
        Args:
            scores: Current prediction scores
            groups: Current group assignments
            reference_scores: Baseline scores (optional)
            reference_groups: Baseline groups (optional)
            quantiles: Quantiles to evaluate
            
        Returns:
            QDD analysis with drift detection
        """
        unique_groups = np.unique(groups)
        current_quantiles = {}
        
        # Calculate current quantiles per group
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > self.min_samples:
                group_scores = scores[mask]
                current_quantiles[group] = np.quantile(group_scores, quantiles)
        
        # If no reference, use first group as baseline
        if reference_scores is None or reference_groups is None:
            reference_quantiles = current_quantiles[unique_groups[0]]
            qdd_scores = {}
            
            for group in unique_groups[1:]:
                if group in current_quantiles:
                    # Calculate drift at each quantile
                    drifts = np.abs(current_quantiles[group] - reference_quantiles)
                    qdd_scores[group] = {
                        'max_drift': np.max(drifts),
                        'mean_drift': np.mean(drifts),
                        'quantile_drifts': dict(zip(quantiles, drifts))
                    }
        else:
            # Compare to historical reference
            reference_quantiles = {}
            for group in np.unique(reference_groups):
                mask = reference_groups == group
                if np.sum(mask) > self.min_samples:
                    group_scores = reference_scores[mask]
                    reference_quantiles[group] = np.quantile(group_scores, quantiles)
            
            qdd_scores = {}
            for group in unique_groups:
                if group in current_quantiles and group in reference_quantiles:
                    drifts = np.abs(current_quantiles[group] - reference_quantiles[group])
                    qdd_scores[group] = {
                        'max_drift': np.max(drifts),
                        'mean_drift': np.mean(drifts),
                        'quantile_drifts': dict(zip(quantiles, drifts))
                    }
        
        # Detect significant drift
        max_drift = max(s['max_drift'] for s in qdd_scores.values()) if qdd_scores else 0
        drift_detected = max_drift > self.qdd_threshold
        
        return {
            'drift_detected': drift_detected,
            'max_drift': max_drift,
            'group_drifts': qdd_scores,
            'current_quantiles': current_quantiles,
            'threshold': self.qdd_threshold,
            'recommendation': 'Investigate score distribution changes' if drift_detected else 'Stable'
        }
    
    def detect_time_of_day_bias(
        self,
        decisions: np.ndarray,
        timestamps: np.ndarray,
        groups: np.ndarray,
        time_bins: int = 4  # Morning, afternoon, evening, night
    ) -> Dict[str, Any]:
        """
        Detect time-of-day bias patterns.
        
        Research shows 40% difference in pain medication during night shifts.
        This method identifies temporal discrimination patterns.
        
        Args:
            decisions: Decision outcomes
            timestamps: Timestamps (datetime or numeric)
            groups: Group assignments
            time_bins: Number of time periods to analyze
            
        Returns:
            Time-of-day bias analysis
        """
        # Extract hour of day if timestamps are datetime
        if isinstance(timestamps[0], (pd.Timestamp, datetime)):
            hours = np.array([t.hour for t in timestamps])
        else:
            # Assume numeric timestamps, extract modulo 24
            hours = timestamps % 24
        
        # Create time bins
        time_labels = ['Night', 'Morning', 'Afternoon', 'Evening'][:time_bins]
        time_periods = np.digitize(hours, np.linspace(0, 24, time_bins + 1)) - 1
        
        # Analyze bias by time period
        time_bias_results = {}
        unique_groups = np.unique(groups)
        
        for period in range(time_bins):
            period_mask = time_periods == period
            period_decisions = decisions[period_mask]
            period_groups = groups[period_mask]
            
            if len(period_decisions) < self.min_samples:
                continue
            
            # Calculate disparities for this time period
            group_rates = {}
            for group in unique_groups:
                group_mask = period_groups == group
                if np.sum(group_mask) > 0:
                    group_rates[group] = np.mean(period_decisions[group_mask])
            
            if len(group_rates) > 1:
                max_rate = max(group_rates.values())
                min_rate = min(group_rates.values())
                disparity = max_rate - min_rate
                
                time_bias_results[time_labels[period] if period < len(time_labels) else f'Period_{period}'] = {
                    'disparity': disparity,
                    'group_rates': group_rates,
                    'n_samples': len(period_decisions),
                    'worst_ratio': max_rate / min_rate if min_rate > 0 else np.inf
                }
        
        # Identify worst time period
        if time_bias_results:
            worst_period = max(time_bias_results.items(), 
                             key=lambda x: x[1]['disparity'])
            
            # Check if night shift has higher bias (research finding)
            night_bias = time_bias_results.get('Night', {}).get('disparity', 0)
            day_bias = np.mean([v['disparity'] for k, v in time_bias_results.items() 
                               if k != 'Night'])
            
            night_shift_bias = night_bias > day_bias * 1.2  # 20% higher
            
            return {
                'detected': worst_period[1]['disparity'] > 0.1,
                'time_periods': time_bias_results,
                'worst_period': worst_period[0],
                'worst_disparity': worst_period[1]['disparity'],
                'night_shift_bias': night_shift_bias,
                'recommendation': f'Increase supervision during {worst_period[0]}' if worst_period[1]['disparity'] > 0.1 else 'No significant time-based bias'
            }
        
        return {'detected': False, 'reason': 'insufficient_data'}
    
    def _calculate_proxy_outcome_disparity(
        self,
        proxy_values: np.ndarray,
        outcomes: np.ndarray,
        protected_attr: np.ndarray
    ) -> float:
        """Calculate outcome disparity when using proxy variable."""
        unique_attrs = np.unique(protected_attr)
        if len(unique_attrs) != 2:
            return 0
        
        # Calculate outcome rates for each protected group at same proxy value
        proxy_bins = np.percentile(proxy_values, [0, 25, 50, 75, 100])
        max_disparity = 0
        
        for i in range(len(proxy_bins) - 1):
            bin_mask = (proxy_values >= proxy_bins[i]) & (proxy_values < proxy_bins[i + 1])
            
            if np.sum(bin_mask) < self.min_samples:
                continue
            
            bin_outcomes = outcomes[bin_mask]
            bin_attrs = protected_attr[bin_mask]
            
            # Calculate outcome rate for each group
            rates = []
            for attr in unique_attrs:
                attr_mask = bin_attrs == attr
                if np.sum(attr_mask) > 0:
                    rates.append(np.mean(bin_outcomes[attr_mask]))
            
            if len(rates) == 2:
                disparity = abs(rates[0] - rates[1])
                max_disparity = max(max_disparity, disparity)
        
        return max_disparity
    
    def generate_urgency_score(
        self,
        wait_times: np.ndarray,
        priorities: Optional[np.ndarray] = None,
        time_factor_exp: float = 2.0
    ) -> np.ndarray:
        """
        Generate urgency scores using research-based formula.
        
        From production SaaS research: urgency = 1/(time_factor²)
        
        Args:
            wait_times: Current wait times
            priorities: Optional priority scores
            time_factor_exp: Exponential factor (default 2.0 from research)
            
        Returns:
            Urgency scores
        """
        # Normalize wait times to [0, 1]
        if len(wait_times) > 0:
            normalized_wait = (wait_times - np.min(wait_times)) / (np.max(wait_times) - np.min(wait_times) + 1e-10)
        else:
            normalized_wait = wait_times
        
        # Apply exponential urgency curve
        urgency = 1 / (normalized_wait + 1e-10) ** time_factor_exp
        
        # Incorporate priorities if provided
        if priorities is not None:
            urgency = urgency * (1 + priorities)
        
        # Normalize to [0, 1]
        urgency = (urgency - np.min(urgency)) / (np.max(urgency) - np.min(urgency) + 1e-10)
        
        return urgency