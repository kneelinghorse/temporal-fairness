"""
Temporal Demographic Parity (TDP) Metric Implementation

This module implements the Temporal Demographic Parity metric for detecting
bias across time windows in algorithmic decision-making systems.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings


class TemporalDemographicParity:
    """
    Temporal Demographic Parity (TDP) metric for fairness measurement.
    
    TDP measures the absolute difference in positive outcome rates between
    demographic groups within specific time windows.
    
    TDP(t) = |P(decision=1|group=A,t) - P(decision=1|group=B,t)|
    
    Attributes:
        threshold (float): Fairness threshold for TDP (default: 0.1)
        min_samples (int): Minimum samples per group per window (default: 30)
    """
    
    def __init__(self, threshold: float = 0.1, min_samples: int = 30):
        """
        Initialize TDP metric calculator.
        
        Args:
            threshold: Maximum acceptable TDP value for fairness
            min_samples: Minimum samples required per group per window
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self.results_cache = {}
        
    def calculate(
        self,
        decisions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None,
        window_size: Optional[int] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate TDP across time windows.
        
        Args:
            decisions: Binary decisions (0 or 1)
            groups: Group membership for each decision
            timestamps: Timestamp for each decision (optional)
            time_windows: List of (start, end) tuples defining windows
            window_size: Size of sliding windows (alternative to time_windows)
            return_details: If True, return detailed statistics
            
        Returns:
            If return_details=False: Maximum TDP value across all windows
            If return_details=True: Dictionary with detailed statistics
            
        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Input validation
        decisions, groups, timestamps = self._validate_inputs(
            decisions, groups, timestamps
        )
        
        # Generate time windows if not provided
        if time_windows is None:
            if timestamps is None:
                # Create single window for all data
                time_windows = [(0, len(decisions))]
            else:
                time_windows = self._generate_windows(
                    timestamps, window_size
                )
        
        # Calculate TDP for each window
        tdp_values = []
        window_stats = []
        
        for window_idx, (start, end) in enumerate(time_windows):
            window_tdp, stats = self._calculate_window_tdp(
                decisions, groups, timestamps, start, end
            )
            
            if window_tdp is not None:
                tdp_values.append(window_tdp)
                stats['window'] = (start, end)
                stats['window_idx'] = window_idx
                window_stats.append(stats)
        
        if not tdp_values:
            warnings.warn(
                f"No valid windows found with minimum {self.min_samples} samples per group"
            )
            return np.nan if not return_details else {
                'max_tdp': np.nan,
                'mean_tdp': np.nan,
                'windows': []
            }
        
        max_tdp = max(tdp_values)
        
        if return_details:
            return {
                'max_tdp': max_tdp,
                'mean_tdp': np.mean(tdp_values),
                'std_tdp': np.std(tdp_values),
                'min_tdp': min(tdp_values),
                'tdp_values': tdp_values,
                'windows': window_stats,
                'is_fair': max_tdp <= self.threshold,
                'threshold': self.threshold
            }
        
        return max_tdp
    
    def calculate_pairwise(
        self,
        decisions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate TDP for all pairs of groups.
        
        Args:
            decisions: Binary decisions
            groups: Group membership
            timestamps: Timestamps (optional)
            time_windows: Time windows for analysis
            
        Returns:
            Dictionary mapping group pairs to their maximum TDP values
        """
        decisions, groups, timestamps = self._validate_inputs(
            decisions, groups, timestamps
        )
        
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("At least 2 groups required for pairwise comparison")
        
        pairwise_tdp = {}
        
        for i, group_a in enumerate(unique_groups):
            for group_b in unique_groups[i+1:]:
                # Filter data for these two groups
                mask = np.isin(groups, [group_a, group_b])
                filtered_decisions = decisions[mask]
                filtered_groups = groups[mask]
                filtered_timestamps = timestamps[mask] if timestamps is not None else None
                
                # Calculate TDP for this pair
                tdp = self.calculate(
                    filtered_decisions,
                    filtered_groups,
                    filtered_timestamps,
                    time_windows
                )
                
                pairwise_tdp[(group_a, group_b)] = tdp
        
        return pairwise_tdp
    
    def _validate_inputs(
        self,
        decisions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate and convert inputs to numpy arrays."""
        # Convert to numpy arrays
        decisions = np.asarray(decisions)
        groups = np.asarray(groups)
        
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
        
        # Check lengths
        if len(decisions) != len(groups):
            raise ValueError("decisions and groups must have same length")
        
        if timestamps is not None and len(timestamps) != len(decisions):
            raise ValueError("timestamps must have same length as decisions")
        
        # Check binary decisions
        unique_decisions = np.unique(decisions[~np.isnan(decisions)])
        if not set(unique_decisions).issubset({0, 1}):
            raise ValueError("decisions must be binary (0 or 1)")
        
        # Check at least 2 groups
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("At least 2 groups required for TDP calculation")
        
        return decisions, groups, timestamps
    
    def _generate_windows(
        self,
        timestamps: np.ndarray,
        window_size: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Generate time windows for analysis."""
        if window_size is None:
            # Use quartiles as default windows
            quartiles = np.percentile(timestamps, [0, 25, 50, 75, 100])
            windows = []
            for i in range(len(quartiles) - 1):
                start, end = quartiles[i], quartiles[i + 1]
                windows.append((start, end))
            return windows
        
        # Generate sliding windows
        min_time, max_time = timestamps.min(), timestamps.max()
        time_range = max_time - min_time
        
        if window_size > time_range:
            return [(min_time, max_time)]
        
        windows = []
        current = min_time
        
        while current < max_time:
            window_end = min(current + window_size, max_time)
            windows.append((current, window_end))
            current += window_size / 2  # 50% overlap
            
        return windows
    
    def _calculate_window_tdp(
        self,
        decisions: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray],
        start: Any,
        end: Any
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate TDP for a single time window."""
        # Filter data for this window
        if timestamps is not None:
            mask = (timestamps >= start) & (timestamps < end)
            window_decisions = decisions[mask]
            window_groups = groups[mask]
        else:
            # Treat start/end as indices
            window_decisions = decisions[int(start):int(end)]
            window_groups = groups[int(start):int(end)]
        
        # Get unique groups in this window
        unique_groups = np.unique(window_groups)
        
        if len(unique_groups) < 2:
            return None, {'n_groups': len(unique_groups), 'samples': len(window_decisions)}
        
        # Calculate positive rates for each group
        group_rates = {}
        group_counts = {}
        
        for group in unique_groups:
            group_mask = window_groups == group
            group_decisions = window_decisions[group_mask]
            
            if len(group_decisions) < self.min_samples:
                continue
                
            positive_rate = np.mean(group_decisions)
            group_rates[group] = positive_rate
            group_counts[group] = len(group_decisions)
        
        if len(group_rates) < 2:
            return None, {
                'n_groups': len(unique_groups),
                'samples': len(window_decisions),
                'insufficient_samples': True
            }
        
        # Calculate maximum pairwise difference
        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates)
        
        stats = {
            'tdp': max_diff,
            'group_rates': group_rates,
            'group_counts': group_counts,
            'n_samples': len(window_decisions),
            'n_groups': len(group_rates)
        }
        
        return max_diff, stats
    
    def detect_bias(
        self,
        decisions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal bias using TDP metric.
        
        Args:
            decisions: Binary decisions
            groups: Group membership
            timestamps: Timestamps (optional)
            time_windows: Time windows for analysis
            
        Returns:
            Dictionary with bias detection results
        """
        results = self.calculate(
            decisions, groups, timestamps, time_windows,
            return_details=True
        )
        
        bias_detected = results['max_tdp'] > self.threshold
        
        detection_results = {
            'bias_detected': bias_detected,
            'metric_value': results['max_tdp'],
            'threshold': self.threshold,
            'confidence': self._calculate_confidence(results),
            'severity': self._calculate_severity(results['max_tdp']),
            'details': results
        }
        
        return detection_results
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in bias detection."""
        if not results['windows']:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of windows analyzed
        # 2. Consistency of TDP across windows
        # 3. Sample sizes
        
        n_windows = len(results['windows'])
        window_factor = min(n_windows / 10, 1.0)  # More windows = higher confidence
        
        if 'std_tdp' in results and results['mean_tdp'] > 0:
            consistency = 1.0 - min(results['std_tdp'] / results['mean_tdp'], 1.0)
        else:
            consistency = 0.5
        
        avg_samples = np.mean([w['n_samples'] for w in results['windows']])
        sample_factor = min(avg_samples / 1000, 1.0)  # More samples = higher confidence
        
        confidence = (window_factor + consistency + sample_factor) / 3.0
        return round(confidence, 3)
    
    def _calculate_severity(self, tdp_value: float) -> str:
        """Categorize bias severity based on TDP value."""
        if np.isnan(tdp_value):
            return 'unknown'
        elif tdp_value <= self.threshold:
            return 'none'
        elif tdp_value <= self.threshold * 2:
            return 'low'
        elif tdp_value <= self.threshold * 4:
            return 'medium'
        else:
            return 'high'


def calculate_tdp(
    decisions: Union[np.ndarray, pd.Series, List],
    groups: Union[np.ndarray, pd.Series, List],
    time_windows: Optional[List[Tuple[Any, Any]]] = None,
    **kwargs
) -> float:
    """
    Convenience function to calculate TDP.
    
    Args:
        decisions: Binary decisions (0 or 1)
        groups: Group membership for each decision
        time_windows: List of (start, end) tuples defining windows
        **kwargs: Additional arguments passed to TemporalDemographicParity
        
    Returns:
        Maximum TDP value across all windows
    """
    tdp = TemporalDemographicParity()
    return tdp.calculate(decisions, groups, time_windows=time_windows, **kwargs)