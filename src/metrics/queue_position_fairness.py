"""
Queue Position Fairness (QPF) Metric Implementation

This module implements the Queue Position Fairness metric for detecting
systematic ordering bias in priority queuing systems like healthcare triage,
customer service routing, or resource allocation.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any
from collections import defaultdict
from scipy import stats
import warnings


class QueuePositionFairness:
    """
    Queue Position Fairness (QPF) metric for priority system bias detection.
    
    QPF measures whether certain demographic groups are systematically
    placed in disadvantageous queue positions, experiencing longer wait
    times or lower priority assignments.
    
    QPF = 1 - |avg_position_groupA - avg_position_groupB| / max_queue_size
    
    Attributes:
        fairness_threshold (float): Threshold for acceptable QPF (default: 0.8)
        min_samples (int): Minimum samples per group (default: 20)
        normalize (bool): Whether to normalize positions to [0,1] (default: True)
    """
    
    def __init__(
        self,
        fairness_threshold: float = 0.8,
        min_samples: int = 20,
        normalize: bool = True
    ):
        """
        Initialize QPF metric calculator.
        
        Args:
            fairness_threshold: Minimum QPF value for fairness (0-1 scale)
            min_samples: Minimum samples required per group
            normalize: Whether to normalize queue positions
        """
        self.fairness_threshold = fairness_threshold
        self.min_samples = min_samples
        self.normalize = normalize
        self.results_cache = {}
    
    def calculate(
        self,
        queue_positions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None,
        max_queue_size: Optional[int] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate QPF across time windows.
        
        Args:
            queue_positions: Queue position assignments (1 = front, higher = back)
            groups: Group membership for each assignment
            timestamps: Timestamp for each assignment (optional)
            time_windows: List of (start, end) tuples defining windows
            max_queue_size: Maximum queue size for normalization
            return_details: If True, return detailed statistics
            
        Returns:
            If return_details=False: Minimum QPF value across all windows
            If return_details=True: Dictionary with detailed statistics
            
        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Input validation
        queue_positions, groups, timestamps = self._validate_inputs(
            queue_positions, groups, timestamps
        )
        
        # Determine max queue size if not provided
        if max_queue_size is None:
            max_queue_size = np.max(queue_positions)
        
        # Generate time windows if not provided
        if time_windows is None:
            if timestamps is None:
                time_windows = [(0, len(queue_positions))]
            else:
                time_windows = self._generate_windows(timestamps)
        
        # Calculate QPF for each window
        qpf_values = []
        window_stats = []
        
        for window_idx, (start, end) in enumerate(time_windows):
            window_qpf, stats = self._calculate_window_qpf(
                queue_positions, groups, timestamps, start, end, max_queue_size
            )
            
            if window_qpf is not None:
                qpf_values.append(window_qpf)
                stats['window'] = (start, end)
                stats['window_idx'] = window_idx
                window_stats.append(stats)
        
        if not qpf_values:
            warnings.warn(
                f"No valid windows found with minimum {self.min_samples} samples per group"
            )
            return np.nan if not return_details else {
                'min_qpf': np.nan,
                'mean_qpf': np.nan,
                'windows': []
            }
        
        min_qpf = min(qpf_values)
        
        if return_details:
            return {
                'min_qpf': min_qpf,
                'mean_qpf': np.mean(qpf_values),
                'std_qpf': np.std(qpf_values),
                'max_qpf': max(qpf_values),
                'qpf_values': qpf_values,
                'windows': window_stats,
                'is_fair': min_qpf >= self.fairness_threshold,
                'threshold': self.fairness_threshold,
                'max_queue_size': max_queue_size
            }
        
        return min_qpf
    
    def calculate_wait_time_disparity(
        self,
        wait_times: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate wait time disparities between groups.
        
        Args:
            wait_times: Actual wait times experienced
            groups: Group membership
            timestamps: Timestamps (optional)
            time_windows: Time windows for analysis
            
        Returns:
            Dictionary with wait time disparity analysis
        """
        wait_times = np.asarray(wait_times)
        groups = np.asarray(groups)
        
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
        
        if time_windows is None:
            if timestamps is None:
                time_windows = [(0, len(wait_times))]
            else:
                time_windows = self._generate_windows(timestamps)
        
        results = {'windows': []}
        
        for window_idx, (start, end) in enumerate(time_windows):
            # Filter data for window
            if timestamps is not None:
                mask = (timestamps >= start) & (timestamps < end)
                window_waits = wait_times[mask]
                window_groups = groups[mask]
            else:
                window_waits = wait_times[int(start):int(end)]
                window_groups = groups[int(start):int(end)]
            
            # Calculate statistics per group
            unique_groups = np.unique(window_groups)
            group_stats = {}
            
            for group in unique_groups:
                group_mask = window_groups == group
                group_waits = window_waits[group_mask]
                
                if len(group_waits) >= self.min_samples:
                    group_stats[group] = {
                        'mean_wait': np.mean(group_waits),
                        'median_wait': np.median(group_waits),
                        'std_wait': np.std(group_waits),
                        'min_wait': np.min(group_waits),
                        'max_wait': np.max(group_waits),
                        'n_samples': len(group_waits)
                    }
            
            if len(group_stats) >= 2:
                # Calculate disparity metrics
                mean_waits = [s['mean_wait'] for s in group_stats.values()]
                max_disparity = max(mean_waits) - min(mean_waits)
                disparity_ratio = max(mean_waits) / min(mean_waits) if min(mean_waits) > 0 else np.inf
                
                results['windows'].append({
                    'window': (start, end),
                    'window_idx': window_idx,
                    'group_stats': group_stats,
                    'max_disparity': max_disparity,
                    'disparity_ratio': disparity_ratio
                })
        
        return results
    
    def _validate_inputs(
        self,
        queue_positions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate and convert inputs to numpy arrays."""
        # Convert to numpy arrays
        queue_positions = np.asarray(queue_positions)
        groups = np.asarray(groups)
        
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
        
        # Check lengths
        if len(queue_positions) != len(groups):
            raise ValueError("queue_positions and groups must have same length")
        
        if timestamps is not None and len(timestamps) != len(queue_positions):
            raise ValueError("timestamps must have same length as queue_positions")
        
        # Check queue positions are positive
        if np.any(queue_positions <= 0):
            raise ValueError("queue_positions must be positive (1 = front of queue)")
        
        # Check at least 2 groups
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("At least 2 groups required for QPF calculation")
        
        return queue_positions, groups, timestamps
    
    def _generate_windows(
        self,
        timestamps: np.ndarray,
        window_size: Optional[float] = None
    ) -> List[Tuple[float, float]]:
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
    
    def _calculate_window_qpf(
        self,
        queue_positions: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray],
        start: Any,
        end: Any,
        max_queue_size: int
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate QPF for a single time window."""
        # Filter data for this window
        if timestamps is not None:
            mask = (timestamps >= start) & (timestamps < end)
            window_positions = queue_positions[mask]
            window_groups = groups[mask]
        else:
            window_positions = queue_positions[int(start):int(end)]
            window_groups = groups[int(start):int(end)]
        
        # Get unique groups in this window
        unique_groups = np.unique(window_groups)
        
        if len(unique_groups) < 2:
            return None, {'n_groups': len(unique_groups), 'samples': len(window_positions)}
        
        # Calculate average position for each group
        group_positions = {}
        group_counts = {}
        
        for group in unique_groups:
            group_mask = window_groups == group
            group_pos = window_positions[group_mask]
            
            if len(group_pos) < self.min_samples:
                continue
            
            # Calculate statistics
            avg_position = np.mean(group_pos)
            group_positions[group] = avg_position
            group_counts[group] = len(group_pos)
        
        if len(group_positions) < 2:
            return None, {
                'n_groups': len(unique_groups),
                'samples': len(window_positions),
                'insufficient_samples': True
            }
        
        # Calculate QPF
        positions = list(group_positions.values())
        max_diff = max(positions) - min(positions)
        
        # QPF = 1 - (max_difference / max_queue_size)
        if self.normalize:
            qpf = 1.0 - (max_diff / max_queue_size)
        else:
            qpf = 1.0 - max_diff
        
        qpf = max(0, min(1, qpf))  # Clamp to [0, 1]
        
        # Additional statistics
        stats = {
            'qpf': qpf,
            'group_positions': group_positions,
            'group_counts': group_counts,
            'position_difference': max_diff,
            'n_samples': len(window_positions),
            'n_groups': len(group_positions),
            'best_group': min(group_positions, key=group_positions.get),
            'worst_group': max(group_positions, key=group_positions.get)
        }
        
        return qpf, stats
    
    def detect_bias(
        self,
        queue_positions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None,
        max_queue_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect queue-based bias using QPF metric.
        
        Args:
            queue_positions: Queue position assignments
            groups: Group membership
            timestamps: Timestamps (optional)
            time_windows: Time windows for analysis
            max_queue_size: Maximum queue size
            
        Returns:
            Dictionary with bias detection results
        """
        results = self.calculate(
            queue_positions, groups, timestamps, time_windows,
            max_queue_size, return_details=True
        )
        
        bias_detected = results['min_qpf'] < self.fairness_threshold
        
        # Identify most disadvantaged groups
        disadvantaged_groups = []
        if results['windows']:
            for window in results['windows']:
                if 'worst_group' in window:
                    disadvantaged_groups.append(window['worst_group'])
        
        most_disadvantaged = max(set(disadvantaged_groups), key=disadvantaged_groups.count) if disadvantaged_groups else None
        
        detection_results = {
            'bias_detected': bias_detected,
            'metric_value': results['min_qpf'],
            'threshold': self.fairness_threshold,
            'most_disadvantaged_group': most_disadvantaged,
            'confidence': self._calculate_confidence(results),
            'severity': self._calculate_severity(results['min_qpf']),
            'details': results
        }
        
        return detection_results
    
    def analyze_priority_patterns(
        self,
        queue_positions: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        priority_levels: Optional[Union[np.ndarray, pd.Series, List]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how priority assignments affect queue positions.
        
        Args:
            queue_positions: Queue positions
            groups: Group membership
            priority_levels: Assigned priority levels (optional)
            
        Returns:
            Analysis of priority patterns by group
        """
        queue_positions = np.asarray(queue_positions)
        groups = np.asarray(groups)
        
        results = {}
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            group_mask = groups == group
            group_positions = queue_positions[group_mask]
            
            results[group] = {
                'mean_position': np.mean(group_positions),
                'median_position': np.median(group_positions),
                'front_quarter_pct': np.mean(group_positions <= np.percentile(queue_positions, 25)),
                'back_quarter_pct': np.mean(group_positions >= np.percentile(queue_positions, 75)),
                'n_samples': len(group_positions)
            }
            
            if priority_levels is not None:
                priority_levels = np.asarray(priority_levels)
                group_priorities = priority_levels[group_mask]
                
                # Correlation between priority and position
                if len(group_priorities) > 1:
                    corr, p_value = stats.spearmanr(group_priorities, group_positions)
                    results[group]['priority_position_correlation'] = corr
                    results[group]['correlation_p_value'] = p_value
                    results[group]['mean_priority'] = np.mean(group_priorities)
        
        # Statistical test for position differences
        if len(unique_groups) == 2:
            group1_positions = queue_positions[groups == unique_groups[0]]
            group2_positions = queue_positions[groups == unique_groups[1]]
            
            if len(group1_positions) > 1 and len(group2_positions) > 1:
                # Mann-Whitney U test for position differences
                u_stat, p_value = stats.mannwhitneyu(group1_positions, group2_positions)
                results['position_difference_test'] = {
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return results
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in bias detection."""
        if not results['windows']:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of windows analyzed
        # 2. Consistency of QPF across windows
        # 3. Sample sizes
        
        n_windows = len(results['windows'])
        window_factor = min(n_windows / 10, 1.0)
        
        if 'std_qpf' in results and results['mean_qpf'] > 0:
            consistency = 1.0 - min(results['std_qpf'] / results['mean_qpf'], 1.0)
        else:
            consistency = 0.5
        
        avg_samples = np.mean([w['n_samples'] for w in results['windows']])
        sample_factor = min(avg_samples / 500, 1.0)
        
        confidence = (window_factor + consistency + sample_factor) / 3.0
        return round(confidence, 3)
    
    def _calculate_severity(self, qpf_value: float) -> str:
        """Categorize bias severity based on QPF value."""
        if np.isnan(qpf_value):
            return 'unknown'
        
        # Invert scale (lower QPF = higher severity)
        severity_score = 1.0 - qpf_value
        
        if severity_score <= (1.0 - self.fairness_threshold):
            return 'none'
        elif severity_score <= 0.3:
            return 'low'
        elif severity_score <= 0.5:
            return 'medium'
        elif severity_score <= 0.7:
            return 'high'
        else:
            return 'critical'


def calculate_qpf(
    queue_positions: Union[np.ndarray, pd.Series, List],
    groups: Union[np.ndarray, pd.Series, List],
    max_queue_size: Optional[int] = None,
    **kwargs
) -> float:
    """
    Convenience function to calculate QPF.
    
    Args:
        queue_positions: Queue position assignments
        groups: Group membership for each assignment
        max_queue_size: Maximum queue size for normalization
        **kwargs: Additional arguments passed to QueuePositionFairness
        
    Returns:
        Minimum QPF value across all windows
    """
    qpf = QueuePositionFairness()
    return qpf.calculate(queue_positions, groups, max_queue_size=max_queue_size, **kwargs)