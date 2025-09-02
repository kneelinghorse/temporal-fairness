"""
Equalized Odds Over Time (EOOT) Metric Implementation

This module implements the Equalized Odds Over Time metric for detecting
fairness violations in True Positive Rate (TPR) and False Positive Rate (FPR)
across demographic groups over time windows.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict


class EqualizedOddsOverTime:
    """
    Equalized Odds Over Time (EOOT) metric for temporal fairness measurement.
    
    EOOT measures the maximum difference in True Positive Rate (TPR) and 
    False Positive Rate (FPR) between demographic groups within time windows.
    
    EOOT(t) = max(
        |TPR(group=A,t) - TPR(group=B,t)|,
        |FPR(group=A,t) - FPR(group=B,t)|
    )
    
    Attributes:
        tpr_threshold (float): Maximum acceptable TPR difference (default: 0.1)
        fpr_threshold (float): Maximum acceptable FPR difference (default: 0.1)
        min_samples (int): Minimum samples per group per window (default: 30)
    """
    
    def __init__(
        self, 
        tpr_threshold: float = 0.1,
        fpr_threshold: float = 0.1,
        min_samples: int = 30
    ):
        """
        Initialize EOOT metric calculator.
        
        Args:
            tpr_threshold: Maximum acceptable TPR difference for fairness
            fpr_threshold: Maximum acceptable FPR difference for fairness
            min_samples: Minimum samples required per group per window
        """
        self.tpr_threshold = tpr_threshold
        self.fpr_threshold = fpr_threshold
        self.min_samples = min_samples
        self.results_cache = {}
    
    def calculate(
        self,
        predictions: Union[np.ndarray, pd.Series, List],
        true_labels: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None,
        window_size: Optional[int] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate EOOT across time windows.
        
        Args:
            predictions: Binary predictions (0 or 1)
            true_labels: Ground truth binary labels (0 or 1)
            groups: Group membership for each sample
            timestamps: Timestamp for each sample (optional)
            time_windows: List of (start, end) tuples defining windows
            window_size: Size of sliding windows (alternative to time_windows)
            return_details: If True, return detailed statistics
            
        Returns:
            If return_details=False: Maximum EOOT value across all windows
            If return_details=True: Dictionary with detailed statistics
            
        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Input validation
        predictions, true_labels, groups, timestamps = self._validate_inputs(
            predictions, true_labels, groups, timestamps
        )
        
        # Generate time windows if not provided
        if time_windows is None:
            if timestamps is None:
                # Create single window for all data
                time_windows = [(0, len(predictions))]
            else:
                time_windows = self._generate_windows(timestamps, window_size)
        
        # Calculate EOOT for each window
        eoot_values = []
        window_stats = []
        
        for window_idx, (start, end) in enumerate(time_windows):
            window_eoot, stats = self._calculate_window_eoot(
                predictions, true_labels, groups, timestamps, start, end
            )
            
            if window_eoot is not None:
                eoot_values.append(window_eoot)
                stats['window'] = (start, end)
                stats['window_idx'] = window_idx
                window_stats.append(stats)
        
        if not eoot_values:
            warnings.warn(
                f"No valid windows found with minimum {self.min_samples} samples per group"
            )
            return np.nan if not return_details else {
                'max_eoot': np.nan,
                'mean_eoot': np.nan,
                'windows': []
            }
        
        max_eoot = max(eoot_values)
        
        if return_details:
            return {
                'max_eoot': max_eoot,
                'mean_eoot': np.mean(eoot_values),
                'std_eoot': np.std(eoot_values),
                'min_eoot': min(eoot_values),
                'eoot_values': eoot_values,
                'windows': window_stats,
                'is_fair': self._check_fairness(window_stats),
                'tpr_threshold': self.tpr_threshold,
                'fpr_threshold': self.fpr_threshold
            }
        
        return max_eoot
    
    def calculate_group_metrics(
        self,
        predictions: Union[np.ndarray, pd.Series, List],
        true_labels: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, Dict[Any, Dict[str, float]]]:
        """
        Calculate TPR and FPR for each group in each time window.
        
        Args:
            predictions: Binary predictions
            true_labels: Ground truth labels
            groups: Group membership
            timestamps: Timestamps (optional)
            time_windows: Time windows for analysis
            
        Returns:
            Dictionary with TPR/FPR for each group in each window
        """
        predictions, true_labels, groups, timestamps = self._validate_inputs(
            predictions, true_labels, groups, timestamps
        )
        
        if time_windows is None:
            if timestamps is None:
                time_windows = [(0, len(predictions))]
            else:
                time_windows = self._generate_windows(timestamps, None)
        
        results = {'windows': []}
        
        for window_idx, (start, end) in enumerate(time_windows):
            window_metrics = self._calculate_window_metrics(
                predictions, true_labels, groups, timestamps, start, end
            )
            
            if window_metrics:
                results['windows'].append({
                    'window': (start, end),
                    'window_idx': window_idx,
                    'metrics': window_metrics
                })
        
        return results
    
    def _validate_inputs(
        self,
        predictions: Union[np.ndarray, pd.Series, List],
        true_labels: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate and convert inputs to numpy arrays."""
        # Convert to numpy arrays
        predictions = np.asarray(predictions)
        true_labels = np.asarray(true_labels)
        groups = np.asarray(groups)
        
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
        
        # Check lengths
        if len(predictions) != len(true_labels):
            raise ValueError("predictions and true_labels must have same length")
        
        if len(predictions) != len(groups):
            raise ValueError("predictions and groups must have same length")
        
        if timestamps is not None and len(timestamps) != len(predictions):
            raise ValueError("timestamps must have same length as predictions")
        
        # Check binary values
        unique_preds = np.unique(predictions[~np.isnan(predictions)])
        unique_labels = np.unique(true_labels[~np.isnan(true_labels)])
        
        if not set(unique_preds).issubset({0, 1}):
            raise ValueError("predictions must be binary (0 or 1)")
        
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("true_labels must be binary (0 or 1)")
        
        # Check at least 2 groups
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("At least 2 groups required for EOOT calculation")
        
        return predictions, true_labels, groups, timestamps
    
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
    
    def _calculate_window_eoot(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray],
        start: Any,
        end: Any
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate EOOT for a single time window."""
        # Filter data for this window
        if timestamps is not None:
            mask = (timestamps >= start) & (timestamps < end)
            window_preds = predictions[mask]
            window_labels = true_labels[mask]
            window_groups = groups[mask]
        else:
            # Treat start/end as indices
            window_preds = predictions[int(start):int(end)]
            window_labels = true_labels[int(start):int(end)]
            window_groups = groups[int(start):int(end)]
        
        # Calculate metrics for each group
        metrics = self._calculate_window_metrics(
            window_preds, window_labels, window_groups, None, 0, len(window_preds)
        )
        
        if not metrics or len(metrics) < 2:
            return None, {'n_groups': len(metrics), 'samples': len(window_preds)}
        
        # Calculate maximum pairwise differences
        tpr_values = [m['tpr'] for m in metrics.values() if m['tpr'] is not None]
        fpr_values = [m['fpr'] for m in metrics.values() if m['fpr'] is not None]
        
        max_tpr_diff = 0 if len(tpr_values) < 2 else max(tpr_values) - min(tpr_values)
        max_fpr_diff = 0 if len(fpr_values) < 2 else max(fpr_values) - min(fpr_values)
        
        # EOOT is the maximum of TPR and FPR differences
        eoot = max(max_tpr_diff, max_fpr_diff)
        
        stats = {
            'eoot': eoot,
            'tpr_diff': max_tpr_diff,
            'fpr_diff': max_fpr_diff,
            'group_metrics': metrics,
            'n_samples': len(window_preds),
            'n_groups': len(metrics)
        }
        
        return eoot, stats
    
    def _calculate_window_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray],
        start: Any,
        end: Any
    ) -> Dict[Any, Dict[str, float]]:
        """Calculate TPR and FPR for each group in a window."""
        # Filter data if needed (when called directly)
        if timestamps is not None:
            mask = (timestamps >= start) & (timestamps < end)
            predictions = predictions[mask]
            true_labels = true_labels[mask]
            groups = groups[mask]
        
        unique_groups = np.unique(groups)
        metrics = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_preds = predictions[group_mask]
            group_labels = true_labels[group_mask]
            
            if len(group_preds) < self.min_samples:
                continue
            
            # Calculate TPR and FPR
            tpr = self._calculate_tpr(group_preds, group_labels)
            fpr = self._calculate_fpr(group_preds, group_labels)
            
            metrics[group] = {
                'tpr': tpr,
                'fpr': fpr,
                'n_samples': len(group_preds),
                'n_positives': np.sum(group_labels == 1),
                'n_negatives': np.sum(group_labels == 0)
            }
        
        return metrics
    
    def _calculate_tpr(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Optional[float]:
        """Calculate True Positive Rate (Sensitivity/Recall)."""
        positive_mask = true_labels == 1
        n_positives = np.sum(positive_mask)
        
        if n_positives == 0:
            return None
        
        true_positives = np.sum((predictions == 1) & positive_mask)
        return true_positives / n_positives
    
    def _calculate_fpr(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Optional[float]:
        """Calculate False Positive Rate."""
        negative_mask = true_labels == 0
        n_negatives = np.sum(negative_mask)
        
        if n_negatives == 0:
            return None
        
        false_positives = np.sum((predictions == 1) & negative_mask)
        return false_positives / n_negatives
    
    def _check_fairness(self, window_stats: List[Dict[str, Any]]) -> bool:
        """Check if fairness criteria are met across all windows."""
        for stats in window_stats:
            if stats.get('tpr_diff', 0) > self.tpr_threshold:
                return False
            if stats.get('fpr_diff', 0) > self.fpr_threshold:
                return False
        return True
    
    def detect_bias(
        self,
        predictions: Union[np.ndarray, pd.Series, List],
        true_labels: Union[np.ndarray, pd.Series, List],
        groups: Union[np.ndarray, pd.Series, List],
        timestamps: Optional[Union[np.ndarray, pd.Series, List]] = None,
        time_windows: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal bias using EOOT metric.
        
        Args:
            predictions: Binary predictions
            true_labels: Ground truth labels
            groups: Group membership
            timestamps: Timestamps (optional)
            time_windows: Time windows for analysis
            
        Returns:
            Dictionary with bias detection results
        """
        results = self.calculate(
            predictions, true_labels, groups, timestamps, time_windows,
            return_details=True
        )
        
        bias_detected = not results['is_fair']
        
        # Determine which metric (TPR or FPR) is causing the bias
        bias_source = []
        if results['windows']:
            max_tpr_diff = max(w.get('tpr_diff', 0) for w in results['windows'])
            max_fpr_diff = max(w.get('fpr_diff', 0) for w in results['windows'])
            
            if max_tpr_diff > self.tpr_threshold:
                bias_source.append('TPR')
            if max_fpr_diff > self.fpr_threshold:
                bias_source.append('FPR')
        
        detection_results = {
            'bias_detected': bias_detected,
            'metric_value': results['max_eoot'],
            'bias_source': bias_source,
            'tpr_threshold': self.tpr_threshold,
            'fpr_threshold': self.fpr_threshold,
            'confidence': self._calculate_confidence(results),
            'severity': self._calculate_severity(results),
            'details': results
        }
        
        return detection_results
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in bias detection."""
        if not results['windows']:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of windows analyzed
        # 2. Consistency of EOOT across windows
        # 3. Sample sizes
        # 4. Presence of both positive and negative examples
        
        n_windows = len(results['windows'])
        window_factor = min(n_windows / 10, 1.0)
        
        if 'std_eoot' in results and results['mean_eoot'] > 0:
            consistency = 1.0 - min(results['std_eoot'] / results['mean_eoot'], 1.0)
        else:
            consistency = 0.5
        
        # Check sample quality
        sample_quality = []
        for window in results['windows']:
            if 'group_metrics' in window:
                for metrics in window['group_metrics'].values():
                    has_pos = metrics.get('n_positives', 0) > 0
                    has_neg = metrics.get('n_negatives', 0) > 0
                    if has_pos and has_neg:
                        sample_quality.append(1.0)
                    elif has_pos or has_neg:
                        sample_quality.append(0.5)
                    else:
                        sample_quality.append(0.0)
        
        quality_factor = np.mean(sample_quality) if sample_quality else 0.5
        
        confidence = (window_factor + consistency + quality_factor) / 3.0
        return round(confidence, 3)
    
    def _calculate_severity(self, results: Dict[str, Any]) -> str:
        """Categorize bias severity based on EOOT value."""
        if not results['windows']:
            return 'unknown'
        
        max_tpr_diff = max(w.get('tpr_diff', 0) for w in results['windows'])
        max_fpr_diff = max(w.get('fpr_diff', 0) for w in results['windows'])
        
        # Calculate severity based on how much thresholds are exceeded
        tpr_severity = max_tpr_diff / self.tpr_threshold if self.tpr_threshold > 0 else 0
        fpr_severity = max_fpr_diff / self.fpr_threshold if self.fpr_threshold > 0 else 0
        
        max_severity = max(tpr_severity, fpr_severity)
        
        if max_severity <= 1.0:
            return 'none'
        elif max_severity <= 2.0:
            return 'low'
        elif max_severity <= 4.0:
            return 'medium'
        else:
            return 'high'


def calculate_eoot(
    predictions: Union[np.ndarray, pd.Series, List],
    true_labels: Union[np.ndarray, pd.Series, List],
    groups: Union[np.ndarray, pd.Series, List],
    time_windows: Optional[List[Tuple[Any, Any]]] = None,
    **kwargs
) -> float:
    """
    Convenience function to calculate EOOT.
    
    Args:
        predictions: Binary predictions (0 or 1)
        true_labels: Ground truth binary labels (0 or 1)
        groups: Group membership for each sample
        time_windows: List of (start, end) tuples defining windows
        **kwargs: Additional arguments passed to EqualizedOddsOverTime
        
    Returns:
        Maximum EOOT value across all windows
    """
    eoot = EqualizedOddsOverTime()
    return eoot.calculate(predictions, true_labels, groups, time_windows=time_windows, **kwargs)