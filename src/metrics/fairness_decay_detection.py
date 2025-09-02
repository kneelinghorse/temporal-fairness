"""
Fairness Decay Detection (FDD) Metric Implementation

This module implements the Fairness Decay Detection metric for monitoring
degradation of fairness metrics over time periods and detecting trends.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from scipy import stats
import warnings


class FairnessDecayDetection:
    """
    Fairness Decay Detection (FDD) for monitoring metric degradation over time.
    
    FDD tracks the temporal evolution of fairness metrics and detects:
    - Gradual degradation trends
    - Sudden shifts in fairness
    - Seasonal patterns
    - Confidence valleys
    
    Attributes:
        decay_threshold (float): Threshold for significant decay (default: 0.05)
        window_months (int): Time window for decay analysis in months (default: 6)
        min_windows (int): Minimum windows for trend detection (default: 3)
        detection_method (str): Method for decay detection ('linear', 'exponential', 'changepoint')
    """
    
    def __init__(
        self,
        decay_threshold: float = 0.05,
        window_months: int = 6,
        min_windows: int = 3,
        detection_method: str = 'linear'
    ):
        """
        Initialize FDD detector.
        
        Args:
            decay_threshold: Minimum change to consider as decay
            window_months: Size of analysis window in months
            min_windows: Minimum windows needed for trend detection
            detection_method: Method to detect decay patterns
        """
        self.decay_threshold = decay_threshold
        self.window_months = window_months
        self.min_windows = min_windows
        self.detection_method = detection_method
        self.history = defaultdict(list)
    
    def detect_fairness_decay(
        self,
        metric_history: Union[List[float], np.ndarray, pd.Series],
        timestamps: Optional[Union[List, np.ndarray, pd.Series]] = None,
        metric_name: str = 'fairness_metric',
        return_details: bool = False
    ) -> Union[bool, Dict[str, Any]]:
        """
        Detect fairness decay in metric history.
        
        Args:
            metric_history: Historical values of fairness metric
            timestamps: Timestamps for each metric value
            metric_name: Name of the metric being tracked
            return_details: If True, return detailed analysis
            
        Returns:
            If return_details=False: Boolean indicating decay detected
            If return_details=True: Dictionary with decay analysis
        """
        # Validate inputs
        metric_history = np.asarray(metric_history)
        
        if len(metric_history) < self.min_windows:
            warnings.warn(f"Insufficient data points ({len(metric_history)}) for decay detection")
            return False if not return_details else {
                'decay_detected': False,
                'reason': 'insufficient_data',
                'n_points': len(metric_history)
            }
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = np.arange(len(metric_history))
        else:
            timestamps = np.asarray(timestamps)
        
        # Detect decay based on method
        if self.detection_method == 'linear':
            decay_info = self._detect_linear_decay(metric_history, timestamps)
        elif self.detection_method == 'exponential':
            decay_info = self._detect_exponential_decay(metric_history, timestamps)
        elif self.detection_method == 'changepoint':
            decay_info = self._detect_changepoint(metric_history, timestamps)
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")
        
        # Calculate additional statistics
        decay_info.update(self._calculate_decay_statistics(metric_history, timestamps))
        
        # Store in history
        self.history[metric_name].append({
            'timestamp': timestamps[-1] if len(timestamps) > 0 else None,
            'value': metric_history[-1] if len(metric_history) > 0 else None,
            'decay_detected': decay_info['decay_detected']
        })
        
        if return_details:
            return decay_info
        
        return decay_info['decay_detected']
    
    def analyze_metric_trends(
        self,
        metrics_data: pd.DataFrame,
        metric_columns: List[str],
        time_column: str = 'timestamp',
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze trends across multiple fairness metrics.
        
        Args:
            metrics_data: DataFrame with metric values over time
            metric_columns: List of column names containing metrics
            time_column: Name of timestamp column
            group_column: Optional column for group-wise analysis
            
        Returns:
            Dictionary with trend analysis for each metric
        """
        results = {}
        
        for metric_col in metric_columns:
            if metric_col not in metrics_data.columns:
                warnings.warn(f"Column {metric_col} not found in data")
                continue
            
            if group_column and group_column in metrics_data.columns:
                # Analyze per group
                group_results = {}
                for group in metrics_data[group_column].unique():
                    group_data = metrics_data[metrics_data[group_column] == group]
                    group_results[group] = self.detect_fairness_decay(
                        group_data[metric_col].values,
                        group_data[time_column].values if time_column in group_data else None,
                        metric_name=f"{metric_col}_{group}",
                        return_details=True
                    )
                results[metric_col] = group_results
            else:
                # Overall analysis
                results[metric_col] = self.detect_fairness_decay(
                    metrics_data[metric_col].values,
                    metrics_data[time_column].values if time_column in metrics_data else None,
                    metric_name=metric_col,
                    return_details=True
                )
        
        return results
    
    def _detect_linear_decay(
        self,
        metric_history: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect linear decay trend using regression."""
        # Normalize timestamps to [0, 1] for stability
        if len(np.unique(timestamps)) > 1:
            norm_timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        else:
            norm_timestamps = np.zeros_like(timestamps)
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(norm_timestamps, metric_history)
        
        # Check for significant negative trend
        decay_detected = (slope < -self.decay_threshold) and (p_value < 0.05)
        
        # Calculate decay rate (change per unit time)
        if timestamps.max() != timestamps.min():
            decay_rate = slope * (timestamps.max() - timestamps.min()) / len(timestamps)
        else:
            decay_rate = 0
        
        return {
            'decay_detected': decay_detected,
            'decay_type': 'linear',
            'slope': slope,
            'decay_rate': decay_rate,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'confidence': 1 - p_value if p_value < 0.05 else 0
        }
    
    def _detect_exponential_decay(
        self,
        metric_history: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect exponential decay pattern."""
        # Transform to log space for exponential fitting
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        positive_metrics = np.maximum(metric_history, epsilon)
        log_metrics = np.log(positive_metrics)
        
        # Normalize timestamps
        if len(np.unique(timestamps)) > 1:
            norm_timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        else:
            norm_timestamps = np.zeros_like(timestamps)
        
        # Fit linear regression in log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(norm_timestamps, log_metrics)
        
        # Check for significant decay
        decay_detected = (slope < -self.decay_threshold) and (p_value < 0.05)
        
        # Calculate half-life if decaying
        half_life = None
        if slope < 0:
            half_life = -np.log(2) / slope
        
        return {
            'decay_detected': decay_detected,
            'decay_type': 'exponential',
            'decay_constant': slope,
            'half_life': half_life,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'confidence': 1 - p_value if p_value < 0.05 else 0
        }
    
    def _detect_changepoint(
        self,
        metric_history: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect sudden changes or shifts in metric values."""
        if len(metric_history) < 4:
            return {
                'decay_detected': False,
                'decay_type': 'changepoint',
                'reason': 'insufficient_data'
            }
        
        # Use simple CUSUM (Cumulative Sum) for changepoint detection
        mean_val = np.mean(metric_history)
        cusum = np.cumsum(metric_history - mean_val)
        
        # Find potential changepoint
        changepoint_idx = np.argmax(np.abs(cusum))
        
        if changepoint_idx == 0 or changepoint_idx == len(metric_history) - 1:
            # No meaningful changepoint found
            return {
                'decay_detected': False,
                'decay_type': 'changepoint',
                'changepoint_index': None
            }
        
        # Compare metrics before and after changepoint
        before_metrics = metric_history[:changepoint_idx]
        after_metrics = metric_history[changepoint_idx:]
        
        # Perform t-test
        if len(before_metrics) > 1 and len(after_metrics) > 1:
            t_stat, p_value = stats.ttest_ind(before_metrics, after_metrics)
            mean_change = np.mean(after_metrics) - np.mean(before_metrics)
            
            # Detect decay if significant decrease
            decay_detected = (mean_change < -self.decay_threshold) and (p_value < 0.05)
        else:
            decay_detected = False
            p_value = 1.0
            mean_change = 0
        
        return {
            'decay_detected': decay_detected,
            'decay_type': 'changepoint',
            'changepoint_index': changepoint_idx,
            'changepoint_time': timestamps[changepoint_idx] if changepoint_idx else None,
            'mean_before': np.mean(before_metrics) if len(before_metrics) > 0 else None,
            'mean_after': np.mean(after_metrics) if len(after_metrics) > 0 else None,
            'mean_change': mean_change,
            'p_value': p_value,
            'confidence': 1 - p_value if p_value < 0.05 else 0
        }
    
    def _calculate_decay_statistics(
        self,
        metric_history: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate additional statistics about decay patterns."""
        stats_dict = {
            'n_observations': len(metric_history),
            'mean_value': np.mean(metric_history),
            'std_value': np.std(metric_history),
            'min_value': np.min(metric_history),
            'max_value': np.max(metric_history),
            'range': np.max(metric_history) - np.min(metric_history)
        }
        
        # Calculate volatility (standard deviation of differences)
        if len(metric_history) > 1:
            differences = np.diff(metric_history)
            stats_dict['volatility'] = np.std(differences)
            stats_dict['max_increase'] = np.max(differences) if len(differences) > 0 else 0
            stats_dict['max_decrease'] = np.min(differences) if len(differences) > 0 else 0
        else:
            stats_dict['volatility'] = 0
            stats_dict['max_increase'] = 0
            stats_dict['max_decrease'] = 0
        
        # Detect monotonic trends
        if len(metric_history) > 1:
            is_monotonic_decrease = all(x >= y for x, y in zip(metric_history[:-1], metric_history[1:]))
            is_monotonic_increase = all(x <= y for x, y in zip(metric_history[:-1], metric_history[1:]))
            stats_dict['is_monotonic'] = is_monotonic_decrease or is_monotonic_increase
            stats_dict['trend_direction'] = 'decreasing' if is_monotonic_decrease else ('increasing' if is_monotonic_increase else 'mixed')
        else:
            stats_dict['is_monotonic'] = True
            stats_dict['trend_direction'] = 'stable'
        
        # Calculate autocorrelation for pattern detection
        if len(metric_history) > 3:
            autocorr = self._calculate_autocorrelation(metric_history, lag=1)
            stats_dict['autocorrelation'] = autocorr
            stats_dict['has_pattern'] = abs(autocorr) > 0.5
        else:
            stats_dict['autocorrelation'] = 0
            stats_dict['has_pattern'] = False
        
        return stats_dict
    
    def _calculate_autocorrelation(self, series: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(series) <= lag:
            return 0
        
        mean = np.mean(series)
        c0 = np.sum((series - mean) ** 2) / len(series)
        
        if c0 == 0:
            return 0
        
        c_lag = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / len(series)
        return c_lag / c0
    
    def predict_future_decay(
        self,
        metric_history: Union[List[float], np.ndarray],
        timestamps: Optional[Union[List, np.ndarray]] = None,
        periods_ahead: int = 3
    ) -> Dict[str, Any]:
        """
        Predict future metric values based on detected decay pattern.
        
        Args:
            metric_history: Historical metric values
            timestamps: Historical timestamps
            periods_ahead: Number of periods to predict ahead
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        metric_history = np.asarray(metric_history)
        
        if len(metric_history) < self.min_windows:
            return {
                'predictions': None,
                'confidence_lower': None,
                'confidence_upper': None,
                'error': 'insufficient_data'
            }
        
        if timestamps is None:
            timestamps = np.arange(len(metric_history))
        else:
            timestamps = np.asarray(timestamps)
        
        # Fit model based on detection method
        decay_info = self._detect_linear_decay(metric_history, timestamps)
        
        if not decay_info.get('slope'):
            return {
                'predictions': None,
                'error': 'no_trend_detected'
            }
        
        # Generate future timestamps
        time_diff = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 1
        future_timestamps = timestamps[-1] + np.arange(1, periods_ahead + 1) * time_diff
        
        # Normalize for prediction
        all_timestamps = np.concatenate([timestamps, future_timestamps])
        norm_timestamps = (all_timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
        # Predict using linear model
        slope = decay_info['slope']
        intercept = metric_history[0]  # Use first value as reference
        
        future_norm = norm_timestamps[len(timestamps):]
        predictions = intercept + slope * future_norm
        
        # Calculate confidence intervals based on historical volatility
        std_residuals = np.std(metric_history - (intercept + slope * norm_timestamps[:len(timestamps)]))
        confidence_lower = predictions - 1.96 * std_residuals
        confidence_upper = predictions + 1.96 * std_residuals
        
        return {
            'predictions': predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'future_timestamps': future_timestamps,
            'decay_rate': decay_info['slope'],
            'confidence': decay_info.get('confidence', 0)
        }
    
    def get_decay_severity(self, decay_info: Dict[str, Any]) -> str:
        """
        Categorize decay severity based on detection results.
        
        Args:
            decay_info: Decay detection results
            
        Returns:
            Severity level: 'none', 'low', 'medium', 'high', 'critical'
        """
        if not decay_info.get('decay_detected', False):
            return 'none'
        
        # Get decay magnitude
        if 'mean_change' in decay_info:
            magnitude = abs(decay_info['mean_change'])
        elif 'decay_rate' in decay_info:
            magnitude = abs(decay_info['decay_rate'])
        elif 'slope' in decay_info:
            magnitude = abs(decay_info['slope'])
        else:
            return 'unknown'
        
        # Get confidence
        confidence = decay_info.get('confidence', 0)
        
        # Calculate severity score
        severity_score = magnitude * confidence
        
        if severity_score < self.decay_threshold:
            return 'low'
        elif severity_score < self.decay_threshold * 2:
            return 'medium'
        elif severity_score < self.decay_threshold * 4:
            return 'high'
        else:
            return 'critical'
    
    def generate_alert(
        self,
        decay_info: Dict[str, Any],
        metric_name: str = 'fairness_metric'
    ) -> Optional[Dict[str, Any]]:
        """
        Generate alert based on decay detection results.
        
        Args:
            decay_info: Decay detection results
            metric_name: Name of the metric
            
        Returns:
            Alert dictionary if alert triggered, None otherwise
        """
        if not decay_info.get('decay_detected', False):
            return None
        
        severity = self.get_decay_severity(decay_info)
        
        if severity in ['none', 'low']:
            return None
        
        alert = {
            'alert_type': 'fairness_decay',
            'metric': metric_name,
            'severity': severity,
            'decay_type': decay_info.get('decay_type', 'unknown'),
            'message': self._generate_alert_message(decay_info, metric_name, severity),
            'timestamp': pd.Timestamp.now(),
            'details': decay_info
        }
        
        return alert
    
    def _generate_alert_message(
        self,
        decay_info: Dict[str, Any],
        metric_name: str,
        severity: str
    ) -> str:
        """Generate human-readable alert message."""
        decay_type = decay_info.get('decay_type', 'unknown')
        
        if decay_type == 'linear':
            rate = decay_info.get('decay_rate', 0)
            message = f"{severity.upper()} ALERT: {metric_name} showing linear decay "
            message += f"(rate: {rate:.4f} per period)"
        elif decay_type == 'exponential':
            half_life = decay_info.get('half_life')
            message = f"{severity.upper()} ALERT: {metric_name} showing exponential decay "
            if half_life:
                message += f"(half-life: {half_life:.1f} periods)"
        elif decay_type == 'changepoint':
            change = decay_info.get('mean_change', 0)
            message = f"{severity.upper()} ALERT: Sudden shift detected in {metric_name} "
            message += f"(change: {change:.4f})"
        else:
            message = f"{severity.upper()} ALERT: Decay detected in {metric_name}"
        
        confidence = decay_info.get('confidence', 0)
        message += f" [Confidence: {confidence:.1%}]"
        
        return message


def detect_fairness_decay(
    metric_history: Union[List[float], np.ndarray, pd.Series],
    threshold: float = 0.05,
    window_months: int = 6,
    **kwargs
) -> bool:
    """
    Convenience function to detect fairness decay.
    
    Args:
        metric_history: Historical values of fairness metric
        threshold: Threshold for significant decay
        window_months: Time window for analysis
        **kwargs: Additional arguments passed to FairnessDecayDetection
        
    Returns:
        Boolean indicating whether decay was detected
    """
    fdd = FairnessDecayDetection(decay_threshold=threshold, window_months=window_months)
    return fdd.detect_fairness_decay(metric_history, **kwargs)