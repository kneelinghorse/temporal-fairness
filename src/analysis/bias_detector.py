"""
Automated bias detection and pattern recognition framework.

This module provides comprehensive tools for detecting various types of
temporal bias patterns in algorithmic decision-making systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from collections import defaultdict
from datetime import datetime, timedelta


class BiasDetector:
    """
    Automated bias detection across temporal patterns.
    
    Detects various bias patterns including:
    - Confidence valleys (U-shaped fairness degradation)
    - Sudden shifts (step changes in fairness)
    - Gradual drift (slow degradation over time)
    - Periodic/seasonal patterns
    - Anomalous spikes
    """
    
    def __init__(
        self,
        sensitivity: float = 0.95,
        min_pattern_length: int = 3,
        anomaly_threshold: float = 3.0
    ):
        """
        Initialize bias detector.
        
        Args:
            sensitivity: Detection sensitivity (0-1, higher = more sensitive)
            min_pattern_length: Minimum time points for pattern detection
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.sensitivity = sensitivity
        self.min_pattern_length = min_pattern_length
        self.anomaly_threshold = anomaly_threshold
        self.detected_patterns = []
        
    def detect_temporal_patterns(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        groups: Union[pd.Series, np.ndarray, List],
        time_column: Optional[str] = None,
        decision_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal bias patterns in data.
        
        Args:
            data: Input data (DataFrame or array)
            groups: Group assignments
            time_column: Name of time column (if DataFrame)
            decision_column: Name of decision column (if DataFrame)
            
        Returns:
            Dictionary with detected patterns and analysis
        """
        # Convert inputs to consistent format
        if isinstance(data, pd.DataFrame):
            if decision_column:
                decisions = data[decision_column].values
            else:
                decisions = data.iloc[:, 0].values
            
            if time_column:
                timestamps = data[time_column].values
            else:
                timestamps = np.arange(len(data))
        else:
            decisions = np.asarray(data)
            timestamps = np.arange(len(decisions))
        
        groups = np.asarray(groups)
        
        # Detect various pattern types
        patterns = {
            'confidence_valley': self.identify_confidence_valleys(decisions, timestamps),
            'sudden_shift': self._detect_sudden_shifts(decisions, timestamps),
            'gradual_drift': self._detect_gradual_drift(decisions, timestamps),
            'periodic': self._detect_periodic_patterns(decisions, timestamps),
            'anomalies': self._detect_anomalies(decisions, timestamps),
            'group_divergence': self._detect_group_divergence(decisions, groups, timestamps)
        }
        
        # Calculate overall bias score
        bias_score = self._calculate_bias_score(patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns)
        
        return {
            'patterns': patterns,
            'bias_score': bias_score,
            'severity': self._classify_severity(bias_score),
            'recommendations': recommendations,
            'summary': self._generate_summary(patterns)
        }
    
    def identify_confidence_valleys(
        self,
        decisions_over_time: Union[np.ndarray, List],
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Identify confidence valley patterns (U-shaped fairness degradation).
        
        Confidence valleys occur when fairness metrics degrade in the middle
        of a time period but improve at the beginning and end.
        
        Args:
            decisions_over_time: Time series of decisions or metrics
            timestamps: Timestamps for each decision
            
        Returns:
            Dictionary with confidence valley detection results
        """
        decisions = np.asarray(decisions_over_time)
        
        if len(decisions) < self.min_pattern_length * 2:
            return {
                'detected': False,
                'reason': 'insufficient_data'
            }
        
        # Smooth the signal to reduce noise
        window_size = max(3, len(decisions) // 10)
        if window_size % 2 == 0:
            window_size += 1
        
        smoothed = signal.savgol_filter(decisions, window_size, 2)
        
        # Find local minima and maxima
        peaks, _ = signal.find_peaks(smoothed)
        valleys, _ = signal.find_peaks(-smoothed)
        
        # Check for U-shape pattern
        if len(valleys) > 0:
            # Find the most prominent valley
            valley_depths = []
            for valley_idx in valleys:
                # Calculate depth relative to surrounding peaks
                left_bound = 0
                right_bound = len(smoothed) - 1
                
                left_peaks = peaks[peaks < valley_idx]
                if len(left_peaks) > 0:
                    left_bound = left_peaks[-1]
                
                right_peaks = peaks[peaks > valley_idx]
                if len(right_peaks) > 0:
                    right_bound = right_peaks[0]
                
                left_height = smoothed[left_bound]
                right_height = smoothed[right_bound]
                valley_depth = min(left_height, right_height) - smoothed[valley_idx]
                
                valley_depths.append({
                    'index': valley_idx,
                    'depth': valley_depth,
                    'left_bound': left_bound,
                    'right_bound': right_bound
                })
            
            # Sort by depth
            valley_depths.sort(key=lambda x: x['depth'], reverse=True)
            
            if valley_depths[0]['depth'] > np.std(smoothed) * (2 - self.sensitivity):
                primary_valley = valley_depths[0]
                
                # Calculate valley characteristics
                valley_start = primary_valley['left_bound']
                valley_end = primary_valley['right_bound']
                valley_center = primary_valley['index']
                
                return {
                    'detected': True,
                    'valley_center': valley_center,
                    'valley_range': (valley_start, valley_end),
                    'valley_depth': primary_valley['depth'],
                    'confidence': min(1.0, primary_valley['depth'] / (2 * np.std(smoothed))),
                    'timestamp': timestamps[valley_center] if timestamps is not None else valley_center,
                    'description': 'Confidence valley detected - fairness degrades in middle period'
                }
        
        return {
            'detected': False,
            'reason': 'no_valley_pattern'
        }
    
    def _detect_sudden_shifts(
        self,
        decisions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect sudden shifts or step changes in bias patterns."""
        if len(decisions) < self.min_pattern_length * 2:
            return {'detected': False, 'shifts': []}
        
        # Use CUSUM for changepoint detection
        mean_val = np.mean(decisions)
        cusum_pos = np.zeros(len(decisions))
        cusum_neg = np.zeros(len(decisions))
        
        for i in range(1, len(decisions)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + decisions[i] - mean_val)
            cusum_neg[i] = max(0, cusum_neg[i-1] + mean_val - decisions[i])
        
        # Find significant changes
        threshold = np.std(decisions) * (2 - self.sensitivity)
        shifts = []
        
        # Detect positive shifts
        pos_peaks, _ = signal.find_peaks(cusum_pos, height=threshold)
        for peak in pos_peaks:
            shifts.append({
                'index': peak,
                'direction': 'increase',
                'magnitude': cusum_pos[peak],
                'timestamp': timestamps[peak]
            })
        
        # Detect negative shifts
        neg_peaks, _ = signal.find_peaks(cusum_neg, height=threshold)
        for peak in neg_peaks:
            shifts.append({
                'index': peak,
                'direction': 'decrease',
                'magnitude': cusum_neg[peak],
                'timestamp': timestamps[peak]
            })
        
        # Sort by magnitude
        shifts.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return {
            'detected': len(shifts) > 0,
            'shifts': shifts[:5],  # Top 5 shifts
            'total_shifts': len(shifts)
        }
    
    def _detect_gradual_drift(
        self,
        decisions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect gradual drift or slow degradation patterns."""
        if len(decisions) < self.min_pattern_length:
            return {'detected': False}
        
        # Fit linear trend
        z = np.polyfit(range(len(decisions)), decisions, 1)
        slope = z[0]
        
        # Calculate trend strength
        predicted = np.polyval(z, range(len(decisions)))
        residuals = decisions - predicted
        r_squared = 1 - (np.sum(residuals**2) / np.sum((decisions - np.mean(decisions))**2))
        
        # Detect significant drift
        drift_threshold = np.std(decisions) / len(decisions) * (2 - self.sensitivity)
        drift_detected = abs(slope) > drift_threshold and r_squared > 0.3
        
        return {
            'detected': drift_detected,
            'drift_rate': slope,
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'r_squared': r_squared,
            'total_change': slope * len(decisions),
            'confidence': r_squared
        }
    
    def _detect_periodic_patterns(
        self,
        decisions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect periodic or seasonal patterns."""
        if len(decisions) < self.min_pattern_length * 4:
            return {'detected': False}
        
        # Remove trend
        detrended = signal.detrend(decisions)
        
        # Compute autocorrelation
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        min_distance = max(2, len(decisions) // 10)
        peaks, properties = signal.find_peaks(
            autocorr[1:],  # Skip lag 0
            height=0.3 * self.sensitivity,
            distance=min_distance
        )
        
        if len(peaks) > 0:
            # Get the strongest periodicity
            peaks = peaks + 1  # Adjust for skipping lag 0
            strongest_peak_idx = np.argmax(properties['peak_heights'])
            period = peaks[strongest_peak_idx]
            strength = properties['peak_heights'][strongest_peak_idx]
            
            return {
                'detected': True,
                'period': period,
                'strength': strength,
                'frequency': 1.0 / period if period > 0 else 0,
                'description': f'Periodic pattern with period of {period} time units'
            }
        
        return {'detected': False}
    
    def _detect_anomalies(
        self,
        decisions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect anomalous spikes or outliers."""
        if len(decisions) < self.min_pattern_length:
            return {'detected': False, 'anomalies': []}
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(decisions))
        
        # Find anomalies
        anomaly_mask = z_scores > self.anomaly_threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
        anomalies = []
        for idx in anomaly_indices:
            anomalies.append({
                'index': idx,
                'value': decisions[idx],
                'z_score': z_scores[idx],
                'timestamp': timestamps[idx],
                'type': 'spike' if decisions[idx] > np.mean(decisions) else 'dip'
            })
        
        # Sort by z-score
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)
        
        return {
            'detected': len(anomalies) > 0,
            'anomalies': anomalies[:10],  # Top 10 anomalies
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(decisions)
        }
    
    def _detect_group_divergence(
        self,
        decisions: np.ndarray,
        groups: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Detect divergence patterns between groups over time."""
        unique_groups = np.unique(groups)
        
        if len(unique_groups) < 2:
            return {'detected': False, 'reason': 'single_group'}
        
        # Calculate metrics for each group over time
        window_size = max(self.min_pattern_length, len(decisions) // 10)
        n_windows = len(decisions) // window_size
        
        if n_windows < 2:
            return {'detected': False, 'reason': 'insufficient_windows'}
        
        group_trajectories = defaultdict(list)
        
        for window_idx in range(n_windows):
            start = window_idx * window_size
            end = min((window_idx + 1) * window_size, len(decisions))
            
            window_decisions = decisions[start:end]
            window_groups = groups[start:end]
            
            for group in unique_groups:
                group_mask = window_groups == group
                if np.sum(group_mask) > 0:
                    group_rate = np.mean(window_decisions[group_mask])
                    group_trajectories[group].append(group_rate)
        
        # Calculate divergence metrics
        trajectories = list(group_trajectories.values())
        
        if len(trajectories) < 2:
            return {'detected': False, 'reason': 'insufficient_group_data'}
        
        # Calculate pairwise divergence
        max_divergence = 0
        diverging_pair = None
        
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                traj1 = np.array(trajectories[i])
                traj2 = np.array(trajectories[j])
                
                # Calculate divergence as increasing difference over time
                differences = np.abs(traj1 - traj2)
                
                if len(differences) > 1:
                    # Check if differences are increasing
                    z = np.polyfit(range(len(differences)), differences, 1)
                    divergence_rate = z[0]
                    
                    if divergence_rate > max_divergence:
                        max_divergence = divergence_rate
                        diverging_pair = (i, j)
        
        divergence_detected = max_divergence > (0.01 * self.sensitivity)
        
        return {
            'detected': divergence_detected,
            'divergence_rate': max_divergence,
            'diverging_groups': diverging_pair,
            'description': 'Groups showing increasing disparity over time' if divergence_detected else None
        }
    
    def _calculate_bias_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate overall bias score from detected patterns."""
        score = 0.0
        weights = {
            'confidence_valley': 0.25,
            'sudden_shift': 0.20,
            'gradual_drift': 0.20,
            'periodic': 0.10,
            'anomalies': 0.10,
            'group_divergence': 0.15
        }
        
        for pattern_type, weight in weights.items():
            if patterns[pattern_type].get('detected', False):
                # Calculate pattern-specific contribution
                if pattern_type == 'confidence_valley':
                    contribution = patterns[pattern_type].get('confidence', 0.5)
                elif pattern_type == 'sudden_shift':
                    contribution = min(1.0, patterns[pattern_type]['total_shifts'] / 3)
                elif pattern_type == 'gradual_drift':
                    contribution = abs(patterns[pattern_type].get('r_squared', 0))
                elif pattern_type == 'periodic':
                    contribution = patterns[pattern_type].get('strength', 0.5)
                elif pattern_type == 'anomalies':
                    contribution = min(1.0, patterns[pattern_type]['anomaly_rate'] * 10)
                elif pattern_type == 'group_divergence':
                    contribution = min(1.0, patterns[pattern_type]['divergence_rate'] * 10)
                else:
                    contribution = 0.5
                
                score += weight * contribution
        
        return min(1.0, score)
    
    def _classify_severity(self, bias_score: float) -> str:
        """Classify bias severity based on score."""
        if bias_score < 0.2:
            return 'minimal'
        elif bias_score < 0.4:
            return 'low'
        elif bias_score < 0.6:
            return 'moderate'
        elif bias_score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on detected patterns."""
        recommendations = []
        
        if patterns['confidence_valley'].get('detected'):
            recommendations.append(
                "Investigate model confidence calibration - fairness degrades in middle periods"
            )
            recommendations.append(
                "Consider retraining with balanced data from valley period"
            )
        
        if patterns['sudden_shift'].get('detected'):
            shifts = patterns['sudden_shift']['shifts']
            if shifts:
                recommendations.append(
                    f"Review system changes around {shifts[0]['timestamp']} - sudden bias shift detected"
                )
        
        if patterns['gradual_drift'].get('detected'):
            direction = patterns['gradual_drift']['direction']
            recommendations.append(
                f"Implement drift monitoring - gradual {direction} trend detected"
            )
            recommendations.append(
                "Consider periodic model retraining to combat drift"
            )
        
        if patterns['periodic'].get('detected'):
            period = patterns['periodic']['period']
            recommendations.append(
                f"Investigate periodic factors with {period} time unit cycle"
            )
        
        if patterns['anomalies'].get('detected'):
            rate = patterns['anomalies']['anomaly_rate']
            if rate > 0.05:
                recommendations.append(
                    "High anomaly rate detected - review data quality and edge cases"
                )
        
        if patterns['group_divergence'].get('detected'):
            recommendations.append(
                "Groups showing increasing disparity - implement group-specific monitoring"
            )
            recommendations.append(
                "Consider fairness-aware retraining to reduce group divergence"
            )
        
        if not recommendations:
            recommendations.append("No significant bias patterns detected - maintain regular monitoring")
        
        return recommendations
    
    def _generate_summary(self, patterns: Dict[str, Any]) -> str:
        """Generate human-readable summary of detected patterns."""
        detected_patterns = []
        
        for pattern_type, result in patterns.items():
            if result.get('detected', False):
                pattern_name = pattern_type.replace('_', ' ').title()
                detected_patterns.append(pattern_name)
        
        if not detected_patterns:
            return "No significant temporal bias patterns detected."
        
        summary = f"Detected {len(detected_patterns)} bias pattern(s): "
        summary += ", ".join(detected_patterns)
        
        # Add most critical finding
        if patterns['confidence_valley'].get('detected'):
            summary += ". PRIMARY CONCERN: Confidence valley indicates model uncertainty in middle periods."
        elif patterns['group_divergence'].get('detected'):
            summary += ". PRIMARY CONCERN: Groups showing increasing disparity over time."
        elif patterns['sudden_shift'].get('detected'):
            shifts = patterns['sudden_shift']['total_shifts']
            summary += f". PRIMARY CONCERN: {shifts} sudden bias shifts detected."
        
        return summary
    
    def detect_complex_patterns(
        self,
        metric_history: np.ndarray,
        pattern_library: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Detect complex patterns using pattern matching and ML techniques.
        
        Args:
            metric_history: Historical metric values
            pattern_library: Optional library of known patterns to match
            
        Returns:
            Dictionary with complex pattern detection results
        """
        if len(metric_history) < self.min_pattern_length * 2:
            return {'detected': False, 'reason': 'insufficient_data'}
        
        results = {}
        
        # Normalize data
        scaler = StandardScaler()
        normalized = scaler.fit_transform(metric_history.reshape(-1, 1)).flatten()
        
        # Detect double valley (W-shape)
        valleys = self._detect_double_valley(normalized)
        if valleys['detected']:
            results['double_valley'] = valleys
        
        # Detect plateau patterns
        plateaus = self._detect_plateaus(normalized)
        if plateaus['detected']:
            results['plateaus'] = plateaus
        
        # Detect oscillating degradation
        osc_degradation = self._detect_oscillating_degradation(normalized)
        if osc_degradation['detected']:
            results['oscillating_degradation'] = osc_degradation
        
        # Pattern matching with library
        if pattern_library:
            matches = self._match_patterns(normalized, pattern_library)
            if matches:
                results['pattern_matches'] = matches
        
        return {
            'complex_patterns': results,
            'detected': len(results) > 0
        }
    
    def _detect_double_valley(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect W-shaped double valley pattern."""
        # Smooth data
        window = max(3, len(data) // 15)
        if window % 2 == 0:
            window += 1
        smoothed = signal.savgol_filter(data, window, 2)
        
        # Find valleys
        valleys, valley_props = signal.find_peaks(-smoothed, prominence=0.3)
        
        if len(valleys) >= 2:
            # Check if valleys are roughly equal depth
            prominences = valley_props['prominences']
            if len(prominences) >= 2:
                depth_ratio = min(prominences[:2]) / max(prominences[:2])
                
                if depth_ratio > 0.6:  # Valleys are similar depth
                    return {
                        'detected': True,
                        'valley_positions': valleys[:2].tolist(),
                        'depth_ratio': depth_ratio,
                        'description': 'Double valley (W-shape) pattern detected'
                    }
        
        return {'detected': False}
    
    def _detect_plateaus(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect plateau patterns (flat regions)."""
        # Calculate local variance
        window_size = max(3, len(data) // 10)
        local_vars = []
        
        for i in range(len(data) - window_size):
            window_var = np.var(data[i:i+window_size])
            local_vars.append(window_var)
        
        local_vars = np.array(local_vars)
        
        # Find low variance regions (plateaus)
        threshold = np.percentile(local_vars, 20)
        plateau_mask = local_vars < threshold
        
        # Find continuous plateau regions
        plateaus = []
        start = None
        
        for i, is_plateau in enumerate(plateau_mask):
            if is_plateau and start is None:
                start = i
            elif not is_plateau and start is not None:
                if i - start >= self.min_pattern_length:
                    plateaus.append((start, i))
                start = None
        
        if plateaus:
            return {
                'detected': True,
                'plateaus': plateaus,
                'total_plateaus': len(plateaus),
                'description': f'Detected {len(plateaus)} plateau region(s)'
            }
        
        return {'detected': False}
    
    def _detect_oscillating_degradation(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect oscillating pattern with overall degradation."""
        # Fit linear trend
        x = np.arange(len(data))
        z = np.polyfit(x, data, 1)
        trend = np.polyval(z, x)
        
        # Remove trend
        detrended = data - trend
        
        # Count oscillations (zero crossings)
        zero_crossings = np.where(np.diff(np.sign(detrended)))[0]
        
        # Check for degradation with oscillation
        if len(zero_crossings) >= 3 and z[0] < -0.01:  # Negative slope
            oscillation_rate = len(zero_crossings) / len(data)
            
            return {
                'detected': True,
                'degradation_rate': z[0],
                'oscillations': len(zero_crossings),
                'oscillation_rate': oscillation_rate,
                'description': 'Oscillating pattern with overall degradation'
            }
        
        return {'detected': False}
    
    def _match_patterns(
        self,
        data: np.ndarray,
        pattern_library: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Match data against known pattern library."""
        matches = []
        
        for pattern_name, pattern_template in pattern_library.items():
            # Resize pattern to match data length if needed
            if len(pattern_template) != len(data):
                pattern_resized = np.interp(
                    np.linspace(0, 1, len(data)),
                    np.linspace(0, 1, len(pattern_template)),
                    pattern_template
                )
            else:
                pattern_resized = pattern_template
            
            # Calculate correlation
            correlation = np.corrcoef(data, pattern_resized)[0, 1]
            
            if correlation > 0.7 * self.sensitivity:
                matches.append({
                    'pattern': pattern_name,
                    'correlation': correlation,
                    'confidence': abs(correlation)
                })
        
        # Sort by correlation
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches[:3]  # Top 3 matches