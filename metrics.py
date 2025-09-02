"""
Temporal Fairness Metrics

Four novel metrics for detecting time-based discrimination in AI systems.
Based on research showing 53% of fairness violations occur in temporal ordering systems.

Author: [Your Name]
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class FairnessResult:
    """Container for fairness metric results"""
    metric_name: str
    value: float
    threshold: float
    is_fair: bool
    details: Dict[str, Any]
    
    def __str__(self):
        status = "✓ FAIR" if self.is_fair else "✗ BIASED"
        return f"{self.metric_name}: {self.value:.3f} [{status}]"


class TemporalFairnessMetrics:
    """
    Temporal Fairness Metrics for detecting bias in time-sensitive AI systems.
    
    These metrics catch discrimination that traditional fairness metrics miss,
    particularly in queue-based systems, urgency scoring, and sequential decisions.
    """
    
    def __init__(self, fairness_threshold: float = 0.1):
        """
        Initialize metrics calculator.
        
        Args:
            fairness_threshold: Maximum acceptable deviation (default 0.1 = 10%)
        """
        self.threshold = fairness_threshold
    
    def tdp(self, 
            decisions: np.ndarray, 
            groups: np.ndarray, 
            timestamps: np.ndarray,
            window_size: int = 3600) -> FairnessResult:
        """
        Temporal Demographic Parity (TDP)
        
        Measures whether different groups receive positive decisions at similar rates
        within specific time windows. Catches bias that emerges over time.
        
        Formula: TDP(t) = |P(decision=1|group=A,t) - P(decision=1|group=B,t)|
        
        Args:
            decisions: Binary decisions (0 or 1)
            groups: Group membership (0 or 1 for binary, multiple values for multi-group)
            timestamps: Unix timestamps of decisions
            window_size: Time window in seconds (default 1 hour)
            
        Returns:
            FairnessResult with TDP value and fairness determination
        """
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return FairnessResult("TDP", 0.0, self.threshold, True, 
                                {"message": "Single group - no comparison needed"})
        
        # Calculate rates within time windows
        time_min, time_max = timestamps.min(), timestamps.max()
        n_windows = int((time_max - time_min) / window_size) + 1
        
        max_disparity = 0.0
        window_disparities = []
        
        for window in range(n_windows):
            window_start = time_min + window * window_size
            window_end = window_start + window_size
            window_mask = (timestamps >= window_start) & (timestamps < window_end)
            
            if not window_mask.any():
                continue
            
            rates = {}
            for group in unique_groups:
                group_mask = (groups == group) & window_mask
                if group_mask.any():
                    rates[group] = decisions[group_mask].mean()
            
            if len(rates) >= 2:
                disparity = max(rates.values()) - min(rates.values())
                window_disparities.append(disparity)
                max_disparity = max(max_disparity, disparity)
        
        avg_disparity = np.mean(window_disparities) if window_disparities else 0.0
        
        return FairnessResult(
            metric_name="Temporal Demographic Parity",
            value=avg_disparity,
            threshold=self.threshold,
            is_fair=avg_disparity <= self.threshold,
            details={
                "max_disparity": max_disparity,
                "n_windows": n_windows,
                "window_size_seconds": window_size
            }
        )
    
    def eoot(self,
             predictions: np.ndarray,
             labels: np.ndarray,
             groups: np.ndarray,
             timestamps: np.ndarray,
             window_size: int = 86400) -> FairnessResult:
        """
        Equalized Odds Over Time (EOOT)
        
        Ensures that true positive rates and false positive rates remain equal
        across groups over time. Critical for maintaining consistent accuracy.
        
        Formula: Equal TPR and FPR across groups at each time interval [t-k, t+k]
        
        Args:
            predictions: Binary predictions
            labels: True labels
            groups: Group membership
            timestamps: Unix timestamps
            window_size: Time window in seconds (default 24 hours)
            
        Returns:
            FairnessResult with EOOT violation measure
        """
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return FairnessResult("EOOT", 0.0, self.threshold, True,
                                {"message": "Single group - no comparison needed"})
        
        time_min, time_max = timestamps.min(), timestamps.max()
        n_windows = int((time_max - time_min) / window_size) + 1
        
        tpr_disparities = []
        fpr_disparities = []
        
        for window in range(n_windows):
            window_start = time_min + window * window_size
            window_end = window_start + window_size
            window_mask = (timestamps >= window_start) & (timestamps < window_end)
            
            if not window_mask.any():
                continue
            
            tprs, fprs = {}, {}
            
            for group in unique_groups:
                group_mask = (groups == group) & window_mask
                if not group_mask.any():
                    continue
                
                group_preds = predictions[group_mask]
                group_labels = labels[group_mask]
                
                # Calculate TPR and FPR
                true_positives = ((group_preds == 1) & (group_labels == 1)).sum()
                false_positives = ((group_preds == 1) & (group_labels == 0)).sum()
                actual_positives = (group_labels == 1).sum()
                actual_negatives = (group_labels == 0).sum()
                
                if actual_positives > 0:
                    tprs[group] = true_positives / actual_positives
                if actual_negatives > 0:
                    fprs[group] = false_positives / actual_negatives
            
            if len(tprs) >= 2:
                tpr_disparities.append(max(tprs.values()) - min(tprs.values()))
            if len(fprs) >= 2:
                fpr_disparities.append(max(fprs.values()) - min(fprs.values()))
        
        avg_tpr_disparity = np.mean(tpr_disparities) if tpr_disparities else 0.0
        avg_fpr_disparity = np.mean(fpr_disparities) if fpr_disparities else 0.0
        max_disparity = max(avg_tpr_disparity, avg_fpr_disparity)
        
        return FairnessResult(
            metric_name="Equalized Odds Over Time",
            value=max_disparity,
            threshold=self.threshold,
            is_fair=max_disparity <= self.threshold,
            details={
                "tpr_disparity": avg_tpr_disparity,
                "fpr_disparity": avg_fpr_disparity,
                "n_windows": n_windows
            }
        )
    
    def fdd(self,
            fairness_scores: List[float],
            timestamps: List[datetime],
            decay_period_days: int = 180) -> FairnessResult:
        """
        Fairness Decay Detection (FDD)
        
        Monitors how fairness metrics degrade over time. Critical for detecting
        systems that start fair but become discriminatory as they learn.
        
        Formula: Linear regression slope of fairness metric over time period
        
        Args:
            fairness_scores: Historical fairness measurements (0=fair, 1=biased)
            timestamps: Timestamps of measurements
            decay_period_days: Period to check for decay (default 6 months)
            
        Returns:
            FairnessResult with decay rate
        """
        if len(fairness_scores) < 2:
            return FairnessResult("FDD", 0.0, self.threshold, True,
                                {"message": "Insufficient data for decay detection"})
        
        # Convert to numpy arrays
        scores = np.array(fairness_scores)
        times = np.array([(t - timestamps[0]).total_seconds() / 86400 
                         for t in timestamps])  # Days since start
        
        # Filter to decay period
        cutoff = max(times) - decay_period_days
        mask = times >= cutoff
        
        if mask.sum() < 2:
            return FairnessResult("FDD", 0.0, self.threshold, True,
                                {"message": "Insufficient recent data"})
        
        recent_times = times[mask]
        recent_scores = scores[mask]
        
        # Calculate decay rate using linear regression
        n = len(recent_times)
        x_mean = recent_times.mean()
        y_mean = recent_scores.mean()
        
        numerator = ((recent_times - x_mean) * (recent_scores - y_mean)).sum()
        denominator = ((recent_times - x_mean) ** 2).sum()
        
        if denominator == 0:
            decay_rate = 0.0
        else:
            decay_rate = numerator / denominator
        
        # Decay rate is per day, multiply by 30 for monthly rate
        monthly_decay = abs(decay_rate * 30)
        
        return FairnessResult(
            metric_name="Fairness Decay Detection",
            value=monthly_decay,
            threshold=self.threshold,
            is_fair=monthly_decay <= self.threshold,
            details={
                "daily_decay_rate": decay_rate,
                "period_days": decay_period_days,
                "n_measurements": n,
                "interpretation": "Points of bias increase per month"
            }
        )
    
    def qpf(self,
            queue_positions: np.ndarray,
            groups: np.ndarray,
            max_queue_size: Optional[int] = None) -> FairnessResult:
        """
        Queue Position Fairness (QPF)
        
        Measures systematic ordering bias in priority systems. Detects when
        certain groups consistently end up at the back of queues.
        
        Formula: QPF = |E[position|group=A] - E[position|group=B]| / max_queue_size
        
        Args:
            queue_positions: Position in queue (1 = front)
            groups: Group membership
            max_queue_size: Maximum queue size for normalization
            
        Returns:
            FairnessResult with QPF score
        """
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return FairnessResult("QPF", 0.0, self.threshold, True,
                                {"message": "Single group - no comparison needed"})
        
        if max_queue_size is None:
            max_queue_size = queue_positions.max()
        
        # Calculate average position per group
        avg_positions = {}
        for group in unique_groups:
            group_mask = groups == group
            avg_positions[group] = queue_positions[group_mask].mean()
        
        # Calculate normalized disparity
        max_diff = max(avg_positions.values()) - min(avg_positions.values())
        normalized_disparity = max_diff / max_queue_size
        
        # Identify which group is disadvantaged
        disadvantaged_group = max(avg_positions.items(), key=lambda x: x[1])[0]
        advantaged_group = min(avg_positions.items(), key=lambda x: x[1])[0]
        
        return FairnessResult(
            metric_name="Queue Position Fairness",
            value=normalized_disparity,
            threshold=self.threshold,
            is_fair=normalized_disparity <= self.threshold,
            details={
                "avg_positions": avg_positions,
                "max_queue_size": max_queue_size,
                "disadvantaged_group": disadvantaged_group,
                "advantaged_group": advantaged_group,
                "position_difference": max_diff
            }
        )
    
    def evaluate_all(self, data: Dict[str, np.ndarray]) -> Dict[str, FairnessResult]:
        """
        Run all applicable metrics on the provided data.
        
        Args:
            data: Dictionary with keys matching metric parameters
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # TDP
        if all(k in data for k in ['decisions', 'groups', 'timestamps']):
            results['TDP'] = self.tdp(
                data['decisions'], 
                data['groups'],
                data['timestamps']
            )
        
        # EOOT
        if all(k in data for k in ['predictions', 'labels', 'groups', 'timestamps']):
            results['EOOT'] = self.eoot(
                data['predictions'],
                data['labels'],
                data['groups'],
                data['timestamps']
            )
        
        # QPF
        if all(k in data for k in ['queue_positions', 'groups']):
            results['QPF'] = self.qpf(
                data['queue_positions'],
                data['groups']
            )
        
        return results
