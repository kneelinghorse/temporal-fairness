"""
Research-Validated Bias Taxonomy Classifier
Identifies 4 categories of temporal bias based on empirical research
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
from scipy import stats
from collections import defaultdict


class BiasCategory(Enum):
    HISTORICAL = "historical_bias"
    REPRESENTATION = "representation_bias"
    MEASUREMENT = "measurement_bias"
    AGGREGATION = "aggregation_bias"


@dataclass
class BiasDetection:
    category: BiasCategory
    confidence: float
    indicators: List[str]
    severity: str
    mitigation_recommendations: List[str]
    evidence: Dict[str, Any]


@dataclass
class ClassificationResult:
    detections: List[BiasDetection]
    overall_risk: str
    processing_time_ms: float
    primary_bias: Optional[BiasCategory]
    interaction_effects: List[str]


class BiasTaxonomyClassifier:
    def __init__(self):
        self.detection_thresholds = {
            BiasCategory.HISTORICAL: 0.6,
            BiasCategory.REPRESENTATION: 0.6,
            BiasCategory.MEASUREMENT: 0.6,
            BiasCategory.AGGREGATION: 0.5
        }
        
        self.mitigation_strategies = {
            BiasCategory.HISTORICAL: [
                "Apply temporal decay to reduce weight of older discriminatory data",
                "Reweight historical samples to match current demographics",
                "Generate synthetic fair historical counterfactuals",
                "Implement sliding window to limit historical influence"
            ],
            BiasCategory.REPRESENTATION: [
                "Remove features with high correlation to protected attributes",
                "Apply adversarial debiasing during training",
                "Add explicit fairness constraints to optimization",
                "Use feature importance analysis to identify proxies"
            ],
            BiasCategory.MEASUREMENT: [
                "Recalibrate scores to achieve group parity",
                "Implement multi-rater systems with diverse evaluators",
                "Replace subjective metrics with objective measures",
                "Apply group-specific calibration functions"
            ],
            BiasCategory.AGGREGATION: [
                "Randomize batch composition to prevent clustering",
                "Implement fair scheduling with proportional representation",
                "Break feedback loops with periodic resets",
                "Monitor and limit cascade effects"
            ]
        }
    
    def classify(self, 
                 data: np.ndarray,
                 temporal_metadata: Optional[Dict] = None,
                 feature_names: Optional[List[str]] = None,
                 protected_attributes: Optional[Dict[str, np.ndarray]] = None) -> ClassificationResult:
        """
        Main classification method for detecting bias categories
        
        Args:
            data: Input data matrix (samples x features)
            temporal_metadata: Time-related information (timestamps, periods, etc.)
            feature_names: Names of features for interpretation
            protected_attributes: Protected attribute values for correlation analysis
        
        Returns:
            ClassificationResult with multi-label classifications and recommendations
        """
        start_time = time.perf_counter()
        
        detections = []
        
        # Detect Historical Bias
        historical_result = self._detect_historical_bias(data, temporal_metadata)
        if historical_result:
            detections.append(historical_result)
        
        # Detect Representation Bias
        representation_result = self._detect_representation_bias(
            data, feature_names, protected_attributes
        )
        if representation_result:
            detections.append(representation_result)
        
        # Detect Measurement Bias
        measurement_result = self._detect_measurement_bias(
            data, protected_attributes
        )
        if measurement_result:
            detections.append(measurement_result)
        
        # Detect Aggregation Bias
        aggregation_result = self._detect_aggregation_bias(
            data, temporal_metadata
        )
        if aggregation_result:
            detections.append(aggregation_result)
        
        # Analyze interaction effects
        interaction_effects = self._analyze_interactions(detections)
        
        # Determine overall risk and primary bias
        overall_risk = self._calculate_overall_risk(detections)
        primary_bias = self._identify_primary_bias(detections)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ClassificationResult(
            detections=detections,
            overall_risk=overall_risk,
            processing_time_ms=processing_time,
            primary_bias=primary_bias,
            interaction_effects=interaction_effects
        )
    
    def _detect_historical_bias(self, 
                                data: np.ndarray,
                                temporal_metadata: Optional[Dict]) -> Optional[BiasDetection]:
        """Detect patterns of historical discrimination in data"""
        
        indicators = []
        evidence = {}
        confidence = 0.0
        
        # Check for temporal patterns in the data itself (gradient effects)
        if data.shape[0] > 100:
            # Look for systematic trends over samples
            for feature_idx in range(min(5, data.shape[1])):
                feature_values = data[:, feature_idx]
                # Check if there's a linear trend
                x = np.arange(len(feature_values))
                correlation = np.corrcoef(x, feature_values)[0, 1]
                
                if abs(correlation) > 0.3:
                    indicators.append(f"Temporal trend in feature {feature_idx}")
                    confidence += 0.25
                    evidence[f'feature_{feature_idx}_trend'] = float(correlation)
        
        if temporal_metadata and 'timestamps' in temporal_metadata:
            timestamps = temporal_metadata['timestamps']
            
            # Check for temporal correlation patterns
            if len(timestamps) > 100:
                # Analyze temporal drift
                early_period = data[:len(data)//3]
                recent_period = data[-len(data)//3:]
                
                drift_score = np.mean(np.abs(
                    np.mean(early_period, axis=0) - np.mean(recent_period, axis=0)
                ))
                
                if drift_score > 0.05:
                    indicators.append("Significant temporal drift detected")
                    confidence += 0.3
                    evidence['temporal_drift'] = float(drift_score)
            
            # Check for compounding effects
            if 'outcomes' in temporal_metadata:
                outcomes = temporal_metadata['outcomes']
                
                # Calculate autocorrelation
                if len(outcomes) > 50:
                    autocorr = np.corrcoef(outcomes[:-1], outcomes[1:])[0, 1]
                    if abs(autocorr) > 0.7:
                        indicators.append("Strong temporal autocorrelation")
                        confidence += 0.3
                        evidence['autocorrelation'] = float(autocorr)
        
        # Check for legacy patterns
        if data.shape[0] > 200:
            # Look for persistent patterns
            variance_by_time = []
            window_size = max(10, len(data) // 20)
            
            for i in range(0, len(data) - window_size, window_size):
                window = data[i:i + window_size]
                variance_by_time.append(np.var(window))
            
            if len(variance_by_time) > 2:
                variance_trend = np.polyfit(range(len(variance_by_time)), variance_by_time, 1)[0]
                
                if abs(variance_trend) > 0.1:
                    indicators.append("Persistent historical patterns")
                    confidence += 0.35
                    evidence['variance_trend'] = float(variance_trend)
        
        if confidence >= self.detection_thresholds[BiasCategory.HISTORICAL]:
            severity = self._calculate_severity(confidence)
            
            return BiasDetection(
                category=BiasCategory.HISTORICAL,
                confidence=min(confidence, 1.0),
                indicators=indicators,
                severity=severity,
                mitigation_recommendations=self.mitigation_strategies[BiasCategory.HISTORICAL][:2],
                evidence=evidence
            )
        
        return None
    
    def _detect_representation_bias(self,
                                   data: np.ndarray,
                                   feature_names: Optional[List[str]],
                                   protected_attributes: Optional[Dict]) -> Optional[BiasDetection]:
        """Detect systematic correlation with protected attributes"""
        
        indicators = []
        evidence = {}
        confidence = 0.0
        
        if protected_attributes:
            for attr_name, attr_values in protected_attributes.items():
                if len(attr_values) != len(data):
                    continue
                
                # Calculate correlation with each feature
                high_correlation_features = []
                
                for feature_idx in range(data.shape[1]):
                    try:
                        correlation = np.corrcoef(data[:, feature_idx], attr_values)[0, 1]
                        
                        if abs(correlation) > 0.4:
                            feature_name = (feature_names[feature_idx] 
                                          if feature_names and feature_idx < len(feature_names)
                                          else f"feature_{feature_idx}")
                            high_correlation_features.append((feature_name, correlation))
                            confidence += 0.15
                    except:
                        continue
                
                if high_correlation_features:
                    indicators.append(f"Features correlate with {attr_name}")
                    evidence[f'{attr_name}_correlations'] = high_correlation_features[:5]
        
        # Check for proxy detection patterns
        if data.shape[1] > 5:
            # Analyze feature interactions
            feature_covariance = np.cov(data.T)
            
            # Look for clusters of correlated features (potential proxies)
            high_covar_count = np.sum(np.abs(feature_covariance) > 0.7)
            
            if high_covar_count > data.shape[1] * 2:
                indicators.append("Detected potential proxy features")
                confidence += 0.3
                evidence['high_covariance_pairs'] = int(high_covar_count)
        
        # Check for multivariate patterns
        if data.shape[1] > 10 and data.shape[0] > 100:
            # Simple PCA to check for systematic variation
            centered_data = data - np.mean(data, axis=0)
            cov_matrix = np.cov(centered_data.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            
            # Check concentration of variance
            variance_ratio = eigenvalues[-3:].sum() / eigenvalues.sum()
            
            if variance_ratio > 0.8:
                indicators.append("High concentration in few principal components")
                confidence += 0.25
                evidence['top_3_variance_ratio'] = float(variance_ratio)
        
        if confidence >= self.detection_thresholds[BiasCategory.REPRESENTATION]:
            severity = self._calculate_severity(confidence)
            
            return BiasDetection(
                category=BiasCategory.REPRESENTATION,
                confidence=min(confidence, 1.0),
                indicators=indicators,
                severity=severity,
                mitigation_recommendations=self.mitigation_strategies[BiasCategory.REPRESENTATION][:2],
                evidence=evidence
            )
        
        return None
    
    def _detect_measurement_bias(self,
                                data: np.ndarray,
                                protected_attributes: Optional[Dict]) -> Optional[BiasDetection]:
        """Detect systematic measurement differences across groups"""
        
        indicators = []
        evidence = {}
        confidence = 0.0
        
        # Check for systematic scaling differences in specific features
        if data.shape[0] > 100:
            # Look for features with bimodal or multimodal distributions
            for feature_idx in range(min(10, data.shape[1])):
                feature_values = data[:, feature_idx]
                # Simple check for systematic differences
                upper_half = feature_values > np.median(feature_values)
                lower_half = ~upper_half
                
                if np.sum(upper_half) > 20 and np.sum(lower_half) > 20:
                    upper_mean = np.mean(feature_values[upper_half])
                    lower_mean = np.mean(feature_values[lower_half])
                    
                    ratio = upper_mean / (lower_mean + 1e-10)
                    if ratio > 1.5 or ratio < 0.7:
                        indicators.append(f"Systematic scaling in feature {feature_idx}")
                        confidence += 0.2
                        evidence[f'feature_{feature_idx}_scaling'] = float(ratio)
        
        if protected_attributes:
            for attr_name, attr_values in protected_attributes.items():
                unique_groups = np.unique(attr_values)
                
                if len(unique_groups) < 2 or len(unique_groups) > 10:
                    continue
                
                # Analyze distribution differences
                group_distributions = []
                
                for group in unique_groups:
                    group_mask = attr_values == group
                    if np.sum(group_mask) < 10:
                        continue
                    
                    group_data = data[group_mask]
                    group_distributions.append({
                        'mean': np.mean(group_data, axis=0),
                        'std': np.std(group_data, axis=0),
                        'size': len(group_data)
                    })
                
                if len(group_distributions) >= 2:
                    # Check for calibration differences
                    means_variance = np.var([g['mean'] for g in group_distributions], axis=0)
                    
                    significant_differences = np.sum(means_variance > 0.1)
                    
                    if significant_differences > data.shape[1] * 0.15:
                        indicators.append(f"Calibration differences across {attr_name}")
                        confidence += 0.35
                        evidence[f'{attr_name}_calibration_diff'] = int(significant_differences)
                    
                    # Check for systematic score differences
                    if data.shape[0] > 50:
                        # Perform simple statistical test
                        group_scores = []
                        for group in unique_groups:
                            group_mask = attr_values == group
                            if np.sum(group_mask) >= 10:
                                group_scores.append(np.mean(data[group_mask]))
                        
                        if len(group_scores) >= 2:
                            score_variance = np.var(group_scores)
                            if score_variance > 0.1:
                                indicators.append("Systematic scoring differences")
                                confidence += 0.35
                                evidence['score_variance'] = float(score_variance)
        
        # Check for missing data patterns
        if hasattr(data, 'mask') or np.any(np.isnan(data)):
            missing_pattern = np.isnan(data) if np.any(np.isnan(data)) else data.mask
            missing_rate = np.mean(missing_pattern, axis=0)
            
            if np.any(missing_rate > 0.1):
                indicators.append("Differential missing data patterns")
                confidence += 0.25
                evidence['max_missing_rate'] = float(np.max(missing_rate))
        
        if confidence >= self.detection_thresholds[BiasCategory.MEASUREMENT]:
            severity = self._calculate_severity(confidence)
            
            return BiasDetection(
                category=BiasCategory.MEASUREMENT,
                confidence=min(confidence, 1.0),
                indicators=indicators,
                severity=severity,
                mitigation_recommendations=self.mitigation_strategies[BiasCategory.MEASUREMENT][:2],
                evidence=evidence
            )
        
        return None
    
    def _detect_aggregation_bias(self,
                                data: np.ndarray,
                                temporal_metadata: Optional[Dict]) -> Optional[BiasDetection]:
        """Detect bias from batch processing and temporal aggregation"""
        
        indicators = []
        evidence = {}
        confidence = 0.0
        
        # Check for batch effects
        if temporal_metadata and 'batch_ids' in temporal_metadata:
            batch_ids = temporal_metadata['batch_ids']
            unique_batches = np.unique(batch_ids)
            
            if len(unique_batches) > 2:
                batch_means = []
                batch_sizes = []
                
                for batch in unique_batches:
                    batch_mask = batch_ids == batch
                    batch_size = np.sum(batch_mask)
                    
                    if batch_size >= 5:
                        batch_means.append(np.mean(data[batch_mask]))
                        batch_sizes.append(batch_size)
                
                if len(batch_means) > 2:
                    # Check for systematic batch differences
                    batch_variance = np.var(batch_means)
                    
                    if batch_variance > 0.15:
                        indicators.append("Significant batch effects detected")
                        confidence += 0.4
                        evidence['batch_variance'] = float(batch_variance)
                    
                    # Check for size imbalance
                    size_ratio = max(batch_sizes) / min(batch_sizes)
                    if size_ratio > 3:
                        indicators.append("Imbalanced batch sizes")
                        confidence += 0.25
                        evidence['max_size_ratio'] = float(size_ratio)
        
        # Check for feedback loops
        if data.shape[0] > 100:
            # Analyze temporal clustering
            window_size = max(10, len(data) // 10)
            window_variances = []
            
            for i in range(0, len(data) - window_size, window_size // 2):
                window = data[i:i + window_size]
                window_variances.append(np.mean(np.var(window, axis=0)))
            
            if len(window_variances) > 3:
                # Check for increasing variance (sign of feedback loops)
                variance_trend = np.polyfit(
                    range(len(window_variances)), 
                    window_variances, 
                    1
                )[0]
                
                if variance_trend > 0.02:
                    indicators.append("Potential feedback loop detected")
                    confidence += 0.35
                    evidence['variance_trend'] = float(variance_trend)
        
        # Check for temporal clustering
        if temporal_metadata and 'timestamps' in temporal_metadata:
            timestamps = np.array(temporal_metadata['timestamps'])
            
            if len(timestamps) > 50:
                # Calculate time gaps
                time_diffs = np.diff(sorted(timestamps))
                
                # Look for clustering patterns
                if len(time_diffs) > 0:
                    clustering_score = np.std(time_diffs) / (np.mean(time_diffs) + 1e-10)
                    
                    if clustering_score > 2:
                        indicators.append("Temporal clustering patterns")
                        confidence += 0.3
                        evidence['clustering_score'] = float(clustering_score)
        
        if confidence >= self.detection_thresholds[BiasCategory.AGGREGATION]:
            severity = self._calculate_severity(confidence)
            
            return BiasDetection(
                category=BiasCategory.AGGREGATION,
                confidence=min(confidence, 1.0),
                indicators=indicators,
                severity=severity,
                mitigation_recommendations=self.mitigation_strategies[BiasCategory.AGGREGATION][:2],
                evidence=evidence
            )
        
        return None
    
    def _analyze_interactions(self, detections: List[BiasDetection]) -> List[str]:
        """Analyze interaction effects between detected bias categories"""
        
        interactions = []
        
        if len(detections) < 2:
            return interactions
        
        categories = {d.category for d in detections}
        
        # Known interaction patterns
        if BiasCategory.HISTORICAL in categories and BiasCategory.REPRESENTATION in categories:
            interactions.append("Historical bias feeding into representation bias")
        
        if BiasCategory.MEASUREMENT in categories and BiasCategory.AGGREGATION in categories:
            interactions.append("Measurement bias amplified through aggregation")
        
        if BiasCategory.REPRESENTATION in categories and BiasCategory.MEASUREMENT in categories:
            interactions.append("Compounded feature and measurement discrimination")
        
        if len(categories) >= 3:
            interactions.append("Multiple bias categories creating cascade effects")
        
        if len(categories) == 4:
            interactions.append("Critical: All bias categories present - system-wide discrimination")
        
        return interactions
    
    def _calculate_overall_risk(self, detections: List[BiasDetection]) -> str:
        """Calculate overall bias risk level"""
        
        if not detections:
            return "LOW"
        
        # Weight by confidence and severity
        risk_score = 0.0
        
        severity_weights = {
            "CRITICAL": 1.0,
            "HIGH": 0.75,
            "MEDIUM": 0.5,
            "LOW": 0.25
        }
        
        for detection in detections:
            weight = severity_weights.get(detection.severity, 0.5)
            risk_score += detection.confidence * weight
        
        # Account for interaction effects
        if len(detections) > 1:
            risk_score *= (1 + 0.2 * (len(detections) - 1))
        
        # Normalize
        risk_score = risk_score / max(len(detections), 1)
        
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_primary_bias(self, detections: List[BiasDetection]) -> Optional[BiasCategory]:
        """Identify the primary bias category"""
        
        if not detections:
            return None
        
        # Sort by confidence and severity
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        sorted_detections = sorted(
            detections,
            key=lambda d: (severity_order.get(d.severity, 0), d.confidence),
            reverse=True
        )
        
        return sorted_detections[0].category if sorted_detections else None
    
    def _calculate_severity(self, confidence: float) -> str:
        """Calculate severity based on confidence score"""
        
        if confidence >= 0.9:
            return "CRITICAL"
        elif confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def validate_case_study(self, case_name: str, data: np.ndarray, 
                           expected_categories: List[BiasCategory],
                           **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate classifier against known case studies
        
        Args:
            case_name: Name of the case study (e.g., "Optum", "MiDAS")
            data: Case study data
            expected_categories: Expected bias categories
            **kwargs: Additional metadata
        
        Returns:
            Tuple of (success, validation_details)
        """
        
        result = self.classify(data, **kwargs)
        
        detected_categories = {d.category for d in result.detections}
        expected_set = set(expected_categories)
        
        # Check if all expected categories were detected
        correctly_detected = expected_set.intersection(detected_categories)
        missed = expected_set - detected_categories
        false_positives = detected_categories - expected_set
        
        # Calculate accuracy
        true_positives = len(correctly_detected)
        false_negatives = len(missed)
        accuracy = true_positives / len(expected_set) if expected_set else 0
        
        # Check confidence levels
        high_confidence = all(
            d.confidence >= 0.9 
            for d in result.detections 
            if d.category in expected_set
        )
        
        success = accuracy >= 0.9 and high_confidence
        
        validation_details = {
            'case_name': case_name,
            'accuracy': accuracy,
            'correctly_detected': list(correctly_detected),
            'missed': list(missed),
            'false_positives': list(false_positives),
            'processing_time_ms': result.processing_time_ms,
            'high_confidence': high_confidence,
            'success': success
        }
        
        return success, validation_details