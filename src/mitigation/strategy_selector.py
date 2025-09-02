"""
Context-Aware Mitigation Strategy Selector
Recommends optimal mitigation techniques based on bias type and system context
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

# Import bias classifier
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.bias_classifier import BiasCategory, BiasDetection


class MitigationTechnique(Enum):
    REWEIGHTING = "reweighting"
    ADVERSARIAL_DEBIASING = "adversarial_debiasing"
    POST_PROCESSING = "post_processing_optimization"
    FAIRNESS_BATCH_SAMPLING = "fairness_aware_batch_sampling"
    FAIRNESS_CONSTRAINTS = "fairness_constraints"
    CALIBRATION = "calibration"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"


class SystemContext(Enum):
    QUEUE_BASED = "queue_based_systems"
    URGENCY_SCORING = "urgency_scoring"
    SEQUENTIAL_DECISIONS = "sequential_decisions"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time_processing"
    HISTORICAL_DATA = "historical_data_processing"


@dataclass
class MitigationStrategy:
    technique: MitigationTechnique
    expected_effectiveness: float  # 0-1 scale
    accuracy_retention: float  # 0-1 scale
    implementation_complexity: str  # LOW, MEDIUM, HIGH
    latency_overhead_ms: float
    configuration: Dict[str, Any]
    rationale: str
    monitoring_metrics: List[str]


@dataclass
class StrategyRecommendation:
    primary_strategy: MitigationStrategy
    alternative_strategies: List[MitigationStrategy]
    combined_approach: Optional[Dict[str, Any]]
    expected_bias_reduction: float
    expected_accuracy_impact: float
    selection_time_ms: float
    confidence: float


@dataclass
class PerformanceMetrics:
    fairness_improvement: float
    accuracy_retention: float
    latency_ms: float
    throughput_per_second: int
    timestamp: float


class MitigationStrategySelector:
    def __init__(self):
        # Technique effectiveness rates from research
        self.technique_profiles = {
            MitigationTechnique.REWEIGHTING: {
                'success_rate': 0.77,
                'accuracy_retention': 0.90,  # 5-15% loss -> ~90% retention
                'latency_overhead': 1.0,
                'complexity': 'LOW',
                'best_contexts': [SystemContext.QUEUE_BASED, SystemContext.HISTORICAL_DATA],
                'bias_types': [BiasCategory.HISTORICAL, BiasCategory.MEASUREMENT]
            },
            MitigationTechnique.ADVERSARIAL_DEBIASING: {
                'success_rate': 0.92,
                'accuracy_retention': 0.945,  # 92-97% retention
                'latency_overhead': 5.0,
                'complexity': 'HIGH',
                'best_contexts': [SystemContext.URGENCY_SCORING, SystemContext.REAL_TIME],
                'bias_types': [BiasCategory.REPRESENTATION, BiasCategory.MEASUREMENT]
            },
            MitigationTechnique.POST_PROCESSING: {
                'success_rate': 0.65,
                'accuracy_retention': 0.95,
                'latency_overhead': 3.0,
                'complexity': 'MEDIUM',
                'best_contexts': [SystemContext.SEQUENTIAL_DECISIONS],
                'bias_types': [BiasCategory.MEASUREMENT, BiasCategory.AGGREGATION]
            },
            MitigationTechnique.FAIRNESS_BATCH_SAMPLING: {
                'success_rate': 0.93,
                'accuracy_retention': 0.98,
                'latency_overhead': 2.0,
                'complexity': 'LOW',
                'best_contexts': [SystemContext.BATCH_PROCESSING, SystemContext.QUEUE_BASED],
                'bias_types': [BiasCategory.AGGREGATION]
            }
        }
        
        # Context-specific optimization mappings
        self.context_optimization = {
            SystemContext.QUEUE_BASED: {
                'primary': MitigationTechnique.FAIRNESS_BATCH_SAMPLING,
                'secondary': MitigationTechnique.REWEIGHTING,
                'effectiveness_range': (0.6, 0.8),
                'config': {
                    'sampling_strategy': 'stratified',
                    'rebalance_frequency': 'per_batch',
                    'representation_target': 'proportional'
                }
            },
            SystemContext.URGENCY_SCORING: {
                'primary': MitigationTechnique.ADVERSARIAL_DEBIASING,
                'secondary': MitigationTechnique.POST_PROCESSING,
                'effectiveness_range': (0.8, 0.95),
                'config': {
                    'adversary_strength': 0.5,
                    'protected_attributes': 'auto_detect',
                    'continuous_features': True
                }
            },
            SystemContext.SEQUENTIAL_DECISIONS: {
                'primary': MitigationTechnique.POST_PROCESSING,
                'secondary': MitigationTechnique.REWEIGHTING,
                'effectiveness_range': (0.4, 0.7),
                'config': {
                    'threshold_optimization': 'bayesian',
                    'per_stage_adjustment': True,
                    'maintain_calibration': False
                }
            },
            SystemContext.BATCH_PROCESSING: {
                'primary': MitigationTechnique.FAIRNESS_BATCH_SAMPLING,
                'secondary': MitigationTechnique.REWEIGHTING,
                'effectiveness_range': (0.7, 0.93),
                'config': {
                    'batch_composition': 'stratified',
                    'fairness_constraint': 'demographic_parity',
                    'real_time_adjustment': True
                }
            }
        }
        
        # Performance monitoring
        self.performance_history = defaultdict(list)
        self.effectiveness_cache = {}
        
    def select_strategy(self,
                        bias_detections: List[BiasDetection],
                        system_context: SystemContext,
                        performance_requirements: Optional[Dict[str, Any]] = None,
                        data_characteristics: Optional[Dict[str, Any]] = None) -> StrategyRecommendation:
        """
        Select optimal mitigation strategy based on detected biases and context
        
        Args:
            bias_detections: List of detected biases from classifier
            system_context: The operational context of the system
            performance_requirements: Optional performance constraints
            data_characteristics: Optional data properties
        
        Returns:
            StrategyRecommendation with primary and alternative strategies
        """
        start_time = time.perf_counter()
        
        # Default performance requirements
        if performance_requirements is None:
            performance_requirements = {
                'min_accuracy_retention': 0.9,
                'max_latency_ms': 10,
                'min_throughput': 1000
            }
        
        # Analyze bias patterns
        bias_categories = {d.category for d in bias_detections}
        primary_bias = self._identify_primary_bias(bias_detections)
        
        # Get context-specific recommendations
        context_strategy = self._get_context_strategy(system_context)
        
        # Score techniques based on bias types and context
        technique_scores = self._score_techniques(
            bias_categories,
            system_context,
            performance_requirements
        )
        
        # Select primary strategy
        primary_technique = self._select_primary_technique(
            technique_scores,
            context_strategy,
            performance_requirements
        )
        
        primary_strategy = self._build_strategy(
            primary_technique,
            system_context,
            bias_detections
        )
        
        # Select alternative strategies
        alternative_strategies = self._select_alternatives(
            technique_scores,
            primary_technique,
            system_context
        )
        
        # Consider combined approach for multiple bias types
        combined_approach = None
        if len(bias_categories) > 2:
            combined_approach = self._design_combined_approach(
                bias_categories,
                system_context
            )
        
        # Calculate expected outcomes
        expected_bias_reduction = self._calculate_expected_reduction(
            primary_strategy,
            bias_detections
        )
        
        expected_accuracy_impact = 1.0 - primary_strategy.accuracy_retention
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate confidence based on context match and historical performance
        confidence = self._calculate_confidence(
            primary_technique,
            system_context,
            bias_categories
        )
        
        return StrategyRecommendation(
            primary_strategy=primary_strategy,
            alternative_strategies=alternative_strategies,
            combined_approach=combined_approach,
            expected_bias_reduction=expected_bias_reduction,
            expected_accuracy_impact=expected_accuracy_impact,
            selection_time_ms=processing_time,
            confidence=confidence
        )
    
    def _identify_primary_bias(self, detections: List[BiasDetection]) -> Optional[BiasCategory]:
        """Identify the most severe bias category"""
        if not detections:
            return None
        
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        sorted_detections = sorted(
            detections,
            key=lambda d: (severity_order.get(d.severity, 0), d.confidence),
            reverse=True
        )
        
        return sorted_detections[0].category if sorted_detections else None
    
    def _get_context_strategy(self, context: SystemContext) -> Dict[str, Any]:
        """Get context-specific strategy configuration"""
        return self.context_optimization.get(
            context,
            {
                'primary': MitigationTechnique.REWEIGHTING,
                'secondary': MitigationTechnique.POST_PROCESSING,
                'effectiveness_range': (0.5, 0.7),
                'config': {}
            }
        )
    
    def _score_techniques(self,
                         bias_categories: set,
                         context: SystemContext,
                         requirements: Dict) -> Dict[MitigationTechnique, float]:
        """Score each technique based on bias types and context"""
        scores = {}
        
        for technique, profile in self.technique_profiles.items():
            score = 0.0
            
            # Score based on bias type match
            matching_biases = len(bias_categories.intersection(set(profile['bias_types'])))
            score += matching_biases * 0.3
            
            # Score based on context match
            if context in profile['best_contexts']:
                score += 0.4
            
            # Score based on performance requirements
            if profile['accuracy_retention'] >= requirements['min_accuracy_retention']:
                score += 0.2
            
            if profile['latency_overhead'] <= requirements['max_latency_ms']:
                score += 0.1
            
            # Apply success rate as multiplier
            score *= profile['success_rate']
            
            scores[technique] = score
        
        return scores
    
    def _select_primary_technique(self,
                                 scores: Dict[MitigationTechnique, float],
                                 context_strategy: Dict,
                                 requirements: Dict) -> MitigationTechnique:
        """Select the primary mitigation technique"""
        
        # Check if context-recommended technique meets requirements
        context_primary = context_strategy.get('primary')
        if context_primary and scores.get(context_primary, 0) > 0.5:
            return context_primary
        
        # Otherwise select highest scoring technique
        if scores:
            return max(scores, key=scores.get)
        
        # Fallback to reweighting (most general)
        return MitigationTechnique.REWEIGHTING
    
    def _build_strategy(self,
                       technique: MitigationTechnique,
                       context: SystemContext,
                       detections: List[BiasDetection]) -> MitigationStrategy:
        """Build a complete mitigation strategy"""
        
        profile = self.technique_profiles[technique]
        context_config = self.context_optimization.get(context, {}).get('config', {})
        
        # Technique-specific configuration
        if technique == MitigationTechnique.REWEIGHTING:
            configuration = {
                'weight_formula': 'inverse_frequency',
                'normalization': 'group_size',
                'update_frequency': 'per_batch',
                **context_config
            }
            monitoring_metrics = [
                'group_weights',
                'effective_sample_size',
                'weight_variance'
            ]
            rationale = "Reweighting addresses historical and measurement bias by adjusting sample importance"
            
        elif technique == MitigationTechnique.ADVERSARIAL_DEBIASING:
            configuration = {
                'adversary_layers': [64, 32],
                'adversary_weight': 0.5,
                'gradient_reversal': True,
                'epochs': 100,
                **context_config
            }
            monitoring_metrics = [
                'adversary_accuracy',
                'predictor_accuracy',
                'representation_fairness'
            ]
            rationale = "Adversarial debiasing removes protected attribute information from learned representations"
            
        elif technique == MitigationTechnique.POST_PROCESSING:
            configuration = {
                'threshold_search': 'grid',
                'optimization_metric': 'equal_opportunity',
                'group_specific': True,
                **context_config
            }
            monitoring_metrics = [
                'threshold_values',
                'group_positive_rates',
                'calibration_error'
            ]
            rationale = "Post-processing optimization adjusts decision boundaries for fairness without retraining"
            
        elif technique == MitigationTechnique.FAIRNESS_BATCH_SAMPLING:
            configuration = {
                'sampling_method': 'stratified',
                'batch_size': 100,
                'representation_guarantee': 'proportional',
                'shuffle': True,
                **context_config
            }
            monitoring_metrics = [
                'batch_composition',
                'representation_deviation',
                'queue_fairness'
            ]
            rationale = "Fair batch sampling ensures proportional representation in processing queues"
            
        else:
            configuration = context_config
            monitoring_metrics = ['fairness_metric', 'accuracy']
            rationale = f"Selected {technique.value} based on context and bias patterns"
        
        return MitigationStrategy(
            technique=technique,
            expected_effectiveness=profile['success_rate'],
            accuracy_retention=profile['accuracy_retention'],
            implementation_complexity=profile['complexity'],
            latency_overhead_ms=profile['latency_overhead'],
            configuration=configuration,
            rationale=rationale,
            monitoring_metrics=monitoring_metrics
        )
    
    def _select_alternatives(self,
                           scores: Dict[MitigationTechnique, float],
                           primary: MitigationTechnique,
                           context: SystemContext) -> List[MitigationStrategy]:
        """Select alternative mitigation strategies"""
        
        alternatives = []
        
        # Sort techniques by score
        sorted_techniques = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top 2 alternatives (excluding primary)
        for technique, score in sorted_techniques:
            if technique != primary and score > 0.3:
                strategy = self._build_strategy(technique, context, [])
                alternatives.append(strategy)
                
                if len(alternatives) >= 2:
                    break
        
        return alternatives
    
    def _design_combined_approach(self,
                                 bias_categories: set,
                                 context: SystemContext) -> Dict[str, Any]:
        """Design a combined mitigation approach for multiple bias types"""
        
        pipeline = []
        
        # Pre-processing stage
        if BiasCategory.HISTORICAL in bias_categories:
            pipeline.append({
                'stage': 'pre_processing',
                'technique': MitigationTechnique.REWEIGHTING.value,
                'target': 'historical_bias'
            })
        
        # In-processing stage
        if BiasCategory.REPRESENTATION in bias_categories:
            pipeline.append({
                'stage': 'in_processing',
                'technique': MitigationTechnique.ADVERSARIAL_DEBIASING.value,
                'target': 'representation_bias'
            })
        
        # Post-processing stage
        if BiasCategory.MEASUREMENT in bias_categories or BiasCategory.AGGREGATION in bias_categories:
            pipeline.append({
                'stage': 'post_processing',
                'technique': MitigationTechnique.POST_PROCESSING.value,
                'target': 'measurement_and_aggregation_bias'
            })
        
        return {
            'approach': 'pipeline',
            'stages': pipeline,
            'coordination': 'sequential',
            'expected_improvement': 0.7 + (0.1 * len(bias_categories))
        }
    
    def _calculate_expected_reduction(self,
                                     strategy: MitigationStrategy,
                                     detections: List[BiasDetection]) -> float:
        """Calculate expected bias reduction"""
        
        if not detections:
            return 0.0
        
        # Base effectiveness from strategy
        base_effectiveness = strategy.expected_effectiveness
        
        # Adjust based on bias severity
        severity_multiplier = {
            'CRITICAL': 0.7,
            'HIGH': 0.8,
            'MEDIUM': 0.9,
            'LOW': 1.0
        }
        
        avg_severity_impact = np.mean([
            severity_multiplier.get(d.severity, 0.85)
            for d in detections
        ])
        
        # Adjust based on confidence
        avg_confidence = np.mean([d.confidence for d in detections])
        
        expected_reduction = base_effectiveness * avg_severity_impact * avg_confidence
        
        return min(expected_reduction, 0.95)  # Cap at 95% reduction
    
    def _calculate_confidence(self,
                            technique: MitigationTechnique,
                            context: SystemContext,
                            bias_categories: set) -> float:
        """Calculate confidence in strategy selection"""
        
        confidence = 0.5  # Base confidence
        
        profile = self.technique_profiles[technique]
        
        # Increase confidence for context match
        if context in profile['best_contexts']:
            confidence += 0.2
        
        # Increase confidence for bias type match
        matching_biases = len(bias_categories.intersection(set(profile['bias_types'])))
        confidence += matching_biases * 0.1
        
        # Increase confidence based on success rate
        confidence += profile['success_rate'] * 0.2
        
        # Check historical performance if available
        technique_key = f"{technique.value}_{context.value}"
        if technique_key in self.effectiveness_cache:
            historical_success = self.effectiveness_cache[technique_key]
            confidence = confidence * 0.7 + historical_success * 0.3
        
        return min(confidence, 1.0)
    
    def monitor_performance(self,
                           strategy: MitigationStrategy,
                           metrics: PerformanceMetrics) -> None:
        """
        Monitor and track strategy performance for adaptive selection
        
        Args:
            strategy: The deployed strategy
            metrics: Measured performance metrics
        """
        
        # Store performance data
        self.performance_history[strategy.technique].append(metrics)
        
        # Update effectiveness cache
        if len(self.performance_history[strategy.technique]) >= 5:
            recent_metrics = self.performance_history[strategy.technique][-5:]
            avg_effectiveness = np.mean([m.fairness_improvement for m in recent_metrics])
            
            # Cache key includes technique
            cache_key = strategy.technique.value
            self.effectiveness_cache[cache_key] = avg_effectiveness
        
        # Alert on performance degradation
        if metrics.fairness_improvement < strategy.expected_effectiveness * 0.7:
            self._trigger_alert(
                f"Strategy {strategy.technique.value} underperforming: "
                f"Expected {strategy.expected_effectiveness:.2%}, "
                f"Actual {metrics.fairness_improvement:.2%}"
            )
        
        if metrics.accuracy_retention < 0.9:
            self._trigger_alert(
                f"Accuracy degradation detected: {metrics.accuracy_retention:.2%}"
            )
    
    def _trigger_alert(self, message: str) -> None:
        """Trigger performance alert"""
        print(f"[ALERT] {message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of mitigation performance"""
        
        summary = {}
        
        for technique, metrics_list in self.performance_history.items():
            if metrics_list:
                recent = metrics_list[-10:] if len(metrics_list) > 10 else metrics_list
                
                summary[technique.value] = {
                    'avg_fairness_improvement': np.mean([m.fairness_improvement for m in recent]),
                    'avg_accuracy_retention': np.mean([m.accuracy_retention for m in recent]),
                    'avg_latency_ms': np.mean([m.latency_ms for m in recent]),
                    'samples': len(recent)
                }
        
        return summary