"""
Comprehensive temporal fairness analysis suite.

This module provides an integrated framework for running full temporal
fairness assessments using all available metrics and detection tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
import json
from collections import defaultdict

# Import all metrics
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.metrics.queue_position_fairness import QueuePositionFairness

# Import analysis tools
from src.analysis.bias_detector import BiasDetector

# Import visualization
from src.visualization.fairness_visualizer import FairnessVisualizer


class TemporalAnalyzer:
    """
    Comprehensive temporal fairness analysis suite.
    
    Integrates all metrics and detection tools to provide complete
    fairness assessment with actionable insights.
    """
    
    def __init__(
        self,
        metrics_config: Optional[Dict[str, Any]] = None,
        detection_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize temporal analyzer.
        
        Args:
            metrics_config: Configuration for metrics (thresholds, parameters)
            detection_config: Configuration for bias detection
        """
        # Default configurations
        self.metrics_config = metrics_config or {
            'tdp_threshold': 0.1,
            'tpr_threshold': 0.15,
            'fpr_threshold': 0.15,
            'qpf_threshold': 0.8,
            'decay_threshold': 0.05,
            'min_samples': 30
        }
        
        self.detection_config = detection_config or {
            'sensitivity': 0.95,
            'min_pattern_length': 3,
            'anomaly_threshold': 3.0
        }
        
        # Initialize metrics
        self.tdp = TemporalDemographicParity(
            threshold=self.metrics_config['tdp_threshold'],
            min_samples=self.metrics_config['min_samples']
        )
        
        self.eoot = EqualizedOddsOverTime(
            tpr_threshold=self.metrics_config['tpr_threshold'],
            fpr_threshold=self.metrics_config['fpr_threshold'],
            min_samples=self.metrics_config['min_samples']
        )
        
        self.qpf = QueuePositionFairness(
            fairness_threshold=self.metrics_config['qpf_threshold'],
            min_samples=self.metrics_config['min_samples']
        )
        
        self.fdd = FairnessDecayDetection(
            decay_threshold=self.metrics_config['decay_threshold']
        )
        
        # Initialize detector
        self.bias_detector = BiasDetector(**self.detection_config)
        
        # Initialize visualizer
        self.visualizer = FairnessVisualizer()
        
        # Store results
        self.analysis_results = {}
        self.metric_history = defaultdict(list)
    
    def run_full_analysis(
        self,
        data: pd.DataFrame,
        groups: Union[str, np.ndarray],
        decision_column: Optional[str] = None,
        prediction_column: Optional[str] = None,
        label_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        queue_column: Optional[str] = None,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive temporal fairness analysis.
        
        Args:
            data: Input DataFrame
            groups: Group column name or array
            decision_column: Column with decisions (for TDP)
            prediction_column: Column with predictions (for EOOT)
            label_column: Column with true labels (for EOOT)
            timestamp_column: Column with timestamps
            queue_column: Column with queue positions (for QPF)
            window_size: Time window size for analysis
            
        Returns:
            Comprehensive analysis results dictionary
        """
        # Extract data
        if isinstance(groups, str):
            group_data = data[groups].values
        else:
            group_data = np.asarray(groups)
        
        timestamps = data[timestamp_column].values if timestamp_column else None
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': self._generate_data_summary(data, group_data),
            'metrics': {},
            'patterns': {},
            'trends': {},
            'recommendations': [],
            'risk_assessment': {}
        }
        
        # Run TDP analysis
        if decision_column and decision_column in data.columns:
            tdp_results = self._analyze_tdp(
                data[decision_column].values,
                group_data,
                timestamps,
                window_size
            )
            results['metrics']['tdp'] = tdp_results
        
        # Run EOOT analysis
        if (prediction_column and label_column and 
            prediction_column in data.columns and 
            label_column in data.columns):
            eoot_results = self._analyze_eoot(
                data[prediction_column].values,
                data[label_column].values,
                group_data,
                timestamps,
                window_size
            )
            results['metrics']['eoot'] = eoot_results
        
        # Run QPF analysis
        if queue_column and queue_column in data.columns:
            qpf_results = self._analyze_qpf(
                data[queue_column].values,
                group_data,
                timestamps
            )
            results['metrics']['qpf'] = qpf_results
        
        # Run pattern detection
        detection_data = data[decision_column] if decision_column else data.iloc[:, 0]
        pattern_results = self.bias_detector.detect_temporal_patterns(
            detection_data,
            group_data,
            timestamp_column,
            decision_column
        )
        results['patterns'] = pattern_results
        
        # Analyze trends across metrics
        if self.metric_history:
            results['trends'] = self._analyze_trends()
        
        # Generate risk assessment
        results['risk_assessment'] = self._assess_fairness_risk(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_comprehensive_recommendations(results)
        
        # Store results
        self.analysis_results = results
        
        return results
    
    def _analyze_tdp(
        self,
        decisions: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray],
        window_size: Optional[int]
    ) -> Dict[str, Any]:
        """Analyze Temporal Demographic Parity."""
        # Calculate overall TDP
        tdp_result = self.tdp.detect_bias(
            decisions=decisions,
            groups=groups,
            timestamps=timestamps
        )
        
        # Calculate TDP over time windows
        tdp_details = self.tdp.calculate(
            decisions=decisions,
            groups=groups,
            timestamps=timestamps,
            window_size=window_size,
            return_details=True
        )
        
        # Store history
        if tdp_details.get('tdp_values'):
            self.metric_history['tdp'].extend(tdp_details['tdp_values'])
        
        # Check for decay
        if len(self.metric_history['tdp']) >= 3:
            decay_result = self.fdd.detect_fairness_decay(
                metric_history=self.metric_history['tdp'],
                return_details=True
            )
        else:
            decay_result = {'decay_detected': False}
        
        return {
            'bias_detected': tdp_result['bias_detected'],
            'metric_value': tdp_result['metric_value'],
            'severity': tdp_result['severity'],
            'confidence': tdp_result['confidence'],
            'details': tdp_details,
            'decay_analysis': decay_result,
            'threshold': self.tdp.threshold
        }
    
    def _analyze_eoot(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray],
        window_size: Optional[int]
    ) -> Dict[str, Any]:
        """Analyze Equalized Odds Over Time."""
        # Calculate EOOT
        eoot_result = self.eoot.detect_bias(
            predictions=predictions,
            true_labels=true_labels,
            groups=groups,
            timestamps=timestamps
        )
        
        # Get detailed metrics
        eoot_details = self.eoot.calculate(
            predictions=predictions,
            true_labels=true_labels,
            groups=groups,
            timestamps=timestamps,
            window_size=window_size,
            return_details=True
        )
        
        # Store history
        if eoot_details.get('eoot_values'):
            self.metric_history['eoot'].extend(eoot_details['eoot_values'])
        
        # Check for decay
        if len(self.metric_history['eoot']) >= 3:
            decay_result = self.fdd.detect_fairness_decay(
                metric_history=self.metric_history['eoot'],
                return_details=True
            )
        else:
            decay_result = {'decay_detected': False}
        
        return {
            'bias_detected': eoot_result['bias_detected'],
            'metric_value': eoot_result['metric_value'],
            'bias_source': eoot_result.get('bias_source', []),
            'severity': eoot_result['severity'],
            'confidence': eoot_result['confidence'],
            'details': eoot_details,
            'decay_analysis': decay_result,
            'tpr_threshold': self.eoot.tpr_threshold,
            'fpr_threshold': self.eoot.fpr_threshold
        }
    
    def _analyze_qpf(
        self,
        queue_positions: np.ndarray,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze Queue Position Fairness."""
        # Calculate QPF
        qpf_result = self.qpf.detect_bias(
            queue_positions=queue_positions,
            groups=groups,
            timestamps=timestamps
        )
        
        # Get detailed analysis
        qpf_details = self.qpf.calculate(
            queue_positions=queue_positions,
            groups=groups,
            timestamps=timestamps,
            return_details=True
        )
        
        # Analyze priority patterns
        priority_patterns = self.qpf.analyze_priority_patterns(
            queue_positions=queue_positions,
            groups=groups
        )
        
        # Store history
        if qpf_details.get('qpf_values'):
            self.metric_history['qpf'].extend(qpf_details['qpf_values'])
        
        return {
            'bias_detected': qpf_result['bias_detected'],
            'metric_value': qpf_result['metric_value'],
            'most_disadvantaged': qpf_result.get('most_disadvantaged_group'),
            'severity': qpf_result['severity'],
            'confidence': qpf_result['confidence'],
            'details': qpf_details,
            'priority_patterns': priority_patterns,
            'threshold': self.qpf.fairness_threshold
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends across all metrics."""
        trends = {}
        
        for metric_name, history in self.metric_history.items():
            if len(history) < 3:
                continue
            
            # Detect decay
            decay_info = self.fdd.detect_fairness_decay(
                metric_history=history,
                return_details=True
            )
            
            # Predict future values
            predictions = self.fdd.predict_future_decay(
                metric_history=history,
                periods_ahead=3
            )
            
            # Calculate trend statistics
            trend_stats = {
                'mean': np.mean(history),
                'std': np.std(history),
                'min': np.min(history),
                'max': np.max(history),
                'range': np.max(history) - np.min(history),
                'trend_direction': 'increasing' if decay_info.get('slope', 0) > 0 else 'decreasing',
                'volatility': np.std(np.diff(history)) if len(history) > 1 else 0
            }
            
            trends[metric_name] = {
                'decay_analysis': decay_info,
                'predictions': predictions,
                'statistics': trend_stats
            }
        
        return trends
    
    def _generate_data_summary(
        self,
        data: pd.DataFrame,
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """Generate summary statistics of the input data."""
        unique_groups = np.unique(groups)
        
        summary = {
            'n_samples': len(data),
            'n_groups': len(unique_groups),
            'groups': unique_groups.tolist(),
            'group_distribution': {}
        }
        
        # Calculate group distribution
        for group in unique_groups:
            count = np.sum(groups == group)
            summary['group_distribution'][str(group)] = {
                'count': int(count),
                'percentage': count / len(groups) * 100
            }
        
        # Add temporal span if timestamps available
        if 'timestamp' in data.columns:
            summary['temporal_span'] = {
                'start': str(data['timestamp'].min()),
                'end': str(data['timestamp'].max()),
                'duration': str(data['timestamp'].max() - data['timestamp'].min())
            }
        
        return summary
    
    def _assess_fairness_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fairness risk based on all analyses."""
        risk_factors = []
        risk_score = 0.0
        
        # Check metric violations
        if 'metrics' in results:
            for metric_name, metric_result in results['metrics'].items():
                if metric_result.get('bias_detected'):
                    risk_factors.append(f"{metric_name.upper()} violation")
                    risk_score += 0.2
                
                # Check for decay
                if metric_result.get('decay_analysis', {}).get('decay_detected'):
                    risk_factors.append(f"{metric_name.upper()} showing decay")
                    risk_score += 0.15
        
        # Check pattern detection
        if 'patterns' in results:
            patterns = results['patterns'].get('patterns', {})
            
            if patterns.get('confidence_valley', {}).get('detected'):
                risk_factors.append("Confidence valley pattern")
                risk_score += 0.25
            
            if patterns.get('group_divergence', {}).get('detected'):
                risk_factors.append("Group divergence detected")
                risk_score += 0.2
            
            if patterns.get('sudden_shift', {}).get('detected'):
                risk_factors.append("Sudden bias shifts")
                risk_score += 0.15
        
        # Classify risk level
        risk_score = min(1.0, risk_score)
        
        if risk_score < 0.2:
            risk_level = 'low'
        elif risk_score < 0.4:
            risk_level = 'moderate'
        elif risk_score < 0.6:
            risk_level = 'high'
        elif risk_score < 0.8:
            risk_level = 'very_high'
        else:
            risk_level = 'critical'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mitigation_priority': self._prioritize_mitigation(risk_factors)
        }
    
    def _prioritize_mitigation(self, risk_factors: List[str]) -> List[str]:
        """Prioritize mitigation strategies based on risk factors."""
        priorities = []
        
        # High priority mitigations
        if any('divergence' in factor.lower() for factor in risk_factors):
            priorities.append("Implement group-specific monitoring and rebalancing")
        
        if any('confidence valley' in factor.lower() for factor in risk_factors):
            priorities.append("Investigate model calibration during middle periods")
        
        if any('decay' in factor.lower() for factor in risk_factors):
            priorities.append("Establish periodic retraining schedule")
        
        # Medium priority
        if any('shift' in factor.lower() for factor in risk_factors):
            priorities.append("Review recent system changes and data drift")
        
        if any('violation' in factor.lower() for factor in risk_factors):
            priorities.append("Adjust decision thresholds for fairness")
        
        # General recommendations
        if not priorities:
            priorities.append("Maintain regular fairness monitoring")
        
        return priorities
    
    def _generate_comprehensive_recommendations(
        self,
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive actionable recommendations."""
        recommendations = []
        
        # Get risk assessment
        risk = results.get('risk_assessment', {})
        risk_level = risk.get('risk_level', 'unknown')
        
        # Critical recommendations for high risk
        if risk_level in ['very_high', 'critical']:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Immediate intervention required',
                'details': 'Multiple fairness violations detected. Consider pausing system for review.',
                'timeline': 'Immediate'
            })
        
        # Metric-specific recommendations
        if 'metrics' in results:
            for metric_name, metric_result in results['metrics'].items():
                if metric_result.get('bias_detected'):
                    severity = metric_result.get('severity', 'unknown')
                    
                    if metric_name == 'tdp':
                        recommendations.append({
                            'priority': 'HIGH' if severity in ['high', 'critical'] else 'MEDIUM',
                            'action': 'Address demographic parity violation',
                            'details': f"TDP value {metric_result['metric_value']:.3f} exceeds threshold",
                            'timeline': '1-2 weeks'
                        })
                    
                    elif metric_name == 'eoot':
                        bias_source = metric_result.get('bias_source', [])
                        recommendations.append({
                            'priority': 'HIGH',
                            'action': 'Investigate equalized odds violation',
                            'details': f"Bias detected in {', '.join(bias_source)} rates",
                            'timeline': '1 week'
                        })
                    
                    elif metric_name == 'qpf':
                        disadvantaged = metric_result.get('most_disadvantaged')
                        recommendations.append({
                            'priority': 'MEDIUM',
                            'action': 'Review queue assignment algorithm',
                            'details': f"Group {disadvantaged} systematically disadvantaged in queue",
                            'timeline': '2-3 weeks'
                        })
        
        # Pattern-based recommendations
        if 'patterns' in results:
            pattern_recs = results['patterns'].get('recommendations', [])
            for rec in pattern_recs[:3]:  # Top 3 pattern recommendations
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Pattern-based intervention',
                    'details': rec,
                    'timeline': '2-4 weeks'
                })
        
        # Trend-based recommendations
        if 'trends' in results:
            for metric_name, trend in results['trends'].items():
                if trend.get('decay_analysis', {}).get('decay_detected'):
                    recommendations.append({
                        'priority': 'HIGH',
                        'action': f'Address {metric_name.upper()} degradation',
                        'details': f"Implement drift monitoring and retraining for {metric_name}",
                        'timeline': '1-2 weeks'
                    })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations
    
    def generate_report(
        self,
        output_format: str = 'dict',
        include_visualizations: bool = False,
        save_path: Optional[str] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Generate comprehensive fairness report.
        
        Args:
            output_format: Format for report ('dict', 'json', 'html')
            include_visualizations: Whether to include visualization paths
            save_path: Path to save report
            
        Returns:
            Report in specified format
        """
        if not self.analysis_results:
            return {'error': 'No analysis results available. Run analysis first.'}
        
        report = {
            'executive_summary': self._generate_executive_summary(),
            'detailed_results': self.analysis_results,
            'action_plan': self._generate_action_plan()
        }
        
        if include_visualizations:
            report['visualizations'] = self._generate_visualizations()
        
        # Format output
        if output_format == 'json':
            report_str = json.dumps(report, indent=2, default=str)
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_str)
            return report_str
        
        elif output_format == 'html':
            html_report = self._generate_html_report(report)
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(html_report)
            return html_report
        
        else:  # dict
            if save_path:
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump(report, f)
            return report
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary of findings."""
        if not self.analysis_results:
            return "No analysis results available."
        
        risk = self.analysis_results.get('risk_assessment', {})
        risk_level = risk.get('risk_level', 'unknown')
        risk_factors = risk.get('risk_factors', [])
        
        summary = f"FAIRNESS ASSESSMENT: {risk_level.upper()} RISK\n\n"
        
        if risk_factors:
            summary += f"Key Issues Identified ({len(risk_factors)}):\n"
            for factor in risk_factors[:5]:
                summary += f"  • {factor}\n"
        else:
            summary += "No significant fairness issues detected.\n"
        
        # Add metric summary
        if 'metrics' in self.analysis_results:
            summary += "\nMetric Status:\n"
            for metric_name, result in self.analysis_results['metrics'].items():
                status = "⚠️ VIOLATION" if result.get('bias_detected') else "✓ PASS"
                value = result.get('metric_value', 'N/A')
                summary += f"  • {metric_name.upper()}: {status} (value: {value:.3f})\n"
        
        # Add top recommendation
        if self.analysis_results.get('recommendations'):
            top_rec = self.analysis_results['recommendations'][0]
            summary += f"\nTop Priority Action: {top_rec['action']}\n"
            summary += f"Timeline: {top_rec['timeline']}\n"
        
        return summary
    
    def _generate_action_plan(self) -> List[Dict[str, Any]]:
        """Generate structured action plan."""
        if not self.analysis_results:
            return []
        
        recommendations = self.analysis_results.get('recommendations', [])
        
        action_plan = []
        for i, rec in enumerate(recommendations[:10], 1):  # Top 10 actions
            action_plan.append({
                'step': i,
                'priority': rec['priority'],
                'action': rec['action'],
                'details': rec['details'],
                'timeline': rec['timeline'],
                'success_criteria': self._define_success_criteria(rec)
            })
        
        return action_plan
    
    def _define_success_criteria(self, recommendation: Dict[str, Any]) -> str:
        """Define success criteria for an action."""
        action = recommendation.get('action', '').lower()
        
        if 'demographic parity' in action:
            return f"TDP < {self.metrics_config['tdp_threshold']}"
        elif 'equalized odds' in action:
            return f"EOOT TPR/FPR differences < {self.metrics_config['tpr_threshold']}"
        elif 'queue' in action:
            return f"QPF > {self.metrics_config['qpf_threshold']}"
        elif 'retraining' in action:
            return "Model retrained and metrics improved by >10%"
        else:
            return "Issue resolved and metrics within acceptable thresholds"
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate visualization paths."""
        viz_paths = {}
        
        # Generate metric evolution plot
        if self.metric_history:
            for metric_name, history in self.metric_history.items():
                if len(history) > 1:
                    fig = self.visualizer.plot_metric_evolution(
                        history,
                        metric_name=metric_name.upper(),
                        threshold=self._get_threshold(metric_name)
                    )
                    path = f"viz_{metric_name}_evolution.png"
                    fig.savefig(path)
                    viz_paths[f'{metric_name}_evolution'] = path
        
        return viz_paths
    
    def _get_threshold(self, metric_name: str) -> float:
        """Get threshold for a metric."""
        thresholds = {
            'tdp': self.metrics_config['tdp_threshold'],
            'eoot': max(self.metrics_config['tpr_threshold'], 
                       self.metrics_config['fpr_threshold']),
            'qpf': 1 - self.metrics_config['qpf_threshold']  # Invert for display
        }
        return thresholds.get(metric_name, 0.1)
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML formatted report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Temporal Fairness Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
                .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                .critical { color: #d32f2f; font-weight: bold; }
                .high { color: #f57c00; font-weight: bold; }
                .medium { color: #fbc02d; }
                .low { color: #388e3c; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f5f5f5; }
                .recommendation { margin: 10px 0; padding: 10px; border-left: 4px solid #2196f3; }
            </style>
        </head>
        <body>
            <h1>Temporal Fairness Analysis Report</h1>
        """
        
        # Add executive summary
        html += f"""
        <div class="summary">
            <h2>Executive Summary</h2>
            <pre>{report.get('executive_summary', 'N/A')}</pre>
        </div>
        """
        
        # Add risk assessment
        if 'detailed_results' in report and 'risk_assessment' in report['detailed_results']:
            risk = report['detailed_results']['risk_assessment']
            risk_class = risk.get('risk_level', 'unknown').lower()
            html += f"""
            <h2>Risk Assessment</h2>
            <p class="{risk_class}">Risk Level: {risk.get('risk_level', 'Unknown').upper()}</p>
            <p>Risk Score: {risk.get('risk_score', 0):.2%}</p>
            """
        
        # Add action plan
        if 'action_plan' in report:
            html += "<h2>Action Plan</h2>"
            for action in report['action_plan']:
                priority_class = action['priority'].lower()
                html += f"""
                <div class="recommendation">
                    <strong class="{priority_class}">Step {action['step']}: {action['action']}</strong>
                    <p>{action['details']}</p>
                    <p><em>Timeline: {action['timeline']}</em></p>
                    <p><em>Success Criteria: {action['success_criteria']}</em></p>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        return html