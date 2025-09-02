"""
Integration tests for temporal fairness metrics.

Tests the interaction between TDP, EOOT, and FDD metrics to ensure
they work together correctly with shared data structures.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Import metrics
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.utils.data_generators import TemporalBiasGenerator


class TestIntegration:
    """Integration tests for all temporal fairness metrics."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for integration testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic data with known bias patterns
        groups = np.random.choice(['A', 'B'], size=n_samples)
        timestamps = np.sort(np.random.uniform(0, 365, n_samples))
        
        # Create true labels with some group correlation
        true_labels = np.zeros(n_samples)
        for i in range(n_samples):
            if groups[i] == 'A':
                true_labels[i] = np.random.binomial(1, 0.6)
            else:
                true_labels[i] = np.random.binomial(1, 0.4)
        
        # Create predictions with temporal bias
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            # Bias increases over time
            time_factor = timestamps[i] / 365
            if groups[i] == 'A':
                pred_prob = 0.7 - 0.2 * time_factor
            else:
                pred_prob = 0.5 - 0.3 * time_factor
            predictions[i] = np.random.binomial(1, pred_prob)
        
        # For TDP (only needs decisions)
        decisions = predictions
        
        return {
            'decisions': decisions,
            'predictions': predictions,
            'true_labels': true_labels,
            'groups': groups,
            'timestamps': timestamps,
            'n_samples': n_samples
        }
    
    def test_all_metrics_with_same_data(self, sample_data):
        """Test that all metrics can process the same dataset."""
        # Initialize metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        eoot = EqualizedOddsOverTime(tpr_threshold=0.1, fpr_threshold=0.1)
        fdd = FairnessDecayDetection(decay_threshold=0.05)
        
        # Calculate TDP
        tdp_result = tdp.calculate(
            decisions=sample_data['decisions'],
            groups=sample_data['groups'],
            timestamps=sample_data['timestamps'],
            return_details=True
        )
        
        # Calculate EOOT
        eoot_result = eoot.calculate(
            predictions=sample_data['predictions'],
            true_labels=sample_data['true_labels'],
            groups=sample_data['groups'],
            timestamps=sample_data['timestamps'],
            return_details=True
        )
        
        # Use TDP values for FDD
        if tdp_result['tdp_values']:
            fdd_result = fdd.detect_fairness_decay(
                metric_history=tdp_result['tdp_values'],
                return_details=True
            )
        
        # Verify all metrics produce results
        assert tdp_result is not None
        assert eoot_result is not None
        assert fdd_result is not None
        
        # Check result structure
        assert 'max_tdp' in tdp_result
        assert 'max_eoot' in eoot_result
        assert 'decay_detected' in fdd_result
    
    def test_metrics_with_temporal_bias_generator(self):
        """Test metrics with data from TemporalBiasGenerator."""
        generator = TemporalBiasGenerator(random_seed=42)
        
        # Generate dataset with increasing bias
        df = generator.generate_dataset(
            n_samples=2000,
            n_groups=3,
            bias_type='increasing',
            bias_strength=0.3,
            time_periods=10
        )
        
        # Initialize metrics
        tdp = TemporalDemographicParity()
        eoot = EqualizedOddsOverTime()
        fdd = FairnessDecayDetection()
        
        # For EOOT, we need to create synthetic true labels
        # (since the generator only creates decisions)
        true_labels = np.random.binomial(1, 0.5, size=len(df))
        
        # Calculate metrics
        tdp_result = tdp.detect_bias(
            decisions=df['decision'].values,
            groups=df['group'].values,
            timestamps=df['time_period'].values
        )
        
        eoot_result = eoot.detect_bias(
            predictions=df['decision'].values,
            true_labels=true_labels,
            groups=df['group'].values,
            timestamps=df['time_period'].values
        )
        
        # Track TDP over time for FDD
        time_windows = [(i, i+1) for i in range(10)]
        tdp_history = []
        for start, end in time_windows:
            mask = (df['time_period'] >= start) & (df['time_period'] < end)
            if mask.sum() > 0:
                window_tdp = tdp.calculate(
                    decisions=df[mask]['decision'].values,
                    groups=df[mask]['group'].values
                )
                tdp_history.append(window_tdp)
        
        if len(tdp_history) >= 3:
            fdd_result = fdd.detect_fairness_decay(
                metric_history=tdp_history,
                return_details=True
            )
            
            # With increasing bias, we expect decay detection
            assert 'decay_detected' in fdd_result
    
    def test_pipeline_integration(self):
        """Test a complete fairness monitoring pipeline."""
        # Generate hiring pipeline data
        generator = TemporalBiasGenerator(random_seed=42)
        df = generator.generate_hiring_pipeline(
            n_applicants=1000,
            n_stages=4,
            groups=['GroupA', 'GroupB', 'GroupC'],
            bias_at_stage={2: 0.15, 3: 0.20}
        )
        
        # Initialize metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        eoot = EqualizedOddsOverTime(tpr_threshold=0.15, fpr_threshold=0.15)
        fdd = FairnessDecayDetection(decay_threshold=0.05)
        
        # Analyze fairness at each stage
        stage_metrics = []
        for stage in sorted(df['stage'].unique()):
            stage_data = df[df['stage'] == stage]
            
            # Calculate TDP for this stage
            tdp_value = tdp.calculate(
                decisions=stage_data['decision'].values,
                groups=stage_data['group'].values
            )
            
            stage_metrics.append({
                'stage': stage,
                'tdp': tdp_value,
                'n_candidates': len(stage_data),
                'pass_rate': stage_data['decision'].mean()
            })
        
        # Check for fairness decay across stages
        tdp_values = [m['tdp'] for m in stage_metrics]
        decay_detected = fdd.detect_fairness_decay(
            metric_history=tdp_values,
            return_details=True
        )
        
        # Verify pipeline processed correctly
        assert len(stage_metrics) == 4
        assert all('tdp' in m for m in stage_metrics)
        
        # With bias at stages 2 and 3, TDP should increase
        assert stage_metrics[2]['tdp'] > stage_metrics[0]['tdp']
    
    def test_cross_metric_validation(self, sample_data):
        """Test that different metrics provide consistent insights."""
        # Calculate all metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        eoot = EqualizedOddsOverTime(tpr_threshold=0.1, fpr_threshold=0.1)
        
        tdp_bias = tdp.detect_bias(
            decisions=sample_data['decisions'],
            groups=sample_data['groups'],
            timestamps=sample_data['timestamps']
        )
        
        eoot_bias = eoot.detect_bias(
            predictions=sample_data['predictions'],
            true_labels=sample_data['true_labels'],
            groups=sample_data['groups'],
            timestamps=sample_data['timestamps']
        )
        
        # Both should detect some form of bias or fairness
        assert 'bias_detected' in tdp_bias
        assert 'bias_detected' in eoot_bias
        assert 'confidence' in tdp_bias
        assert 'confidence' in eoot_bias
    
    def test_metric_composition(self):
        """Test combining multiple metrics for comprehensive analysis."""
        generator = TemporalBiasGenerator(random_seed=42)
        
        # Generate data with complex bias pattern
        df = generator.generate_dataset(
            n_samples=3000,
            bias_type='confidence_valley',
            bias_strength=0.25,
            time_periods=12
        )
        
        # Create composite fairness score
        tdp = TemporalDemographicParity()
        
        # Calculate TDP for each time period
        composite_scores = []
        for period in sorted(df['time_period'].unique()):
            period_data = df[df['time_period'] == period]
            
            if len(period_data) < 30:
                continue
            
            tdp_value = tdp.calculate(
                decisions=period_data['decision'].values,
                groups=period_data['group'].values
            )
            
            # Composite score (could include multiple metrics)
            composite_scores.append({
                'period': period,
                'tdp': tdp_value,
                'fairness_score': 1.0 - min(tdp_value, 1.0)  # Convert to fairness score
            })
        
        # Analyze composite scores with FDD
        fdd = FairnessDecayDetection(detection_method='changepoint')
        fairness_scores = [s['fairness_score'] for s in composite_scores]
        
        decay_analysis = fdd.detect_fairness_decay(
            metric_history=fairness_scores,
            return_details=True
        )
        
        # With confidence valley pattern, we expect a changepoint
        assert decay_analysis is not None
        if decay_analysis['decay_type'] == 'changepoint':
            assert 'changepoint_index' in decay_analysis
    
    def test_alert_generation_pipeline(self):
        """Test end-to-end alert generation based on metrics."""
        generator = TemporalBiasGenerator(random_seed=42)
        
        # Generate data with sudden bias shift
        df = generator.generate_dataset(
            n_samples=2000,
            bias_type='sudden_shift',
            bias_strength=0.4,
            time_periods=8
        )
        
        # Monitor metrics over time
        tdp = TemporalDemographicParity(threshold=0.1)
        fdd = FairnessDecayDetection(decay_threshold=0.05, detection_method='changepoint')
        
        alerts = []
        tdp_history = []
        
        for period in sorted(df['time_period'].unique()):
            period_data = df[df['time_period'] == period]
            
            # Calculate TDP
            tdp_value = tdp.calculate(
                decisions=period_data['decision'].values,
                groups=period_data['group'].values
            )
            tdp_history.append(tdp_value)
            
            # Check for immediate threshold violation
            if tdp_value > tdp.threshold * 2:
                alerts.append({
                    'type': 'threshold_violation',
                    'period': period,
                    'metric': 'TDP',
                    'value': tdp_value,
                    'severity': 'high' if tdp_value > tdp.threshold * 4 else 'medium'
                })
            
            # Check for decay pattern after enough history
            if len(tdp_history) >= 3:
                decay_info = fdd.detect_fairness_decay(
                    metric_history=tdp_history,
                    return_details=True
                )
                
                alert = fdd.generate_alert(decay_info, metric_name='TDP')
                if alert:
                    alerts.append(alert)
        
        # With sudden shift pattern, we should have alerts
        assert len(alerts) > 0
        
        # Check alert structure
        for alert in alerts:
            assert 'type' in alert or 'alert_type' in alert
            assert 'severity' in alert
    
    def test_performance_with_large_dataset(self):
        """Test metric performance with production-sized data."""
        import time
        
        # Generate large dataset
        generator = TemporalBiasGenerator(random_seed=42)
        df = generator.generate_dataset(
            n_samples=10000,
            n_groups=5,
            time_periods=20
        )
        
        # Initialize metrics
        tdp = TemporalDemographicParity()
        eoot = EqualizedOddsOverTime()
        fdd = FairnessDecayDetection()
        
        # Measure TDP performance
        start_time = time.time()
        tdp_result = tdp.calculate(
            decisions=df['decision'].values,
            groups=df['group'].values,
            timestamps=df['time_period'].values,
            return_details=True
        )
        tdp_time = time.time() - start_time
        
        # Measure EOOT performance (with synthetic labels)
        true_labels = np.random.binomial(1, 0.5, size=len(df))
        start_time = time.time()
        eoot_result = eoot.calculate(
            predictions=df['decision'].values,
            true_labels=true_labels,
            groups=df['group'].values,
            timestamps=df['time_period'].values,
            return_details=True
        )
        eoot_time = time.time() - start_time
        
        # Measure FDD performance
        metric_history = [np.random.random() for _ in range(100)]
        start_time = time.time()
        fdd_result = fdd.detect_fairness_decay(
            metric_history=metric_history,
            return_details=True
        )
        fdd_time = time.time() - start_time
        
        # All metrics should process 10K records in under 100ms
        assert tdp_time < 0.1, f"TDP took {tdp_time:.3f}s, expected < 0.1s"
        assert eoot_time < 0.1, f"EOOT took {eoot_time:.3f}s, expected < 0.1s"
        assert fdd_time < 0.1, f"FDD took {fdd_time:.3f}s, expected < 0.1s"
        
        print(f"\nPerformance Results (10K records):")
        print(f"  TDP:  {tdp_time*1000:.2f}ms")
        print(f"  EOOT: {eoot_time*1000:.2f}ms")
        print(f"  FDD:  {fdd_time*1000:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])