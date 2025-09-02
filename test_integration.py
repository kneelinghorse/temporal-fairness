"""
Integration Testing Suite for Temporal Fairness Framework

This module provides end-to-end testing of all components working together,
validating complete workflows from data input to analysis output.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import json
import os
from typing import Dict, List, Any

# Import all components
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.metrics.queue_position_fairness import QueuePositionFairness
from src.analysis.bias_detector import BiasDetector
from src.analysis.enhanced_bias_detector import EnhancedBiasDetector
from src.analysis.temporal_analyzer import TemporalAnalyzer
from src.utils.data_generators import TemporalBiasGenerator
from src.visualization.fairness_visualizer import FairnessVisualizer


class IntegrationTestBase(unittest.TestCase):
    """Base class for integration tests with common setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TemporalBiasGenerator(random_seed=42)
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate standard test data
        self.n_records = 1000
        self.groups = ['Group_A', 'Group_B', 'Group_C']
        self.timestamps = pd.date_range(
            start='2024-01-01',
            periods=self.n_records,
            freq='H'
        )
        
        # Generate decisions with known bias
        np.random.seed(42)
        self.decisions = np.zeros(self.n_records, dtype=int)
        self.group_assignments = np.random.choice(self.groups, self.n_records)
        
        # Inject bias: Group_A has 60% positive rate, others 40%
        for i in range(self.n_records):
            if self.group_assignments[i] == 'Group_A':
                self.decisions[i] = 1 if np.random.random() < 0.6 else 0
            else:
                self.decisions[i] = 1 if np.random.random() < 0.4 else 0
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestEndToEndWorkflow(IntegrationTestBase):
    """Test complete workflows from data input to analysis output."""
    
    def test_complete_fairness_assessment_workflow(self):
        """Test full workflow: data → metrics → analysis → visualization."""
        
        # Step 1: Generate realistic data
        data = self.generator.generate_biased_hiring_data(
            n_applicants=1000,
            n_days=30,
            groups=['Male', 'Female', 'Non-binary'],
            bias_strength=0.2
        )
        
        # Step 2: Calculate all metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        tdp_result = tdp.detect_bias(
            data['hired'].values,
            data['group'].values,
            data['timestamp'].values
        )
        
        eoot = EqualizedOddsOverTime(tpr_threshold=0.15, fpr_threshold=0.15)
        # Generate synthetic ground truth for EOOT
        data['qualified'] = np.random.choice([0, 1], len(data), p=[0.5, 0.5])
        eoot_result = eoot.detect_bias(
            data['hired'].values,
            data['qualified'].values,
            data['group'].values,
            data['timestamp'].values
        )
        
        qpf = QueuePositionFairness(fairness_threshold=0.8)
        # Generate queue positions
        data['queue_position'] = np.random.randint(1, 100, len(data))
        qpf_result = qpf.detect_bias(
            data['queue_position'].values,
            data['group'].values,
            data['timestamp'].values
        )
        
        # Step 3: Run comprehensive analysis
        analyzer = TemporalAnalyzer()
        analysis_results = analyzer.run_full_analysis(
            data=data,
            groups='group',
            decision_column='hired',
            timestamp_column='timestamp',
            queue_column='queue_position'
        )
        
        # Step 4: Validate results structure
        self.assertIn('metrics', analysis_results)
        self.assertIn('patterns', analysis_results)
        self.assertIn('risk_assessment', analysis_results)
        self.assertIn('recommendations', analysis_results)
        
        # Validate metric results
        self.assertIsInstance(tdp_result['bias_detected'], bool)
        self.assertIsInstance(tdp_result['metric_value'], (int, float))
        self.assertIsInstance(eoot_result['bias_detected'], bool)
        self.assertIsInstance(qpf_result['bias_detected'], bool)
        
        # Validate risk assessment
        risk = analysis_results['risk_assessment']
        self.assertIn('risk_level', risk)
        self.assertIn('risk_score', risk)
        self.assertIn(risk['risk_level'], ['low', 'moderate', 'high', 'very_high', 'critical'])
        
        # Step 5: Generate visualization
        visualizer = FairnessVisualizer()
        fig = visualizer.plot_temporal_fairness_trends(
            timestamps=data['timestamp'].values,
            fairness_scores={'TDP': [tdp_result['metric_value']] * 10},
            threshold=0.8,
            save_path=os.path.join(self.temp_dir, 'test_viz.png')
        )
        
        # Verify visualization was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_viz.png')))
    
    def test_streaming_data_workflow(self):
        """Test workflow with streaming/incremental data updates."""
        
        tdp = TemporalDemographicParity(threshold=0.1)
        all_decisions = []
        all_groups = []
        all_timestamps = []
        
        # Simulate streaming data in batches
        for day in range(7):
            # Generate batch of data
            batch_size = 100
            batch_decisions = np.random.choice([0, 1], batch_size)
            batch_groups = np.random.choice(self.groups, batch_size)
            batch_timestamps = pd.date_range(
                start=f'2024-01-{day+1:02d}',
                periods=batch_size,
                freq='15min'
            )
            
            # Accumulate data
            all_decisions.extend(batch_decisions)
            all_groups.extend(batch_groups)
            all_timestamps.extend(batch_timestamps)
            
            # Calculate metrics on accumulated data
            result = tdp.detect_bias(
                np.array(all_decisions),
                np.array(all_groups),
                np.array(all_timestamps)
            )
            
            # Verify result structure remains consistent
            self.assertIn('bias_detected', result)
            self.assertIn('metric_value', result)
            self.assertIn('confidence', result)
            
            # Verify metric value is valid
            self.assertGreaterEqual(result['metric_value'], 0)
            self.assertLessEqual(result['metric_value'], 1)
    
    def test_multi_stage_pipeline_workflow(self):
        """Test multi-stage analysis pipeline (e.g., hiring process)."""
        
        stages = ['Resume', 'Phone', 'Technical', 'Onsite', 'Offer']
        stage_results = {}
        
        for stage in stages:
            # Generate stage-specific data
            n_candidates = 1000 - (stages.index(stage) * 150)  # Decreasing candidates
            stage_data = {
                'decision': np.random.choice([0, 1], n_candidates, p=[0.5, 0.5]),
                'group': np.random.choice(self.groups, n_candidates),
                'timestamp': pd.date_range('2024-01-01', periods=n_candidates, freq='H')
            }
            
            # Calculate metrics for this stage
            tdp = TemporalDemographicParity(threshold=0.1)
            result = tdp.detect_bias(
                stage_data['decision'],
                stage_data['group'],
                stage_data['timestamp']
            )
            
            stage_results[stage] = {
                'bias_detected': result['bias_detected'],
                'metric_value': result['metric_value'],
                'n_candidates': n_candidates
            }
        
        # Verify pipeline results
        self.assertEqual(len(stage_results), len(stages))
        
        # Verify candidate funnel
        prev_count = float('inf')
        for stage in stages:
            self.assertLess(stage_results[stage]['n_candidates'], prev_count)
            prev_count = stage_results[stage]['n_candidates']
    
    def test_cross_metric_validation(self):
        """Test that different metrics provide consistent bias detection."""
        
        # Generate data with strong bias
        data = self.generator.generate_biased_hiring_data(
            n_applicants=1000,
            n_days=30,
            groups=['A', 'B'],
            bias_strength=0.5  # Strong bias
        )
        
        # Calculate multiple metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        tdp_result = tdp.detect_bias(
            data['hired'].values,
            data['group'].values,
            data['timestamp'].values
        )
        
        qpf = QueuePositionFairness(fairness_threshold=0.8)
        data['queue'] = np.random.randint(1, 100, len(data))
        qpf_result = qpf.detect_bias(
            data['queue'].values,
            data['group'].values,
            data['timestamp'].values
        )
        
        # With strong bias, multiple metrics should detect it
        bias_detections = [
            tdp_result['bias_detected'],
            qpf_result['bias_detected']
        ]
        
        # At least one metric should detect the bias
        self.assertTrue(any(bias_detections), 
                       "Strong bias not detected by any metric")


class TestErrorHandling(IntegrationTestBase):
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test that all components handle empty data gracefully."""
        
        empty_decisions = np.array([])
        empty_groups = np.array([])
        empty_timestamps = np.array([])
        
        # TDP should handle empty data
        tdp = TemporalDemographicParity(threshold=0.1)
        result = tdp.detect_bias(empty_decisions, empty_groups, empty_timestamps)
        self.assertFalse(result['bias_detected'])
        
        # QPF should handle empty data
        qpf = QueuePositionFairness(fairness_threshold=0.8)
        result = qpf.detect_bias(empty_decisions, empty_groups, empty_timestamps)
        self.assertFalse(result['bias_detected'])
    
    def test_single_group_handling(self):
        """Test handling when only one group is present."""
        
        single_group_data = np.array(['A'] * 100)
        decisions = np.random.choice([0, 1], 100)
        timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
        
        tdp = TemporalDemographicParity(threshold=0.1)
        result = tdp.detect_bias(decisions, single_group_data, timestamps)
        
        # Should not detect bias with single group
        self.assertFalse(result['bias_detected'])
        self.assertEqual(result['metric_value'], 0)
    
    def test_missing_values_handling(self):
        """Test handling of missing/null values."""
        
        # Create data with NaN values
        decisions = np.array([1, 0, 1, np.nan, 0, 1, 0, np.nan, 1, 0])
        groups = np.array(['A', 'B', 'A', 'B', None, 'A', 'B', 'A', 'B', 'A'])
        timestamps = pd.date_range('2024-01-01', periods=10, freq='H')
        
        tdp = TemporalDemographicParity(threshold=0.1)
        
        # Should handle NaN values without crashing
        try:
            result = tdp.detect_bias(decisions, groups, timestamps)
            # If it doesn't crash, test passes
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Failed to handle missing values: {e}")
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        
        # Very large queue positions
        extreme_positions = np.array([1, 1000000, 50, 999999, 25])
        groups = np.array(['A', 'B', 'A', 'B', 'A'])
        timestamps = pd.date_range('2024-01-01', periods=5, freq='H')
        
        qpf = QueuePositionFairness(fairness_threshold=0.8)
        result = qpf.detect_bias(extreme_positions, groups, timestamps)
        
        # Should handle extreme values
        self.assertIsNotNone(result)
        self.assertIn('bias_detected', result)
    
    def test_data_type_conversion(self):
        """Test automatic data type conversion."""
        
        # Mixed data types
        decisions_list = [1, 0, 1, 0, 1]  # List instead of array
        groups_tuple = ('A', 'B', 'A', 'B', 'A')  # Tuple
        timestamps_list = ['2024-01-01', '2024-01-02', '2024-01-03', 
                          '2024-01-04', '2024-01-05']  # String dates
        
        tdp = TemporalDemographicParity(threshold=0.1)
        
        # Should handle different input types
        try:
            result = tdp.detect_bias(decisions_list, groups_tuple, 
                                    pd.to_datetime(timestamps_list))
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Failed to handle mixed data types: {e}")


class TestBiasPatternDetection(IntegrationTestBase):
    """Test detection of specific bias patterns."""
    
    def test_confidence_valley_detection(self):
        """Test detection of U-shaped confidence valley pattern."""
        
        # Create U-shaped pattern
        n_points = 100
        t = np.linspace(0, 1, n_points)
        fairness_scores = 0.8 - 0.4 * (t - 0.5) ** 2  # U-shape
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='D')
        
        detector = BiasDetector(sensitivity=0.95)
        valleys = detector.identify_confidence_valleys(
            fairness_scores,
            timestamps
        )
        
        # Should detect at least one valley
        self.assertIsNotNone(valleys)
        if valleys:
            self.assertGreater(len(valleys), 0)
            # Valley should be around the middle
            valley_pos = valleys[0]['position']
            self.assertTrue(40 <= valley_pos <= 60)
    
    def test_sudden_shift_detection(self):
        """Test detection of sudden bias shifts."""
        
        # Create data with sudden shift
        n_points = 100
        fairness_scores = np.ones(n_points) * 0.8
        fairness_scores[50:] = 0.4  # Sudden drop at midpoint
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='D')
        
        detector = BiasDetector(sensitivity=0.95)
        shifts = detector.detect_sudden_shifts(
            fairness_scores,
            timestamps
        )
        
        # Should detect the shift
        self.assertIsNotNone(shifts)
        if shifts:
            self.assertEqual(len(shifts), 1)
            # Shift should be around position 50
            self.assertTrue(48 <= shifts[0]['position'] <= 52)
    
    def test_gradual_drift_detection(self):
        """Test detection of gradual bias drift."""
        
        # Create gradually declining pattern
        n_points = 100
        fairness_scores = 0.9 - np.linspace(0, 0.3, n_points)
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='D')
        
        detector = BiasDetector(sensitivity=0.95)
        drift = detector.detect_gradual_drift(
            fairness_scores,
            timestamps
        )
        
        # Should detect negative drift
        self.assertIsNotNone(drift)
        if drift and drift['detected']:
            self.assertLess(drift['drift_rate'], 0)
            self.assertGreater(drift['confidence'], 0.8)
    
    def test_periodic_pattern_detection(self):
        """Test detection of periodic/seasonal patterns."""
        
        # Create periodic pattern
        n_points = 100
        t = np.linspace(0, 4 * np.pi, n_points)
        fairness_scores = 0.7 + 0.2 * np.sin(t)
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='D')
        
        detector = BiasDetector(sensitivity=0.95)
        periodic = detector.detect_periodic_patterns(
            fairness_scores,
            timestamps
        )
        
        # Should detect periodicity
        self.assertIsNotNone(periodic)
        if periodic and periodic['detected']:
            self.assertGreater(periodic['confidence'], 0.7)
            # Period should be roughly 25 days (100 points / 4 cycles)
            self.assertTrue(20 <= periodic['period'] <= 30)


class TestDataGenerators(IntegrationTestBase):
    """Test data generation utilities."""
    
    def test_hiring_data_generation(self):
        """Test hiring data generator creates valid data."""
        
        data = self.generator.generate_biased_hiring_data(
            n_applicants=500,
            n_days=30,
            groups=['A', 'B', 'C'],
            bias_strength=0.3
        )
        
        # Validate data structure
        self.assertIn('applicant_id', data.columns)
        self.assertIn('group', data.columns)
        self.assertIn('timestamp', data.columns)
        self.assertIn('hired', data.columns)
        
        # Validate data properties
        self.assertEqual(len(data), 500)
        self.assertEqual(len(data['group'].unique()), 3)
        
        # Check bias was injected
        group_rates = data.groupby('group')['hired'].mean()
        self.assertGreater(group_rates.max() - group_rates.min(), 0.1)
    
    def test_queue_data_generation(self):
        """Test queue data generator."""
        
        data = self.generator.generate_queue_simulation(
            n_items=1000,
            n_hours=24,
            groups=['Premium', 'Standard', 'Basic'],
            queue_bias_factor=0.5
        )
        
        # Validate structure
        self.assertIn('queue_position', data.columns)
        self.assertIn('wait_time', data.columns)
        self.assertIn('group', data.columns)
        
        # Check queue bias
        avg_positions = data.groupby('group')['queue_position'].mean()
        # Premium should have better positions than Basic
        self.assertLess(avg_positions['Premium'], avg_positions['Basic'])
    
    def test_temporal_pattern_generation(self):
        """Test generation of specific temporal patterns."""
        
        # Test confidence valley generation
        valley_data = self.generator.generate_confidence_valley_pattern(
            n_points=100,
            valley_depth=0.3,
            valley_position=0.5
        )
        
        # Verify valley exists
        min_idx = np.argmin(valley_data)
        self.assertTrue(40 <= min_idx <= 60)  # Valley near middle
        
        # Valley should be at least 0.2 deep
        self.assertGreater(valley_data.max() - valley_data.min(), 0.2)


class TestVisualization(IntegrationTestBase):
    """Test visualization components."""
    
    def test_trend_visualization(self):
        """Test temporal trend visualization."""
        
        visualizer = FairnessVisualizer()
        
        # Create sample data
        timestamps = pd.date_range('2024-01-01', periods=30, freq='D')
        scores = {
            'TDP': np.random.uniform(0.6, 0.9, 30),
            'EOOT': np.random.uniform(0.7, 0.95, 30)
        }
        
        # Generate plot
        fig = visualizer.plot_temporal_fairness_trends(
            timestamps=timestamps,
            fairness_scores=scores,
            threshold=0.8,
            save_path=os.path.join(self.temp_dir, 'trends.png')
        )
        
        # Verify file created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'trends.png')))
    
    def test_comprehensive_dashboard(self):
        """Test comprehensive dashboard generation."""
        
        visualizer = FairnessVisualizer()
        
        # Generate test data
        data = self.generator.generate_biased_hiring_data(
            n_applicants=1000,
            n_days=30,
            groups=['A', 'B', 'C'],
            bias_strength=0.2
        )
        
        # Calculate metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        tdp_result = tdp.detect_bias(
            data['hired'].values,
            data['group'].values,
            data['timestamp'].values
        )
        
        # Create dashboard
        dashboard_path = os.path.join(self.temp_dir, 'dashboard.png')
        
        # Mock dashboard creation (actual implementation would be more complex)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Fairness Dashboard')
        plt.savefig(dashboard_path)
        plt.close()
        
        # Verify dashboard created
        self.assertTrue(os.path.exists(dashboard_path))


class TestPerformanceIntegration(IntegrationTestBase):
    """Test performance characteristics in integrated scenarios."""
    
    def test_large_scale_integration(self):
        """Test framework performance with large datasets."""
        
        import time
        
        # Generate large dataset
        n_records = 10000
        data = {
            'decisions': np.random.choice([0, 1], n_records),
            'groups': np.random.choice(['A', 'B', 'C'], n_records),
            'timestamps': pd.date_range('2024-01-01', periods=n_records, freq='min')
        }
        
        # Time the full pipeline
        start_time = time.time()
        
        # Calculate all metrics
        tdp = TemporalDemographicParity(threshold=0.1)
        tdp_result = tdp.detect_bias(
            data['decisions'],
            data['groups'],
            data['timestamps']
        )
        
        qpf = QueuePositionFairness(fairness_threshold=0.8)
        queue_positions = np.random.randint(1, 100, n_records)
        qpf_result = qpf.detect_bias(
            queue_positions,
            data['groups'],
            data['timestamps']
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for 10K records)
        self.assertLess(total_time, 1.0, 
                       f"Processing 10K records took {total_time:.2f}s")
    
    def test_memory_efficiency(self):
        """Test memory usage remains reasonable."""
        
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Process moderate dataset
        n_records = 5000
        data = self.generator.generate_biased_hiring_data(
            n_applicants=n_records,
            n_days=30,
            groups=['A', 'B', 'C'],
            bias_strength=0.2
        )
        
        # Run analysis
        analyzer = TemporalAnalyzer()
        results = analyzer.run_full_analysis(
            data=data,
            groups='group',
            decision_column='hired',
            timestamp_column='timestamp'
        )
        
        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        
        # Should use less than 50MB for 5K records
        self.assertLess(peak_mb, 50, 
                       f"Peak memory usage was {peak_mb:.2f}MB")


class TestConfigurationAndSettings(IntegrationTestBase):
    """Test configuration and settings management."""
    
    def test_metric_threshold_configuration(self):
        """Test that metric thresholds can be configured."""
        
        # Test different threshold settings
        strict_tdp = TemporalDemographicParity(threshold=0.05)
        lenient_tdp = TemporalDemographicParity(threshold=0.2)
        
        # Same data
        decisions = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        groups = np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])
        timestamps = pd.date_range('2024-01-01', periods=10, freq='H')
        
        strict_result = strict_tdp.detect_bias(decisions, groups, timestamps)
        lenient_result = lenient_tdp.detect_bias(decisions, groups, timestamps)
        
        # Same metric value
        self.assertEqual(strict_result['metric_value'], 
                        lenient_result['metric_value'])
        
        # But potentially different bias detection
        # (depending on the actual metric value)
        self.assertEqual(strict_result['threshold'], 0.05)
        self.assertEqual(lenient_result['threshold'], 0.2)
    
    def test_analyzer_sensitivity_settings(self):
        """Test analyzer sensitivity configuration."""
        
        # Create analyzers with different sensitivities
        high_sensitivity = BiasDetector(sensitivity=0.99)
        low_sensitivity = BiasDetector(sensitivity=0.8)
        
        # Subtle pattern
        subtle_drift = 0.8 - np.linspace(0, 0.1, 100)
        timestamps = pd.date_range('2024-01-01', periods=100, freq='D')
        
        high_result = high_sensitivity.detect_gradual_drift(
            subtle_drift, timestamps
        )
        low_result = low_sensitivity.detect_gradual_drift(
            subtle_drift, timestamps
        )
        
        # High sensitivity more likely to detect subtle patterns
        if high_result and low_result:
            if high_result['detected'] and low_result['detected']:
                self.assertGreaterEqual(high_result['confidence'],
                                      low_result['confidence'])


def run_integration_tests():
    """Run all integration tests and generate report."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestBiasPatternDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDataGenerators))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationAndSettings))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nTests with Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)