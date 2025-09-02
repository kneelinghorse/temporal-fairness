"""
Test suite for Mitigation Strategy Selector
Validates context-aware strategy selection and effectiveness
"""

import numpy as np
import time
import sys
import os
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.bias_classifier import (
    BiasCategory,
    BiasDetection
)
from src.mitigation.strategy_selector import (
    MitigationStrategySelector,
    SystemContext,
    MitigationTechnique,
    PerformanceMetrics
)


def create_mock_bias_detections(categories: List[BiasCategory], 
                               severities: List[str] = None) -> List[BiasDetection]:
    """Create mock bias detections for testing"""
    
    if severities is None:
        severities = ['HIGH'] * len(categories)
    
    detections = []
    for cat, severity in zip(categories, severities):
        detection = BiasDetection(
            category=cat,
            confidence=0.85 + np.random.random() * 0.15,
            indicators=[f"Test indicator for {cat.value}"],
            severity=severity,
            mitigation_recommendations=["Test recommendation"],
            evidence={"test": "evidence"}
        )
        detections.append(detection)
    
    return detections


def test_queue_system_strategy():
    """Test strategy selection for queue-based systems"""
    print("\n" + "="*60)
    print("Testing Queue-Based System Strategy")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    # Create bias detections typical for queue systems
    detections = create_mock_bias_detections(
        [BiasCategory.AGGREGATION, BiasCategory.HISTORICAL],
        ['HIGH', 'MEDIUM']
    )
    
    # Select strategy
    recommendation = selector.select_strategy(
        bias_detections=detections,
        system_context=SystemContext.QUEUE_BASED,
        performance_requirements={
            'min_accuracy_retention': 0.95,
            'max_latency_ms': 5,
            'min_throughput': 2000
        }
    )
    
    print(f"\nPrimary Strategy: {recommendation.primary_strategy.technique.value}")
    print(f"Expected Effectiveness: {recommendation.primary_strategy.expected_effectiveness:.2%}")
    print(f"Accuracy Retention: {recommendation.primary_strategy.accuracy_retention:.2%}")
    print(f"Latency Overhead: {recommendation.primary_strategy.latency_overhead_ms:.1f}ms")
    print(f"Complexity: {recommendation.primary_strategy.implementation_complexity}")
    
    print(f"\nExpected Bias Reduction: {recommendation.expected_bias_reduction:.2%}")
    print(f"Selection Confidence: {recommendation.confidence:.2%}")
    print(f"Selection Time: {recommendation.selection_time_ms:.2f}ms")
    
    # Check for expected strategy (should be fairness batch sampling)
    expected = MitigationTechnique.FAIRNESS_BATCH_SAMPLING
    success = recommendation.primary_strategy.technique == expected
    
    print(f"\nValidation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"Expected: {expected.value}, Got: {recommendation.primary_strategy.technique.value}")
    
    return success, recommendation.selection_time_ms


def test_urgency_scoring_strategy():
    """Test strategy selection for urgency scoring systems"""
    print("\n" + "="*60)
    print("Testing Urgency Scoring System Strategy")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    # Create bias detections typical for urgency scoring
    detections = create_mock_bias_detections(
        [BiasCategory.REPRESENTATION, BiasCategory.MEASUREMENT],
        ['CRITICAL', 'HIGH']
    )
    
    # Select strategy
    recommendation = selector.select_strategy(
        bias_detections=detections,
        system_context=SystemContext.URGENCY_SCORING,
        performance_requirements={
            'min_accuracy_retention': 0.92,
            'max_latency_ms': 10,
            'min_throughput': 1000
        }
    )
    
    print(f"\nPrimary Strategy: {recommendation.primary_strategy.technique.value}")
    print(f"Expected Effectiveness: {recommendation.primary_strategy.expected_effectiveness:.2%}")
    print(f"Configuration: {recommendation.primary_strategy.configuration}")
    
    print(f"\nAlternative Strategies:")
    for alt in recommendation.alternative_strategies[:2]:
        print(f"  - {alt.technique.value}: {alt.expected_effectiveness:.2%} effectiveness")
    
    # Check for expected strategy (should be adversarial debiasing)
    expected = MitigationTechnique.ADVERSARIAL_DEBIASING
    success = recommendation.primary_strategy.technique == expected
    
    print(f"\nValidation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"Expected: {expected.value}, Got: {recommendation.primary_strategy.technique.value}")
    
    return success, recommendation.selection_time_ms


def test_sequential_decisions_strategy():
    """Test strategy selection for sequential decision systems"""
    print("\n" + "="*60)
    print("Testing Sequential Decisions Strategy")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    # Create bias detections
    detections = create_mock_bias_detections(
        [BiasCategory.MEASUREMENT, BiasCategory.AGGREGATION],
        ['MEDIUM', 'MEDIUM']
    )
    
    # Select strategy
    recommendation = selector.select_strategy(
        bias_detections=detections,
        system_context=SystemContext.SEQUENTIAL_DECISIONS
    )
    
    print(f"\nPrimary Strategy: {recommendation.primary_strategy.technique.value}")
    print(f"Rationale: {recommendation.primary_strategy.rationale}")
    print(f"Monitoring Metrics: {', '.join(recommendation.primary_strategy.monitoring_metrics)}")
    
    # Check for expected strategy (should be post-processing)
    expected = MitigationTechnique.POST_PROCESSING
    success = recommendation.primary_strategy.technique == expected
    
    print(f"\nValidation: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return success, recommendation.selection_time_ms


def test_batch_processing_strategy():
    """Test strategy selection for batch processing systems"""
    print("\n" + "="*60)
    print("Testing Batch Processing Strategy")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    # Create bias detections
    detections = create_mock_bias_detections(
        [BiasCategory.AGGREGATION],
        ['HIGH']
    )
    
    # Select strategy
    recommendation = selector.select_strategy(
        bias_detections=detections,
        system_context=SystemContext.BATCH_PROCESSING
    )
    
    print(f"\nPrimary Strategy: {recommendation.primary_strategy.technique.value}")
    print(f"Expected Effectiveness: {recommendation.primary_strategy.expected_effectiveness:.2%}")
    
    # Check for expected strategy
    expected = MitigationTechnique.FAIRNESS_BATCH_SAMPLING
    success = recommendation.primary_strategy.technique == expected
    
    # Verify effectiveness is in expected range (93% for batch sampling)
    effectiveness_check = recommendation.primary_strategy.expected_effectiveness >= 0.90
    
    print(f"\nValidation: {'✓ PASSED' if success and effectiveness_check else '✗ FAILED'}")
    print(f"Effectiveness Check: {recommendation.primary_strategy.expected_effectiveness:.2%} >= 90%")
    
    return success and effectiveness_check, recommendation.selection_time_ms


def test_combined_approach():
    """Test combined mitigation approach for multiple bias types"""
    print("\n" + "="*60)
    print("Testing Combined Mitigation Approach")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    # Create multiple bias detections
    detections = create_mock_bias_detections(
        [BiasCategory.HISTORICAL, BiasCategory.REPRESENTATION, 
         BiasCategory.MEASUREMENT, BiasCategory.AGGREGATION],
        ['HIGH', 'CRITICAL', 'MEDIUM', 'HIGH']
    )
    
    # Select strategy
    recommendation = selector.select_strategy(
        bias_detections=detections,
        system_context=SystemContext.QUEUE_BASED
    )
    
    print(f"\nMultiple Bias Categories Detected: {len(detections)}")
    print(f"Primary Strategy: {recommendation.primary_strategy.technique.value}")
    
    if recommendation.combined_approach:
        print("\nCombined Approach Pipeline:")
        for stage in recommendation.combined_approach['stages']:
            print(f"  - {stage['stage']}: {stage['technique']} → {stage['target']}")
        print(f"Expected Improvement: {recommendation.combined_approach['expected_improvement']:.2%}")
    
    success = recommendation.combined_approach is not None
    print(f"\nCombined Approach: {'✓ CREATED' if success else '✗ NOT CREATED'}")
    
    return success, recommendation.selection_time_ms


def test_performance_requirements():
    """Test performance requirements (<100ms selection time)"""
    print("\n" + "="*60)
    print("Testing Performance Requirements")
    print("Target: <100ms selection time")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    test_cases = [
        (SystemContext.QUEUE_BASED, [BiasCategory.AGGREGATION]),
        (SystemContext.URGENCY_SCORING, [BiasCategory.REPRESENTATION, BiasCategory.MEASUREMENT]),
        (SystemContext.SEQUENTIAL_DECISIONS, [BiasCategory.MEASUREMENT]),
        (SystemContext.BATCH_PROCESSING, [BiasCategory.AGGREGATION, BiasCategory.HISTORICAL])
    ]
    
    times = []
    
    for context, categories in test_cases:
        detections = create_mock_bias_detections(categories)
        
        # Run multiple times for average
        case_times = []
        for _ in range(10):
            start = time.perf_counter()
            recommendation = selector.select_strategy(
                bias_detections=detections,
                system_context=context
            )
            elapsed = (time.perf_counter() - start) * 1000
            case_times.append(elapsed)
        
        avg_time = np.mean(case_times)
        times.append(avg_time)
        
        print(f"\n{context.value}: {avg_time:.2f}ms (avg)")
        print(f"  Min: {min(case_times):.2f}ms, Max: {max(case_times):.2f}ms")
    
    overall_avg = np.mean(times)
    success = all(t < 100 for t in times)
    
    print(f"\nOverall Average: {overall_avg:.2f}ms")
    print(f"Performance: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return success, overall_avg


def test_performance_monitoring():
    """Test performance monitoring and alerting"""
    print("\n" + "="*60)
    print("Testing Performance Monitoring")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    # Create a strategy
    detections = create_mock_bias_detections([BiasCategory.HISTORICAL])
    recommendation = selector.select_strategy(
        bias_detections=detections,
        system_context=SystemContext.QUEUE_BASED
    )
    
    # Simulate performance metrics
    print("\nSimulating performance tracking...")
    
    # Good performance
    good_metrics = PerformanceMetrics(
        fairness_improvement=0.75,
        accuracy_retention=0.92,
        latency_ms=3.5,
        throughput_per_second=1500,
        timestamp=time.time()
    )
    
    selector.monitor_performance(recommendation.primary_strategy, good_metrics)
    print("  ✓ Logged good performance")
    
    # Poor performance (should trigger alert)
    poor_metrics = PerformanceMetrics(
        fairness_improvement=0.35,  # Below 70% of expected
        accuracy_retention=0.85,  # Below 90% threshold
        latency_ms=15.0,
        throughput_per_second=800,
        timestamp=time.time()
    )
    
    print("\nExpecting alerts for poor performance:")
    selector.monitor_performance(recommendation.primary_strategy, poor_metrics)
    
    # Get performance summary
    summary = selector.get_performance_summary()
    
    print("\nPerformance Summary:")
    for technique, stats in summary.items():
        print(f"  {technique}:")
        print(f"    Fairness Improvement: {stats['avg_fairness_improvement']:.2%}")
        print(f"    Accuracy Retention: {stats['avg_accuracy_retention']:.2%}")
        print(f"    Latency: {stats['avg_latency_ms']:.2f}ms")
    
    success = len(summary) > 0
    print(f"\nMonitoring System: {'✓ ACTIVE' if success else '✗ INACTIVE'}")
    
    return success, 0


def test_effectiveness_rates():
    """Test that strategies achieve documented effectiveness rates"""
    print("\n" + "="*60)
    print("Testing Documented Effectiveness Rates")
    print("="*60)
    
    selector = MitigationStrategySelector()
    
    test_cases = {
        MitigationTechnique.REWEIGHTING: (0.77, SystemContext.HISTORICAL_DATA),
        MitigationTechnique.ADVERSARIAL_DEBIASING: (0.92, SystemContext.URGENCY_SCORING),
        MitigationTechnique.POST_PROCESSING: (0.65, SystemContext.SEQUENTIAL_DECISIONS),
        MitigationTechnique.FAIRNESS_BATCH_SAMPLING: (0.93, SystemContext.BATCH_PROCESSING)
    }
    
    results = []
    
    for technique, (expected_rate, context) in test_cases.items():
        # Create appropriate bias detections for each technique
        if technique == MitigationTechnique.REWEIGHTING:
            categories = [BiasCategory.HISTORICAL]
        elif technique == MitigationTechnique.ADVERSARIAL_DEBIASING:
            categories = [BiasCategory.REPRESENTATION]
        elif technique == MitigationTechnique.POST_PROCESSING:
            categories = [BiasCategory.MEASUREMENT]
        else:  # FAIRNESS_BATCH_SAMPLING
            categories = [BiasCategory.AGGREGATION]
        
        detections = create_mock_bias_detections(categories)
        
        recommendation = selector.select_strategy(
            bias_detections=detections,
            system_context=context
        )
        
        actual_rate = recommendation.primary_strategy.expected_effectiveness
        matches = abs(actual_rate - expected_rate) < 0.05
        
        print(f"\n{technique.value}:")
        print(f"  Expected: {expected_rate:.2%}")
        print(f"  Actual: {actual_rate:.2%}")
        print(f"  Status: {'✓ MATCH' if matches else '✗ MISMATCH'}")
        
        results.append(matches)
    
    success = all(results)
    print(f"\nEffectiveness Validation: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return success, 0


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MITIGATION STRATEGY SELECTOR TEST SUITE")
    print("="*60)
    
    tests = [
        ("Queue System Strategy", test_queue_system_strategy),
        ("Urgency Scoring Strategy", test_urgency_scoring_strategy),
        ("Sequential Decisions Strategy", test_sequential_decisions_strategy),
        ("Batch Processing Strategy", test_batch_processing_strategy),
        ("Combined Approach", test_combined_approach),
        ("Performance Requirements", test_performance_requirements),
        ("Performance Monitoring", test_performance_monitoring),
        ("Effectiveness Rates", test_effectiveness_rates)
    ]
    
    results = []
    total_time = 0
    
    for test_name, test_func in tests:
        try:
            success, elapsed = test_func()
            results.append((test_name, success))
            total_time += elapsed
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Average Selection Time: {total_time/len([r for r in results if r[0] != 'Performance Monitoring']):.2f}ms")
    
    if passed == total:
        print("\n🎉 All tests passed! Strategy selector ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues.")


if __name__ == "__main__":
    main()