"""
Test suite for Bias Taxonomy Classifier
Validates against known case studies: Optum and MiDAS
"""

import numpy as np
import time
from src.analysis.bias_classifier import (
    BiasTaxonomyClassifier, 
    BiasCategory,
    ClassificationResult
)


def generate_optum_case_data():
    """
    Generate synthetic data mimicking Optum case patterns
    Should detect: Historical bias + Measurement bias
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create base data
    data = np.random.randn(n_samples, n_features)
    
    # Add historical bias pattern (spending patterns) - stronger signal
    historical_effect = np.linspace(0, 3, n_samples).reshape(-1, 1)
    data[:, :3] += historical_effect * 0.8
    
    # Add measurement bias pattern (systematic differences)
    group_assignment = np.random.choice([0, 1], n_samples)
    for i, group in enumerate(group_assignment):
        if group == 1:
            data[i, 3:6] *= 0.7  # Systematic undervaluation
    
    # Create temporal metadata
    temporal_metadata = {
        'timestamps': np.arange(n_samples),
        'outcomes': data[:, 0] + np.random.randn(n_samples) * 0.1
    }
    
    # Protected attributes showing correlation
    protected_attributes = {
        'race': group_assignment,
        'income_level': np.random.randn(n_samples) + group_assignment * 0.8
    }
    
    return data, temporal_metadata, protected_attributes


def generate_midas_case_data():
    """
    Generate synthetic data mimicking MiDAS case patterns
    Should detect: Aggregation bias + Measurement bias
    """
    np.random.seed(43)
    n_samples = 1200
    n_features = 12
    
    # Create base data
    data = np.random.randn(n_samples, n_features)
    
    # Add aggregation bias pattern (batch effects) - stronger signal
    # But don't add linear trends to avoid historical bias detection
    batch_size = 100
    for batch_idx in range(n_samples // batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        # Create batch effects with batch-specific variance (not linearly increasing)
        batch_effect = np.random.randn(1, n_features) * (1.5 if batch_idx % 2 == 0 else 0.5)
        data[start_idx:end_idx] += batch_effect
    
    # Add measurement bias (differential scoring)
    group_assignment = np.random.choice([0, 1, 2], n_samples)
    for i, group in enumerate(group_assignment):
        if group == 0:
            data[i, 6:9] *= 1.3  # Overvaluation
        elif group == 2:
            data[i, 6:9] *= 0.6  # Undervaluation
    
    # Create temporal metadata with batch information
    batch_ids = np.repeat(np.arange(n_samples // batch_size), batch_size)
    if len(batch_ids) < n_samples:
        batch_ids = np.concatenate([batch_ids, [batch_ids[-1]] * (n_samples - len(batch_ids))])
    
    temporal_metadata = {
        'batch_ids': batch_ids,
        'timestamps': np.arange(n_samples) + np.random.randn(n_samples) * 10
    }
    
    # Protected attributes
    protected_attributes = {
        'disability_status': group_assignment
    }
    
    return data, temporal_metadata, protected_attributes


def test_optum_case():
    """Test Optum case study detection"""
    print("\n" + "="*60)
    print("Testing Optum Case Study")
    print("Expected: Historical Bias + Measurement Bias")
    print("="*60)
    
    classifier = BiasTaxonomyClassifier()
    data, temporal_metadata, protected_attributes = generate_optum_case_data()
    
    # Run classification
    result = classifier.classify(
        data=data,
        temporal_metadata=temporal_metadata,
        protected_attributes=protected_attributes
    )
    
    # Display results
    print(f"\nProcessing Time: {result.processing_time_ms:.2f}ms")
    print(f"Overall Risk: {result.overall_risk}")
    print(f"Primary Bias: {result.primary_bias.value if result.primary_bias else 'None'}")
    
    print("\nDetected Biases:")
    for detection in result.detections:
        print(f"  - {detection.category.value}")
        print(f"    Confidence: {detection.confidence:.2%}")
        print(f"    Severity: {detection.severity}")
        print(f"    Indicators: {', '.join(detection.indicators[:2])}")
    
    # Validate
    expected = [BiasCategory.HISTORICAL, BiasCategory.MEASUREMENT]
    success, details = classifier.validate_case_study(
        "Optum", data, expected,
        temporal_metadata=temporal_metadata,
        protected_attributes=protected_attributes
    )
    
    print(f"\nValidation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"Accuracy: {details['accuracy']:.2%}")
    
    if details['missed']:
        print(f"Missed: {[b.value for b in details['missed']]}")
    if details['false_positives']:
        print(f"False Positives: {[b.value for b in details['false_positives']]}")
    
    return success


def test_midas_case():
    """Test MiDAS case study detection"""
    print("\n" + "="*60)
    print("Testing MiDAS Case Study")
    print("Expected: Aggregation Bias + Measurement Bias")
    print("="*60)
    
    classifier = BiasTaxonomyClassifier()
    data, temporal_metadata, protected_attributes = generate_midas_case_data()
    
    # Run classification
    result = classifier.classify(
        data=data,
        temporal_metadata=temporal_metadata,
        protected_attributes=protected_attributes
    )
    
    # Display results
    print(f"\nProcessing Time: {result.processing_time_ms:.2f}ms")
    print(f"Overall Risk: {result.overall_risk}")
    print(f"Primary Bias: {result.primary_bias.value if result.primary_bias else 'None'}")
    
    print("\nDetected Biases:")
    for detection in result.detections:
        print(f"  - {detection.category.value}")
        print(f"    Confidence: {detection.confidence:.2%}")
        print(f"    Severity: {detection.severity}")
        print(f"    Indicators: {', '.join(detection.indicators[:2])}")
    
    # Validate
    expected = [BiasCategory.AGGREGATION, BiasCategory.MEASUREMENT]
    success, details = classifier.validate_case_study(
        "MiDAS", data, expected,
        temporal_metadata=temporal_metadata,
        protected_attributes=protected_attributes
    )
    
    print(f"\nValidation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"Accuracy: {details['accuracy']:.2%}")
    
    if details['missed']:
        print(f"Missed: {[b.value for b in details['missed']]}")
    if details['false_positives']:
        print(f"False Positives: {[b.value for b in details['false_positives']]}")
    
    return success


def test_performance():
    """Test performance requirements (<10ms)"""
    print("\n" + "="*60)
    print("Testing Performance Requirements")
    print("Target: <10ms per classification")
    print("="*60)
    
    classifier = BiasTaxonomyClassifier()
    
    # Test with various data sizes
    test_sizes = [(100, 10), (500, 20), (1000, 30)]
    
    for n_samples, n_features in test_sizes:
        data = np.random.randn(n_samples, n_features)
        
        # Run multiple times for average
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = classifier.classify(data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\nSize {n_samples}x{n_features}: {avg_time:.2f}ms (avg)")
        print(f"  Min: {min(times):.2f}ms, Max: {max(times):.2f}ms")
        
        if avg_time < 10:
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED")
    
    return all(np.mean(times) < 10 for times in [times])


def test_multi_label_detection():
    """Test multi-label classification capability"""
    print("\n" + "="*60)
    print("Testing Multi-Label Classification")
    print("="*60)
    
    classifier = BiasTaxonomyClassifier()
    
    # Create data with multiple bias patterns
    np.random.seed(44)
    n_samples = 800
    n_features = 20
    
    data = np.random.randn(n_samples, n_features)
    
    # Add all four bias patterns with stronger signals
    # Historical - stronger linear trend
    data[:, :3] += np.linspace(0, 4, n_samples).reshape(-1, 1) * 0.8
    
    # Representation - stronger correlation
    protected_attr = np.random.choice([0, 1], n_samples)
    data[:, 5:8] += protected_attr.reshape(-1, 1) * 2.0
    
    # Measurement - stronger systematic differences
    for i in range(n_samples):
        if i % 3 == 0:
            data[i, 10:13] *= 0.3
    
    # Aggregation - stronger batch effects
    batch_ids = np.repeat(np.arange(n_samples // 50), 50)
    if len(batch_ids) < n_samples:
        batch_ids = np.concatenate([batch_ids, [batch_ids[-1]] * (n_samples - len(batch_ids))])
    
    # Add batch-specific effects
    for batch_idx in range(n_samples // 50):
        start_idx = batch_idx * 50
        end_idx = min(start_idx + 50, n_samples)
        data[start_idx:end_idx] += batch_idx * 0.3
    
    temporal_metadata = {
        'timestamps': np.arange(n_samples),
        'batch_ids': batch_ids,
        'outcomes': data[:, 0]
    }
    
    protected_attributes = {
        'test_attribute': protected_attr
    }
    
    result = classifier.classify(
        data=data,
        temporal_metadata=temporal_metadata,
        protected_attributes=protected_attributes
    )
    
    detected_categories = [d.category for d in result.detections]
    
    print(f"\nDetected {len(detected_categories)} bias categories:")
    for detection in result.detections:
        print(f"  - {detection.category.value}: {detection.confidence:.2%} confidence")
    
    print(f"\nInteraction Effects:")
    for effect in result.interaction_effects:
        print(f"  - {effect}")
    
    success = len(detected_categories) >= 2
    print(f"\nMulti-label Detection: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return success


def test_mitigation_recommendations():
    """Test mitigation recommendation system"""
    print("\n" + "="*60)
    print("Testing Mitigation Recommendations")
    print("="*60)
    
    classifier = BiasTaxonomyClassifier()
    data, temporal_metadata, protected_attributes = generate_optum_case_data()
    
    result = classifier.classify(
        data=data,
        temporal_metadata=temporal_metadata,
        protected_attributes=protected_attributes
    )
    
    print("\nMitigation Recommendations by Category:")
    for detection in result.detections:
        print(f"\n{detection.category.value}:")
        for i, recommendation in enumerate(detection.mitigation_recommendations, 1):
            print(f"  {i}. {recommendation}")
    
    # Check that recommendations are provided
    has_recommendations = all(
        len(d.mitigation_recommendations) > 0 
        for d in result.detections
    )
    
    print(f"\nRecommendation System: {'✓ PASSED' if has_recommendations else '✗ FAILED'}")
    
    return has_recommendations


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BIAS TAXONOMY CLASSIFIER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Optum Case Study", test_optum_case),
        ("MiDAS Case Study", test_midas_case),
        ("Performance (<10ms)", test_performance),
        ("Multi-Label Detection", test_multi_label_detection),
        ("Mitigation Recommendations", test_mitigation_recommendations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
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
    
    if passed == total:
        print("\n🎉 All tests passed! Classifier ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues.")


if __name__ == "__main__":
    main()