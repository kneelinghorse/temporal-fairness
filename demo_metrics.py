"""
Demo script showcasing EOOT and FDD metrics integration.
"""

import numpy as np
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.utils.data_generators import TemporalBiasGenerator


def main():
    print("=" * 60)
    print("TEMPORAL FAIRNESS METRICS DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data with increasing bias
    generator = TemporalBiasGenerator(random_seed=42)
    df = generator.generate_dataset(
        n_samples=2000,
        n_groups=3,
        bias_type='increasing',
        bias_strength=0.3,
        time_periods=10
    )
    
    # Create synthetic true labels for EOOT
    np.random.seed(42)
    true_labels = np.zeros(len(df))
    for i in range(len(df)):
        if df.iloc[i]['group'] == 0:
            true_labels[i] = np.random.binomial(1, 0.6)
        else:
            true_labels[i] = np.random.binomial(1, 0.4)
    
    print("\n1. EQUALIZED ODDS OVER TIME (EOOT)")
    print("-" * 40)
    
    eoot = EqualizedOddsOverTime(tpr_threshold=0.1, fpr_threshold=0.1)
    eoot_result = eoot.detect_bias(
        predictions=df['decision'].values,
        true_labels=true_labels,
        groups=df['group'].values,
        timestamps=df['time_period'].values
    )
    
    print(f"Bias Detected: {eoot_result['bias_detected']}")
    print(f"Max EOOT Value: {eoot_result['metric_value']:.3f}")
    print(f"Bias Source: {', '.join(eoot_result['bias_source']) if eoot_result['bias_source'] else 'None'}")
    print(f"Severity: {eoot_result['severity']}")
    print(f"Confidence: {eoot_result['confidence']:.1%}")
    
    print("\n2. FAIRNESS DECAY DETECTION (FDD)")
    print("-" * 40)
    
    # Calculate TDP values over time for FDD analysis
    tdp = TemporalDemographicParity(threshold=0.1)
    tdp_history = []
    
    for period in sorted(df['time_period'].unique()):
        period_data = df[df['time_period'] == period]
        if len(period_data) >= 30:
            tdp_value = tdp.calculate(
                decisions=period_data['decision'].values,
                groups=period_data['group'].values
            )
            tdp_history.append(tdp_value)
            print(f"Period {int(period)}: TDP = {tdp_value:.3f}")
    
    print("\nAnalyzing TDP trend for decay...")
    fdd = FairnessDecayDetection(decay_threshold=0.05, detection_method='linear')
    decay_result = fdd.detect_fairness_decay(
        metric_history=tdp_history,
        return_details=True
    )
    
    print(f"Decay Detected: {decay_result['decay_detected']}")
    print(f"Decay Type: {decay_result['decay_type']}")
    if 'slope' in decay_result:
        print(f"Decay Rate: {decay_result['slope']:.4f}")
    print(f"Confidence: {decay_result.get('confidence', 0):.1%}")
    
    # Generate alert if needed
    alert = fdd.generate_alert(decay_result, metric_name='TDP')
    if alert:
        print(f"\n⚠️  ALERT: {alert['message']}")
    
    print("\n3. PREDICTIVE ANALYSIS")
    print("-" * 40)
    
    prediction = fdd.predict_future_decay(
        metric_history=tdp_history,
        periods_ahead=3
    )
    
    if prediction['predictions'] is not None:
        print("Predicted TDP values for next 3 periods:")
        for i, pred in enumerate(prediction['predictions']):
            lower = prediction['confidence_lower'][i]
            upper = prediction['confidence_upper'][i]
            print(f"  Period +{i+1}: {pred:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
    
    print("\n4. INTEGRATED ANALYSIS SUMMARY")
    print("-" * 40)
    
    # Overall fairness assessment
    if eoot_result['bias_detected'] and decay_result['decay_detected']:
        print("⚠️  CRITICAL: Both current bias and degradation trend detected")
        print("   Immediate intervention recommended")
    elif eoot_result['bias_detected']:
        print("⚠️  WARNING: Current bias detected but no degradation trend")
        print("   Monitor closely and consider mitigation")
    elif decay_result['decay_detected']:
        print("⚠️  WARNING: Fairness degradation trend detected")
        print("   Preventive measures recommended")
    else:
        print("✓  System appears fair with stable metrics")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()