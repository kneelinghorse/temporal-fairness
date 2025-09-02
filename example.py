"""
Temporal Fairness Metrics - Example Usage

This example demonstrates how temporal bias can hide in systems that appear fair
using traditional metrics, and how our temporal metrics reveal this hidden discrimination.
"""

import numpy as np
from datetime import datetime, timedelta
from metrics import TemporalFairnessMetrics, FairnessResult


def generate_biased_queue_data(n_customers=1000):
    """
    Simulate a customer support queue with subtle temporal bias.
    
    Group 0 customers systematically get pushed back during peak hours,
    even though overall service rates appear equal.
    """
    np.random.seed(42)
    
    # Generate customer data
    groups = np.random.choice([0, 1], size=n_customers, p=[0.4, 0.6])
    base_time = datetime.now() - timedelta(days=7)
    
    timestamps = []
    queue_positions = []
    
    for i in range(n_customers):
        # Add some time variation
        time_offset = timedelta(hours=np.random.uniform(0, 168))  # Week of data
        timestamps.append(base_time + time_offset)
        
        # During business hours (9-5), Group 0 gets pushed back
        hour = (base_time + time_offset).hour
        is_business_hours = 9 <= hour <= 17
        
        if groups[i] == 0 and is_business_hours:
            # Group 0 during peak: positions 15-30
            position = np.random.randint(15, 31)
        elif groups[i] == 0:
            # Group 0 off-peak: positions 5-15
            position = np.random.randint(5, 16)
        else:
            # Group 1 anytime: positions 1-20
            position = np.random.randint(1, 21)
        
        queue_positions.append(position)
    
    return {
        'groups': np.array(groups),
        'queue_positions': np.array(queue_positions),
        'timestamps': np.array([int(t.timestamp()) for t in timestamps])
    }


def generate_degrading_fairness_data():
    """
    Simulate a loan approval system that starts fair but degrades over time.
    
    This pattern is common when models are retrained on biased feedback loops.
    """
    np.random.seed(42)
    
    # Generate 6 months of measurements
    n_measurements = 180  # Daily measurements
    timestamps = []
    fairness_scores = []
    
    base_time = datetime.now() - timedelta(days=180)
    
    for day in range(n_measurements):
        timestamp = base_time + timedelta(days=day)
        timestamps.append(timestamp)
        
        # Fairness starts good (0.05) and degrades to bad (0.25)
        # with some noise
        base_score = 0.05 + (day / n_measurements) * 0.20
        noise = np.random.normal(0, 0.02)
        score = max(0, min(1, base_score + noise))
        
        fairness_scores.append(score)
    
    return fairness_scores, timestamps


def generate_temporal_decision_data(n_decisions=5000):
    """
    Simulate an automated hiring system with temporal discrimination.
    
    Group 0 applications get processed more slowly, creating temporal bias
    even though the final approval rates are similar.
    """
    np.random.seed(42)
    
    groups = np.random.choice([0, 1], size=n_decisions, p=[0.3, 0.7])
    
    # Generate over a month
    base_time = int((datetime.now() - timedelta(days=30)).timestamp())
    timestamps = base_time + np.sort(np.random.randint(0, 30*24*3600, size=n_decisions))
    
    decisions = []
    predictions = []
    labels = []
    
    for i in range(n_decisions):
        # True qualification is similar across groups
        qualified = np.random.random() < 0.3
        labels.append(int(qualified))
        
        # Model has slight bias that varies by time of day
        hour = datetime.fromtimestamp(timestamps[i]).hour
        
        if groups[i] == 0:
            # Group 0: Lower acceptance during business hours
            if 9 <= hour <= 17:
                threshold = 0.35 if qualified else 0.70
            else:
                threshold = 0.30 if qualified else 0.65
        else:
            # Group 1: Consistent threshold
            threshold = 0.30 if qualified else 0.65
        
        prediction = int(np.random.random() < threshold)
        predictions.append(prediction)
        
        # Actual decision (with some human override)
        if np.random.random() < 0.9:
            decisions.append(prediction)
        else:
            decisions.append(int(qualified))
    
    return {
        'decisions': np.array(decisions),
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'groups': np.array(groups),
        'timestamps': np.array(timestamps)
    }


def main():
    """Run comprehensive temporal fairness analysis."""
    
    print("=" * 70)
    print("TEMPORAL FAIRNESS METRICS DEMONSTRATION")
    print("=" * 70)
    print("\nTraditional metrics often miss temporal discrimination.")
    print("These examples show how our metrics catch what others miss.\n")
    
    # Initialize metrics calculator
    metrics = TemporalFairnessMetrics(fairness_threshold=0.1)
    
    # Example 1: Queue Position Fairness
    print("-" * 70)
    print("EXAMPLE 1: Customer Support Queue Discrimination")
    print("-" * 70)
    
    queue_data = generate_biased_queue_data()
    qpf_result = metrics.qpf(
        queue_data['queue_positions'],
        queue_data['groups']
    )
    
    print(f"\n{qpf_result}")
    print(f"\nDetails:")
    for key, value in qpf_result.details.items():
        if key == 'avg_positions':
            print(f"  Average queue positions by group:")
            for group, pos in value.items():
                print(f"    Group {group}: {pos:.1f}")
        else:
            print(f"  {key}: {value}")
    
    if not qpf_result.is_fair:
        print("\n⚠️  BIAS DETECTED: One group systematically waits longer!")
    
    # Example 2: Temporal Demographic Parity
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Hiring Decision Temporal Bias")
    print("-" * 70)
    
    hiring_data = generate_temporal_decision_data()
    tdp_result = metrics.tdp(
        hiring_data['decisions'],
        hiring_data['groups'],
        hiring_data['timestamps'],
        window_size=3600*24  # Daily windows
    )
    
    print(f"\n{tdp_result}")
    print(f"\nDetails:")
    print(f"  Maximum disparity in any time window: {tdp_result.details['max_disparity']:.3f}")
    print(f"  Number of time windows analyzed: {tdp_result.details['n_windows']}")
    
    # Example 3: Equalized Odds Over Time
    print("\n" + "-" * 70)
    print("EXAMPLE 3: Model Accuracy Consistency Over Time")
    print("-" * 70)
    
    eoot_result = metrics.eoot(
        hiring_data['predictions'],
        hiring_data['labels'],
        hiring_data['groups'],
        hiring_data['timestamps'],
        window_size=3600*24*7  # Weekly windows
    )
    
    print(f"\n{eoot_result}")
    print(f"\nDetails:")
    print(f"  True Positive Rate disparity: {eoot_result.details['tpr_disparity']:.3f}")
    print(f"  False Positive Rate disparity: {eoot_result.details['fpr_disparity']:.3f}")
    
    # Example 4: Fairness Decay Detection
    print("\n" + "-" * 70)
    print("EXAMPLE 4: Model Fairness Degradation Over Time")
    print("-" * 70)
    
    fairness_scores, timestamps = generate_degrading_fairness_data()
    fdd_result = metrics.fdd(
        fairness_scores,
        timestamps,
        decay_period_days=90  # Check last 3 months
    )
    
    print(f"\n{fdd_result}")
    print(f"\nDetails:")
    print(f"  Daily decay rate: {fdd_result.details['daily_decay_rate']:.5f}")
    print(f"  Interpretation: Bias increases by {fdd_result.value:.3f} points per month")
    
    if not fdd_result.is_fair:
        print("\n⚠️  WARNING: Model fairness is degrading over time!")
        print("  Consider retraining with fairness constraints.")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Why Temporal Fairness Matters")
    print("=" * 70)
    
    print("""
Traditional fairness metrics would show these systems as "fair" because:
- Overall approval/service rates are similar between groups
- When averaged across all time, disparities disappear
- Point-in-time measurements miss temporal patterns

But temporal fairness metrics reveal:
- Systematic delays and queue discrimination (QPF)
- Time-varying bias in decisions (TDP)
- Inconsistent model accuracy across groups over time (EOOT)
- Gradual fairness degradation that compounds over months (FDD)

Real-world impact:
- Michigan unemployment system: 40,000 false fraud accusations
- Healthcare algorithms: 200M Americans affected by temporal bias
- Both passed traditional fairness tests but would fail temporal checks
    """)
    
    print("\n📊 Run all metrics on your data:")
    print("   results = metrics.evaluate_all(your_data)")
    print("\n📚 Learn more about the research:")
    print("   53% of AI bias is temporal - traditional metrics miss most discrimination")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
