#!/usr/bin/env python3
"""
Quick demonstration of Temporal Demographic Parity metric.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.utils.data_generators import TemporalBiasGenerator
import numpy as np

def main():
    print("=" * 60)
    print("Temporal Demographic Parity (TDP) Metric Demo")
    print("=" * 60)
    
    # Generate synthetic data with increasing bias over time
    print("\n1. Generating synthetic data with increasing temporal bias...")
    generator = TemporalBiasGenerator(random_seed=42)
    df = generator.generate_dataset(
        n_samples=5000,
        n_groups=2,
        bias_type='increasing',
        bias_strength=0.3,
        time_periods=10
    )
    
    print(f"   - Generated {len(df)} samples")
    print(f"   - Groups: {df['group'].unique()}")
    print(f"   - Time periods: {df['time_period'].unique()}")
    
    # Calculate TDP
    print("\n2. Calculating Temporal Demographic Parity...")
    tdp = TemporalDemographicParity(threshold=0.1)
    
    # Get detailed results
    results = tdp.calculate(
        df['decision'].values,
        df['group'].values,
        df['time_period'].values,
        return_details=True
    )
    
    print(f"   - Maximum TDP: {results['max_tdp']:.3f}")
    print(f"   - Mean TDP: {results['mean_tdp']:.3f}")
    print(f"   - Is Fair (threshold={tdp.threshold})?: {results['is_fair']}")
    
    # Show TDP progression over time
    print("\n3. TDP values across time windows:")
    for i, window in enumerate(results['windows'][:5]):  # Show first 5 windows
        print(f"   Window {i}: TDP = {window['tdp']:.3f}, " +
              f"Samples = {window['n_samples']}")
    
    # Detect bias
    print("\n4. Bias Detection Analysis...")
    detection = tdp.detect_bias(
        df['decision'].values,
        df['group'].values,
        df['time_period'].values
    )
    
    print(f"   - Bias Detected: {detection['bias_detected']}")
    print(f"   - Severity: {detection['severity']}")
    print(f"   - Confidence: {detection['confidence']:.2%}")
    
    # Compare different bias patterns
    print("\n5. Comparing Different Bias Patterns:")
    bias_patterns = ['constant', 'increasing', 'confidence_valley']
    
    for pattern in bias_patterns:
        df_pattern = generator.generate_dataset(
            n_samples=1000,
            bias_type=pattern,
            bias_strength=0.2
        )
        tdp_value = tdp.calculate(
            df_pattern['decision'].values,
            df_pattern['group'].values,
            df_pattern['time_period'].values
        )
        print(f"   - {pattern:20s}: TDP = {tdp_value:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()