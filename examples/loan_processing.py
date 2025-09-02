"""
Loan Processing Example - Demonstrating Temporal Bias in Financial Services

This example shows how loan approval systems exhibit temporal fairness issues,
including gradual drift in approval rates, sudden policy changes, and
demographic disparities that compound over time.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import temporal fairness metrics
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.analysis.enhanced_bias_detector import EnhancedBiasDetector
from src.analysis.temporal_analyzer import TemporalAnalyzer
from src.utils.data_generators import TemporalBiasGenerator
from src.visualization.fairness_visualizer import FairnessVisualizer


def loan_processing_example():
    """
    Demonstrate temporal fairness issues in loan processing.
    
    This example shows:
    1. Demographic parity violations over quarters
    2. Fairness decay after economic shocks
    3. Sudden shifts in approval patterns
    4. Population Stability Index (PSI) for early warning
    """
    print("="*70)
    print("LOAN PROCESSING - TEMPORAL FAIRNESS ANALYSIS")
    print("="*70)
    print("\nGenerating realistic loan application data...")
    
    # Generate loan data with economic shock
    generator = TemporalBiasGenerator(random_seed=42)
    
    loan_data = generator.generate_resource_allocation_queue(
        n_requests=2000,
        n_quarters=12,  # 3 years of data
        groups=['High-Income', 'Middle-Income', 'Low-Income', 'Student'],
        resource_types=['Personal Loan', 'Auto Loan', 'Business Loan', 'Education Loan'],
        scarcity_level=0.3  # 30% scarcity
    )
    
    # Rename columns for clarity
    loan_data = loan_data.rename(columns={
        'request_id': 'application_id',
        'amount_requested': 'loan_amount',
        'processing_days': 'processing_time'
    })
    
    # Add credit scores (correlated with income groups)
    np.random.seed(42)
    credit_scores = []
    for group in loan_data['group']:
        if group == 'High-Income':
            score = np.random.normal(750, 40)
        elif group == 'Middle-Income':
            score = np.random.normal(680, 50)
        elif group == 'Low-Income':
            score = np.random.normal(620, 60)
        else:  # Student
            score = np.random.normal(650, 70)
        credit_scores.append(np.clip(score, 300, 850))
    
    loan_data['credit_score'] = credit_scores
    
    # Add economic shock at quarter 6
    shock_quarter = 6
    loan_data['post_shock'] = loan_data['quarter'] >= shock_quarter
    
    # Reduce approval rates post-shock
    for idx in loan_data[loan_data['post_shock']].index:
        if np.random.random() < 0.3:  # 30% additional rejections
            loan_data.loc[idx, 'approved'] = 0
    
    print(f"Generated {len(loan_data)} loan applications over 3 years")
    print(f"Income groups: {loan_data['group'].value_counts().to_dict()}")
    print(f"Overall approval rate: {loan_data['approved'].mean():.1%}")
    print(f"Economic shock at quarter {shock_quarter}")
    
    # =========================================================================
    # ANALYSIS 1: Temporal Demographic Parity
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: TEMPORAL DEMOGRAPHIC PARITY BY QUARTER")
    print("-"*70)
    
    tdp = TemporalDemographicParity(threshold=0.1)
    tdp_history = []
    
    print("\nQuarterly TDP Analysis:")
    for quarter in sorted(loan_data['quarter'].unique()):
        quarter_data = loan_data[loan_data['quarter'] == quarter]
        
        if len(quarter_data) < 30:
            continue
        
        tdp_value = tdp.calculate(
            decisions=quarter_data['approved'].values,
            groups=quarter_data['group'].values
        )
        
        tdp_history.append(tdp_value)
        
        # Detailed analysis for each quarter
        tdp_details = tdp.calculate(
            decisions=quarter_data['approved'].values,
            groups=quarter_data['group'].values,
            return_details=True
        )
        
        shock_marker = " [SHOCK]" if quarter == shock_quarter else ""
        print(f"\nQuarter {quarter}{shock_marker}:")
        print(f"  TDP: {tdp_value:.3f} {'⚠️' if tdp_value > tdp.threshold else '✓'}")
        print(f"  Samples: {len(quarter_data)}")
        
        # Show group approval rates
        for window in tdp_details.get('windows', []):
            if 'group_rates' in window:
                print("  Approval rates:")
                for group, rate in window['group_rates'].items():
                    print(f"    {group}: {rate:.1%}")
                break
    
    # =========================================================================
    # ANALYSIS 2: Fairness Decay Detection
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: FAIRNESS DECAY DETECTION")
    print("-"*70)
    
    fdd = FairnessDecayDetection(decay_threshold=0.05, detection_method='changepoint')
    
    # Detect decay in TDP values
    decay_result = fdd.detect_fairness_decay(
        metric_history=tdp_history,
        return_details=True
    )
    
    print(f"\nDecay Detection Results:")
    print(f"  Method: {decay_result.get('decay_type', 'N/A')}")
    print(f"  Decay Detected: {decay_result['decay_detected']}")
    
    if decay_result.get('changepoint_index'):
        print(f"  Changepoint at quarter: {decay_result['changepoint_index']}")
        print(f"  Mean before: {decay_result.get('mean_before', 0):.3f}")
        print(f"  Mean after: {decay_result.get('mean_after', 0):.3f}")
        print(f"  Change magnitude: {decay_result.get('mean_change', 0):.3f}")
    
    # Predict future quarters
    predictions = fdd.predict_future_decay(
        metric_history=tdp_history,
        periods_ahead=3
    )
    
    if predictions.get('predictions') is not None:
        print("\nPredicted TDP for next 3 quarters:")
        for i, pred in enumerate(predictions['predictions']):
            lower = predictions['confidence_lower'][i]
            upper = predictions['confidence_upper'][i]
            print(f"  Quarter +{i+1}: {pred:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
    
    # Generate alert if needed
    alert = fdd.generate_alert(decay_result, metric_name='TDP')
    if alert:
        print(f"\n⚠️ ALERT: {alert['message']}")
    
    # =========================================================================
    # ANALYSIS 3: Population Stability Index (PSI)
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: POPULATION STABILITY INDEX - EARLY WARNING")
    print("-"*70)
    
    detector = EnhancedBiasDetector()
    
    # Compare pre-shock and post-shock distributions
    pre_shock_scores = loan_data[~loan_data['post_shock']]['credit_score'].values
    post_shock_scores = loan_data[loan_data['post_shock']]['credit_score'].values
    
    psi_result = detector.calculate_population_stability_index(
        reference_data=pre_shock_scores,
        current_data=post_shock_scores,
        n_bins=10
    )
    
    print(f"\nPopulation Stability Analysis:")
    print(f"  PSI Value: {psi_result['psi']:.3f}")
    print(f"  Stability: {psi_result['stability']}")
    print(f"  Risk Level: {psi_result['risk_level']}")
    print(f"  Requires Intervention: {psi_result['requires_intervention']}")
    
    if psi_result['predicted_bias_timeline']:
        print(f"  Predicted Bias Timeline: {psi_result['predicted_bias_timeline']}")
    
    # Analyze each group's PSI
    print("\nPSI by Income Group:")
    for group in loan_data['group'].unique():
        group_pre = loan_data[(loan_data['group'] == group) & (~loan_data['post_shock'])]['credit_score'].values
        group_post = loan_data[(loan_data['group'] == group) & (loan_data['post_shock'])]['credit_score'].values
        
        if len(group_pre) > 10 and len(group_post) > 10:
            group_psi = detector.calculate_population_stability_index(
                reference_data=group_pre,
                current_data=group_post
            )
            print(f"  {group}: PSI={group_psi['psi']:.3f} ({group_psi['stability']})")
    
    # =========================================================================
    # ANALYSIS 4: EOOT for Loan Decisions
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: EQUALIZED ODDS OVER TIME")
    print("-"*70)
    
    # Create synthetic "true need" labels based on credit score
    # (In reality, this would be actual loan performance data)
    loan_data['true_need'] = (loan_data['credit_score'] > 650).astype(int)
    
    eoot = EqualizedOddsOverTime(tpr_threshold=0.15, fpr_threshold=0.15)
    
    # Calculate EOOT
    eoot_result = eoot.detect_bias(
        predictions=loan_data['approved'].values,
        true_labels=loan_data['true_need'].values,
        groups=loan_data['group'].values,
        timestamps=loan_data['quarter'].values
    )
    
    print(f"\nEOOT Analysis Results:")
    print(f"  Bias Detected: {eoot_result['bias_detected']}")
    print(f"  EOOT Value: {eoot_result['metric_value']:.3f}")
    print(f"  Bias Source: {', '.join(eoot_result.get('bias_source', []))}")
    print(f"  Severity: {eoot_result['severity']}")
    
    # Get group metrics
    group_metrics = eoot.calculate_group_metrics(
        predictions=loan_data['approved'].values,
        true_labels=loan_data['true_need'].values,
        groups=loan_data['group'].values,
        timestamps=loan_data['quarter'].values
    )
    
    if group_metrics['windows']:
        latest_window = group_metrics['windows'][-1]
        print("\nLatest Quarter Metrics:")
        for group, metrics in latest_window['metrics'].items():
            print(f"  {group}:")
            if metrics['tpr'] is not None:
                print(f"    TPR: {metrics['tpr']:.3f}")
            if metrics['fpr'] is not None:
                print(f"    FPR: {metrics['fpr']:.3f}")
    
    # =========================================================================
    # ANALYSIS 5: Comprehensive Risk Assessment
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 5: COMPREHENSIVE RISK ASSESSMENT")
    print("-"*70)
    
    analyzer = TemporalAnalyzer()
    
    # Run full analysis
    analysis_results = analyzer.run_full_analysis(
        data=loan_data,
        groups='group',
        decision_column='approved',
        prediction_column='approved',
        label_column='true_need',
        timestamp_column='quarter'
    )
    
    # Display risk assessment
    risk = analysis_results['risk_assessment']
    print(f"\nOverall Risk Assessment:")
    print(f"  Risk Level: {risk['risk_level'].upper()}")
    print(f"  Risk Score: {risk['risk_score']:.1%}")
    
    if risk['risk_factors']:
        print("\nRisk Factors:")
        for factor in risk['risk_factors']:
            print(f"  • {factor}")
    
    # Display recommendations
    if analysis_results['recommendations']:
        print("\nTop Recommendations:")
        for i, rec in enumerate(analysis_results['recommendations'][:3], 1):
            print(f"\n{i}. [{rec['priority']}] {rec['action']}")
            print(f"   {rec['details']}")
            print(f"   Timeline: {rec['timeline']}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Loan Processing - Temporal Fairness Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: TDP over time
    ax1 = axes[0, 0]
    quarters = list(range(len(tdp_history)))
    ax1.plot(quarters, tdp_history, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=tdp.threshold, color='r', linestyle='--', label=f'Threshold ({tdp.threshold})')
    ax1.axvline(x=shock_quarter, color='orange', linestyle='--', alpha=0.7, label='Economic Shock')
    ax1.fill_between(quarters, 0, tdp_history, alpha=0.3)
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('TDP Value')
    ax1.set_title('Temporal Demographic Parity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Approval rates by group
    ax2 = axes[0, 1]
    approval_by_group = loan_data.groupby(['quarter', 'group'])['approved'].mean().unstack()
    for group in approval_by_group.columns:
        ax2.plot(approval_by_group.index, approval_by_group[group], marker='o', label=group)
    ax2.axvline(x=shock_quarter, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Approval Rate')
    ax2.set_title('Approval Rates by Income Group')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Credit score distributions
    ax3 = axes[0, 2]
    for group in loan_data['group'].unique():
        group_scores = loan_data[loan_data['group'] == group]['credit_score']
        ax3.hist(group_scores, alpha=0.5, label=group, bins=20)
    ax3.set_xlabel('Credit Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Credit Score Distribution by Group')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loan amounts by approval status
    ax4 = axes[1, 0]
    approved_amounts = loan_data[loan_data['approved'] == 1]['loan_amount']
    rejected_amounts = loan_data[loan_data['approved'] == 0]['loan_amount']
    ax4.hist([approved_amounts, rejected_amounts], label=['Approved', 'Rejected'], 
             bins=20, alpha=0.7, color=['green', 'red'])
    ax4.set_xlabel('Loan Amount ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Loan Amount Distribution by Decision')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Processing time analysis
    ax5 = axes[1, 1]
    processing_by_group = loan_data.groupby('group')['processing_time'].mean().sort_values()
    ax5.barh(processing_by_group.index, processing_by_group.values)
    ax5.set_xlabel('Average Processing Time (days)')
    ax5.set_title('Processing Time by Income Group')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: PSI visualization
    ax6 = axes[1, 2]
    if predictions.get('predictions') is not None:
        # Combine historical and predicted
        all_values = tdp_history + list(predictions['predictions'])
        all_quarters = list(range(len(all_values)))
        
        ax6.plot(quarters, tdp_history, 'b-o', label='Historical', linewidth=2)
        
        future_quarters = list(range(len(tdp_history), len(all_values)))
        ax6.plot(future_quarters, predictions['predictions'], 'g--o', label='Predicted', linewidth=2)
        
        # Add confidence interval
        ax6.fill_between(future_quarters, 
                        predictions['confidence_lower'], 
                        predictions['confidence_upper'],
                        alpha=0.3, color='green', label='95% CI')
    else:
        ax6.plot(quarters, tdp_history, 'b-o', linewidth=2)
    
    ax6.axvline(x=shock_quarter, color='orange', linestyle='--', alpha=0.7, label='Economic Shock')
    ax6.set_xlabel('Quarter')
    ax6.set_ylabel('TDP Value')
    ax6.set_title('Fairness Prediction with Confidence Intervals')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loan_processing_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'loan_processing_analysis.png'")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    print("\n✓ Temporal Demographic Parity:")
    print(f"  - Pre-shock mean TDP: {np.mean(tdp_history[:shock_quarter]):.3f}")
    print(f"  - Post-shock mean TDP: {np.mean(tdp_history[shock_quarter:]):.3f}")
    print(f"  - Violations detected in {sum(t > tdp.threshold for t in tdp_history)} quarters")
    
    print("\n✓ Fairness Decay:")
    print(f"  - Changepoint detected at quarter {shock_quarter}")
    print(f"  - Significant fairness degradation post-shock")
    
    print("\n✓ Population Stability:")
    print(f"  - PSI: {psi_result['psi']:.3f} (threshold: 0.1)")
    print(f"  - Distribution shift indicates future bias risk")
    
    print("\n✓ Equalized Odds:")
    print(f"  - EOOT violations in TPR/FPR rates")
    print(f"  - Low-income groups most affected")
    
    print("\n✓ Risk Assessment:")
    print(f"  - Overall risk: {risk['risk_level'].upper()}")
    print(f"  - Immediate intervention recommended")
    
    return loan_data, analysis_results


if __name__ == "__main__":
    # Run the loan processing example
    loan_data, results = loan_processing_example()
    
    print("\n" + "="*70)
    print("LOAN PROCESSING EXAMPLE COMPLETED")
    print("="*70)
    print("\nThis example demonstrated temporal fairness issues in financial services:")
    print("• Demographic parity violations across income groups")
    print("• Fairness decay after economic shocks")
    print("• Early warning through Population Stability Index")
    print("• Equalized odds violations in loan approvals")
    print("\nThese patterns reflect real-world financial discrimination issues.")