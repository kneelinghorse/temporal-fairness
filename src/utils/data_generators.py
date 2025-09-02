"""
Synthetic data generators for testing temporal fairness metrics.

This module provides utilities to generate synthetic datasets with
controlled bias patterns for testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict


class TemporalBiasGenerator:
    """Generate synthetic data with controlled temporal bias patterns."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def generate_dataset(
        self,
        n_samples: int = 10000,
        n_groups: int = 2,
        bias_type: str = 'constant',
        bias_strength: float = 0.2,
        time_periods: int = 10,
        base_positive_rate: float = 0.5,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset with temporal patterns.
        
        Args:
            n_samples: Total number of samples to generate
            n_groups: Number of demographic groups
            bias_type: Type of bias pattern ('constant', 'increasing', 'decreasing', 
                      'oscillating', 'sudden_shift', 'confidence_valley')
            bias_strength: Strength of the bias (0 to 1)
            time_periods: Number of time periods
            base_positive_rate: Base rate of positive outcomes
            noise_level: Amount of random noise to add
            
        Returns:
            DataFrame with columns: decision, group, timestamp, time_period
        """
        # Generate group assignments
        groups = self.rng.choice(range(n_groups), size=n_samples)
        
        # Generate timestamps
        time_period = self.rng.choice(range(time_periods), size=n_samples)
        base_time = datetime.now()
        timestamps = [
            base_time + timedelta(days=int(tp * 30))
            for tp in time_period
        ]
        
        # Generate decisions based on bias type
        decisions = self._generate_biased_decisions(
            groups, time_period, n_groups, time_periods,
            bias_type, bias_strength, base_positive_rate, noise_level
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'decision': decisions,
            'group': groups,
            'timestamp': timestamps,
            'time_period': time_period
        })
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def _generate_biased_decisions(
        self,
        groups: np.ndarray,
        time_periods: np.ndarray,
        n_groups: int,
        n_time_periods: int,
        bias_type: str,
        bias_strength: float,
        base_rate: float,
        noise: float
    ) -> np.ndarray:
        """Generate decisions with specified bias pattern."""
        n_samples = len(groups)
        decisions = np.zeros(n_samples)
        
        # Get bias pattern function
        bias_func = self._get_bias_function(bias_type, bias_strength)
        
        # Generate decisions for each group and time period
        for g in range(n_groups):
            for t in range(n_time_periods):
                mask = (groups == g) & (time_periods == t)
                n_masked = np.sum(mask)
                
                if n_masked == 0:
                    continue
                
                # Calculate positive rate for this group and time
                if g == 0:
                    # Reference group - use base rate
                    positive_rate = base_rate
                else:
                    # Apply bias to other groups
                    bias_adjustment = bias_func(t / n_time_periods)
                    positive_rate = base_rate - bias_adjustment
                
                # Add noise
                positive_rate += self.rng.normal(0, noise)
                positive_rate = np.clip(positive_rate, 0, 1)
                
                # Generate decisions
                decisions[mask] = self.rng.binomial(1, positive_rate, n_masked)
        
        return decisions
    
    def _get_bias_function(self, bias_type: str, strength: float) -> Callable:
        """Get the bias function for a given type."""
        bias_functions = {
            'constant': lambda t: strength,
            'increasing': lambda t: strength * t,
            'decreasing': lambda t: strength * (1 - t),
            'oscillating': lambda t: strength * np.sin(2 * np.pi * t),
            'sudden_shift': lambda t: strength if t > 0.5 else 0,
            'confidence_valley': lambda t: strength * (1 - 4 * (t - 0.5) ** 2)
        }
        
        if bias_type not in bias_functions:
            raise ValueError(f"Unknown bias type: {bias_type}")
        
        return bias_functions[bias_type]
    
    def generate_hiring_pipeline(
        self,
        n_applicants: int = 5000,
        n_stages: int = 4,
        groups: List[str] = None,
        bias_at_stage: Dict[int, float] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic hiring pipeline data.
        
        Args:
            n_applicants: Number of initial applicants
            n_stages: Number of hiring stages
            groups: List of group names (default: ['A', 'B'])
            bias_at_stage: Dictionary mapping stage to bias strength
            
        Returns:
            DataFrame with hiring pipeline data
        """
        if groups is None:
            groups = ['A', 'B']
        
        if bias_at_stage is None:
            bias_at_stage = {2: 0.15}  # Default bias at stage 2
        
        # Initialize data
        group_assignments = self.rng.choice(groups, size=n_applicants)
        stage_data = []
        
        remaining_candidates = list(range(n_applicants))
        
        for stage in range(n_stages):
            n_current = len(remaining_candidates)
            
            # Calculate pass rates
            base_pass_rate = 0.7 - (stage * 0.15)  # Decreasing pass rate
            
            decisions = []
            for idx in remaining_candidates:
                group = group_assignments[idx]
                
                # Apply bias if specified for this stage
                if stage in bias_at_stage and group != groups[0]:
                    pass_rate = max(0.05, base_pass_rate - bias_at_stage[stage])  # Ensure minimum 5% pass rate
                else:
                    pass_rate = base_pass_rate
                
                passed = self.rng.binomial(1, pass_rate)
                decisions.append(passed)
                
                stage_data.append({
                    'applicant_id': idx,
                    'group': group,
                    'stage': stage,
                    'decision': passed,
                    'timestamp': datetime.now() + timedelta(days=stage * 7)
                })
            
            # Keep only those who passed for next stage
            remaining_candidates = [
                remaining_candidates[i] 
                for i, d in enumerate(decisions) if d == 1
            ]
        
        return pd.DataFrame(stage_data)
    
    def generate_emergency_room_queue(
        self,
        n_patients: int = 1000,
        n_hours: int = 168,  # One week
        groups: List[str] = None,
        bias_strength: float = 0.2,
        include_severity: bool = True
    ) -> pd.DataFrame:
        """
        Generate emergency room queue data with realistic bias patterns.
        
        Args:
            n_patients: Number of patients
            n_hours: Number of hours to simulate
            groups: Demographic groups
            bias_strength: Strength of queue position bias
            include_severity: Include medical severity scores
            
        Returns:
            DataFrame with ER queue data
        """
        if groups is None:
            groups = ['GroupA', 'GroupB', 'GroupC']
        
        data = []
        base_time = datetime.now() - timedelta(hours=n_hours)
        
        # Generate arrival times (Poisson process)
        arrival_rate = n_patients / n_hours
        current_hour = 0
        queue_positions = defaultdict(list)
        
        for i in range(n_patients):
            # Random arrival time with rush hour patterns
            hour_of_day = current_hour % 24
            
            # Rush periods: 8-10am, 6-8pm
            if hour_of_day in [8, 9, 18, 19]:
                rush_factor = 1.5
            else:
                rush_factor = 1.0
            
            # Generate inter-arrival time
            inter_arrival = self.rng.exponential(1 / (arrival_rate * rush_factor))
            current_hour += inter_arrival
            
            if current_hour >= n_hours:
                break
            
            timestamp = base_time + timedelta(hours=current_hour)
            
            # Assign group
            group = self.rng.choice(groups)
            
            # Generate severity score (1-5, 5 being most severe)
            if include_severity:
                # Groups might have different severity distributions
                if group == groups[0]:
                    severity = self.rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
                elif group == groups[1]:
                    severity = self.rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.35, 0.15, 0.1, 0.05])
                else:
                    severity = self.rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.3, 0.25, 0.15, 0.05])
            else:
                severity = 3
            
            # Calculate base queue position based on severity
            base_position = 6 - severity  # Higher severity = lower position (front of queue)
            
            # Apply group-based bias to queue position
            if group != groups[0]:
                # Add bias: push certain groups further back
                position_adjustment = self.rng.poisson(bias_strength * 5)
                queue_position = base_position + position_adjustment
            else:
                queue_position = base_position
            
            # Add some randomness
            queue_position += self.rng.randint(-1, 2)
            queue_position = max(1, queue_position)  # Ensure positive
            
            # Calculate wait time based on queue position
            base_wait = queue_position * 15  # 15 minutes per position
            wait_time = base_wait + self.rng.normal(0, 5)  # Add variance
            wait_time = max(5, wait_time)  # Minimum 5 minutes
            
            data.append({
                'patient_id': i,
                'group': group,
                'timestamp': timestamp,
                'hour_of_day': hour_of_day,
                'severity': severity if include_severity else None,
                'queue_position': int(queue_position),
                'wait_time_minutes': wait_time,
                'seen_by_doctor': timestamp + timedelta(minutes=wait_time)
            })
        
        return pd.DataFrame(data)
    
    def generate_customer_service_queue(
        self,
        n_customers: int = 2000,
        n_days: int = 30,
        groups: List[str] = None,
        priority_tiers: List[str] = None,
        bias_type: str = 'systematic'
    ) -> pd.DataFrame:
        """
        Generate customer service queue data with priority tiers.
        
        Args:
            n_customers: Number of customers
            n_days: Number of days to simulate
            groups: Customer demographic groups
            priority_tiers: Service tier levels
            bias_type: Type of bias pattern
            
        Returns:
            DataFrame with customer service queue data
        """
        if groups is None:
            groups = ['Standard', 'Premium', 'Enterprise']
        
        if priority_tiers is None:
            priority_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
        
        data = []
        base_time = datetime.now() - timedelta(days=n_days)
        
        for i in range(n_customers):
            # Random arrival time during business hours
            day = self.rng.randint(0, n_days)
            hour = self.rng.choice(range(9, 18))  # 9am to 6pm
            minute = self.rng.randint(0, 60)
            
            timestamp = base_time + timedelta(days=int(day), hours=int(hour), minutes=int(minute))
            
            # Assign group and tier
            group = self.rng.choice(groups)
            
            # Tier distribution depends on group
            if group == 'Enterprise':
                tier_probs = [0.05, 0.15, 0.40, 0.40]  # More Gold/Platinum
            elif group == 'Premium':
                tier_probs = [0.15, 0.35, 0.35, 0.15]  # Balanced
            else:
                tier_probs = [0.40, 0.40, 0.15, 0.05]  # More Bronze/Silver
            
            tier = self.rng.choice(priority_tiers, p=tier_probs)
            
            # Base queue position by tier
            tier_positions = {'Platinum': 1, 'Gold': 10, 'Silver': 30, 'Bronze': 50}
            base_position = tier_positions[tier]
            
            # Apply systematic bias
            if bias_type == 'systematic' and group == 'Standard':
                # Standard customers pushed back even within their tier
                position_penalty = self.rng.randint(5, 15)
                queue_position = base_position + position_penalty
            else:
                queue_position = base_position + self.rng.randint(0, 5)
            
            # Calculate wait time
            wait_time = queue_position * 2 + self.rng.exponential(10)
            
            # Issue complexity
            complexity = self.rng.choice(['Simple', 'Medium', 'Complex'], p=[0.5, 0.35, 0.15])
            
            # Resolution time
            resolution_times = {'Simple': 10, 'Medium': 25, 'Complex': 60}
            resolution_time = resolution_times[complexity] + self.rng.normal(0, 5)
            
            data.append({
                'customer_id': i,
                'group': group,
                'priority_tier': tier,
                'timestamp': timestamp,
                'queue_position': int(queue_position),
                'wait_time_minutes': wait_time,
                'issue_complexity': complexity,
                'resolution_time_minutes': max(5, resolution_time),
                'satisfaction_score': max(1, min(5, 5 - (wait_time / 30)))
            })
        
        return pd.DataFrame(data)
    
    def generate_resource_allocation_queue(
        self,
        n_requests: int = 1500,
        n_quarters: int = 8,
        groups: List[str] = None,
        resource_types: List[str] = None,
        scarcity_level: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate resource allocation queue data (e.g., grant funding, housing).
        
        Args:
            n_requests: Number of resource requests
            n_quarters: Number of quarters to simulate
            groups: Applicant groups
            resource_types: Types of resources
            scarcity_level: Resource scarcity (0=abundant, 1=very scarce)
            
        Returns:
            DataFrame with resource allocation data
        """
        if groups is None:
            groups = ['NonProfit', 'SmallBusiness', 'Individual', 'Corporation']
        
        if resource_types is None:
            resource_types = ['Grant', 'Loan', 'Subsidy', 'TaxCredit']
        
        data = []
        base_time = datetime.now() - timedelta(days=n_quarters * 90)
        
        # Track available resources per quarter
        resources_per_quarter = int(n_requests * (1 - scarcity_level) / n_quarters)
        
        for i in range(n_requests):
            # Assign to quarter
            quarter = self.rng.randint(0, n_quarters)
            timestamp = base_time + timedelta(days=quarter * 90 + self.rng.randint(0, 90))
            
            # Assign group and resource type
            group = self.rng.choice(groups)
            resource_type = self.rng.choice(resource_types)
            
            # Generate merit score (objective criteria)
            if group == 'NonProfit':
                merit_score = self.rng.beta(8, 2) * 100  # Higher scores
            elif group == 'SmallBusiness':
                merit_score = self.rng.beta(6, 4) * 100  # Medium scores
            elif group == 'Individual':
                merit_score = self.rng.beta(5, 5) * 100  # Varied scores
            else:  # Corporation
                merit_score = self.rng.beta(7, 3) * 100  # Higher scores
            
            # Calculate base queue position (lower is better)
            base_position = int(101 - merit_score)  # Invert score
            
            # Apply bias in queue assignment
            if group in ['Individual', 'SmallBusiness']:
                # Disadvantage certain groups
                position_penalty = self.rng.poisson(scarcity_level * 20)
                queue_position = base_position + position_penalty
            else:
                queue_position = base_position
            
            # Determine if approved (based on position and resources)
            position_threshold = resources_per_quarter / (n_requests / n_quarters)
            approved = queue_position <= (100 * position_threshold)
            
            # Amount requested
            if resource_type == 'Grant':
                amount = self.rng.lognormal(10, 1)  # ~$22k median
            elif resource_type == 'Loan':
                amount = self.rng.lognormal(11, 1)  # ~$60k median
            elif resource_type == 'Subsidy':
                amount = self.rng.lognormal(9, 0.5)  # ~$8k median
            else:  # Tax Credit
                amount = self.rng.lognormal(8, 0.5)  # ~$3k median
            
            data.append({
                'request_id': i,
                'group': group,
                'resource_type': resource_type,
                'timestamp': timestamp,
                'quarter': quarter,
                'merit_score': round(merit_score, 2),
                'queue_position': int(queue_position),
                'amount_requested': round(amount, 2),
                'approved': int(approved),
                'processing_days': queue_position / 2 + self.rng.exponential(10)
            })
        
        return pd.DataFrame(data)
    
    def generate_healthcare_triage(
        self,
        n_patients: int = 10000,
        n_days: int = 365,
        groups: List[str] = None,
        seasonal_bias: bool = True,
        emergency_rate: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate synthetic healthcare triage data.
        
        Args:
            n_patients: Number of patients
            n_days: Number of days to simulate
            groups: List of demographic groups
            seasonal_bias: Whether to include seasonal patterns
            emergency_rate: Base rate of emergency classification
            
        Returns:
            DataFrame with triage decisions
        """
        if groups is None:
            groups = ['GroupA', 'GroupB', 'GroupC']
        
        data = []
        base_time = datetime.now() - timedelta(days=n_days)
        
        for i in range(n_patients):
            # Random arrival time
            arrival_day = self.rng.randint(0, n_days)
            timestamp = base_time + timedelta(days=arrival_day)
            
            # Assign group
            group = self.rng.choice(groups)
            
            # Calculate emergency probability
            if seasonal_bias:
                # Higher emergency rate in winter months
                month = timestamp.month
                seasonal_factor = 1.3 if month in [12, 1, 2] else 1.0
            else:
                seasonal_factor = 1.0
            
            # Apply group-based bias
            if group == groups[0]:
                group_factor = 1.0
            elif group == groups[1]:
                group_factor = 0.85  # 15% less likely to be classified as emergency
            else:
                group_factor = 0.9
            
            emergency_prob = emergency_rate * seasonal_factor * group_factor
            emergency_prob = np.clip(emergency_prob, 0, 1)
            
            # Generate decision
            is_emergency = self.rng.binomial(1, emergency_prob)
            
            # Generate wait time (inversely related to emergency status)
            if is_emergency:
                wait_time = self.rng.exponential(30)  # 30 minute average
            else:
                wait_time = self.rng.exponential(120)  # 2 hour average
            
            data.append({
                'patient_id': i,
                'group': group,
                'timestamp': timestamp,
                'emergency_decision': is_emergency,
                'wait_time_minutes': wait_time,
                'day_of_week': timestamp.weekday(),
                'month': timestamp.month
            })
        
        return pd.DataFrame(data)
    
    def generate_loan_approval(
        self,
        n_applications: int = 8000,
        n_quarters: int = 8,
        groups: List[str] = None,
        economic_shock_quarter: Optional[int] = None,
        base_approval_rate: float = 0.6
    ) -> pd.DataFrame:
        """
        Generate synthetic loan approval data.
        
        Args:
            n_applications: Number of loan applications
            n_quarters: Number of quarters to simulate
            groups: List of demographic groups
            economic_shock_quarter: Quarter when economic shock occurs
            base_approval_rate: Base approval rate
            
        Returns:
            DataFrame with loan approval decisions
        """
        if groups is None:
            groups = ['High-Income', 'Middle-Income', 'Low-Income']
        
        data = []
        base_time = datetime.now() - timedelta(days=n_quarters * 90)
        
        for i in range(n_applications):
            # Assign to quarter
            quarter = self.rng.randint(0, n_quarters)
            timestamp = base_time + timedelta(days=quarter * 90 + self.rng.randint(0, 90))
            
            # Assign group
            group = self.rng.choice(groups)
            
            # Generate credit score (correlated with group)
            if group == 'High-Income':
                credit_score = self.rng.normal(720, 50)
            elif group == 'Middle-Income':
                credit_score = self.rng.normal(650, 60)
            else:
                credit_score = self.rng.normal(580, 70)
            
            credit_score = np.clip(credit_score, 300, 850)
            
            # Calculate approval probability
            score_factor = (credit_score - 300) / 550  # Normalize to 0-1
            approval_prob = base_approval_rate * score_factor
            
            # Apply economic shock if specified
            if economic_shock_quarter is not None and quarter >= economic_shock_quarter:
                approval_prob *= 0.7  # 30% reduction in approval rates
            
            # Add group-based bias
            if group == 'Low-Income':
                approval_prob *= 0.85  # Additional 15% reduction
            
            approval_prob = np.clip(approval_prob, 0, 1)
            approved = self.rng.binomial(1, approval_prob)
            
            # Generate loan amount
            if group == 'High-Income':
                loan_amount = self.rng.lognormal(12, 0.5)  # ~$160k median
            elif group == 'Middle-Income':
                loan_amount = self.rng.lognormal(11, 0.5)  # ~$60k median
            else:
                loan_amount = self.rng.lognormal(10, 0.5)  # ~$22k median
            
            data.append({
                'application_id': i,
                'group': group,
                'timestamp': timestamp,
                'quarter': quarter,
                'credit_score': int(credit_score),
                'loan_amount': round(loan_amount, 2),
                'approved': approved
            })
        
        return pd.DataFrame(data)


def generate_test_data(
    n_samples: int = 1000,
    bias_strength: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick function to generate test data for TDP metric.
    
    Args:
        n_samples: Number of samples
        bias_strength: Strength of bias
        random_seed: Random seed
        
    Returns:
        Tuple of (decisions, groups, timestamps)
    """
    generator = TemporalBiasGenerator(random_seed)
    df = generator.generate_dataset(
        n_samples=n_samples,
        bias_strength=bias_strength,
        bias_type='increasing'
    )
    
    return df['decision'].values, df['group'].values, df['time_period'].values