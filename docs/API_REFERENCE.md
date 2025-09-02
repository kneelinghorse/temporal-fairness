# API Reference

Complete API documentation for the Temporal Fairness Framework.

## Core Metrics

### TemporalDemographicParity

**Class**: `src.metrics.temporal_demographic_parity.TemporalDemographicParity`

Measures fairness across demographic groups over time using demographic parity principles.

#### Constructor

```python
TemporalDemographicParity(threshold=0.1, min_samples=30)
```

**Parameters:**
- `threshold` (float): Maximum acceptable TDP value for fairness detection. Default: 0.1
- `min_samples` (int): Minimum samples required per group for reliable calculation. Default: 30

#### Methods

##### `detect_bias(decisions, groups, timestamps, time_windows=None)`

Detect temporal demographic parity violations.

**Parameters:**
- `decisions` (array-like): Binary decisions (0/1) or positive class labels
- `groups` (array-like): Group identifiers for each decision
- `timestamps` (array-like): Timestamps for each decision (datetime objects)
- `time_windows` (list, optional): Custom time windows as (start, end) tuples

**Returns:**
```python
{
    'bias_detected': bool,           # Whether bias exceeds threshold
    'metric_value': float,           # TDP score (0=fair, 1=maximum bias)  
    'threshold': float,              # Threshold used for detection
    'confidence': float,             # Confidence in bias detection (0-1)
    'severity': str,                 # 'none', 'low', 'medium', 'high'
    'details': dict                  # Detailed analysis results
}
```

**Example:**
```python
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
import numpy as np
import pandas as pd

tdp = TemporalDemographicParity(threshold=0.1)
decisions = np.array([1, 0, 1, 0, 1, 1, 0, 0])
groups = np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])
timestamps = pd.date_range('2024-01-01', periods=8, freq='H')

result = tdp.detect_bias(decisions, groups, timestamps)
```

##### `calculate_pairwise(decisions, groups)`

Calculate pairwise TDP between all group combinations.

**Parameters:**
- `decisions` (array-like): Binary decisions
- `groups` (array-like): Group identifiers

**Returns:**
- `dict`: Pairwise TDP values with group pairs as keys

---

### EqualizedOddsOverTime

**Class**: `src.metrics.equalized_odds_over_time.EqualizedOddsOverTime`

Ensures equal true positive and false positive rates across groups over time.

#### Constructor

```python
EqualizedOddsOverTime(tpr_threshold=0.1, fpr_threshold=0.1, min_samples=30)
```

**Parameters:**
- `tpr_threshold` (float): Maximum acceptable TPR disparity. Default: 0.1
- `fpr_threshold` (float): Maximum acceptable FPR disparity. Default: 0.1
- `min_samples` (int): Minimum samples per group. Default: 30

#### Methods

##### `detect_bias(predictions, true_labels, groups, timestamps, time_windows=None)`

Detect equalized odds violations over time.

**Parameters:**
- `predictions` (array-like): Model predictions (0/1)
- `true_labels` (array-like): Ground truth labels (0/1)  
- `groups` (array-like): Group identifiers
- `timestamps` (array-like): Timestamps for each prediction
- `time_windows` (list, optional): Custom time windows

**Returns:**
```python
{
    'bias_detected': bool,           # Whether bias exceeds thresholds
    'metric_value': float,           # Maximum of TPR and FPR disparities
    'bias_source': str/list,         # 'TPR', 'FPR', or ['TPR', 'FPR']
    'tpr_threshold': float,          # TPR threshold used
    'fpr_threshold': float,          # FPR threshold used
    'confidence': float,             # Confidence in detection
    'severity': str,                 # Severity level
    'details': dict                  # Detailed TPR/FPR analysis
}
```

**Example:**
```python
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime

eoot = EqualizedOddsOverTime(tpr_threshold=0.15, fpr_threshold=0.15)
predictions = np.array([1, 0, 1, 1, 0, 0, 1, 0])
labels = np.array([1, 0, 0, 1, 0, 1, 1, 0])
groups = np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])
timestamps = pd.date_range('2024-01-01', periods=8, freq='H')

result = eoot.detect_bias(predictions, labels, groups, timestamps)
```

---

### FairnessDecayDetection

**Class**: `src.metrics.fairness_decay_detection.FairnessDecayDetection`

Monitors degradation of fairness metrics over time with predictive capabilities.

#### Constructor

```python
FairnessDecayDetection(decay_threshold=0.1, min_points=10)
```

**Parameters:**
- `decay_threshold` (float): Minimum decay rate to trigger detection. Default: 0.1
- `min_points` (int): Minimum time points for analysis. Default: 10

#### Methods

##### `detect_fairness_decay(fairness_scores, timestamps)`

Detect fairness decay patterns over time.

**Parameters:**
- `fairness_scores` (array-like): Time series of fairness scores
- `timestamps` (array-like): Corresponding timestamps

**Returns:**
```python
{
    'decay_detected': bool,          # Whether decay is detected
    'decay_type': str,               # 'linear', 'exponential', 'changepoint', 'none'
    'decay_rate': float,             # Rate of decay per time unit
    'changepoint_detected': bool,    # Whether sudden change detected
    'changepoint_index': int,        # Index of changepoint if detected
    'alert_level': str,              # 'none', 'low', 'medium', 'high', 'critical'
    'confidence': float,             # Confidence in detection
    'details': dict                  # Detailed analysis results
}
```

##### `predict_future_fairness(fairness_scores, timestamps, horizon_days=30)`

Predict future fairness evolution.

**Parameters:**
- `fairness_scores` (array-like): Historical fairness scores
- `timestamps` (array-like): Timestamps
- `horizon_days` (int): Days to predict ahead. Default: 30

**Returns:**
```python
{
    'predictions': array,            # Predicted fairness scores
    'lower_bound': array,            # Lower confidence bound
    'upper_bound': array,            # Upper confidence bound
    'prediction_timestamps': array,  # Timestamps for predictions
    'alert_level': str,              # Alert level for predictions
    'model_confidence': float        # Confidence in prediction model
}
```

**Example:**
```python
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from datetime import datetime, timedelta

fdd = FairnessDecayDetection(decay_threshold=0.05)
fairness_scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
timestamps = [datetime.now() + timedelta(days=i) for i in range(6)]

decay_result = fdd.detect_fairness_decay(fairness_scores, timestamps)
prediction = fdd.predict_future_fairness(fairness_scores, timestamps, horizon_days=90)
```

---

### QueuePositionFairness

**Class**: `src.metrics.queue_position_fairness.QueuePositionFairness`

Detects systematic bias in queue ordering and position allocation.

#### Constructor

```python
QueuePositionFairness(fairness_threshold=0.8, min_samples=20)
```

**Parameters:**
- `fairness_threshold` (float): Minimum fairness score (0-1). Default: 0.8
- `min_samples` (int): Minimum samples per group. Default: 20

#### Methods

##### `detect_bias(queue_positions, groups, timestamps, time_windows=None)`

Detect queue position bias across groups.

**Parameters:**
- `queue_positions` (array-like): Queue positions (lower = better)
- `groups` (array-like): Group identifiers
- `timestamps` (array-like): Timestamps
- `time_windows` (list, optional): Custom time windows

**Returns:**
```python
{
    'bias_detected': bool,           # Whether bias exceeds threshold
    'metric_value': float,           # QPF score (1=fair, 0=maximum bias)
    'fairness_threshold': float,     # Threshold used
    'most_disadvantaged_group': str, # Group with worst positions
    'confidence': float,             # Confidence in detection
    'details': dict                  # Detailed position analysis
}
```

##### `calculate_wait_time_disparity(wait_times, groups, time_windows=None)`

Analyze wait time disparities between groups.

**Parameters:**
- `wait_times` (array-like): Wait times for each item
- `groups` (array-like): Group identifiers
- `time_windows` (list, optional): Time windows for analysis

**Returns:**
```python
{
    'windows': list,                 # Analysis for each time window
    'overall_disparity': float,      # Overall disparity measure
    'statistical_significance': dict # Statistical test results
}
```

##### `analyze_priority_patterns(queue_positions, groups, priority_levels=None)`

Analyze priority patterns and front/back queue distributions.

**Parameters:**
- `queue_positions` (array-like): Queue positions
- `groups` (array-like): Group identifiers  
- `priority_levels` (array-like, optional): Priority levels for analysis

**Returns:**
- `dict`: Priority analysis with statistics for each group

**Example:**
```python
from src.metrics.queue_position_fairness import QueuePositionFairness

qpf = QueuePositionFairness(fairness_threshold=0.8)
positions = np.array([1, 5, 2, 8, 3, 10, 4, 12])
groups = np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])
timestamps = pd.date_range('2024-01-01', periods=8, freq='H')

result = qpf.detect_bias(positions, groups, timestamps)
```

---

## Analysis Tools

### BiasDetector

**Class**: `src.analysis.bias_detector.BiasDetector`

Automated detection of specific bias patterns in fairness time series.

#### Constructor

```python
BiasDetector(sensitivity=0.95, min_pattern_length=10)
```

**Parameters:**
- `sensitivity` (float): Detection sensitivity (0-1). Default: 0.95
- `min_pattern_length` (int): Minimum points for pattern detection. Default: 10

#### Methods

##### `identify_confidence_valleys(fairness_over_time, timestamps=None)`

Detect U-shaped confidence valleys in fairness scores.

**Parameters:**
- `fairness_over_time` (array-like): Time series of fairness scores
- `timestamps` (array-like, optional): Corresponding timestamps

**Returns:**
```python
[
    {
        'position': int,             # Index of valley center
        'depth': float,              # Depth of valley (0-1)
        'width': int,                # Width in time points
        'confidence': float,         # Detection confidence
        'start_time': datetime,      # Valley start time
        'end_time': datetime         # Valley end time
    }
]
```

##### `detect_sudden_shifts(fairness_over_time, timestamps=None)`

Detect sudden changes in fairness levels using CUSUM algorithm.

**Parameters:**
- `fairness_over_time` (array-like): Fairness time series
- `timestamps` (array-like, optional): Timestamps

**Returns:**
```python
[
    {
        'position': int,             # Index of shift
        'magnitude': float,          # Size of shift
        'direction': str,            # 'increase' or 'decrease'
        'confidence': float,         # Detection confidence
        'timestamp': datetime        # Time of shift
    }
]
```

##### `detect_gradual_drift(fairness_over_time, timestamps=None)`

Detect gradual drift in fairness using linear regression.

**Parameters:**
- `fairness_over_time` (array-like): Fairness time series
- `timestamps` (array-like, optional): Timestamps

**Returns:**
```python
{
    'detected': bool,                # Whether drift is detected
    'drift_rate': float,             # Rate of drift per time unit
    'direction': str,                # 'improving', 'degrading', 'stable'
    'confidence': float,             # Statistical confidence
    'r_squared': float,              # R² of linear fit
    'p_value': float                 # Statistical p-value
}
```

##### `detect_periodic_patterns(fairness_over_time, timestamps=None)`

Detect periodic/cyclical patterns using autocorrelation.

**Parameters:**
- `fairness_over_time` (array-like): Fairness time series
- `timestamps` (array-like, optional): Timestamps

**Returns:**
```python
{
    'detected': bool,                # Whether periodicity detected
    'period': float,                 # Period length in time units
    'strength': float,               # Strength of periodicity (0-1)
    'confidence': float,             # Detection confidence
    'phase': float                   # Phase offset
}
```

**Example:**
```python
from src.analysis.bias_detector import BiasDetector
import numpy as np

detector = BiasDetector(sensitivity=0.95)
fairness_scores = 0.8 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 100))

valleys = detector.identify_confidence_valleys(fairness_scores)
shifts = detector.detect_sudden_shifts(fairness_scores)
drift = detector.detect_gradual_drift(fairness_scores)
periodic = detector.detect_periodic_patterns(fairness_scores)
```

---

### TemporalAnalyzer

**Class**: `src.analysis.temporal_analyzer.TemporalAnalyzer`

Comprehensive analysis orchestrator that runs multiple metrics and provides unified results.

#### Constructor

```python
TemporalAnalyzer(config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary for customizing analysis

#### Methods

##### `run_full_analysis(data, groups, decision_column, timestamp_column, queue_column=None)`

Run comprehensive temporal fairness analysis on a dataset.

**Parameters:**
- `data` (DataFrame): Input dataset
- `groups` (str): Column name for group identifiers
- `decision_column` (str): Column name for decisions/outcomes  
- `timestamp_column` (str): Column name for timestamps
- `queue_column` (str, optional): Column name for queue positions

**Returns:**
```python
{
    'metrics': {
        'TDP': dict,                 # TDP analysis results
        'EOOT': dict,                # EOOT results (if labels available)
        'QPF': dict,                 # QPF results (if queue data available)
        'FDD': dict                  # FDD results
    },
    'patterns': {
        'valleys': list,             # Confidence valleys detected
        'shifts': list,              # Sudden shifts detected
        'drift': dict,               # Gradual drift analysis
        'periodic': dict             # Periodic pattern analysis
    },
    'risk_assessment': {
        'risk_level': str,           # 'low', 'moderate', 'high', 'very_high', 'critical'
        'risk_score': float,         # Overall risk score (0-1)
        'risk_factors': list,        # List of identified risk factors
        'critical_issues': list      # Critical issues requiring immediate attention
    },
    'recommendations': [
        {
            'action': str,           # Recommended action
            'priority': str,         # 'low', 'medium', 'high', 'critical'
            'timeline': str,         # Suggested implementation timeline
            'details': str,          # Detailed explanation
            'expected_impact': str   # Expected impact of action
        }
    ],
    'summary': {
        'total_records': int,        # Number of records analyzed
        'time_span': str,            # Time span of analysis
        'groups_analyzed': list,     # List of groups
        'metrics_calculated': list,  # Metrics successfully calculated
        'analysis_timestamp': datetime # When analysis was performed
    }
}
```

**Example:**
```python
from src.analysis.temporal_analyzer import TemporalAnalyzer
import pandas as pd

analyzer = TemporalAnalyzer()

# Your data
data = pd.DataFrame({
    'outcome': [1, 0, 1, 0, 1, 0],
    'group': ['A', 'B', 'A', 'B', 'A', 'B'],
    'timestamp': pd.date_range('2024-01-01', periods=6, freq='H'),
    'queue_pos': [1, 3, 2, 5, 1, 4]
})

results = analyzer.run_full_analysis(
    data=data,
    groups='group',
    decision_column='outcome',
    timestamp_column='timestamp',
    queue_column='queue_pos'
)

print(f"Risk level: {results['risk_assessment']['risk_level']}")
for rec in results['recommendations']:
    print(f"Action: {rec['action']} (Priority: {rec['priority']})")
```

---

## Visualization

### FairnessVisualizer

**Class**: `src.visualization.fairness_visualizer.FairnessVisualizer`

Create visualizations for temporal fairness analysis.

#### Constructor

```python
FairnessVisualizer(figsize=(12, 8), style='default')
```

**Parameters:**
- `figsize` (tuple): Figure size as (width, height). Default: (12, 8)
- `style` (str): Matplotlib style. Default: 'default'

#### Methods

##### `plot_temporal_fairness_trends(timestamps, fairness_scores, threshold=0.8, save_path=None)`

Plot fairness metrics over time with threshold lines.

**Parameters:**
- `timestamps` (array-like): Time points
- `fairness_scores` (dict): Dictionary mapping metric names to score arrays
- `threshold` (float): Fairness threshold line. Default: 0.8
- `save_path` (str, optional): Path to save plot

**Returns:**
- `Figure`: Matplotlib figure object

##### `create_fairness_dashboard(data, metrics=['TDP', 'QPF'], save_path=None)`

Create comprehensive fairness dashboard.

**Parameters:**  
- `data` (DataFrame): Dataset for analysis
- `metrics` (list): Metrics to include in dashboard
- `save_path` (str, optional): Path to save dashboard

**Returns:**
- `Figure`: Matplotlib figure object

**Example:**
```python
from src.visualization.fairness_visualizer import FairnessVisualizer
import pandas as pd
import numpy as np

visualizer = FairnessVisualizer(figsize=(15, 10))

timestamps = pd.date_range('2024-01-01', periods=30, freq='D')
fairness_scores = {
    'TDP': np.random.uniform(0.6, 0.9, 30),
    'QPF': np.random.uniform(0.7, 0.95, 30)
}

fig = visualizer.plot_temporal_fairness_trends(
    timestamps=timestamps,
    fairness_scores=fairness_scores,
    threshold=0.8,
    save_path='fairness_trends.png'
)
```

---

## Data Generation

### TemporalBiasGenerator

**Class**: `src.utils.data_generators.TemporalBiasGenerator`

Generate realistic datasets with controllable temporal bias patterns for testing.

#### Constructor

```python
TemporalBiasGenerator(random_seed=None)
```

**Parameters:**
- `random_seed` (int, optional): Random seed for reproducible generation

#### Methods

##### `generate_biased_hiring_data(n_applicants, n_days, groups, bias_strength=0.2)`

Generate realistic hiring data with temporal bias.

**Parameters:**
- `n_applicants` (int): Number of applicants to generate
- `n_days` (int): Time span in days
- `groups` (list): List of demographic groups
- `bias_strength` (float): Strength of bias injection (0-1). Default: 0.2

**Returns:**
- `DataFrame`: Generated hiring dataset with columns: applicant_id, group, timestamp, hired

##### `generate_queue_simulation(n_items, n_hours, groups, queue_bias_factor=0.3)`

Generate queue simulation data with position bias.

**Parameters:**
- `n_items` (int): Number of items in queue
- `n_hours` (int): Simulation time span in hours  
- `groups` (list): List of groups
- `queue_bias_factor` (float): Bias strength in queue positioning. Default: 0.3

**Returns:**
- `DataFrame`: Queue data with columns: item_id, group, timestamp, queue_position, wait_time

**Example:**
```python
from src.utils.data_generators import TemporalBiasGenerator

generator = TemporalBiasGenerator(random_seed=42)

# Generate hiring data
hiring_data = generator.generate_biased_hiring_data(
    n_applicants=1000,
    n_days=30,
    groups=['Male', 'Female', 'Non-binary'],
    bias_strength=0.3
)

# Generate queue data  
queue_data = generator.generate_queue_simulation(
    n_items=500,
    n_hours=24,
    groups=['Premium', 'Standard', 'Basic'],
    queue_bias_factor=0.4
)
```

---

## Error Handling

### Common Exceptions

**ValueError**: Raised for invalid input parameters
```python
# Example: Insufficient groups
try:
    result = tdp.detect_bias(decisions, ['A'], timestamps)
except ValueError as e:
    print(f"Error: {e}")  # "At least 2 groups required for TDP calculation"
```

**DataValidationError**: Raised for inconsistent data
```python  
# Example: Mismatched array lengths
try:
    result = tdp.detect_bias([1, 0], ['A', 'B', 'C'], timestamps)
except ValueError as e:
    print(f"Error: {e}")  # "Array length mismatch"
```

### Input Validation

All metrics perform automatic input validation:
- Array length consistency checking
- Data type conversion (lists to numpy arrays)  
- Timestamp format standardization
- Missing value handling
- Group identifier validation

---

## Performance Notes

- All metrics are optimized for production use with O(n) or O(n log n) complexity
- Memory usage scales linearly with input size
- Recommended batch size: 10,000 records for optimal performance
- Use time windows for very large datasets to control memory usage
- Vectorized operations throughout for maximum performance

---

## Version Compatibility

- Python 3.7+
- NumPy 1.19+
- Pandas 1.3+
- SciPy 1.7+
- Matplotlib 3.3+ (for visualizations)
- Scikit-learn 1.0+ (for some analysis features)

For full installation requirements, see `requirements.txt`.