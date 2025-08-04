# Shared Data and Preprocessing

## Overview

This folder contains common data processing utilities, preprocessing scripts, and shared components used across all modeling approaches. It provides the foundation for consistent data handling and feature engineering throughout the project.

## 📁 Files

- `preprocess_data.py` - Main preprocessing pipeline and data preparation
- `scaler.pkl` - Trained StandardScaler for consistent feature scaling
- `README.md` - This documentation

## 🔧 Data Processing Pipeline

### Main Preprocessing (`preprocess_data.py`)
```python
Core Functions:
- load_and_clean_data(): Data loading and initial cleaning
- create_features(): Feature engineering and temporal features
- split_data(): Train/validation/test splitting
- scale_features(): Standardization and normalization
- create_sequences(): Sequence generation for LSTM models
```

### Feature Engineering Pipeline
```python
Feature Creation:
1. Temporal features (hour, day, month, cyclical encoding)
2. Rolling statistics (moving averages, volatility)
3. Lag features (historical values)
4. Rate of change indicators
5. Peak/valley detection features
```

## 📊 Data Characteristics

### Raw Data Format
```python
Source Data (EventsMetricsMarJul.csv):
- Columns: timestamp, queue_length, processing_capacity
- Frequency: 2-minute intervals
- Period: March-July 2025
- Records: 107,144 total observations
```

### Processed Data Output
```python
Processed Features:
- queue_length: Target variable (scaled)
- hour_sin/hour_cos: Cyclical hour encoding
- day_of_week_sin/day_of_week_cos: Cyclical day encoding
- month_sin/month_cos: Cyclical month encoding
- rolling_mean_6/12/24: Moving averages
- rolling_std_6/12/24: Rolling volatility
- lag_1/6/12/24: Historical lag features
- rate_of_change: Velocity indicator
```

## 🚀 Usage

### Basic Preprocessing
```python
python preprocess_data.py
# Creates: processed_data/train.csv, val.csv, test.csv
```

### Custom Preprocessing
```python
from preprocess_data import DataPreprocessor

preprocessor = DataPreprocessor()
train_data, val_data, test_data = preprocessor.process_all_data()
```

### Feature Engineering Only
```python
from preprocess_data import create_advanced_features

enhanced_data = create_advanced_features(raw_data)
```

## 🔧 Preprocessing Components

### 1. Data Loading and Cleaning
```python
def load_and_clean_data(filepath):
    """
    - Load CSV with proper datetime parsing
    - Handle missing values
    - Remove outliers (optional)
    - Validate data integrity
    """
```

### 2. Temporal Feature Engineering
```python
def create_temporal_features(data):
    """
    - Hour of day (0-23) with cyclical encoding
    - Day of week (0-6) with cyclical encoding
    - Month (1-12) with cyclical encoding
    - Business hours indicator
    - Weekend/holiday flags
    """
```

### 3. Statistical Features
```python
def create_rolling_features(data, windows=[6, 12, 24]):
    """
    - Rolling means for trend capture
    - Rolling standard deviations for volatility
    - Rolling min/max for range analysis
    - Exponential moving averages
    """
```

### 4. Lag Features
```python
def create_lag_features(data, lags=[1, 6, 12, 24]):
    """
    - Historical values at specified lags
    - Useful for capturing temporal dependencies
    - Handles edge cases and missing values
    """
```

## 📈 Feature Engineering Details

### Cyclical Encoding
```python
# Handle cyclical nature of time features
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)
```

### Rolling Statistics
```python
# Multi-window rolling features
for window in [6, 12, 24]:  # 12 min, 24 min, 48 min
    data[f'rolling_mean_{window}'] = data['queue_length'].rolling(window).mean()
    data[f'rolling_std_{window}'] = data['queue_length'].rolling(window).std()
```

### Rate of Change
```python
# Velocity indicators
data['rate_of_change'] = data['queue_length'].diff()
data['rate_of_change_pct'] = data['queue_length'].pct_change()
```

## 🔄 Data Splitting Strategy

### Time-Based Splitting
```python
Split Strategy:
- Training: 70% (oldest data)
- Validation: 15% (middle period)
- Test: 15% (most recent data)
- Ensures temporal integrity
```

### Stratified Considerations
```python
Additional Considerations:
- Preserve seasonal patterns in all splits
- Balance peak/non-peak periods
- Handle holiday periods appropriately
- Maintain day-of-week distributions
```

## 📊 Scaling and Normalization

### StandardScaler Usage
```python
# Consistent scaling across all models
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# Saved as scaler.pkl for inference consistency
```

### Feature-Specific Scaling
```python
Scaling Strategy:
- Target variable: StandardScaler (for regression)
- Temporal features: Already normalized (cyclical)
- Rolling features: StandardScaler
- Lag features: StandardScaler
- Binary features: No scaling needed
```

## 🛠️ Sequence Generation

### LSTM Sequence Creation
```python
def create_sequences(data, sequence_length=30, target_col='queue_length'):
    """
    - Creates sliding window sequences
    - Handles variable sequence lengths
    - Preserves temporal order
    - Efficient memory usage
    """
```

### Sequence Parameters
```python
Default Configuration:
- sequence_length: 30 (for basic model)
- sequence_length: 48 (for improved/tiered models)
- step_size: 1 (overlapping sequences)
- target_offset: 1 (predict next time step)
```

## 🔍 Data Quality Assurance

### Validation Checks
```python
Quality Assurance:
- Missing value detection and handling
- Outlier identification (IQR method)
- Data type validation
- Temporal consistency checks
- Feature distribution analysis
```

### Data Integrity
```python
Integrity Checks:
- Timestamp ordering validation
- Duplicate detection
- Range validation for all features
- Correlation sanity checks
```

## 📋 Configuration Management

### Preprocessing Parameters
```python
PREPROCESSING_CONFIG = {
    'sequence_length': 30,
    'test_size': 0.15,
    'val_size': 0.15,
    'rolling_windows': [6, 12, 24],
    'lag_periods': [1, 6, 12, 24],
    'outlier_method': 'iqr',
    'outlier_threshold': 1.5
}
```

### Feature Selection
```python
FEATURE_CONFIG = {
    'temporal_features': True,
    'rolling_features': True,
    'lag_features': True,
    'rate_features': True,
    'cyclical_encoding': True
}
```

## 🔄 Pipeline Integration

### Model Integration
```python
# Used by all modeling approaches
from shared_data.preprocess_data import DataPreprocessor

# Basic model
preprocessor = DataPreprocessor()
train_loader, val_loader, test_loader = preprocessor.get_data_loaders()

# Improved model (with enhanced features)
enhanced_preprocessor = DataPreprocessor(enhanced_features=True)
data = enhanced_preprocessor.process_all_data()
```

### Consistent Scaling
```python
# Ensure consistent scaling across training and inference
import pickle

# During training
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# During inference
scaler = pickle.load(open('scaler.pkl', 'rb'))
scaled_new_data = scaler.transform(new_data)
```

## 📈 Performance Considerations

### Memory Optimization
```python
Memory Strategies:
- Chunk processing for large datasets
- Generator-based sequence creation
- Efficient data types (float32 vs float64)
- Memory mapping for very large files
```

### Speed Optimization
```python
Speed Improvements:
- Vectorized operations with NumPy
- Pandas optimizations
- Parallel processing where applicable
- Cached intermediate results
```

## 🔧 Utility Functions

### Data Validation
```python
def validate_data_quality(data):
    """
    Comprehensive data quality checks
    - Missing value analysis
    - Outlier detection
    - Distribution analysis
    - Temporal consistency
    """
```

### Feature Analysis
```python
def analyze_feature_importance(data, target):
    """
    Feature importance analysis
    - Correlation with target
    - Mutual information
    - Statistical significance
    - Feature stability over time
    """
```

## 🎯 Best Practices

### Data Handling
- Always preserve temporal order
- Handle missing values appropriately
- Validate data quality at each step
- Maintain reproducible preprocessing

### Feature Engineering
- Create meaningful features based on domain knowledge
- Avoid data leakage in temporal data
- Balance feature complexity vs interpretability
- Validate feature usefulness

### Scaling and Normalization
- Use consistent scaling across train/val/test
- Save scalers for production inference
- Handle new categories in categorical features
- Monitor feature drift over time

## 🔮 Future Enhancements

### Advanced Features
- External data integration (weather, holidays)
- Automated feature selection
- Feature interaction detection
- Domain-specific transformations

### Pipeline Improvements
- Real-time data processing
- Incremental preprocessing
- Data versioning
- Automated data quality monitoring
