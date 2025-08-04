# Data Analysis and Visualization Tools

## Overview

This folder contains comprehensive analysis and visualization tools for understanding event processing patterns, model performance, and forecasting insights. These tools support all modeling approaches and provide both exploratory data analysis and model evaluation capabilities.

## 📁 Files

- `analyze_data.py` - Core data analysis and statistical exploration
- `detailed_analysis.py` - In-depth pattern analysis and feature engineering insights
- `visualize_patterns.py` - Pattern discovery and temporal analysis visualizations
- `visualize_results.py` - Model performance and prediction analysis
- `README.md` - This documentation

## 🔍 Analysis Capabilities

### Core Data Analysis (`analyze_data.py`)
```python
Key Functions:
- basic_statistics(): Descriptive statistics and distributions
- temporal_patterns(): Time-based pattern identification
- correlation_analysis(): Feature relationship analysis
- outlier_detection(): Anomaly identification
- seasonality_analysis(): Seasonal pattern decomposition
```

### Detailed Analysis (`detailed_analysis.py`)
```python
Advanced Functions:
- feature_importance_analysis(): Variable significance
- time_series_decomposition(): Trend/seasonal/residual
- peak_detection_analysis(): Spike pattern identification
- volatility_analysis(): Variance pattern understanding
- cross_validation_analysis(): Model stability assessment
```

## 📊 Visualization Categories

### 1. Exploratory Data Analysis
```python
Available Plots:
- Distribution histograms and box plots
- Time series overview plots
- Correlation heatmaps
- Seasonal decomposition plots
- Anomaly detection visualizations
```

### 2. Pattern Analysis (`visualize_patterns.py`)
```python
Pattern Visualizations:
- Daily pattern analysis
- Weekly seasonality
- Hourly usage patterns
- Peak occurrence timing
- Event clustering analysis
- Autocorrelation plots
```

### 3. Model Performance (`visualize_results.py`)
```python
Performance Visualizations:
- Prediction vs actual comparisons
- Error analysis plots
- Residual analysis
- Uncertainty visualization
- Learning curves
- Feature importance plots
```

## 🚀 Usage Examples

### Basic Data Exploration
```python
python analyze_data.py
# Outputs: Basic statistics, distributions, temporal overview
```

### Pattern Discovery
```python
python visualize_patterns.py
# Outputs: Seasonal patterns, daily cycles, peak analysis
```

### Model Evaluation
```python
python visualize_results.py
# Outputs: Prediction accuracy, error analysis, performance metrics
```

### Comprehensive Analysis
```python
python detailed_analysis.py
# Outputs: In-depth feature analysis, cross-validation, stability
```

## 📈 Key Analysis Features

### Statistical Analysis
- **Descriptive Statistics**: Mean, median, std, quartiles
- **Distribution Analysis**: Normality tests, skewness, kurtosis
- **Correlation Analysis**: Pearson, Spearman correlations
- **Time Series Analysis**: Stationarity, autocorrelation, seasonality

### Pattern Recognition
- **Seasonal Decomposition**: Trend, seasonal, residual components
- **Peak Detection**: Automatic spike identification
- **Clustering**: Event pattern grouping
- **Anomaly Detection**: Outlier identification and analysis

### Model Diagnostics
- **Prediction Quality**: MAE, MSE, R², MAPE metrics
- **Error Analysis**: Residual patterns, heteroscedasticity
- **Uncertainty Analysis**: Confidence interval evaluation
- **Feature Importance**: Variable contribution analysis

## 🔧 Visualization Tools

### Time Series Plotting
```python
def plot_time_series_overview(data, target_col='queue_length'):
    """
    Comprehensive time series visualization
    - Full time series plot
    - Zoomed sections for detail
    - Statistical annotations
    - Trend lines and patterns
    """
```

### Pattern Analysis
```python
def analyze_daily_patterns(data):
    """
    Daily pattern analysis
    - Hour-of-day averages
    - Weekend vs weekday patterns
    - Peak timing analysis
    - Variability by time of day
    """
```

### Model Performance
```python
def plot_prediction_analysis(predictions, actuals, uncertainties=None):
    """
    Comprehensive prediction evaluation
    - Scatter plots with correlation
    - Time series comparison
    - Error distribution analysis
    - Uncertainty visualization if available
    """
```

## 📊 Output Examples

### 1. Data Overview Dashboard
- Time series plot with key statistics
- Distribution analysis
- Missing data patterns
- Basic correlation matrix

### 2. Seasonal Analysis Report
- Monthly/weekly/daily patterns
- Holiday effect analysis
- Business hours vs after-hours
- Peak timing distributions

### 3. Model Performance Report
- Prediction accuracy metrics
- Error analysis by time period
- Residual pattern analysis
- Uncertainty calibration plots

### 4. Feature Analysis Report
- Feature importance rankings
- Correlation with target variable
- Feature stability over time
- Interaction effects

## 🛠️ Technical Features

### Data Handling
```python
Capabilities:
- Large dataset processing
- Memory-efficient operations
- Missing data handling
- Outlier-robust analysis
```

### Visualization Quality
```python
Features:
- High-resolution plot generation
- Consistent styling and themes
- Interactive plots where beneficial
- Publication-ready figures
```

### Statistical Rigor
```python
Methods:
- Proper significance testing
- Confidence intervals
- Robust statistical measures
- Multiple testing corrections
```

## 📋 Analysis Workflow

### 1. Exploratory Phase
```python
Workflow:
1. Load and validate data
2. Basic statistical summary
3. Distribution analysis
4. Initial visualization
5. Outlier identification
```

### 2. Pattern Discovery
```python
Workflow:
1. Temporal pattern analysis
2. Seasonal decomposition
3. Peak detection
4. Clustering analysis
5. Feature engineering insights
```

### 3. Model Evaluation
```python
Workflow:
1. Load model predictions
2. Calculate performance metrics
3. Error pattern analysis
4. Uncertainty evaluation
5. Comparison across models
```

## 🔍 Specific Analysis Tools

### Peak Analysis
```python
def analyze_peak_events(data, threshold_percentile=95):
    """
    Detailed peak event analysis
    - Peak detection and characterization
    - Duration and intensity analysis
    - Frequency and timing patterns
    - Predictive indicators
    """
```

### Seasonality Detection
```python
def detect_seasonality_patterns(data, periods=[24, 168, 720]):
    """
    Multi-scale seasonality analysis
    - Hourly patterns (24 hours)
    - Weekly patterns (168 hours)
    - Monthly patterns (720 hours)
    - Statistical significance testing
    """
```

### Model Comparison
```python
def compare_model_performance(models_results):
    """
    Comprehensive model comparison
    - Performance metric comparison
    - Error pattern analysis
    - Prediction interval analysis
    - Statistical significance tests
    """
```

## 📈 Business Intelligence

### Operational Insights
- **Peak Timing**: When do highest loads occur?
- **Capacity Planning**: What are baseline requirements?
- **Variability**: How much does demand fluctuate?
- **Predictability**: Which patterns are most reliable?

### Performance Monitoring
- **Model Accuracy**: How well do predictions match reality?
- **Uncertainty**: When is the model confident/uncertain?
- **Drift Detection**: Are patterns changing over time?
- **Feature Stability**: Which features remain predictive?

## 🔄 Integration

### With Modeling Approaches
- **Basic Model**: Simple performance evaluation
- **Improved Model**: Advanced uncertainty analysis
- **Tiered Approach**: Multi-scale pattern analysis
- **Lightning Framework**: Automated analysis integration

### Data Pipeline Integration
```python
# Automatic analysis after model training
def run_comprehensive_analysis(model_results):
    analyze_data.main()
    visualize_patterns.main()
    visualize_results.main(model_results)
    detailed_analysis.main()
```

## 🎯 Output Management

### File Organization
```python
Analysis Outputs:
- Statistical summaries (CSV/JSON)
- Visualization plots (PNG/PDF)
- Analysis reports (HTML/Markdown)
- Interactive dashboards (HTML)
```

### Report Generation
```python
def generate_analysis_report(data, model_results=None):
    """
    Automated report generation
    - Executive summary
    - Detailed findings
    - Visualizations
    - Recommendations
    """
```

## 🔮 Advanced Features

### Interactive Analysis
- Jupyter notebook integration
- Plotly interactive visualizations
- Real-time data exploration
- Parameter sensitivity analysis

### Automated Insights
- Pattern change detection
- Anomaly alerting
- Performance degradation warnings
- Recommendation generation

### Custom Analysis
- Flexible framework for new analyses
- Plugin architecture for extensions
- Custom metric definitions
- Domain-specific insights
