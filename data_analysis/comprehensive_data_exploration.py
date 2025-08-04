"""
Comprehensive Data Exploration and Understanding
===============================================

This script provides deep insights into the event processing data to understand:
1. Data characteristics and quality
2. Temporal patterns and seasonality
3. Statistical properties and distributions
4. Feature relationships and correlations
5. Peak/anomaly patterns
6. Business insights and recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDataExplorer:
    def __init__(self, data_path='../shared_data/EventsMetricsMarJul.csv'):
        """Initialize the data explorer with the dataset."""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis."""
        print("📊 Loading and preparing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['DateTime'])
        
        # Rename queue length column for consistency
        self.data['queue_length'] = self.data['avg_unprocessed_events_count']
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Create time-based features
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['day_of_month'] = self.data['timestamp'].dt.day
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        self.data['is_business_hours'] = ((self.data['hour'] >= 9) & (self.data['hour'] <= 17)).astype(int)
        
        # Calculate derived metrics
        self.data['queue_change'] = self.data['queue_length'].diff()
        self.data['queue_pct_change'] = self.data['queue_length'].pct_change()
        
        print(f"✅ Data loaded: {len(self.data)} records from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        return self.data
    
    def basic_data_quality_analysis(self):
        """Perform comprehensive data quality analysis."""
        print("\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"Total duration: {self.data['timestamp'].max() - self.data['timestamp'].min()}")
        
        # Missing values
        missing_data = self.data.isnull().sum()
        print(f"\nMissing values:\n{missing_data}")
        
        # Data types
        print(f"\nData types:\n{self.data.dtypes}")
        
        # Basic statistics
        print(f"\nBasic statistics for queue_length:")
        print(self.data['queue_length'].describe())
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Time series frequency check
        time_diffs = self.data['timestamp'].diff().dropna()
        print(f"\nTime frequency analysis:")
        print(f"Most common interval: {time_diffs.mode().iloc[0]}")
        print(f"Median interval: {time_diffs.median()}")
        print(f"Frequency consistency: {(time_diffs == time_diffs.mode().iloc[0]).mean():.2%}")
        
        # Outlier detection using IQR
        Q1 = self.data['queue_length'].quantile(0.25)
        Q3 = self.data['queue_length'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.data['queue_length'] < Q1 - 1.5 * IQR) | 
                   (self.data['queue_length'] > Q3 + 1.5 * IQR)).sum()
        print(f"\nOutliers (IQR method): {outliers} ({outliers/len(self.data):.2%})")
    
    def temporal_pattern_analysis(self):
        """Analyze temporal patterns in the data."""
        print("\n📈 TEMPORAL PATTERN ANALYSIS")
        print("=" * 50)
        
        # Daily patterns
        hourly_avg = self.data.groupby('hour')['queue_length'].agg(['mean', 'std', 'min', 'max'])
        print("Hourly patterns (average queue length):")
        print(hourly_avg)
        
        # Weekly patterns
        daily_avg = self.data.groupby('day_of_week')['queue_length'].agg(['mean', 'std', 'min', 'max'])
        daily_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg.index = daily_names
        print(f"\nWeekly patterns:")
        print(daily_avg)
        
        # Monthly patterns
        monthly_avg = self.data.groupby('month')['queue_length'].agg(['mean', 'std', 'min', 'max'])
        print(f"\nMonthly patterns:")
        print(monthly_avg)
        
        # Business hours vs non-business hours
        business_comparison = self.data.groupby('is_business_hours')['queue_length'].agg(['mean', 'std', 'min', 'max'])
        business_comparison.index = ['Non-Business Hours', 'Business Hours']
        print(f"\nBusiness hours comparison:")
        print(business_comparison)
        
        # Weekend vs weekday
        weekend_comparison = self.data.groupby('is_weekend')['queue_length'].agg(['mean', 'std', 'min', 'max'])
        weekend_comparison.index = ['Weekday', 'Weekend']
        print(f"\nWeekend comparison:")
        print(weekend_comparison)
    
    def statistical_analysis(self):
        """Perform detailed statistical analysis."""
        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 50)
        
        queue_data = self.data['queue_length']
        
        # Distribution analysis
        print("Distribution Analysis:")
        print(f"Mean: {queue_data.mean():.2f}")
        print(f"Median: {queue_data.median():.2f}")
        print(f"Mode: {queue_data.mode().iloc[0]:.2f}")
        print(f"Standard Deviation: {queue_data.std():.2f}")
        print(f"Variance: {queue_data.var():.2f}")
        print(f"Skewness: {stats.skew(queue_data):.2f}")
        print(f"Kurtosis: {stats.kurtosis(queue_data):.2f}")
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(queue_data.sample(5000))  # Sample for performance
        print(f"\nNormality Tests:")
        print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")
        
        # Stationarity test
        adf_result = adfuller(queue_data.dropna())
        print(f"\nStationarity Test (Augmented Dickey-Fuller):")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4e}")
        print(f"Critical Values: {adf_result[4]}")
        
        if adf_result[1] <= 0.05:
            print("✅ Series is stationary")
        else:
            print("❌ Series is non-stationary")
    
    def peak_and_anomaly_analysis(self):
        """Analyze peaks and anomalies in the data."""
        print("\n🔺 PEAK AND ANOMALY ANALYSIS")
        print("=" * 50)
        
        queue_data = self.data['queue_length'].values
        
        # Peak detection
        # Use 95th percentile as threshold for peaks
        peak_threshold = np.percentile(queue_data, 95)
        peaks, properties = find_peaks(queue_data, height=peak_threshold, distance=6)  # 12 minutes apart
        
        print(f"Peak Analysis:")
        print(f"Peak threshold (95th percentile): {peak_threshold:.2f}")
        print(f"Number of peaks found: {len(peaks)}")
        print(f"Peak frequency: {len(peaks) / (len(queue_data) / (24 * 30)):.2f} peaks per hour")
        
        if len(peaks) > 0:
            peak_values = queue_data[peaks]
            print(f"Average peak value: {peak_values.mean():.2f}")
            print(f"Max peak value: {peak_values.max():.2f}")
            print(f"Peak value std: {peak_values.std():.2f}")
            
            # Peak timing analysis
            peak_times = self.data.iloc[peaks]['timestamp']
            peak_hours = peak_times.dt.hour
            print(f"\nPeak timing distribution (by hour):")
            peak_hour_dist = peak_hours.value_counts().sort_index()
            print(peak_hour_dist)
        
        # Anomaly detection using z-score
        z_scores = np.abs(stats.zscore(queue_data))
        anomalies = z_scores > 3  # 3 standard deviations
        print(f"\nAnomaly Analysis (z-score > 3):")
        print(f"Number of anomalies: {anomalies.sum()}")
        print(f"Anomaly percentage: {anomalies.mean() * 100:.2f}%")
    
    def correlation_and_feature_analysis(self):
        """Analyze correlations between features."""
        print("\n🔗 CORRELATION AND FEATURE ANALYSIS")
        print("=" * 50)
        
        # Select numeric columns for correlation
        numeric_cols = ['queue_length', 'hour', 'day_of_week', 'day_of_month', 'month',
                       'is_weekend', 'is_business_hours', 'queue_change', 'queue_pct_change']
        
        correlation_matrix = self.data[numeric_cols].corr()
        
        print("Correlation with queue_length:")
        queue_correlations = correlation_matrix['queue_length'].sort_values(key=abs, ascending=False)
        print(queue_correlations)
        
        # Feature importance insights
        print(f"\nKey insights:")
        print(f"- Strongest time-based correlation: {queue_correlations.drop('queue_length').abs().idxmax()} "
              f"({queue_correlations.drop('queue_length').abs().max():.3f})")
        
        # Lag correlation analysis
        print(f"\nLag correlation analysis:")
        for lag in [1, 6, 12, 24, 48]:  # Various lags
            lag_corr = self.data['queue_length'].corr(self.data['queue_length'].shift(lag))
            print(f"Lag {lag} (2-min intervals): {lag_corr:.3f}")
    
    def seasonality_decomposition(self):
        """Perform time series decomposition."""
        print("\n📅 SEASONALITY DECOMPOSITION")
        print("=" * 50)
        
        # Prepare data for decomposition
        ts_data = self.data.set_index('timestamp')['queue_length']
        
        # Resample to hourly for better decomposition
        hourly_data = ts_data.resample('H').mean()
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(hourly_data, model='additive', period=24)  # 24-hour cycle
            
            trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(hourly_data.dropna())
            seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var((hourly_data - decomposition.trend).dropna())
            
            print(f"Seasonality Analysis (24-hour cycle):")
            print(f"Trend strength: {trend_strength:.3f}")
            print(f"Seasonal strength: {seasonal_strength:.3f}")
            
            # Weekly seasonality
            weekly_decomposition = seasonal_decompose(hourly_data, model='additive', period=168)  # 168-hour cycle
            weekly_seasonal_strength = 1 - np.var(weekly_decomposition.resid.dropna()) / np.var((hourly_data - weekly_decomposition.trend).dropna())
            print(f"Weekly seasonal strength: {weekly_seasonal_strength:.3f}")
            
        except Exception as e:
            print(f"Decomposition failed: {e}")
    
    def business_insights_analysis(self):
        """Generate business insights from the data."""
        print("\n💼 BUSINESS INSIGHTS")
        print("=" * 50)
        
        # Capacity utilization analysis
        if 'processing_capacity' in self.data.columns:
            utilization = self.data['queue_length'] / self.data['processing_capacity']
            print(f"Capacity Utilization Analysis:")
            print(f"Average utilization: {utilization.mean():.2%}")
            print(f"Max utilization: {utilization.max():.2%}")
            print(f"Over-capacity periods: {(utilization > 1).mean():.2%}")
        
        # Peak hours identification
        hourly_stats = self.data.groupby('hour')['queue_length'].agg(['mean', 'max', 'std'])
        peak_hours = hourly_stats.nlargest(3, 'mean').index.tolist()
        low_hours = hourly_stats.nsmallest(3, 'mean').index.tolist()
        
        print(f"\nOperational Insights:")
        print(f"Peak hours (highest average): {peak_hours}")
        print(f"Low hours (lowest average): {low_hours}")
        
        # Variability analysis
        cv_by_hour = (hourly_stats['std'] / hourly_stats['mean']).sort_values(ascending=False)
        print(f"\nMost variable hours (coefficient of variation):")
        print(cv_by_hour.head())
        
        # Growth trends
        monthly_growth = self.data.groupby('month')['queue_length'].mean().pct_change()
        print(f"\nMonthly growth trends:")
        print(monthly_growth)
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"1. Scale up during peak hours: {peak_hours}")
        print(f"2. Monitor high variability hours: {cv_by_hour.head(3).index.tolist()}")
        print(f"3. Consider different strategies for business vs non-business hours")
        print(f"4. Weekly patterns suggest different weekend scaling needs")
    
    def generate_comprehensive_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n📊 Generating comprehensive visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Time series overview
        ax1 = plt.subplot(6, 2, 1)
        self.data.set_index('timestamp')['queue_length'].plot(ax=ax1, alpha=0.7)
        plt.title('Complete Time Series Overview')
        plt.ylabel('Queue Length')
        
        # 2. Distribution
        ax2 = plt.subplot(6, 2, 2)
        plt.hist(self.data['queue_length'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(self.data['queue_length'].mean(), color='red', linestyle='--', label='Mean')
        plt.axvline(self.data['queue_length'].median(), color='green', linestyle='--', label='Median')
        plt.title('Queue Length Distribution')
        plt.xlabel('Queue Length')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 3. Hourly patterns
        ax3 = plt.subplot(6, 2, 3)
        hourly_stats = self.data.groupby('hour')['queue_length'].agg(['mean', 'std'])
        plt.errorbar(hourly_stats.index, hourly_stats['mean'], yerr=hourly_stats['std'], 
                    capsize=3, capthick=1, alpha=0.8)
        plt.title('Average Queue Length by Hour (with std)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Queue Length')
        plt.grid(True, alpha=0.3)
        
        # 4. Weekly patterns
        ax4 = plt.subplot(6, 2, 4)
        daily_stats = self.data.groupby('day_of_week')['queue_length'].agg(['mean', 'std'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.errorbar(days, daily_stats['mean'], yerr=daily_stats['std'], 
                    capsize=3, capthick=1, alpha=0.8)
        plt.title('Average Queue Length by Day of Week')
        plt.ylabel('Queue Length')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Monthly trends
        ax5 = plt.subplot(6, 2, 5)
        monthly_stats = self.data.groupby('month')['queue_length'].agg(['mean', 'std'])
        months = ['Mar', 'Apr', 'May', 'Jun', 'Jul']
        plt.errorbar(months, monthly_stats['mean'], yerr=monthly_stats['std'], 
                    capsize=3, capthick=1, alpha=0.8)
        plt.title('Average Queue Length by Month')
        plt.ylabel('Queue Length')
        plt.grid(True, alpha=0.3)
        
        # 6. Business hours comparison
        ax6 = plt.subplot(6, 2, 6)
        business_stats = self.data.groupby('is_business_hours')['queue_length'].agg(['mean', 'std'])
        labels = ['Non-Business', 'Business Hours']
        plt.bar(labels, business_stats['mean'], yerr=business_stats['std'], 
               capsize=5, alpha=0.8, color=['skyblue', 'orange'])
        plt.title('Business Hours vs Non-Business Hours')
        plt.ylabel('Average Queue Length')
        
        # 7. Correlation heatmap
        ax7 = plt.subplot(6, 2, 7)
        numeric_cols = ['queue_length', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours']
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 8. Box plot by hour
        ax8 = plt.subplot(6, 2, 8)
        hourly_data = [self.data[self.data['hour'] == h]['queue_length'].values 
                      for h in range(24)]
        plt.boxplot(hourly_data, labels=range(24))
        plt.title('Queue Length Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Queue Length')
        plt.xticks(range(1, 25, 2), range(0, 24, 2))
        
        # 9. Autocorrelation plot
        ax9 = plt.subplot(6, 2, 9)
        autocorr = acf(self.data['queue_length'].dropna(), nlags=100)
        plt.plot(autocorr)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        plt.title('Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True, alpha=0.3)
        
        # 10. Rolling statistics
        ax10 = plt.subplot(6, 2, 10)
        rolling_mean = self.data['queue_length'].rolling(window=72).mean()  # 2.4 hours
        rolling_std = self.data['queue_length'].rolling(window=72).std()
        
        plt.plot(self.data['timestamp'], rolling_mean, label='Rolling Mean (2.4h)', alpha=0.8)
        plt.fill_between(self.data['timestamp'], 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.3, label='±1 Std')
        plt.title('Rolling Statistics (2.4 hour window)')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 11. Peak analysis
        ax11 = plt.subplot(6, 2, 11)
        # Sample a week for peak visualization
        week_data = self.data[self.data['timestamp'].dt.date == self.data['timestamp'].dt.date.iloc[0]]
        queue_values = week_data['queue_length'].values
        peak_threshold = np.percentile(self.data['queue_length'], 95)
        peaks, _ = find_peaks(queue_values, height=peak_threshold, distance=6)
        
        plt.plot(week_data['timestamp'], queue_values, alpha=0.8)
        if len(peaks) > 0:
            plt.scatter(week_data['timestamp'].iloc[peaks], queue_values[peaks], 
                       color='red', s=50, zorder=5, label='Detected Peaks')
        plt.axhline(y=peak_threshold, color='red', linestyle='--', alpha=0.7, label='Peak Threshold')
        plt.title('Peak Detection Example (First Day)')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 12. Change analysis
        ax12 = plt.subplot(6, 2, 12)
        changes = self.data['queue_change'].dropna()
        plt.hist(changes, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(changes.mean(), color='red', linestyle='--', label='Mean Change')
        plt.axvline(0, color='green', linestyle='-', alpha=0.7, label='No Change')
        plt.title('Queue Length Change Distribution')
        plt.xlabel('Change in Queue Length')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('../results/comprehensive_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualization saved as 'comprehensive_data_analysis.png'")
    
    def run_complete_analysis(self):
        """Run the complete data exploration pipeline."""
        print("🚀 Starting Comprehensive Data Exploration")
        print("=" * 60)
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all analyses
        self.basic_data_quality_analysis()
        self.temporal_pattern_analysis()
        self.statistical_analysis()
        self.peak_and_anomaly_analysis()
        self.correlation_and_feature_analysis()
        self.seasonality_decomposition()
        self.business_insights_analysis()
        
        # Generate visualizations
        self.generate_comprehensive_visualizations()
        
        print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Key findings and model recommendations available in the output above.")
        print("Detailed visualizations saved to results folder.")

if __name__ == "__main__":
    explorer = ComprehensiveDataExplorer()
    explorer.run_complete_analysis()
