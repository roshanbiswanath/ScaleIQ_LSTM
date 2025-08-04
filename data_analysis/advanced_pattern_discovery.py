"""
Advanced Pattern Discovery and Feature Engineering Analysis
=========================================================

This script performs deep pattern discovery to understand:
1. Multi-scale temporal patterns (intraday, daily, weekly, monthly)
2. Event clustering and behavioral segmentation
3. Advanced feature engineering opportunities
4. Model-specific insights for optimal architecture selection
5. Forecasting horizon analysis
6. Feature importance and selection recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, periodogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternAnalyzer:
    def __init__(self, data_path='../shared_data/EventsMetricsMarJul.csv'):
        """Initialize the advanced pattern analyzer."""
        self.data_path = data_path
        self.data = None
        self.features = None
        
    def load_and_engineer_features(self):
        """Load data and create comprehensive features."""
        print("🔧 Loading data and engineering advanced features...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['DateTime'])
        self.data['queue_length'] = self.data['avg_unprocessed_events_count']
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Basic temporal features
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['day_of_month'] = self.data['timestamp'].dt.day
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['week_of_year'] = self.data['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        
        # Time-based indicators
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        self.data['is_business_hours'] = ((self.data['hour'] >= 9) & (self.data['hour'] <= 17)).astype(int)
        self.data['is_peak_hour'] = self.data['hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
        self.data['is_lunch_hour'] = self.data['hour'].isin([12, 13]).astype(int)
        
        # Rolling statistics (multiple windows)
        windows = [6, 12, 24, 72, 168]  # 12min, 24min, 48min, 2.4h, 5.6h
        for window in windows:
            self.data[f'rolling_mean_{window}'] = self.data['queue_length'].rolling(window, min_periods=1).mean()
            self.data[f'rolling_std_{window}'] = self.data['queue_length'].rolling(window, min_periods=1).std()
            self.data[f'rolling_min_{window}'] = self.data['queue_length'].rolling(window, min_periods=1).min()
            self.data[f'rolling_max_{window}'] = self.data['queue_length'].rolling(window, min_periods=1).max()
        
        # Lag features
        lags = [1, 3, 6, 12, 24, 72, 168]  # Various time horizons
        for lag in lags:
            self.data[f'lag_{lag}'] = self.data['queue_length'].shift(lag)
        
        # Change features
        self.data['change_1'] = self.data['queue_length'].diff(1)
        self.data['change_pct_1'] = self.data['queue_length'].pct_change(1)
        self.data['change_12'] = self.data['queue_length'].diff(12)  # 24-minute change
        self.data['change_pct_12'] = self.data['queue_length'].pct_change(12)
        
        # Velocity and acceleration
        self.data['velocity'] = self.data['change_1']
        self.data['acceleration'] = self.data['velocity'].diff(1)
        
        # Exponential moving averages
        alphas = [0.1, 0.3, 0.5]
        for alpha in alphas:
            self.data[f'ema_{int(alpha*10)}'] = self.data['queue_length'].ewm(alpha=alpha).mean()
        
        print(f"✅ Feature engineering complete: {self.data.shape[1]} features created")
        return self.data
    
    def multi_scale_temporal_analysis(self):
        """Analyze patterns at multiple temporal scales."""
        print("\n⏰ MULTI-SCALE TEMPORAL ANALYSIS")
        print("=" * 50)
        
        # Intraday patterns (within day)
        print("INTRADAY PATTERNS:")
        hourly_stats = self.data.groupby('hour')['queue_length'].agg(['mean', 'std', 'min', 'max'])
        peak_hours = hourly_stats['mean'].nlargest(5).index.tolist()
        valley_hours = hourly_stats['mean'].nsmallest(5).index.tolist()
        print(f"Peak hours: {peak_hours}")
        print(f"Valley hours: {valley_hours}")
        print(f"Intraday variability (CV): {(hourly_stats['std'] / hourly_stats['mean']).mean():.3f}")
        
        # Daily patterns (across days)
        print(f"\nDAILY PATTERNS:")
        daily_agg = self.data.groupby(self.data['timestamp'].dt.date)['queue_length'].agg(['mean', 'std', 'max'])
        print(f"Daily average variability: {daily_agg['mean'].std():.2f}")
        print(f"Average daily peak: {daily_agg['max'].mean():.2f}")
        print(f"Daily peak variability: {daily_agg['max'].std():.2f}")
        
        # Weekly patterns
        print(f"\nWEEKLY PATTERNS:")
        weekly_stats = self.data.groupby('day_of_week')['queue_length'].agg(['mean', 'std'])
        weekday_avg = weekly_stats.loc[0:4, 'mean'].mean()  # Mon-Fri
        weekend_avg = weekly_stats.loc[5:6, 'mean'].mean()  # Sat-Sun
        print(f"Weekday average: {weekday_avg:.2f}")
        print(f"Weekend average: {weekend_avg:.2f}")
        print(f"Weekend vs Weekday ratio: {weekend_avg/weekday_avg:.3f}")
        
        # Monthly trends
        print(f"\nMONTHLY TRENDS:")
        monthly_stats = self.data.groupby('month')['queue_length'].agg(['mean', 'std'])
        monthly_growth = monthly_stats['mean'].pct_change().fillna(0)
        print(f"Monthly growth rates:")
        months = ['Mar', 'Apr', 'May', 'Jun', 'Jul']
        for i, month in enumerate(months, 3):
            if i in monthly_growth.index:
                print(f"  {month}: {monthly_growth[i]:.2%}")
    
    def event_clustering_analysis(self):
        """Perform clustering analysis to identify behavioral patterns."""
        print("\n🎯 EVENT CLUSTERING ANALYSIS")
        print("=" * 50)
        
        # Create features for clustering
        cluster_features = [
            'hour', 'day_of_week', 'queue_length', 'rolling_mean_24', 'rolling_std_24',
            'change_1', 'change_pct_1', 'is_business_hours', 'is_weekend'
        ]
        
        cluster_data = self.data[cluster_features].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Find optimal number of clusters
        silhouette_scores = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        print(f"\nCluster Analysis:")
        for cluster in range(optimal_k):
            cluster_data = self.data[self.data['cluster'] == cluster]
            print(f"\nCluster {cluster} ({len(cluster_data)} samples, {len(cluster_data)/len(self.data):.1%}):")
            print(f"  Avg queue length: {cluster_data['queue_length'].mean():.2f}")
            print(f"  Common hours: {cluster_data['hour'].mode().values}")
            print(f"  Weekday/Weekend: {cluster_data['is_weekend'].value_counts().to_dict()}")
            print(f"  Business hours: {cluster_data['is_business_hours'].mean():.2%}")
    
    def frequency_domain_analysis(self):
        """Analyze patterns in frequency domain."""
        print("\n📊 FREQUENCY DOMAIN ANALYSIS")
        print("=" * 50)
        
        # Compute periodogram
        queue_data = self.data['queue_length'].fillna(method='ffill')
        frequencies, power = periodogram(queue_data, fs=1/2)  # 2-minute sampling
        
        # Find dominant frequencies
        peak_indices = find_peaks(power, height=np.percentile(power, 95))[0]
        dominant_periods = 1 / frequencies[peak_indices] if len(peak_indices) > 0 else []
        
        print(f"Dominant periods found:")
        for period in sorted(dominant_periods, reverse=True):
            if period > 1:  # Only periods longer than 2 minutes
                hours = period * 2 / 60  # Convert to hours
                if hours < 48:  # Only show periods up to 2 days
                    print(f"  {period:.1f} samples ({hours:.1f} hours)")
        
        # Expected periods for validation
        expected_periods = {
            '30 minutes': 15,   # 15 samples * 2 min = 30 min
            '1 hour': 30,       # 30 samples * 2 min = 1 hour
            '12 hours': 360,    # 360 samples * 2 min = 12 hours
            '24 hours': 720,    # 720 samples * 2 min = 24 hours
            '1 week': 5040      # 5040 samples * 2 min = 1 week
        }
        
        print(f"\nExpected vs Found periods:")
        for name, expected in expected_periods.items():
            closest_period = min(dominant_periods, key=lambda x: abs(x - expected)) if dominant_periods else None
            if closest_period and abs(closest_period - expected) / expected < 0.1:  # Within 10%
                print(f"  ✅ {name}: Found {closest_period:.1f} (expected {expected})")
            else:
                print(f"  ❌ {name}: Not clearly detected (expected {expected})")
    
    def feature_importance_analysis(self):
        """Analyze feature importance for forecasting."""
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Prepare feature matrix
        feature_cols = [col for col in self.data.columns if col not in 
                       ['timestamp', 'queue_length', 'cluster'] and not col.startswith('lag_')]
        
        # Add only relevant lag features to avoid data leakage
        target_lags = [1, 3, 6, 12, 24]  # These are okay for forecasting
        for lag in target_lags:
            if f'lag_{lag}' in self.data.columns:
                feature_cols.append(f'lag_{lag}')
        
        # Create target (next step prediction)
        X = self.data[feature_cols].fillna(0)
        y = self.data['queue_length'].shift(-1).fillna(method='ffill')  # Predict next step
        
        # Remove last row (no target)
        X = X[:-1]
        y = y[:-1]
        
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        print(f"Top 15 features (Random Forest importance):")
        for i, (feature, importance) in enumerate(rf_importance.head(15).items(), 1):
            print(f"{i:2d}. {feature:<25}: {importance:.4f}")
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        print(f"\nTop 15 features (Mutual Information):")
        for i, (feature, score) in enumerate(mi_importance.head(15).items(), 1):
            print(f"{i:2d}. {feature:<25}: {score:.4f}")
        
        # Feature categories analysis
        temporal_features = [col for col in feature_cols if any(x in col for x in ['hour', 'day', 'month', 'week', 'sin', 'cos', 'is_'])]
        rolling_features = [col for col in feature_cols if 'rolling' in col]
        lag_features = [col for col in feature_cols if 'lag_' in col]
        change_features = [col for col in feature_cols if 'change' in col or 'velocity' in col or 'acceleration' in col]
        ema_features = [col for col in feature_cols if 'ema' in col]
        
        print(f"\nFeature category importance (Random Forest):")
        print(f"Temporal features: {rf_importance[temporal_features].sum():.3f}")
        print(f"Rolling statistics: {rf_importance[rolling_features].sum():.3f}")
        print(f"Lag features: {rf_importance[lag_features].sum():.3f}")
        print(f"Change features: {rf_importance[change_features].sum():.3f}")
        print(f"EMA features: {rf_importance[ema_features].sum():.3f}")
    
    def forecasting_horizon_analysis(self):
        """Analyze optimal forecasting horizons."""
        print("\n🔮 FORECASTING HORIZON ANALYSIS")
        print("=" * 50)
        
        # Test different prediction horizons
        horizons = [1, 3, 6, 12, 24, 48]  # 2min to 96min ahead
        horizon_performance = {}
        
        # Use simple baseline (last value) for quick analysis
        for horizon in horizons:
            # Create target at different horizons
            target = self.data['queue_length'].shift(-horizon)
            baseline_pred = self.data['queue_length']  # Naive forecast
            
            # Calculate performance (remove NaN values)
            valid_mask = ~(target.isna() | baseline_pred.isna())
            if valid_mask.sum() > 0:
                mse = np.mean((target[valid_mask] - baseline_pred[valid_mask]) ** 2)
                mae = np.mean(np.abs(target[valid_mask] - baseline_pred[valid_mask]))
                horizon_performance[horizon] = {'MSE': mse, 'MAE': mae}
        
        print(f"Baseline performance by horizon (naive forecast):")
        print(f"{'Horizon':<8} {'Minutes':<8} {'MSE':<10} {'MAE':<10}")
        print("-" * 40)
        for horizon, perf in horizon_performance.items():
            minutes = horizon * 2
            print(f"{horizon:<8} {minutes:<8} {perf['MSE']:<10.2f} {perf['MAE']:<10.2f}")
        
        # Autocorrelation analysis for optimal sequence length
        autocorr = acf(self.data['queue_length'].dropna(), nlags=200, fft=True)
        
        # Find where autocorrelation drops below threshold
        threshold = 0.1
        optimal_sequence = np.argmax(autocorr < threshold)
        if optimal_sequence == 0:  # If never drops below threshold
            optimal_sequence = len(autocorr) - 1
        
        print(f"\nSequence length analysis:")
        print(f"Autocorrelation drops below {threshold} at lag {optimal_sequence} ({optimal_sequence * 2} minutes)")
        print(f"Recommended sequence length: {min(optimal_sequence, 72)} (max 144 minutes)")
    
    def model_architecture_recommendations(self):
        """Provide model architecture recommendations based on analysis."""
        print("\n🏗️ MODEL ARCHITECTURE RECOMMENDATIONS")
        print("=" * 50)
        
        # Analyze data characteristics
        data_size = len(self.data)
        feature_count = len([col for col in self.data.columns if col not in ['timestamp', 'queue_length']])
        seasonality_strength = self._calculate_seasonality_strength()
        trend_strength = self._calculate_trend_strength()
        volatility = self.data['queue_length'].std() / self.data['queue_length'].mean()
        
        print(f"Data Characteristics:")
        print(f"  Dataset size: {data_size:,} samples")
        print(f"  Available features: {feature_count}")
        print(f"  Seasonality strength: {seasonality_strength:.3f}")
        print(f"  Trend strength: {trend_strength:.3f}")
        print(f"  Coefficient of variation: {volatility:.3f}")
        
        # Recommendations based on analysis
        print(f"\nModel Recommendations:")
        
        # 1. Architecture complexity
        if data_size > 50000:
            print("✅ Deep models: Sufficient data for complex architectures")
        else:
            print("⚠️  Simpler models: Limited data, avoid overfitting")
        
        # 2. Sequence modeling
        if seasonality_strength > 0.3:
            print("✅ LSTM/GRU: Strong temporal patterns detected")
        else:
            print("⚠️  Consider feedforward: Weak temporal dependencies")
        
        # 3. Attention mechanism
        if feature_count > 20 and seasonality_strength > 0.2:
            print("✅ Attention: Multiple features with temporal relationships")
        else:
            print("⚠️  Skip attention: May not provide significant benefit")
        
        # 4. Multi-scale approach
        if seasonality_strength > 0.4 and volatility > 0.5:
            print("✅ Tiered approach: Strong seasonality + high variability")
        else:
            print("⚠️  Single-scale: Patterns may not warrant complexity")
        
        # 5. Uncertainty quantification
        if volatility > 0.3:
            print("✅ Uncertainty estimation: High variability requires confidence intervals")
        else:
            print("⚠️  Point estimates: Low variability, simpler outputs sufficient")
        
        # 6. Recommended architecture
        print(f"\n🎯 OPTIMAL MODEL ARCHITECTURE:")
        if seasonality_strength > 0.4 and volatility > 0.5:
            print("TIER 1: Advanced Tiered Forecasting System")
            print("  - Daily forecaster: LSTM (128 hidden, 2 layers)")
            print("  - Intraday forecaster: BiLSTM + Attention (256 hidden, 3 layers)")
            print("  - Uncertainty quantification: Yes")
            print("  - Sequence length: 48 (intraday), 30 (daily)")
        elif seasonality_strength > 0.3:
            print("TIER 2: Enhanced LSTM with Attention")
            print("  - Architecture: BiLSTM + Multi-head attention")
            print("  - Hidden size: 256, Layers: 3")
            print("  - Uncertainty quantification: Yes")
            print("  - Sequence length: 48")
        else:
            print("TIER 3: Standard LSTM")
            print("  - Architecture: LSTM")
            print("  - Hidden size: 128, Layers: 2")
            print("  - Uncertainty quantification: Optional")
            print("  - Sequence length: 30")
    
    def _calculate_seasonality_strength(self):
        """Calculate seasonality strength metric."""
        try:
            # Resample to hourly for decomposition
            hourly_data = self.data.set_index('timestamp')['queue_length'].resample('H').mean()
            decomposition = seasonal_decompose(hourly_data, model='additive', period=24)
            seasonal_var = np.var(decomposition.seasonal.dropna())
            total_var = np.var(hourly_data.dropna())
            return seasonal_var / total_var if total_var > 0 else 0
        except:
            return 0
    
    def _calculate_trend_strength(self):
        """Calculate trend strength metric."""
        try:
            # Simple trend strength using linear regression
            x = np.arange(len(self.data))
            y = self.data['queue_length'].values
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return abs(r_value)
        except:
            return 0
    
    def generate_pattern_visualizations(self):
        """Generate comprehensive pattern visualizations."""
        print("\n📊 Generating pattern analysis visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Multi-scale decomposition
        ax1 = plt.subplot(4, 3, 1)
        hourly_data = self.data.set_index('timestamp')['queue_length'].resample('H').mean()
        try:
            decomposition = seasonal_decompose(hourly_data, model='additive', period=24)
            decomposition.seasonal.plot(ax=ax1, title='Daily Seasonality')
        except:
            plt.title('Seasonality Analysis Failed')
        
        # 2. Autocorrelation
        ax2 = plt.subplot(4, 3, 2)
        autocorr = acf(self.data['queue_length'].dropna(), nlags=100)
        plt.plot(autocorr)
        plt.title('Autocorrelation Function')
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        
        # 3. Feature importance
        ax3 = plt.subplot(4, 3, 3)
        # Quick feature importance (simplified)
        corr_with_target = self.data.corr()['queue_length'].abs().sort_values(ascending=False)[1:11]
        corr_with_target.plot(kind='barh', ax=ax3)
        plt.title('Top Feature Correlations')
        
        # 4. Hourly patterns by day type
        ax4 = plt.subplot(4, 3, 4)
        weekday_pattern = self.data[self.data['is_weekend'] == 0].groupby('hour')['queue_length'].mean()
        weekend_pattern = self.data[self.data['is_weekend'] == 1].groupby('hour')['queue_length'].mean()
        plt.plot(weekday_pattern.index, weekday_pattern.values, label='Weekday', marker='o')
        plt.plot(weekend_pattern.index, weekend_pattern.values, label='Weekend', marker='s')
        plt.title('Hourly Patterns by Day Type')
        plt.legend()
        plt.xlabel('Hour')
        plt.ylabel('Avg Queue Length')
        
        # 5. Monthly trends
        ax5 = plt.subplot(4, 3, 5)
        monthly_avg = self.data.groupby('month')['queue_length'].mean()
        monthly_std = self.data.groupby('month')['queue_length'].std()
        months = ['Mar', 'Apr', 'May', 'Jun', 'Jul']
        plt.errorbar(months, monthly_avg.values, yerr=monthly_std.values, capsize=5)
        plt.title('Monthly Trends')
        plt.ylabel('Queue Length')
        
        # 6. Volatility analysis
        ax6 = plt.subplot(4, 3, 6)
        rolling_std = self.data['queue_length'].rolling(72).std()  # 2.4 hour window
        plt.plot(self.data['timestamp'], rolling_std, alpha=0.7)
        plt.title('Rolling Volatility (2.4h window)')
        plt.xticks(rotation=45)
        
        # 7. Change distribution
        ax7 = plt.subplot(4, 3, 7)
        changes = self.data['change_1'].dropna()
        plt.hist(changes, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='No change')
        plt.title('Queue Length Changes')
        plt.xlabel('Change')
        plt.legend()
        
        # 8. Peak timing analysis
        ax8 = plt.subplot(4, 3, 8)
        peak_threshold = self.data['queue_length'].quantile(0.95)
        peak_data = self.data[self.data['queue_length'] > peak_threshold]
        peak_hours = peak_data['hour'].value_counts().sort_index()
        peak_hours.plot(kind='bar', ax=ax8)
        plt.title('Peak Event Timing')
        plt.xlabel('Hour')
        plt.ylabel('Peak Count')
        
        # 9. Business hours impact
        ax9 = plt.subplot(4, 3, 9)
        business_comparison = self.data.groupby(['hour', 'is_business_hours'])['queue_length'].mean().unstack()
        business_comparison.plot(ax=ax9, kind='line')
        plt.title('Business Hours Effect')
        plt.legend(['Non-Business', 'Business'])
        
        # 10. Rolling correlation
        ax10 = plt.subplot(4, 3, 10)
        if 'lag_24' in self.data.columns:
            rolling_corr = self.data['queue_length'].rolling(72).corr(self.data['lag_24'])
            plt.plot(self.data['timestamp'], rolling_corr, alpha=0.7)
            plt.title('Rolling 24-step Autocorrelation')
            plt.xticks(rotation=45)
        
        # 11. Weekly seasonality
        ax11 = plt.subplot(4, 3, 11)
        daily_patterns = self.data.groupby('day_of_week')['queue_length'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(days, daily_patterns.values)
        plt.title('Weekly Seasonality')
        plt.xticks(rotation=45)
        
        # 12. Feature correlation heatmap
        ax12 = plt.subplot(4, 3, 12)
        important_features = ['queue_length', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                            'is_business_hours', 'rolling_mean_24', 'rolling_std_24']
        if all(col in self.data.columns for col in important_features):
            corr_matrix = self.data[important_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', ax=ax12)
            plt.title('Key Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('../results/advanced_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Advanced pattern visualizations saved as 'advanced_pattern_analysis.png'")
    
    def run_complete_analysis(self):
        """Run the complete advanced pattern analysis."""
        print("🚀 Starting Advanced Pattern Discovery Analysis")
        print("=" * 60)
        
        # Load and prepare data
        self.load_and_engineer_features()
        
        # Run all analyses
        self.multi_scale_temporal_analysis()
        self.event_clustering_analysis()
        self.frequency_domain_analysis()
        self.feature_importance_analysis()
        self.forecasting_horizon_analysis()
        self.model_architecture_recommendations()
        
        # Generate visualizations
        self.generate_pattern_visualizations()
        
        print("\n✅ ADVANCED PATTERN ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Detailed recommendations for optimal model architecture provided above.")

if __name__ == "__main__":
    analyzer = AdvancedPatternAnalyzer()
    analyzer.run_complete_analysis()
