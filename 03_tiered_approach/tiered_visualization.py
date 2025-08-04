import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import torch
from tiered_forecasting_system import TieredForecaster, DailyAggregator, IntradayPatternExtractor

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TieredVisualization:
    """Visualization system for tiered forecasting results"""
    
    def __init__(self):
        self.aggregator = DailyAggregator()
        self.pattern_extractor = IntradayPatternExtractor()
    
    def plot_daily_analysis(self, daily_data, daily_forecasts=None):
        """Plot daily-level analysis and forecasts"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # 1. Daily volume trend
        axes[0, 0].plot(daily_data['DateTime'], daily_data['logged_events_sum'], 
                       'b-', linewidth=2, label='Actual Daily Volume')
        
        if daily_forecasts is not None:
            axes[0, 0].plot(daily_forecasts['DateTime'], daily_forecasts['forecast'], 
                           'r--', linewidth=2, label='Forecasted Daily Volume')
        
        axes[0, 0].set_title('Daily Event Volume Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Daily Events')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Weekly seasonality
        daily_data['day_name'] = daily_data['DateTime'].dt.day_name()
        weekly_pattern = daily_data.groupby('day_of_week')['logged_events_sum'].mean()
        
        axes[0, 1].bar(range(7), weekly_pattern.values, alpha=0.7, 
                      color=['lightblue' if i < 5 else 'lightcoral' for i in range(7)])
        axes[0, 1].set_title('Weekly Seasonality Pattern')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Daily Events')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly growth trend
        monthly_data = daily_data.groupby(daily_data['DateTime'].dt.to_period('M'))['logged_events_sum'].sum()
        
        axes[1, 0].plot(monthly_data.index.astype(str), monthly_data.values, 
                       'g-o', linewidth=2, markersize=8)
        axes[1, 0].set_title('Monthly Growth Trend')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Monthly Total Events')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Daily statistics distribution
        daily_stats = ['logged_events_mean', 'logged_events_std', 'logged_events_max']
        for i, stat in enumerate(daily_stats):
            axes[1, 1].hist(daily_data[stat], bins=30, alpha=0.7, label=stat.replace('logged_events_', ''))
        
        axes[1, 1].set_title('Daily Statistics Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Growth rate analysis
        axes[2, 0].plot(daily_data['DateTime'], daily_data['daily_growth_rate'], 
                       'purple', linewidth=1, alpha=0.7)
        axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[2, 0].set_title('Daily Growth Rate')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Growth Rate')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Weekend vs weekday comparison
        weekend_data = daily_data[daily_data['is_weekend'] == 1]['logged_events_sum']
        weekday_data = daily_data[daily_data['is_weekend'] == 0]['logged_events_sum']
        
        axes[2, 1].boxplot([weekday_data, weekend_data], labels=['Weekdays', 'Weekends'])
        axes[2, 1].set_title('Weekday vs Weekend Volume')
        axes[2, 1].set_ylabel('Daily Events')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_intraday_analysis(self, intraday_data, sample_days=3):
        """Plot intraday pattern analysis"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # 1. Hourly pattern (average across all days)
        hourly_pattern = intraday_data.groupby('hour')['logged_events'].mean()
        
        axes[0, 0].plot(hourly_pattern.index, hourly_pattern.values, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Average Hourly Pattern')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Events per 2-min Interval')
        axes[0, 0].set_xticks(range(0, 24, 2))
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Intraday ratio patterns
        axes[0, 1].scatter(intraday_data['minute_of_day'], intraday_data['intraday_ratio'], 
                          alpha=0.1, s=1, c='blue')
        
        # Add rolling average
        rolling_avg = intraday_data.groupby('minute_of_day')['intraday_ratio'].mean()
        axes[0, 1].plot(rolling_avg.index, rolling_avg.values, 'r-', linewidth=2, label='Average Pattern')
        
        axes[0, 1].set_title('Intraday Ratio Pattern')
        axes[0, 1].set_xlabel('Minute of Day')
        axes[0, 1].set_ylabel('Intraday Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sample daily patterns
        unique_dates = intraday_data['date'].unique()
        sample_dates = np.random.choice(unique_dates, min(sample_days, len(unique_dates)), replace=False)
        
        for i, date in enumerate(sample_dates):
            day_data = intraday_data[intraday_data['date'] == date]
            color = plt.cm.Set1(i)
            axes[1, 0].plot(day_data['minute_of_day'], day_data['logged_events'], 
                           color=color, linewidth=1, alpha=0.8, label=f'{date}')
        
        axes[1, 0].set_title(f'Sample Daily Patterns ({sample_days} days)')
        axes[1, 0].set_xlabel('Minute of Day')
        axes[1, 0].set_ylabel('Events per 2-min Interval')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Weekday vs weekend intraday patterns
        weekday_pattern = intraday_data[intraday_data['DateTime'].dt.dayofweek < 5].groupby('hour')['logged_events'].mean()
        weekend_pattern = intraday_data[intraday_data['DateTime'].dt.dayofweek >= 5].groupby('hour')['logged_events'].mean()
        
        axes[1, 1].plot(weekday_pattern.index, weekday_pattern.values, 'b-', linewidth=2, label='Weekdays')
        axes[1, 1].plot(weekend_pattern.index, weekend_pattern.values, 'r-', linewidth=2, label='Weekends')
        axes[1, 1].set_title('Weekday vs Weekend Intraday Patterns')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Events per 2-min Interval')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Intraday variability
        hourly_std = intraday_data.groupby('hour')['logged_events'].std()
        hourly_cv = hourly_std / hourly_pattern  # Coefficient of variation
        
        axes[2, 0].bar(hourly_cv.index, hourly_cv.values, alpha=0.7, color='green')
        axes[2, 0].set_title('Intraday Variability by Hour (Coefficient of Variation)')
        axes[2, 0].set_xlabel('Hour of Day')
        axes[2, 0].set_ylabel('CV (Std/Mean)')
        axes[2, 0].set_xticks(range(0, 24, 2))
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Peak detection
        peak_threshold = hourly_pattern.mean() + 2 * hourly_pattern.std()
        peak_hours = hourly_pattern[hourly_pattern > peak_threshold].index
        
        axes[2, 1].plot(hourly_pattern.index, hourly_pattern.values, 'b-', linewidth=2)
        axes[2, 1].axhline(y=peak_threshold, color='red', linestyle='--', alpha=0.7, label='Peak Threshold')
        
        for hour in peak_hours:
            axes[2, 1].axvline(x=hour, color='red', alpha=0.3)
            axes[2, 1].text(hour, hourly_pattern[hour], f'{hour}:00', 
                           rotation=90, ha='center', va='bottom')
        
        axes[2, 1].set_title('Peak Hour Detection')
        axes[2, 1].set_xlabel('Hour of Day')
        axes[2, 1].set_ylabel('Average Events per 2-min Interval')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_tiered_forecasts(self, recent_data, forecasts, forecast_dates):
        """Plot combined tiered forecasting results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Daily forecast with components
        daily_dates = pd.date_range(start=forecast_dates['daily_start'], periods=len(forecasts['daily_forecast']))
        
        axes[0, 0].plot(daily_dates, forecasts['daily_forecast'], 'b-o', linewidth=2, label='Total Forecast')
        axes[0, 0].plot(daily_dates, forecasts['daily_trend'], 'g--', linewidth=2, label='Trend Component')
        axes[0, 0].plot(daily_dates, forecasts['daily_seasonal'], 'r:', linewidth=2, label='Seasonal Component')
        
        axes[0, 0].set_title('Daily Forecast Decomposition')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Daily Events')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Intraday forecast with uncertainty
        intraday_times = pd.date_range(start=forecast_dates['intraday_start'], 
                                      periods=len(forecasts['intraday_forecast']), freq='2T')
        
        axes[0, 1].plot(intraday_times, forecasts['intraday_forecast'], 'b-', linewidth=2, label='Intraday Forecast')
        
        # Add uncertainty bands
        upper_bound = forecasts['intraday_forecast'] + forecasts['intraday_uncertainty']
        lower_bound = forecasts['intraday_forecast'] - forecasts['intraday_uncertainty']
        
        axes[0, 1].fill_between(intraday_times, lower_bound, upper_bound, 
                               alpha=0.3, color='blue', label='Uncertainty Band')
        
        axes[0, 1].set_title('Intraday Forecast with Uncertainty')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Events per 2-min Interval')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scaling decision matrix
        # Create a heatmap showing recommended scaling levels
        scaling_matrix = self._create_scaling_matrix(forecasts)
        
        im = axes[1, 0].imshow(scaling_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[1, 0].set_title('Recommended Scaling Levels')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Scaling Tier')
        axes[1, 0].set_yticks(range(len(['Low', 'Medium', 'High', 'Peak'])))
        axes[1, 0].set_yticklabels(['Low', 'Medium', 'High', 'Peak'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Scaling Intensity')
        
        # 4. Combined forecast timeline
        # Show both daily and intraday on same timeline
        recent_daily = recent_data['daily'].tail(7)
        recent_intraday = recent_data['intraday'].tail(24)
        
        # Plot recent actual data
        axes[1, 1].plot(recent_daily['DateTime'], recent_daily['logged_events_sum'], 
                       'k-', linewidth=2, label='Recent Daily Actual')
        
        # Plot forecasts
        axes[1, 1].plot(daily_dates, forecasts['daily_forecast'], 
                       'b--', linewidth=2, label='Daily Forecast')
        
        axes[1, 1].set_title('Combined Forecast Timeline')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Events')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_scaling_matrix(self, forecasts):
        """Create scaling decision matrix based on forecasts"""
        daily_forecast = forecasts['daily_forecast']
        intraday_forecast = forecasts['intraday_forecast']
        
        # Normalize forecasts to scaling levels (0-1)
        daily_normalized = (daily_forecast - daily_forecast.min()) / (daily_forecast.max() - daily_forecast.min())
        intraday_normalized = (intraday_forecast - intraday_forecast.min()) / (intraday_forecast.max() - intraday_forecast.min())
        
        # Create matrix
        scaling_levels = 4  # Low, Medium, High, Peak
        time_periods = max(len(daily_normalized), len(intraday_normalized))
        
        matrix = np.zeros((scaling_levels, time_periods))
        
        # Fill matrix based on forecast levels
        for i in range(time_periods):
            daily_level = daily_normalized[min(i, len(daily_normalized)-1)]
            intraday_level = intraday_normalized[min(i, len(intraday_normalized)-1)] if i < len(intraday_normalized) else 0.5
            
            combined_level = (daily_level + intraday_level) / 2
            
            if combined_level < 0.25:
                matrix[0, i] = 1  # Low
            elif combined_level < 0.5:
                matrix[1, i] = 1  # Medium
            elif combined_level < 0.75:
                matrix[2, i] = 1  # High
            else:
                matrix[3, i] = 1  # Peak
        
        return matrix
    
    def create_comprehensive_tiered_report(self, df):
        """Create comprehensive report for tiered forecasting system"""
        
        print("=== CREATING TIERED FORECASTING ANALYSIS ===")
        
        # Prepare data
        daily_data = self.aggregator.aggregate_to_daily(df)
        intraday_data = self.pattern_extractor.extract_patterns(df)
        
        print(f"Daily data points: {len(daily_data)}")
        print(f"Intraday data points: {len(intraday_data)}")
        
        figures = []
        
        # 1. Daily analysis
        print("Creating daily analysis...")
        fig1 = self.plot_daily_analysis(daily_data)
        figures.append(('tiered_daily_analysis', fig1))
        
        # 2. Intraday analysis
        print("Creating intraday analysis...")
        fig2 = self.plot_intraday_analysis(intraday_data, sample_days=5)
        figures.append(('tiered_intraday_analysis', fig2))
        
        # Save figures
        print("\\n=== SAVING TIERED ANALYSIS FIGURES ===")
        for name, fig in figures:
            filename = f'{name}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        # Show plots
        plt.show()
        
        print(f"\\n🎯 TIERED ANALYSIS COMPLETE!")
        print(f"✅ {len(figures)} analysis categories created")
        print("✅ Daily-level: Trends, seasonality, growth patterns")
        print("✅ Intraday-level: Hourly patterns, peak detection, variability")
        
        return daily_data, intraday_data, figures

def main():
    print("=== TIERED FORECASTING VISUALIZATION ===")
    print()
    
    # Load data
    df = pd.read_csv('EventsMetricsMarJul.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.rename(columns={
        'avg_average_processing_duration_ms': 'processing_duration',
        'avg_unprocessed_events_count': 'queue_size',
        'avg_processed_events_in_interval': 'processed_events',
        'avg_logged_events_in_interval': 'logged_events',
        'avg_queued_events_in_interval': 'queued_events'
    })
    
    # Create visualization
    visualizer = TieredVisualization()
    daily_data, intraday_data, figures = visualizer.create_comprehensive_tiered_report(df)
    
    # Print insights
    print("\\n=== KEY INSIGHTS ===")
    
    # Daily insights
    total_events = daily_data['logged_events_sum'].sum()
    avg_daily = daily_data['logged_events_sum'].mean()
    growth_rate = daily_data['daily_growth_rate'].mean() * 100
    
    print(f"📊 Daily Level:")
    print(f"  Total events processed: {total_events:,.0f}")
    print(f"  Average daily volume: {avg_daily:,.0f} events")
    print(f"  Average daily growth: {growth_rate:.2f}%")
    
    # Intraday insights
    peak_hour = intraday_data.groupby('hour')['logged_events'].mean().idxmax()
    peak_volume = intraday_data.groupby('hour')['logged_events'].mean().max()
    quiet_hour = intraday_data.groupby('hour')['logged_events'].mean().idxmin()
    quiet_volume = intraday_data.groupby('hour')['logged_events'].mean().min()
    
    print(f"📊 Intraday Level:")
    print(f"  Peak hour: {peak_hour}:00 ({peak_volume:.0f} events/2min)")
    print(f"  Quiet hour: {quiet_hour}:00 ({quiet_volume:.0f} events/2min)")
    print(f"  Peak/Quiet ratio: {peak_volume/quiet_volume:.1f}x")

if __name__ == "__main__":
    main()
