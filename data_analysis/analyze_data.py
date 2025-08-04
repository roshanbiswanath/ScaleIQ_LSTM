import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv('EventsMetricsMarJul.csv')

print('=== DATA OVERVIEW ===')
print(f'Total records: {len(df):,}')
print(f'Date range: {df.DateTime.min()} to {df.DateTime.max()}')
print(f'Columns: {list(df.columns)}')
print()

print('=== DATA STATISTICS ===')
print(df.describe())
print()

print('=== MISSING VALUES ===')
print(df.isnull().sum())
print()

print('=== TIME INTERVAL ANALYSIS ===')
df['DateTime'] = pd.to_datetime(df['DateTime'])
time_diffs = df['DateTime'].diff().dropna()
print(f'Average interval: {time_diffs.mean()}')
print(f'Most common interval: {time_diffs.mode().iloc[0]}')
print()

print('=== DATA QUALITY ===')
total_days = (df['DateTime'].max() - df['DateTime'].min()).days
expected_intervals = total_days * 24 * 30  # 30 intervals per hour
actual_intervals = len(df)
completeness = actual_intervals / expected_intervals
print(f'Expected intervals (2-min): {expected_intervals:,}')
print(f'Actual intervals: {actual_intervals:,}')
print(f'Data completeness: {completeness:.2%}')
print()

print('=== PATTERN ANALYSIS ===')
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['date'] = df['DateTime'].dt.date

print('Hourly coverage:')
hourly_counts = df['hour'].value_counts().sort_index()
print(f'Hours covered: {len(hourly_counts)}/24')
print(f'Min records per hour: {hourly_counts.min():,}')
print(f'Max records per hour: {hourly_counts.max():,}')
print()

print('Daily coverage:')
daily_counts = df['day_of_week'].value_counts().sort_index()
print(f'Days of week covered: {len(daily_counts)}/7')
print(f'Weekend vs weekday balance: {daily_counts.std():.1f} std dev')
print()

print('=== EVENT PROCESSING METRICS ===')
print(f"Average processing duration: {df['avg_average_processing_duration_ms'].mean():.1f}ms")
print(f"Average queue size: {df['avg_unprocessed_events_count'].mean():.1f}")
print(f"Average throughput: {df['avg_processed_events_in_interval'].mean():.1f} events/interval")
print(f"Processing efficiency: {(df['avg_processed_events_in_interval'] / df['avg_queued_events_in_interval']).mean():.2%}")
print()

print('=== LOAD PATTERNS ===')
queue_percentiles = np.percentile(df['avg_unprocessed_events_count'], [25, 50, 75, 90, 95, 99])
print('Queue size percentiles:')
for i, p in enumerate([25, 50, 75, 90, 95, 99]):
    print(f'  {p}th percentile: {queue_percentiles[i]:.0f} events')
