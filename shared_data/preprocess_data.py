import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Configuration
INPUT_FILE = 'EventsMetricsMarJul.csv'
OUTPUT_DIR = 'processed_data'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.csv')
VAL_FILE = os.path.join(OUTPUT_DIR, 'val.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.csv')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load data
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# --- Feature Engineering ---
print("Starting feature engineering...")

# Convert DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime').reset_index(drop=True)

# Rename columns for easier access
df = df.rename(columns={
    'avg_average_processing_duration_ms': 'processing_duration',
    'avg_unprocessed_events_count': 'queue_size',
    'avg_processed_events_in_interval': 'processed_events',
    'avg_logged_events_in_interval': 'logged_events',
    'avg_queued_events_in_interval': 'queued_events'
})

# Temporal Features
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['day_of_month'] = df['DateTime'].dt.day
df['week_of_year'] = df['DateTime'].dt.isocalendar().week
df['month'] = df['DateTime'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Lag Features (for key metrics)
print("Creating lag features...")
for col in ['queue_size', 'processing_duration', 'processed_events']:
    for lag in [1, 2, 3, 6, 12]:  # Lags for 2, 4, 6, 12, 24 minutes
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# Rolling Window Features
print("Creating rolling window features...")
for window in [5, 15, 30]: # 10-min, 30-min, 1-hour windows
    for col in ['queue_size', 'processing_duration']:
        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

# Drop rows with NaN values created by lags and rolling windows
df = df.dropna().reset_index(drop=True)

print(f"Data shape after feature engineering: {df.shape}")

# --- Data Splitting ---
print("Splitting data into train, validation, and test sets...")
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.9)

train_df = df[:train_end]
val_df = df[train_end:val_end]
test_df = df[val_end:]

print(f"Train set size: {len(train_df):,}")
print(f"Validation set size: {len(val_df):,}")
print(f"Test set size: {len(test_df):,}")

# --- Scaling ---
print("Scaling data...")
scaler = MinMaxScaler()
feature_cols = [col for col in df.columns if col not in ['DateTime']]

# Fit on training data only
scaler.fit(train_df[feature_cols])

# Transform all sets
train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# --- Save Processed Data ---
print(f"Saving processed data to '{OUTPUT_DIR}'...")
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("Preprocessing complete!")
