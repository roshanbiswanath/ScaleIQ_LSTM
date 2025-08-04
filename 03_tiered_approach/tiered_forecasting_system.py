import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pickle
import math
from typing import Tuple, Optional
from datetime import datetime, timedelta
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DailyAggregator:
    """Aggregates intraday data to daily level for trend analysis"""
    
    def __init__(self):
        pass
    
    def aggregate_to_daily(self, df):
        """Aggregate 2-minute data to daily statistics"""
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['date'] = df['DateTime'].dt.date
        
        # Daily aggregations
        daily_stats = df.groupby('date').agg({
            'logged_events': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'processed_events': ['sum', 'mean', 'std'],
            'queue_size': ['mean', 'max', 'std'],
            'processing_duration': ['mean', 'max', 'std'],
            'queued_events': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        
        # Add temporal features
        daily_stats = daily_stats.reset_index()
        daily_stats['DateTime'] = pd.to_datetime(daily_stats['date'])
        daily_stats['day_of_week'] = daily_stats['DateTime'].dt.dayofweek
        daily_stats['day_of_month'] = daily_stats['DateTime'].dt.day
        daily_stats['week_of_year'] = daily_stats['DateTime'].dt.isocalendar().week
        daily_stats['month'] = daily_stats['DateTime'].dt.month
        daily_stats['is_weekend'] = daily_stats['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate daily growth rates
        daily_stats['logged_events_sum_lag1'] = daily_stats['logged_events_sum'].shift(1)
        daily_stats['logged_events_sum_lag7'] = daily_stats['logged_events_sum'].shift(7)
        daily_stats['daily_growth_rate'] = (daily_stats['logged_events_sum'] - daily_stats['logged_events_sum_lag1']) / (daily_stats['logged_events_sum_lag1'] + 1)
        daily_stats['weekly_growth_rate'] = (daily_stats['logged_events_sum'] - daily_stats['logged_events_sum_lag7']) / (daily_stats['logged_events_sum_lag7'] + 1)
        
        # Rolling averages
        for window in [3, 7, 14]:
            daily_stats[f'logged_events_sum_ma_{window}'] = daily_stats['logged_events_sum'].rolling(window).mean()
            daily_stats[f'logged_events_mean_ma_{window}'] = daily_stats['logged_events_mean'].rolling(window).mean()
        
        return daily_stats.dropna()

class IntradayPatternExtractor:
    """Extracts intraday patterns and residuals after removing daily trends"""
    
    def __init__(self):
        pass
    
    def extract_patterns(self, df, daily_forecasts=None):
        """Extract intraday patterns relative to daily trends"""
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['date'] = df['DateTime'].dt.date
        df['time_of_day'] = df['DateTime'].dt.time
        df['hour'] = df['DateTime'].dt.hour
        df['minute'] = df['DateTime'].dt.minute
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        
        # Calculate daily totals for normalization
        daily_totals = df.groupby('date')['logged_events'].sum().reset_index()
        daily_totals.columns = ['date', 'daily_total']
        
        # Merge back
        df = df.merge(daily_totals, on='date', how='left')
        
        # If daily forecasts provided, use them; otherwise use actual daily totals
        if daily_forecasts is not None:
            daily_forecasts['date'] = pd.to_datetime(daily_forecasts['date']).dt.date
            df = df.merge(daily_forecasts[['date', 'predicted_daily_total']], on='date', how='left')
            df['daily_baseline'] = df['predicted_daily_total'].fillna(df['daily_total'])
        else:
            df['daily_baseline'] = df['daily_total']
        
        # Calculate intraday patterns
        df['intraday_ratio'] = df['logged_events'] / (df['daily_baseline'] / (24 * 30))  # 30 intervals per hour
        df['intraday_residual'] = df['logged_events'] - (df['daily_baseline'] / (24 * 30))
        
        # Add intraday temporal features
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_minute_of_day'] = np.sin(2 * np.pi * df['minute_of_day'] / (24 * 60))
        df['cos_minute_of_day'] = np.cos(2 * np.pi * df['minute_of_day'] / (24 * 60))
        
        # Historical intraday patterns
        hourly_patterns = df.groupby(['hour', 'minute'])['intraday_ratio'].mean().reset_index()
        hourly_patterns.columns = ['hour', 'minute', 'avg_intraday_ratio']
        df = df.merge(hourly_patterns, on=['hour', 'minute'], how='left')
        
        return df

class DailyForecaster(nn.Module):
    """Model for daily-level forecasting (trend, seasonality, growth)"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, forecast_horizon=7):
        super(DailyForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM for capturing daily trends
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Trend component
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        # Seasonal component
        self.seasonal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        # Growth component
        self.growth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Tanh()  # Growth rate between -1 and 1
        )
        
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Generate components
        trend = self.trend_head(last_output)
        seasonal = self.seasonal_head(last_output)
        growth = self.growth_head(last_output)
        
        # Combine components (additive decomposition)
        forecast = trend + seasonal + growth * trend
        
        return forecast, trend, seasonal, growth

class IntradayForecaster(nn.Module):
    """Model for intraday patterns and short-term fluctuations"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, forecast_horizon=12):
        super(IntradayForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Input projection with batch norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Bidirectional LSTM for intraday patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention for focusing on relevant time periods
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Pattern head (for ratio forecasting)
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Residual head (for absolute adjustments)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Softplus()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x_proj = self.input_proj(x.view(-1, x.size(-1)))
        x_proj = x_proj.view(batch_size, x.size(1), -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_proj)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        combined = lstm_out + attended_out  # Residual connection
        
        # Use last output
        last_output = combined[:, -1, :]
        
        # Generate forecasts
        pattern_forecast = self.pattern_head(last_output)
        residual_forecast = self.residual_head(last_output)
        uncertainty = self.uncertainty_head(last_output)
        
        return pattern_forecast, residual_forecast, uncertainty

class TieredDataset(Dataset):
    """Dataset for tiered forecasting approach"""
    
    def __init__(self, data, sequence_length, forecast_horizon, level='daily', target_col='logged_events_sum'):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.level = level
        self.target_col = target_col
        
        # Define feature columns based on level
        if level == 'daily':
            exclude_cols = ['DateTime', 'date', target_col]
        else:  # intraday
            exclude_cols = ['DateTime', 'date', 'time_of_day', target_col]
        
        self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _create_sequences(self):
        feature_data = self.data[self.feature_cols].values
        target_data = self.data[self.target_col].values
        
        n_sequences = len(self.data) - self.sequence_length - self.forecast_horizon + 1
        
        if n_sequences <= 0:
            return np.array([]), np.array([])
        
        sequences = np.zeros((n_sequences, self.sequence_length, len(self.feature_cols)))
        targets = np.zeros((n_sequences, self.forecast_horizon))
        
        for i in range(n_sequences):
            sequences[i] = feature_data[i:i + self.sequence_length]
            targets[i] = target_data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
        
        return sequences, targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )

class TieredForecaster:
    """Combined tiered forecasting system"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.daily_model = None
        self.intraday_model = None
        self.daily_scaler = None
        self.intraday_scaler = None
        self.aggregator = DailyAggregator()
        self.pattern_extractor = IntradayPatternExtractor()
    
    def prepare_data(self, df):
        """Prepare both daily and intraday datasets"""
        print("Preparing tiered datasets...")
        
        # Prepare daily data
        daily_data = self.aggregator.aggregate_to_daily(df)
        print(f"Daily data shape: {daily_data.shape}")
        
        # Prepare intraday data
        intraday_data = self.pattern_extractor.extract_patterns(df)
        print(f"Intraday data shape: {intraday_data.shape}")
        
        return daily_data, intraday_data
    
    def train_daily_model(self, daily_data, epochs=100):
        """Train the daily-level forecasting model"""
        print("\\n=== TRAINING DAILY MODEL ===")
        
        # Scale daily data
        feature_cols = [col for col in daily_data.columns 
                       if col not in ['DateTime', 'date', 'logged_events_sum']]
        
        self.daily_scaler = StandardScaler()
        daily_data_scaled = daily_data.copy()
        daily_data_scaled[feature_cols] = self.daily_scaler.fit_transform(daily_data[feature_cols])
        
        # Create dataset
        daily_dataset = TieredDataset(
            daily_data_scaled, 
            sequence_length=14,  # 2 weeks history
            forecast_horizon=7,   # 1 week ahead
            level='daily',
            target_col='logged_events_sum'
        )
        
        if len(daily_dataset) == 0:
            print("❌ Not enough daily data for training")
            return None
        
        # Split data
        train_size = int(0.8 * len(daily_dataset))
        val_size = len(daily_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            daily_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Initialize model
        input_dim = len(daily_dataset.feature_cols)
        self.daily_model = DailyForecaster(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            forecast_horizon=7
        ).to(self.device)
        
        # Train model
        optimizer = optim.AdamW(self.daily_model.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.daily_model.train()
            total_train_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                forecast, trend, seasonal, growth = self.daily_model(data)
                loss = criterion(forecast, target)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation
            self.daily_model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    forecast, _, _, _ = self.daily_model(data)
                    val_loss = criterion(forecast, target)
                    total_val_loss += val_loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save model
        torch.save(self.daily_model.state_dict(), 'daily_forecaster.pth')
        print("✅ Daily model training complete")
        
        return train_losses, val_losses
    
    def train_intraday_model(self, intraday_data, epochs=50):
        """Train the intraday pattern forecasting model"""
        print("\\n=== TRAINING INTRADAY MODEL ===")
        
        # Scale intraday data
        feature_cols = [col for col in intraday_data.columns 
                       if col not in ['DateTime', 'date', 'time_of_day', 'logged_events']]
        
        self.intraday_scaler = StandardScaler()
        intraday_data_scaled = intraday_data.copy()
        intraday_data_scaled[feature_cols] = self.intraday_scaler.fit_transform(intraday_data[feature_cols])
        
        # Create dataset
        intraday_dataset = TieredDataset(
            intraday_data_scaled,
            sequence_length=24,  # 48 minutes history
            forecast_horizon=12,  # 24 minutes ahead
            level='intraday',
            target_col='logged_events'
        )
        
        if len(intraday_dataset) == 0:
            print("❌ Not enough intraday data for training")
            return None
        
        # Split data (time-based split for intraday)
        train_size = int(0.8 * len(intraday_dataset))
        train_dataset = torch.utils.data.Subset(intraday_dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(intraday_dataset, range(train_size, len(intraday_dataset)))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        input_dim = len(intraday_dataset.feature_cols)
        self.intraday_model = IntradayForecaster(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            forecast_horizon=12
        ).to(self.device)
        
        # Train model
        optimizer = optim.AdamW(self.intraday_model.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.intraday_model.train()
            total_train_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                pattern_forecast, residual_forecast, uncertainty = self.intraday_model(data)
                
                # Combined forecast
                combined_forecast = pattern_forecast + residual_forecast
                loss = criterion(combined_forecast, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.intraday_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation
            self.intraday_model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    pattern_forecast, residual_forecast, uncertainty = self.intraday_model(data)
                    combined_forecast = pattern_forecast + residual_forecast
                    val_loss = criterion(combined_forecast, target)
                    total_val_loss += val_loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save model
        torch.save(self.intraday_model.state_dict(), 'intraday_forecaster.pth')
        print("✅ Intraday model training complete")
        
        return train_losses, val_losses
    
    def forecast(self, recent_daily_data, recent_intraday_data, daily_horizon=7, intraday_horizon=12):
        """Generate tiered forecasts"""
        if self.daily_model is None or self.intraday_model is None:
            raise ValueError("Models not trained yet")
        
        # Daily forecast
        daily_features = recent_daily_data[self.daily_model.feature_cols].values
        daily_features_scaled = self.daily_scaler.transform(daily_features)
        
        with torch.no_grad():
            daily_input = torch.FloatTensor(daily_features_scaled[-14:]).unsqueeze(0).to(self.device)
            daily_forecast, trend, seasonal, growth = self.daily_model(daily_input)
            daily_forecast = daily_forecast.cpu().numpy().flatten()
        
        # Intraday forecast
        intraday_features = recent_intraday_data[self.intraday_model.feature_cols].values
        intraday_features_scaled = self.intraday_scaler.transform(intraday_features)
        
        with torch.no_grad():
            intraday_input = torch.FloatTensor(intraday_features_scaled[-24:]).unsqueeze(0).to(self.device)
            pattern_forecast, residual_forecast, uncertainty = self.intraday_model(intraday_input)
            
            intraday_forecast = (pattern_forecast + residual_forecast).cpu().numpy().flatten()
            intraday_uncertainty = uncertainty.cpu().numpy().flatten()
        
        return {
            'daily_forecast': daily_forecast,
            'daily_trend': trend.cpu().numpy().flatten(),
            'daily_seasonal': seasonal.cpu().numpy().flatten(),
            'daily_growth': growth.cpu().numpy().flatten(),
            'intraday_forecast': intraday_forecast,
            'intraday_uncertainty': intraday_uncertainty
        }

def main():
    print("=== TIERED FORECASTING SYSTEM ===")
    print("Daily-level: Trend and capacity planning")
    print("Intraday-level: Fine-grained scaling decisions\\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('EventsMetricsMarJul.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.rename(columns={
        'avg_average_processing_duration_ms': 'processing_duration',
        'avg_unprocessed_events_count': 'queue_size',
        'avg_processed_events_in_interval': 'processed_events',
        'avg_logged_events_in_interval': 'logged_events',
        'avg_queued_events_in_interval': 'queued_events'
    })
    
    print(f"Original data: {len(df):,} records")
    
    # Initialize tiered forecaster
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forecaster = TieredForecaster(device)
    
    # Prepare data
    daily_data, intraday_data = forecaster.prepare_data(df)
    
    # Train models
    daily_losses = forecaster.train_daily_model(daily_data, epochs=100)
    intraday_losses = forecaster.train_intraday_model(intraday_data, epochs=50)
    
    # Save scalers
    with open('daily_scaler.pkl', 'wb') as f:
        pickle.dump(forecaster.daily_scaler, f)
    with open('intraday_scaler.pkl', 'wb') as f:
        pickle.dump(forecaster.intraday_scaler, f)
    
    print("\\n🎯 TIERED FORECASTING SYSTEM COMPLETE!")
    print("✅ Daily model: Captures trends, seasonality, growth")
    print("✅ Intraday model: Captures within-day patterns and fluctuations")
    print("✅ Models and scalers saved for production use")

if __name__ == "__main__":
    main()
