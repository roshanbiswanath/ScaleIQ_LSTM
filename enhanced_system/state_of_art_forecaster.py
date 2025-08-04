"""
State-of-the-Art Event Volume Forecasting Model
==============================================

This module implements a hybrid Transformer-LSTM model for multi-horizon event volume forecasting
with uncertainty quantification and anomaly detection capabilities.

Key Features:
- Multi-scale attention mechanisms
- Uncertainty quantification
- Multiple forecasting horizons
- Anomaly detection integration
- Advanced feature engineering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import math
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for the forecasting model."""
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    lstm_hidden: int = 256
    lstm_layers: int = 3
    dropout: float = 0.1
    
    # Data configuration
    sequence_length: int = 96  # 3.2 hours of 2-min intervals
    forecast_horizons: List[int] = None  # Multiple horizons: [6, 12, 24, 48, 96]
    feature_dim: int = 32
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_epochs: int = 100
    batch_size: int = 64
    
    def __post_init__(self):
        if self.forecast_horizons is None:
            self.forecast_horizons = [6, 12, 24, 48, 96]  # 12min to 3.2hr ahead


class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable components."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable position embedding
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for capturing patterns at different time scales."""
    
    def __init__(self, d_model: int, n_heads: int, scales: List[int] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales or [1, 2, 4, 8]  # Different temporal scales
        
        self.attention_layers = nn.ModuleDict()
        for scale in self.scales:
            self.attention_layers[f'scale_{scale}'] = nn.MultiheadAttention(
                d_model, n_heads, dropout=0.1, batch_first=True
            )
        
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))
        self.output_proj = nn.Linear(d_model * len(self.scales), d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # Downsample for different scales
            if scale > 1:
                # Average pooling for downsampling
                downsampled_len = seq_len // scale
                if downsampled_len > 0:
                    x_scaled = F.avg_pool1d(
                        x.transpose(1, 2), 
                        kernel_size=scale, 
                        stride=scale
                    ).transpose(1, 2)
                else:
                    x_scaled = x  # Fallback to original
            else:
                x_scaled = x
            
            # Apply attention
            attn_out, _ = self.attention_layers[f'scale_{scale}'](
                x_scaled, x_scaled, x_scaled, attn_mask=mask
            )
            
            # Upsample back if needed
            if scale > 1 and downsampled_len > 0:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(attn_out * self.scale_weights[i])
        
        # Combine multi-scale outputs
        combined = torch.cat(scale_outputs, dim=-1)
        return self.output_proj(combined)


class TransformerBlock(nn.Module):
    """Enhanced transformer block with multi-scale attention."""
    
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.multi_scale_attn = MultiScaleAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-scale attention with residual connection
        attn_out = self.multi_scale_attn(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class UncertaintyHead(nn.Module):
    """Uncertainty quantification head using evidential learning."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim * 4)  # 4 evidential parameters
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns evidential parameters for uncertainty quantification."""
        params = self.dense(x)
        batch_size, seq_len = x.shape[:2]
        output_dim = params.shape[-1] // 4
        
        params = params.view(batch_size, seq_len, output_dim, 4)
        
        # Evidential parameters (alpha, beta, lambda, nu)
        alpha = F.softplus(params[..., 0]) + 1  # > 1
        beta = F.softplus(params[..., 1])       # > 0  
        lambda_param = F.softplus(params[..., 2]) + 1  # > 1
        nu = F.softplus(params[..., 3]) + 1     # > 1
        
        # Mean prediction
        mean = alpha / beta
        
        # Uncertainty (epistemic + aleatoric)
        epistemic = alpha / (beta * (lambda_param - 1))
        aleatoric = 1 / nu
        
        return {
            'mean': mean,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': epistemic + aleatoric
        }


class StateOfTheArtForecaster(nn.Module):
    """
    State-of-the-art event volume forecasting model combining:
    - Multi-scale Transformer attention
    - LSTM for sequential modeling
    - Multi-horizon forecasting
    - Uncertainty quantification
    - Anomaly detection
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, 
                config.n_heads, 
                config.d_model * 4, 
                config.dropout
            ) for _ in range(config.n_layers)
        ])
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            config.d_model, 
            config.lstm_hidden, 
            config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fusion layer
        self.fusion = nn.Linear(config.d_model + config.lstm_hidden * 2, config.d_model)
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict()
        self.uncertainty_heads = nn.ModuleDict()
        
        for horizon in config.forecast_horizons:
            self.prediction_heads[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, horizon)
            )
            
            self.uncertainty_heads[f'horizon_{horizon}'] = UncertaintyHead(
                config.d_model, horizon
            )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-horizon predictions and uncertainty.
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions, uncertainties, and anomaly scores
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Transformer processing
        transformer_out = x
        attention_weights = [] if return_attention else None
        
        for layer in self.transformer_layers:
            transformer_out = layer(transformer_out)
        
        # LSTM processing
        lstm_out, _ = self.lstm(transformer_out)
        
        # Fusion
        fused = torch.cat([transformer_out, lstm_out], dim=-1)
        fused = self.fusion(fused)
        
        # Use the last time step for predictions
        final_repr = fused[:, -1, :]  # [batch_size, d_model]
        
        # Multi-horizon predictions
        predictions = {}
        uncertainties = {}
        
        for horizon in self.config.forecast_horizons:
            # Point predictions
            pred = self.prediction_heads[f'horizon_{horizon}'](final_repr)
            predictions[f'horizon_{horizon}'] = pred
            
            # Uncertainty quantification
            unc = self.uncertainty_heads[f'horizon_{horizon}'](final_repr.unsqueeze(1))
            uncertainties[f'horizon_{horizon}'] = unc
        
        # Anomaly detection
        anomaly_score = self.anomaly_head(final_repr)
        
        results = {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'anomaly_score': anomaly_score
        }
        
        if return_attention:
            results['attention_weights'] = attention_weights
            
        return results


class AdvancedFeatureEngineer:
    """Advanced feature engineering for event processing data."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for forecasting."""
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        # Rename columns for easier handling
        df = df.rename(columns={
            'avg_average_processing_duration_ms': 'proc_duration',
            'avg_unprocessed_events_count': 'queue_size',
            'avg_processed_events_in_interval': 'processed_events',
            'avg_logged_events_in_interval': 'logged_events',
            'avg_queued_events_in_interval': 'queued_events'
        })
        
        # Core temporal features
        df = self._add_temporal_features(df)
        
        # Event processing features
        df = self._add_processing_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # System state features
        df = self._add_system_features(df)
        
        # Advanced pattern features
        df = self._add_pattern_features(df)
        
        # Clean up
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive temporal features."""
        # Basic time components
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['day_of_month'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        df['quarter'] = df['DateTime'].dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def _add_processing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event processing related features."""
        # Processing efficiency
        df['processing_rate'] = df['processed_events'] / (df['proc_duration'] + 1e-6)
        df['queue_ratio'] = df['queue_size'] / (df['logged_events'] + 1e-6)
        df['utilization'] = df['processed_events'] / (df['queued_events'] + 1e-6)
        
        # Event flow
        df['event_flow_balance'] = df['logged_events'] - df['processed_events']
        df['queue_growth_rate'] = df['queue_size'].pct_change().fillna(0)
        
        # Processing load
        df['total_load'] = df['logged_events'] + df['queue_size']
        df['processing_pressure'] = df['queue_size'] / (df['processing_rate'] + 1e-6)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features across different time windows."""
        windows = [6, 12, 24, 48, 96]  # Different time windows
        
        for col in ['logged_events', 'processed_events', 'queue_size', 'proc_duration']:
            for window in windows:
                # Rolling statistics
                df[f'{col}_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
                # Percentiles
                df[f'{col}_q25_{window}'] = df[col].rolling(window=window, min_periods=1).quantile(0.25)
                df[f'{col}_q75_{window}'] = df[col].rolling(window=window, min_periods=1).quantile(0.75)
                
                # Exponential moving average
                df[f'{col}_ema_{window}'] = df[col].ewm(span=window).mean()
        
        return df
    
    def _add_system_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add system state features."""
        # Lag features
        lags = [1, 3, 6, 12, 24, 48]
        for col in ['logged_events', 'queue_size', 'processing_rate']:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rate of change
        for col in ['logged_events', 'queue_size', 'processed_events']:
            df[f'{col}_roc_1'] = df[col].pct_change(1).fillna(0)
            df[f'{col}_roc_6'] = df[col].pct_change(6).fillna(0)
            df[f'{col}_roc_24'] = df[col].pct_change(24).fillna(0)
        
        # Momentum indicators
        df['event_momentum'] = df['logged_events'].rolling(12).apply(
            lambda x: (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0
        ).fillna(0)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced pattern recognition features."""
        # Volatility features
        df['event_volatility'] = df['logged_events'].rolling(24).std().fillna(0)
        df['queue_volatility'] = df['queue_size'].rolling(24).std().fillna(0)
        
        # Anomaly indicators
        df['is_high_load'] = (df['logged_events'] > 
                             df['logged_events'].rolling(96).quantile(0.95)).astype(int)
        df['is_queue_buildup'] = (df['queue_size'] > 
                                 df['queue_size'].rolling(48).quantile(0.9)).astype(int)
        
        # Pattern recognition
        df['daily_hour_avg'] = df.groupby('hour')['logged_events'].transform(
            lambda x: x.expanding().mean()
        )
        df['hourly_deviation'] = df['logged_events'] - df['daily_hour_avg']
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for model input."""
        exclude_cols = ['DateTime', 'logged_events']  # target is logged_events
        return [col for col in df.columns if col not in exclude_cols]


if __name__ == "__main__":
    # Example usage
    config = ModelConfig()
    model = StateOfTheArtForecaster(config)
    
    print("State-of-the-Art Event Forecasting Model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Forecast horizons: {config.forecast_horizons}")
    print("✅ Model ready for training!")
