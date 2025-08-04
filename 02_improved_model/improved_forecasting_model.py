import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import math
from typing import Tuple, Optional

# Configuration
SEQUENCE_LENGTH = 24  # Increased to 48 minutes of history
FORECAST_HORIZON = 6   # 6 intervals = 12 minutes ahead
BATCH_SIZE = 64       # Increased batch size
EPOCHS = 50           # More epochs
LEARNING_RATE = 0.001 # Lower learning rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class EventDataset(Dataset):
    def __init__(self, data, sequence_length, forecast_horizon, target_col='logged_events'):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        
        # Define feature columns (excluding DateTime and target)
        self.feature_cols = [col for col in data.columns 
                           if col not in ['DateTime', target_col]]
        
        # Prepare sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _create_sequences(self):
        # Pre-convert to numpy for efficiency
        feature_data = self.data[self.feature_cols].values
        target_data = self.data[self.target_col].values
        
        n_sequences = len(self.data) - self.sequence_length - self.forecast_horizon + 1
        
        # Pre-allocate arrays
        sequences = np.zeros((n_sequences, self.sequence_length, len(self.feature_cols)))
        targets = np.zeros((n_sequences, self.forecast_horizon))
        
        for i in range(n_sequences):
            # Input sequence (features)
            sequences[i] = feature_data[i:i + self.sequence_length]
            
            # Target sequence (logged_events for next forecast_horizon steps)
            targets[i] = target_data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
        
        return sequences, targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal patterns"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ImprovedEventForecaster(nn.Module):
    """Improved forecasting model with attention and residual connections"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, forecast_horizon=6, 
                 dropout=0.1, use_attention=True):
        super(ImprovedEventForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.use_attention = use_attention
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for temporal patterns
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Bidirectional LSTM for better context capture
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # *2 for bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Feature fusion
        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.feature_fusion = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-step forecast head with residual connections
        self.forecast_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(forecast_horizon)
        ])
        
        # Uncertainty estimation with separate head for each step
        self.uncertainty_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            ) for _ in range(forecast_horizon)
        ])
        
        # Global forecast head for overall trend
        self.global_forecast = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, hidden)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, hidden)
        
        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attended_out  # Residual connection
        
        # Feature fusion
        fused_features = self.feature_fusion(lstm_out)
        
        # Use last output for forecasting
        last_output = fused_features[:, -1, :]
        
        # Generate step-wise forecasts
        step_forecasts = []
        step_uncertainties = []
        
        for i in range(self.forecast_horizon):
            forecast = self.forecast_layers[i](last_output)
            uncertainty = self.uncertainty_layers[i](last_output)
            step_forecasts.append(forecast)
            step_uncertainties.append(uncertainty)
        
        forecasts = torch.cat(step_forecasts, dim=1)
        uncertainties = torch.cat(step_uncertainties, dim=1)
        
        # Add global trend
        global_trend = self.global_forecast(last_output)
        forecasts = forecasts + 0.1 * global_trend  # Small contribution from global trend
        
        return forecasts, uncertainties

class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced predictions"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        # Compute relative error for weighting
        relative_error = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
        # Apply focal weighting
        weights = self.alpha * (relative_error ** self.gamma)
        return torch.mean(weights * mse_loss)

def train_model(model, train_loader, val_loader, epochs, device):
    # Use focal loss for better handling of outliers
    criterion = FocalLoss(alpha=1.0, gamma=1.5)
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # Warm restart scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            forecasts, uncertainties = model(data)
            
            # Combined loss: focal loss + uncertainty regularization
            forecast_loss = criterion(forecasts, target)
            uncertainty_loss = torch.mean(uncertainties)  # Encourage reasonable uncertainty
            
            loss = forecast_loss + 0.01 * uncertainty_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                forecasts, uncertainties = model(data)
                
                val_loss = criterion(forecasts, target)
                total_val_loss += val_loss.item()
                val_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_improved_forecasting_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, scaler=None, target_col='logged_events'):
    """Evaluate model and return predictions in original scale"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    # Map target column to scaler index
    basic_cols = ['processing_duration', 'queue_size', 'processed_events', 'logged_events', 'queued_events']
    target_idx = basic_cols.index(target_col) if target_col in basic_cols else 3
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            forecasts, uncertainties = model(data)
            
            all_predictions.append(forecasts.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    uncertainties = np.concatenate(all_uncertainties, axis=0)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Create dummy data with same shape as original to use scaler
        dummy_data = np.zeros((predictions.shape[0], scaler.n_features_in_))
        
        # Find the column index for logged_events in the original data
        feature_names = ['processing_duration', 'queue_size', 'processed_events', 'logged_events', 'queued_events']
        if target_col in feature_names:
            target_col_idx = feature_names.index(target_col)
            
            # Inverse transform predictions
            for i in range(predictions.shape[1]):  # For each forecast step
                dummy_data[:, target_col_idx] = predictions[:, i]
                predictions[:, i] = scaler.inverse_transform(dummy_data)[:, target_col_idx]
            
            # Inverse transform targets
            for i in range(targets.shape[1]):  # For each forecast step
                dummy_data[:, target_col_idx] = targets[:, i]
                targets[:, i] = scaler.inverse_transform(dummy_data)[:, target_col_idx]
    
    # Calculate metrics on original scale
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # Calculate per-step metrics
    step_maes = []
    step_rmses = []
    for i in range(predictions.shape[1]):
        step_mae = mean_absolute_error(targets[:, i], predictions[:, i])
        step_rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
        step_maes.append(step_mae)
        step_rmses.append(step_rmse)
    
    print(f"\n=== IMPROVED MODEL EVALUATION (Original Scale) ===")
    print(f"Overall Metrics:")
    print(f"  MAE: {mae:.2f} events")
    print(f"  RMSE: {rmse:.2f} events")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Average actual events: {np.mean(targets):.2f}")
    print(f"  Average predicted events: {np.mean(predictions):.2f}")
    
    print(f"\nPer-Step Metrics (2-min intervals ahead):")
    for i, (step_mae, step_rmse) in enumerate(zip(step_maes, step_rmses)):
        print(f"  Step {i+1} ({(i+1)*2} min): MAE={step_mae:.1f}, RMSE={step_rmse:.1f}")
    
    return predictions, targets, uncertainties

def main():
    print("=== IMPROVED EVENT QUEUE FORECASTING MODEL ===")
    print()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv('processed_data/train.csv')
    val_df = pd.read_csv('processed_data/val.csv')
    test_df = pd.read_csv('processed_data/test.csv')
    
    # Load original data to get scaler info
    print("Loading original data for scaling reference...")
    original_df = pd.read_csv('EventsMetricsMarJul.csv')
    original_df['DateTime'] = pd.to_datetime(original_df['DateTime'])
    original_df = original_df.rename(columns={
        'avg_average_processing_duration_ms': 'processing_duration',
        'avg_unprocessed_events_count': 'queue_size',
        'avg_processed_events_in_interval': 'processed_events',
        'avg_logged_events_in_interval': 'logged_events',
        'avg_queued_events_in_interval': 'queued_events'
    })
    
    # Create scaler with basic columns to inverse transform
    basic_cols = ['processing_duration', 'queue_size', 'processed_events', 'logged_events', 'queued_events']
    scaler = MinMaxScaler()
    scaler.fit(original_df[basic_cols])
    
    print(f"Train set: {len(train_df):,} records")
    print(f"Validation set: {len(val_df):,} records")
    print(f"Test set: {len(test_df):,} records")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EventDataset(train_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
    val_dataset = EventDataset(val_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
    test_dataset = EventDataset(test_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize improved model
    input_dim = len(train_dataset.feature_cols)
    print(f"\nInput features: {input_dim}")
    
    model = ImprovedEventForecaster(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        forecast_horizon=FORECAST_HORIZON,
        dropout=0.1,
        use_attention=True
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n=== TRAINING IMPROVED MODEL ===")
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, DEVICE)
    
    # Load best model
    model.load_state_dict(torch.load('best_improved_forecasting_model.pth'))
    
    # Evaluate model with original scale
    print("\n=== EVALUATING IMPROVED MODEL ===")
    predictions, targets, uncertainties = evaluate_model(model, test_loader, DEVICE, scaler, 'logged_events')
    
    # Save results
    print("\nSaving improved results...")
    np.save('improved_predictions.npy', predictions)
    np.save('improved_targets.npy', targets)
    np.save('improved_uncertainties.npy', uncertainties)
    
    # Save scaler for visualization script
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save training history
    np.save('improved_train_losses.npy', train_losses)
    np.save('improved_val_losses.npy', val_losses)
    
    print("\n🎯 IMPROVED MODEL TRAINING COMPLETE!")
    print("✅ Best model saved as 'best_improved_forecasting_model.pth'")
    print("✅ Results saved as numpy arrays (original scale)")
    print("✅ Scaler saved for visualization script")

if __name__ == "__main__":
    main()
