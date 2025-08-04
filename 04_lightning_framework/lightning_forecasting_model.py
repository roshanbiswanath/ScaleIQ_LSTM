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

# PyTorch Lightning imports
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    LIGHTNING_AVAILABLE = True
    print("✅ PyTorch Lightning available")
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("❌ PyTorch Lightning not available. Install with: pip install pytorch-lightning")

# Configuration
SEQUENCE_LENGTH = 24  # 48 minutes of history
FORECAST_HORIZON = 6   # 12 minutes ahead
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
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

class LightningEventForecaster(pl.LightningModule):
    """PyTorch Lightning version of the event forecasting model"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, forecast_horizon=6, 
                 dropout=0.1, use_attention=True, learning_rate=0.001):
        super(LightningEventForecaster, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        
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
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
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
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        forecasts, uncertainties = self(data)
        
        # Combined loss: MSE + uncertainty regularization
        forecast_loss = self.mse_loss(forecasts, target)
        uncertainty_loss = torch.mean(uncertainties)  # Encourage reasonable uncertainty
        
        loss = forecast_loss + 0.01 * uncertainty_loss
        
        # Calculate metrics
        mae = self.mae_loss(forecasts, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_forecast_loss', forecast_loss, on_step=False, on_epoch=True)
        self.log('train_uncertainty', uncertainty_loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        forecasts, uncertainties = self(data)
        
        # Calculate losses
        forecast_loss = self.mse_loss(forecasts, target)
        uncertainty_loss = torch.mean(uncertainties)
        loss = forecast_loss + 0.01 * uncertainty_loss
        
        # Calculate metrics
        mae = self.mae_loss(forecasts, target)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_forecast_loss', forecast_loss, on_step=False, on_epoch=True)
        self.log('val_uncertainty', uncertainty_loss, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        forecasts, uncertainties = self(data)
        
        # Calculate losses
        forecast_loss = self.mse_loss(forecasts, target)
        mae = self.mae_loss(forecasts, target)
        
        # Log metrics
        self.log('test_loss', forecast_loss, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        
        return {'forecasts': forecasts, 'targets': target, 'uncertainties': uncertainties}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }

def create_lightning_trainer():
    """Create PyTorch Lightning trainer with callbacks"""
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_checkpoints/',
        filename='event-forecaster-{epoch:02d}-{val_loss:.6f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = CSVLogger('lightning_logs/', name='event_forecaster')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer

def train_with_lightning():
    """Train model using PyTorch Lightning"""
    
    print("=== LIGHTNING-POWERED EVENT FORECASTING ===")
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
    
    # Create scaler
    basic_cols = ['processing_duration', 'queue_size', 'processed_events', 'logged_events', 'queued_events']
    scaler = MinMaxScaler()
    scaler.fit(original_df[basic_cols])
    
    print(f"Train set: {len(train_df):,} records")
    print(f"Validation set: {len(val_df):,} records")
    print(f"Test set: {len(test_df):,} records")
    
    # Create datasets
    print("\\nCreating datasets...")
    train_dataset = EventDataset(train_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
    val_dataset = EventDataset(val_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
    test_dataset = EventDataset(test_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, persistent_workers=True)
    
    # Initialize Lightning model
    input_dim = len(train_dataset.feature_cols)
    print(f"\\nInput features: {input_dim}")
    
    model = LightningEventForecaster(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        forecast_horizon=FORECAST_HORIZON,
        dropout=0.1,
        use_attention=True,
        learning_rate=LEARNING_RATE
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = create_lightning_trainer()
    
    # Train model
    print("\\n=== TRAINING WITH PYTORCH LIGHTNING ===")
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    print("\\n=== TESTING MODEL ===")
    test_results = trainer.test(model, test_loader)
    
    # Load best model for evaluation
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        model = LightningEventForecaster.load_from_checkpoint(best_model_path)
    
    # Evaluate with original scale
    print("\\n=== EVALUATING WITH ORIGINAL SCALE ===")
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for data, target in test_loader:
            forecasts, uncertainties = model(data)
            
            all_predictions.append(forecasts.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    uncertainties = np.concatenate(all_uncertainties, axis=0)
    
    # Inverse transform to original scale
    if scaler is not None:
        dummy_data = np.zeros((predictions.shape[0], scaler.n_features_in_))
        feature_names = ['processing_duration', 'queue_size', 'processed_events', 'logged_events', 'queued_events']
        target_col_idx = feature_names.index('logged_events')
        
        # Inverse transform predictions
        for i in range(predictions.shape[1]):
            dummy_data[:, target_col_idx] = predictions[:, i]
            predictions[:, i] = scaler.inverse_transform(dummy_data)[:, target_col_idx]
        
        # Inverse transform targets
        for i in range(targets.shape[1]):
            dummy_data[:, target_col_idx] = targets[:, i]
            targets[:, i] = scaler.inverse_transform(dummy_data)[:, target_col_idx]
    
    # Calculate final metrics
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    print(f"\\n=== LIGHTNING MODEL EVALUATION (Original Scale) ===")
    print(f"MAE: {mae:.2f} events")
    print(f"RMSE: {rmse:.2f} events")
    print(f"MAPE: {mape:.2f}%")
    print(f"Average actual events: {np.mean(targets):.2f}")
    print(f"Average predicted events: {np.mean(predictions):.2f}")
    
    # Save results
    print("\\nSaving Lightning results...")
    np.save('lightning_predictions.npy', predictions)
    np.save('lightning_targets.npy', targets)
    np.save('lightning_uncertainties.npy', uncertainties)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\\n🎯 LIGHTNING TRAINING COMPLETE!")
    print("✅ Best model saved in lightning_checkpoints/")
    print("✅ Results saved as numpy arrays (original scale)")
    print("✅ Training logs saved in lightning_logs/")
    print(f"✅ Tensorboard logs: tensorboard --logdir lightning_logs/")
    
    return model, predictions, targets, uncertainties

def main():
    if not LIGHTNING_AVAILABLE:
        print("\\n❌ PyTorch Lightning not available.")
        print("Install with: pip install pytorch-lightning")
        print("Or use the improved_forecasting_model.py instead.")
        return
    
    try:
        train_with_lightning()
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        print("You may want to try the improved_forecasting_model.py instead.")

if __name__ == "__main__":
    main()
