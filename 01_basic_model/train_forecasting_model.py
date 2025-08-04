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

# Configuration
SEQUENCE_LENGTH = 12  # 12 intervals = 24 minutes of history
FORECAST_HORIZON = 6   # 6 intervals = 12 minutes ahead
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.003
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

class EventQueueForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, forecast_horizon=10, dropout=0.1):
        super(EventQueueForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Forecast head with residual connection
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Softplus()  # Ensures positive uncertainty
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output for forecasting
        last_output = lstm_out[:, -1, :]
        
        # Generate forecasts and uncertainties
        forecasts = self.forecast_head(last_output)
        uncertainties = self.uncertainty_head(last_output)
        
        return forecasts, uncertainties

def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            forecasts, uncertainties = model(data)
            
            # Simple MSE loss for better training stability
            loss = criterion(forecasts, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                forecasts, uncertainties = model(data)
                loss = criterion(forecasts, target)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_forecasting_model.pth')
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, scaler=None, target_col='logged_events'):
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
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
        feature_names = ['processing_duration', 'queue_size', 'processed_events', 'logged_events', 'queued_events']  # From preprocessing
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
    
    print(f"\n=== MODEL EVALUATION (Original Scale) ===")
    print(f"MAE: {mae:.2f} events")
    print(f"RMSE: {rmse:.2f} events")
    print(f"MAPE: {mape:.2f}%")
    print(f"Average actual events: {np.mean(targets):.2f}")
    print(f"Average predicted events: {np.mean(predictions):.2f}")
    
    return predictions, targets, uncertainties

def main():
    print("=== EVENT QUEUE FORECASTING MODEL ===")
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_dim = len(train_dataset.feature_cols)
    print(f"\nInput features: {input_dim}")
    
    model = EventQueueForecaster(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        forecast_horizon=FORECAST_HORIZON
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n=== TRAINING MODEL ===")
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, DEVICE)
    
    # Load best model
    model.load_state_dict(torch.load('best_forecasting_model.pth'))
    
    # Evaluate model with original scale
    print("\n=== EVALUATING MODEL ===")
    predictions, targets, uncertainties = evaluate_model(model, test_loader, DEVICE, scaler, 'logged_events')
    
    # Save results
    print("\nSaving results...")
    np.save('predictions.npy', predictions)
    np.save('targets.npy', targets)
    np.save('uncertainties.npy', uncertainties)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot sample predictions (original scale)
    plt.subplot(1, 3, 2)
    sample_idx = 0
    time_steps = range(FORECAST_HORIZON)
    plt.plot(time_steps, targets[sample_idx], 'b-', label='Actual', linewidth=2, marker='o')
    plt.plot(time_steps, predictions[sample_idx], 'r--', label='Predicted', linewidth=2, marker='s')
    plt.xlabel('Time Steps (2-min intervals)')
    plt.ylabel('Logged Events (actual count)')
    plt.title('Sample Forecast: Incoming Event Load')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.subplot(1, 3, 3)
    errors = (predictions - targets).flatten()
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (events)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print sample predictions for better understanding
    print(f"\n=== SAMPLE PREDICTIONS (Original Scale) ===")
    print("First 3 test samples:")
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Actual:    {targets[i]}")
        print(f"  Predicted: {predictions[i]}")
        print(f"  Error:     {predictions[i] - targets[i]}")
    
    print("\n🎯 MODEL TRAINING COMPLETE!")
    print("✅ Best model saved as 'best_forecasting_model.pth'")
    print("✅ Results saved as numpy arrays (original scale)")
    print("✅ Training plots saved as 'training_results.png'")

if __name__ == "__main__":
    main()
