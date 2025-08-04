"""
Model Performance Comparison and Optimization Analysis
=====================================================

This script compares different model architectures and provides recommendations for:
1. Optimal model selection based on data characteristics
2. Hyperparameter optimization strategies
3. Performance vs inference time trade-offs
4. Feature selection for efficiency
5. Model ensemble strategies
6. Deployment optimization recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataModule(L.LightningDataModule):
    """Data module for time series data."""
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader

class ModelComparisonAnalyzer:
    def __init__(self, data_path='../shared_data/EventsMetricsMarJul.csv'):
        """Initialize the model comparison analyzer."""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
            print(f"🔥 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            torch.set_num_threads(4)  # Optimize CPU performance if no GPU
        
        # Create models directory if it doesn't exist
        self.models_dir = '../models'
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"📁 Models will be saved to: {self.models_dir}")
        
    def save_model_and_results(self, name, model, results):
        """Save trained model and results to disk."""
        model_path = os.path.join(self.models_dir, f"{name.replace(' ', '_').lower()}")
        os.makedirs(model_path, exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
        
        # Save model architecture info
        model_info = {
            'class_name': model.__class__.__name__,
            'input_size': model.lstm.input_size if hasattr(model, 'lstm') else None,
            'hidden_size': getattr(model, 'hidden_size', None),
            'num_layers': getattr(model, 'num_layers', None)
        }
        
        with open(os.path.join(model_path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save results
        with open(os.path.join(model_path, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Saved {name} model and results to {model_path}")
    
    def load_model_and_results(self, name, input_size):
        """Load trained model and results from disk if they exist."""
        model_path = os.path.join(self.models_dir, f"{name.replace(' ', '_').lower()}")
        model_file = os.path.join(model_path, 'model.pth')
        results_file = os.path.join(model_path, 'results.pkl')
        info_file = os.path.join(model_path, 'model_info.json')
        
        if not (os.path.exists(model_file) and os.path.exists(results_file) and os.path.exists(info_file)):
            return None, None
        
        try:
            # Load model info
            with open(info_file, 'r') as f:
                model_info = json.load(f)
            
            # Recreate model based on saved info
            class_name = model_info['class_name']
            if class_name == 'LightweightLSTM':
                model = LightweightLSTM(input_size, 
                                      hidden_size=model_info['hidden_size'],
                                      num_layers=model_info['num_layers'])
            elif class_name == 'SimpleLSTM':
                model = SimpleLSTM(input_size,
                                 hidden_size=model_info['hidden_size'],
                                 num_layers=model_info['num_layers'])
            elif class_name == 'AttentionLSTM':
                model = AttentionLSTM(input_size,
                                    hidden_size=model_info['hidden_size'],
                                    num_layers=model_info['num_layers'])
            else:
                return None, None
            
            # Load model weights
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model = model.to(self.device)
            
            # Load results
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            print(f"📂 Loaded {name} model and results from {model_path}")
            return model, results
            
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            return None, None
        
    def load_and_prepare_data(self):
        """Load and prepare data for model comparison."""
        print("📊 Loading and preparing data for model comparison...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['DateTime'])
        self.data['queue_length'] = self.data['avg_unprocessed_events_count']
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Create essential features
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        self.data['is_business_hours'] = ((self.data['hour'] >= 9) & (self.data['hour'] <= 17)).astype(int)
        
        # Cyclical encoding
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        # Rolling features
        for window in [6, 12, 24]:
            self.data[f'rolling_mean_{window}'] = self.data['queue_length'].rolling(window, min_periods=1).mean()
            self.data[f'rolling_std_{window}'] = self.data['queue_length'].rolling(window, min_periods=1).std()
        
        # Lag features
        for lag in [1, 6, 12, 24]:
            self.data[f'lag_{lag}'] = self.data['queue_length'].shift(lag)
        
        print(f"✅ Data prepared: {len(self.data)} samples with {self.data.shape[1]} features")
        return self.data
    
    def create_datasets(self, sequence_length=30, test_size=0.2):
        """Create train/test datasets for model comparison."""
        print(f"🔧 Creating datasets with sequence length {sequence_length}...")
        
        # Select features
        feature_cols = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'is_weekend', 'is_business_hours',
            'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24',
            'rolling_std_6', 'rolling_std_12', 'rolling_std_24',
            'lag_1', 'lag_6', 'lag_12', 'lag_24'
        ]
        
        # Prepare data
        data_clean = self.data[['queue_length'] + feature_cols].fillna(method='ffill').fillna(0)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data_clean[feature_cols])
        target_scaler = StandardScaler()
        target_scaled = target_scaler.fit_transform(data_clean[['queue_length']])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Convert to PyTorch tensors and move to device (GPU if available)
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        print(f"✅ Datasets created - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"📍 Data loaded on: {self.device}")
        
        return X_train, X_test, y_train, y_test, scaler, target_scaler
    
    def train_model(self, model, model_name, train_loader, val_loader, epochs=50):
        """Train a Lightning model with automatic checkpointing."""
        print(f"\nTraining {model_name}...")
        
        # Setup Lightning trainer with checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.models_dir, model_name),
            filename=f'{model_name}_best',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            save_last=True,
            verbose=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=10,
            verbose=True,
            mode='min'
        )
        
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=TensorBoardLogger(
                save_dir=os.path.join(self.models_dir, 'lightning_logs'),
                name=model_name
            ),
            enable_checkpointing=True,
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            deterministic=True
        )
        
        # Create data module
        data_module = TimeSeriesDataModule(train_loader, val_loader, val_loader)
        
        try:
            # Train the model (will resume from checkpoint if exists)
            trainer.fit(model, data_module)
            
            # Get the best model using the class method
            model_class = model.__class__
            best_model = model_class.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                input_size=model.hparams.input_size
            )
            
            return best_model, trainer.logged_metrics
            
        except Exception as e:
            print(f"Training failed for {model_name}: {str(e)}")
            return None, None
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate a trained Lightning model."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                
                predictions = model(batch_x)
                all_predictions.append(predictions)
                all_targets.append(batch_y)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate MSE
        test_mse = F.mse_loss(all_predictions, all_targets).item()
        
        evaluation_time = time.time() - start_time
        
        # Inference time measurement (single prediction)
        start_time = time.time()
        with torch.no_grad():
            single_batch = next(iter(test_loader))
            single_x = single_batch[0][:1]  # Take first sample
            if torch.cuda.is_available():
                single_x = single_x.cuda()
            
            for _ in range(100):  # 100 predictions for average
                _ = model(single_x)
        
        inference_time = (time.time() - start_time) / 100 * 1000  # ms per prediction
        
        # Model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        results = {
            'test_mse': test_mse,
            'evaluation_time': evaluation_time,
            'inference_time_ms': inference_time,
            'parameters': param_count,
            'model_size_mb': model_size_mb,
        }
        
        print(f"✅ Evaluation complete for {model_name} - Test MSE: {test_mse:.4f}, Inference: {inference_time:.2f}ms")
        return results

    def compare_models(self, force_retrain=False):
        """Compare different model architectures using Lightning."""
        print("\n🏆 MODEL ARCHITECTURE COMPARISON (Lightning Framework)")
        print("=" * 50)
        
        if force_retrain:
            print("🔄 Force retraining enabled - will train all models from scratch")
        
        # Prepare data - use shorter sequence for memory efficiency
        self.load_and_prepare_data()
        sequence_length = 15 if torch.cuda.is_available() else 30
        X_train, X_test, y_train, y_test, scaler, target_scaler = self.create_datasets(sequence_length=sequence_length)
        
        input_size = X_train.shape[2]
        
        # Create data loaders
        batch_size = 16 if torch.cuda.is_available() else 32
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Split train into train/val for Lightning
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Define models to compare
        models_to_test = {
            'Lightweight LSTM': LightweightLSTM(input_size, hidden_size=16, num_layers=1),
            'Simple LSTM': SimpleLSTM(input_size, hidden_size=32, num_layers=1),
            'Standard LSTM': SimpleLSTM(input_size, hidden_size=64, num_layers=2),
            'Attention LSTM': AttentionLSTM(input_size, hidden_size=64, num_layers=1),
        }
        
        # Train and evaluate each model
        for name, model in models_to_test.items():
            print(f"\n--- {name} ---")
            
            # Check if trained model exists (unless force_retrain is True)
            checkpoint_path = os.path.join(self.models_dir, name, f'{name}_best.ckpt')
            
            if not force_retrain and os.path.exists(checkpoint_path):
                print(f"🔄 Loading existing checkpoint for {name}")
                try:
                    model_class = model.__class__
                    trained_model = model_class.load_from_checkpoint(checkpoint_path, input_size=input_size)
                    training_metrics = None  # We don't have training metrics from checkpoint
                except Exception as e:
                    print(f"❌ Failed to load checkpoint: {e}. Training from scratch...")
                    trained_model, training_metrics = self.train_model(model, name, train_loader, val_loader)
            else:
                # Train model
                trained_model, training_metrics = self.train_model(model, name, train_loader, val_loader)
            
            if trained_model is not None:
                # Evaluate model
                eval_results = self.evaluate_model(trained_model, test_loader, name)
                
                # Store results
                self.results[name] = eval_results
                self.models[name] = trained_model
                
                # Save model and results for future use
                self.save_model_and_results(name, trained_model, eval_results)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"❌ Failed to train {name}")
        
        # Display comparison results
        self.display_comparison_results()

    def display_comparison_results(self):
        """Display comprehensive comparison results."""
        print("\n📊 MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        if not self.results:
            print("No model results to display.")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Test MSE': results['test_mse'],
                'Inference Time (ms)': results['inference_time_ms'],
                'Parameters': results['parameters'],
                'Size (MB)': results['model_size_mb']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test MSE')
        
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Performance-Efficiency Analysis
        print(f"\n🎯 PERFORMANCE-EFFICIENCY ANALYSIS")
        print("=" * 50)
        
        best_accuracy = df.iloc[0]
        fastest_inference = df.loc[df['Inference Time (ms)'].idxmin()]
        smallest_model = df.loc[df['Size (MB)'].idxmin()]
        
        print(f"🏆 Best Accuracy: {best_accuracy['Model']} (MSE: {best_accuracy['Test MSE']:.4f})")
        print(f"⚡ Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference Time (ms)']:.2f}ms)")
        print(f"💾 Smallest Model: {smallest_model['Model']} ({smallest_model['Size (MB)']:.2f}MB)")
        
        return df
        """Create a comprehensive comparison summary."""
        print("\n📊 MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Test MSE': results['test_mse'],
                'Training Time (s)': results['training_time'],
                'Inference Time (ms)': results['inference_time_ms'],
                'Parameters': results['parameters'],
                'Size (MB)': results['model_size_mb']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test MSE')
        
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Performance-Efficiency Analysis
        print(f"\n🎯 PERFORMANCE-EFFICIENCY ANALYSIS")
        print("=" * 50)
        
        best_accuracy = df.iloc[0]
        fastest_inference = df.loc[df['Inference Time (ms)'].idxmin()]
        smallest_model = df.loc[df['Size (MB)'].idxmin()]
        
        print(f"🏆 Best Accuracy: {best_accuracy['Model']} (MSE: {best_accuracy['Test MSE']:.4f})")
        print(f"⚡ Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference Time (ms)']:.2f}ms)")
        print(f"💾 Smallest Model: {smallest_model['Model']} ({smallest_model['Size (MB)']:.2f}MB)")
        
        # Efficiency ratios
        print(f"\n📈 EFFICIENCY METRICS")
        print("-" * 30)
        df['Accuracy/Speed Ratio'] = 1 / (df['Test MSE'] * df['Inference Time (ms)'])
        df['Accuracy/Size Ratio'] = 1 / (df['Test MSE'] * df['Size (MB)'])
        
        best_speed_ratio = df.loc[df['Accuracy/Speed Ratio'].idxmax()]
        best_size_ratio = df.loc[df['Accuracy/Size Ratio'].idxmax()]
        
        print(f"Best Speed-Accuracy Trade-off: {best_speed_ratio['Model']}")
        print(f"Best Size-Accuracy Trade-off: {best_size_ratio['Model']}")
        
        return df

    def run_complete_comparison(self, force_retrain=False):
        """Run the complete model comparison analysis."""
        print("🚀 Starting Model Performance Comparison Analysis")
        print("=" * 60)
        
        # Run comparison
        self.compare_models(force_retrain=force_retrain)
        
        print("\n✅ MODEL COMPARISON ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Models trained and saved. Results ready for comparison.")

class SimpleLSTM(L.LightningModule):
    """Simple LSTM model for comparison."""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1, lr=0.001):
        super(SimpleLSTM, self).__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class AttentionLSTM(L.LightningModule):
    """LSTM with attention mechanism."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1, lr=0.001):
        super(AttentionLSTM, self).__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = self.fc(attn_out[:, -1, :])
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class LightweightLSTM(L.LightningModule):
    """Lightweight LSTM for fast inference."""
    def __init__(self, input_size, hidden_size=32, num_layers=1, lr=0.001):
        super(LightweightLSTM, self).__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Performance Comparison Analysis')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Force retrain all models even if cached versions exist')
    args = parser.parse_args()
    
    analyzer = ModelComparisonAnalyzer()
    analyzer.run_complete_comparison(force_retrain=args.force_retrain)
