"""
Comprehensive Training Pipeline for Enhanced Event Processing System
==================================================================

This module orchestrates the training of both the forecasting model and the job allocation
controller, providing a complete end-to-end training pipeline with evaluation and monitoring.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from state_of_art_forecaster import StateOfTheArtForecaster, ModelConfig, AdvancedFeatureEngineer


class EventProcessingDataset(Dataset):
    """Dataset for training the enhanced forecasting model."""
    
    def __init__(self, data: pd.DataFrame, config: ModelConfig, scaler: Optional[StandardScaler] = None):
        self.config = config
        self.data = data.copy().sort_values('DateTime').reset_index(drop=True)
        
        # Feature engineering
        feature_engineer = AdvancedFeatureEngineer()
        self.data = feature_engineer.engineer_features(self.data)
        
        # Get feature columns
        self.feature_cols = feature_engineer.get_feature_columns(self.data)
        self.target_col = 'logged_events'
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            self.data[self.feature_cols] = self.scaler.fit_transform(self.data[self.feature_cols])
        else:
            self.scaler = scaler
            self.data[self.feature_cols] = self.scaler.transform(self.data[self.feature_cols])
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
        
    def _create_sequences(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create sequences for multi-horizon forecasting."""
        feature_data = self.data[self.feature_cols].values
        target_data = self.data[self.target_col].values
        
        n_sequences = len(self.data) - self.config.sequence_length - max(self.config.forecast_horizons) + 1
        
        if n_sequences <= 0:
            return np.array([]), {}
        
        # Input sequences
        sequences = np.zeros((n_sequences, self.config.sequence_length, len(self.feature_cols)))
        
        # Multi-horizon targets
        targets = {}
        for horizon in self.config.forecast_horizons:
            targets[f'horizon_{horizon}'] = np.zeros((n_sequences, horizon))
        
        for i in range(n_sequences):
            # Input sequence
            sequences[i] = feature_data[i:i + self.config.sequence_length]
            
            # Target sequences for each horizon
            for horizon in self.config.forecast_horizons:
                start_idx = i + self.config.sequence_length
                end_idx = start_idx + horizon
                targets[f'horizon_{horizon}'][i] = target_data[start_idx:end_idx]
        
        return sequences, targets
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sequence = torch.FloatTensor(self.sequences[idx])
        targets = {key: torch.FloatTensor(values[idx]) for key, values in self.targets.items()}
        return sequence, targets


class EnhancedTrainingPipeline:
    """Complete training pipeline for the enhanced event processing system."""
    
    def __init__(self, data_path: str, config: Optional[ModelConfig] = None):
        self.data_path = data_path
        self.config = config or ModelConfig()
        self.results_dir = "enhanced_system/results"
        self.models_dir = "enhanced_system/models"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def load_and_prepare_data(self) -> None:
        """Load and prepare data for training."""
        print("📊 Loading and preparing data...")
        
        # Load raw data
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df):,} records from {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        # Create dataset
        dataset = EventProcessingDataset(df, self.config)
        self.scaler = dataset.scaler
        
        print(f"Created {len(dataset):,} sequences with {len(dataset.feature_cols)} features")
        print(f"Feature columns: {dataset.feature_cols[:10]}...")  # Show first 10
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        
        print(f"Split: Train={len(train_dataset):,}, Val={len(val_dataset):,}, Test={len(test_dataset):,}")
        
        # Save scaler
        with open(f"{self.models_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Update config with actual feature dimension
        self.config.feature_dim = len(dataset.feature_cols)
        
    def create_model(self) -> None:
        """Initialize the forecasting model."""
        print("🏗️ Creating state-of-the-art forecasting model...")
        
        self.model = StateOfTheArtForecaster(self.config)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Using device: {self.device}")
        
    def train_forecasting_model(self) -> Dict[str, List[float]]:
        """Train the state-of-the-art forecasting model."""
        print("🚀 Training forecasting model...")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate * 5,
            steps_per_epoch=len(self.train_loader),
            epochs=self.config.max_epochs,
            pct_start=0.1
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            train_metrics = self._train_epoch(optimizer, scheduler)
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_mae'].append(train_metrics['mae'])
            history['val_mae'].append(val_metrics['mae'])
            
            # Print progress
            if epoch % 5 == 0 or epoch == self.config.max_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss={train_metrics['loss']:.6f}, "
                      f"Val Loss={val_metrics['loss']:.6f}, "
                      f"Train MAE={train_metrics['mae']:.2f}, "
                      f"Val MAE={val_metrics['mae']:.2f}")
            
            # Early stopping and checkpointing
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{self.models_dir}/best_forecaster.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{self.models_dir}/best_forecaster.pth"))
        
        # Save training history
        with open(f"{self.results_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def _train_epoch(self, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        n_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(self.train_loader):
            sequences = sequences.to(self.device)
            targets = {key: value.to(self.device) for key, value in targets.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            
            # Calculate multi-horizon loss
            loss = 0
            mae = 0
            
            for horizon in self.config.forecast_horizons:
                horizon_key = f'horizon_{horizon}'
                pred_mean = outputs['predictions'][horizon_key]
                target = targets[horizon_key]
                
                # MSE loss for predictions
                horizon_loss = nn.MSELoss()(pred_mean, target)
                loss += horizon_loss
                
                # MAE for monitoring
                with torch.no_grad():
                    horizon_mae = torch.mean(torch.abs(pred_mean - target))
                    mae += horizon_mae
            
            # Average across horizons
            loss = loss / len(self.config.forecast_horizons)
            mae = mae / len(self.config.forecast_horizons)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'mae': total_mae / n_batches
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_batches = 0
        
        with torch.no_grad():
            for sequences, targets in self.val_loader:
                sequences = sequences.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate multi-horizon loss
                loss = 0
                mae = 0
                
                for horizon in self.config.forecast_horizons:
                    horizon_key = f'horizon_{horizon}'
                    pred_mean = outputs['predictions'][horizon_key]
                    target = targets[horizon_key]
                    
                    # MSE loss
                    horizon_loss = nn.MSELoss()(pred_mean, target)
                    loss += horizon_loss
                    
                    # MAE
                    horizon_mae = torch.mean(torch.abs(pred_mean - target))
                    mae += horizon_mae
                
                # Average across horizons
                loss = loss / len(self.config.forecast_horizons)
                mae = mae / len(self.config.forecast_horizons)
                
                total_loss += loss.item()
                total_mae += mae.item()
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'mae': total_mae / n_batches
        }
    
    def evaluate_model(self) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        print("📈 Evaluating trained model...")
        
        self.model.eval()
        all_predictions = {f'horizon_{h}': [] for h in self.config.forecast_horizons}
        all_targets = {f'horizon_{h}': [] for h in self.config.forecast_horizons}
        all_uncertainties = {f'horizon_{h}': [] for h in self.config.forecast_horizons}
        anomaly_scores = []
        
        with torch.no_grad():
            for sequences, targets in self.test_loader:
                sequences = sequences.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Collect predictions and targets
                for horizon in self.config.forecast_horizons:
                    horizon_key = f'horizon_{horizon}'
                    
                    pred_mean = outputs['predictions'][horizon_key].cpu().numpy()
                    target = targets[horizon_key].numpy()
                    uncertainty = outputs['uncertainties'][horizon_key]['total_uncertainty'].cpu().numpy()
                    
                    all_predictions[horizon_key].append(pred_mean)
                    all_targets[horizon_key].append(target)
                    all_uncertainties[horizon_key].append(uncertainty)
                
                # Anomaly scores
                anomaly_scores.append(outputs['anomaly_score'].cpu().numpy())
        
        # Concatenate results
        for horizon in self.config.forecast_horizons:
            horizon_key = f'horizon_{horizon}'
            all_predictions[horizon_key] = np.concatenate(all_predictions[horizon_key])
            all_targets[horizon_key] = np.concatenate(all_targets[horizon_key])
            all_uncertainties[horizon_key] = np.concatenate(all_uncertainties[horizon_key])
        
        anomaly_scores = np.concatenate(anomaly_scores)
        
        # Calculate metrics for each horizon
        metrics = {}
        
        for horizon in self.config.forecast_horizons:
            horizon_key = f'horizon_{horizon}'
            pred = all_predictions[horizon_key]
            target = all_targets[horizon_key]
            
            # Flatten for metrics calculation
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(target_flat, pred_flat)
            rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
            mape = np.mean(np.abs((target_flat - pred_flat) / (target_flat + 1e-8))) * 100
            r2 = r2_score(target_flat, pred_flat)
            
            metrics[f'{horizon_key}_mae'] = mae
            metrics[f'{horizon_key}_rmse'] = rmse
            metrics[f'{horizon_key}_mape'] = mape
            metrics[f'{horizon_key}_r2'] = r2
            
            print(f"Horizon {horizon} ({horizon*2} min): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R²={r2:.3f}")
        
        # Overall metrics (average across horizons)
        metrics['overall_mae'] = np.mean([metrics[f'horizon_{h}_mae'] for h in self.config.forecast_horizons])
        metrics['overall_rmse'] = np.mean([metrics[f'horizon_{h}_rmse'] for h in self.config.forecast_horizons])
        metrics['overall_mape'] = np.mean([metrics[f'horizon_{h}_mape'] for h in self.config.forecast_horizons])
        metrics['overall_r2'] = np.mean([metrics[f'horizon_{h}_r2'] for h in self.config.forecast_horizons])
        
        print(f"\\nOverall Performance:")
        print(f"MAE: {metrics['overall_mae']:.2f} events")
        print(f"RMSE: {metrics['overall_rmse']:.2f} events")
        print(f"MAPE: {metrics['overall_mape']:.2f}%")
        print(f"R²: {metrics['overall_r2']:.3f}")
        
        # Save metrics
        with open(f"{self.results_dir}/evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(all_predictions, all_targets, all_uncertainties, anomaly_scores)
        
        return metrics
    
    def _create_evaluation_plots(self, predictions: Dict, targets: Dict, uncertainties: Dict, anomaly_scores: np.ndarray):
        """Create comprehensive evaluation plots."""
        print("📊 Creating evaluation visualizations...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Multi-horizon performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE by horizon
        horizons = self.config.forecast_horizons
        maes = []
        
        for horizon in horizons:
            horizon_key = f'horizon_{horizon}'
            pred_flat = predictions[horizon_key].flatten()
            target_flat = targets[horizon_key].flatten()
            mae = mean_absolute_error(target_flat, pred_flat)
            maes.append(mae)
        
        axes[0, 0].bar(horizons, maes, color='skyblue', alpha=0.7)
        axes[0, 0].set_xlabel('Forecast Horizon (intervals)')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Forecast Accuracy by Horizon')
        
        # Prediction vs actual scatter for shortest horizon
        shortest_horizon = min(horizons)
        horizon_key = f'horizon_{shortest_horizon}'
        pred_sample = predictions[horizon_key][:, 0]  # First step of shortest horizon
        target_sample = targets[horizon_key][:, 0]
        
        axes[0, 1].scatter(target_sample, pred_sample, alpha=0.6)
        axes[0, 1].plot([target_sample.min(), target_sample.max()], 
                       [target_sample.min(), target_sample.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Events')
        axes[0, 1].set_ylabel('Predicted Events')
        axes[0, 1].set_title(f'Predictions vs Actual (Horizon {shortest_horizon})')
        
        # Uncertainty distribution
        uncertainty_sample = uncertainties[horizon_key][:, 0].flatten()
        axes[1, 0].hist(uncertainty_sample, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Prediction Uncertainty')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Uncertainty Distribution')
        
        # Anomaly scores
        axes[1, 1].hist(anomaly_scores.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Anomaly Score Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/model_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Time series forecast examples
        fig, axes = plt.subplots(len(horizons), 1, figsize=(15, 4*len(horizons)))
        if len(horizons) == 1:
            axes = [axes]
        
        # Show first 100 predictions for each horizon
        n_samples = min(100, predictions[f'horizon_{horizons[0]}'].shape[0])
        
        for i, horizon in enumerate(horizons):
            horizon_key = f'horizon_{horizon}'
            
            # Get sample predictions
            pred_sample = predictions[horizon_key][:n_samples]
            target_sample = targets[horizon_key][:n_samples]
            uncertainty_sample = uncertainties[horizon_key][:n_samples]
            
            # Plot time series
            time_steps = np.arange(n_samples)
            
            # Plot actual vs predicted for first forecast step
            axes[i].plot(time_steps, target_sample[:, 0], label='Actual', color='blue', alpha=0.7)
            axes[i].plot(time_steps, pred_sample[:, 0], label='Predicted', color='red', alpha=0.7)
            
            # Add uncertainty bands
            uncertainty_mean = uncertainty_sample[:, 0]
            axes[i].fill_between(time_steps, 
                                pred_sample[:, 0] - uncertainty_mean,
                                pred_sample[:, 0] + uncertainty_mean,
                                alpha=0.2, color='red', label='Uncertainty')
            
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Events')
            axes[i].set_title(f'Forecast Examples - Horizon {horizon} ({horizon*2} minutes)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/forecast_examples.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {self.results_dir}/")
    
    def run_complete_training(self):
        """Run the complete training pipeline."""
        print("=" * 60)
        print("🚀 ENHANCED EVENT PROCESSING SYSTEM TRAINING")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. Data preparation
            self.load_and_prepare_data()
            
            # 2. Model creation
            self.create_model()
            
            # 3. Model training
            training_history = self.train_forecasting_model()
            
            # 4. Model evaluation
            evaluation_metrics = self.evaluate_model()
            
            # 5. Save final model and config
            torch.save(self.model.state_dict(), f"{self.models_dir}/final_forecaster.pth")
            
            with open(f"{self.models_dir}/model_config.json", 'w') as f:
                config_dict = {
                    'd_model': self.config.d_model,
                    'n_heads': self.config.n_heads,
                    'n_layers': self.config.n_layers,
                    'lstm_hidden': self.config.lstm_hidden,
                    'lstm_layers': self.config.lstm_layers,
                    'dropout': self.config.dropout,
                    'sequence_length': self.config.sequence_length,
                    'forecast_horizons': self.config.forecast_horizons,
                    'feature_dim': self.config.feature_dim
                }
                json.dump(config_dict, f, indent=2)
            
            training_time = datetime.now() - start_time
            
            print("\\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"⏱️ Total training time: {training_time}")
            print(f"📊 Overall MAE: {evaluation_metrics['overall_mae']:.2f} events")
            print(f"📊 Overall RMSE: {evaluation_metrics['overall_rmse']:.2f} events")
            print(f"📊 Overall R²: {evaluation_metrics['overall_r2']:.3f}")
            print(f"📁 Models saved to: {self.models_dir}/")
            print(f"📁 Results saved to: {self.results_dir}/")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise


def main():
    """Main function to run the enhanced training pipeline."""
    
    # Configuration
    config = ModelConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        lstm_hidden=256,
        lstm_layers=2,
        sequence_length=48,  # 1.6 hours of 2-min intervals
        forecast_horizons=[6, 12, 24, 48],  # 12min to 1.6hr ahead
        learning_rate=1e-4,
        max_epochs=50,
        batch_size=32
    )
    
    # Data path
    data_path = "shared_data/EventsMetricsMarJul.csv"
    
    # Create and run training pipeline
    pipeline = EnhancedTrainingPipeline(data_path, config)
    pipeline.run_complete_training()


if __name__ == "__main__":
    main()
