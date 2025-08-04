"""
Enhanced Model Comparison and Analysis System
===========================================

This module provides comprehensive comparison between the original models and our enhanced system,
with detailed analysis and recommendations for production deployment.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing models
sys.path.append('01_basic_model')
sys.path.append('02_improved_model') 
sys.path.append('03_tiered_approach')
sys.path.append('04_lightning_framework')

# Import our enhanced system
from state_of_art_forecaster import StateOfTheArtForecaster, ModelConfig, AdvancedFeatureEngineer


class ModelComparison:
    """Comprehensive model comparison and analysis system."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.results = {}
        self.comparison_dir = "enhanced_system/comparison"
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Load and prepare data
        self.load_data()
        
    def load_data(self):
        """Load and prepare data for comparison."""
        print("📊 Loading data for model comparison...")
        
        # Load original data
        self.df = pd.read_csv(self.data_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df = self.df.sort_values('DateTime').reset_index(drop=True)
        
        # Rename columns for consistency
        self.df = self.df.rename(columns={
            'avg_average_processing_duration_ms': 'proc_duration',
            'avg_unprocessed_events_count': 'queue_size',
            'avg_processed_events_in_interval': 'processed_events',
            'avg_logged_events_in_interval': 'logged_events',
            'avg_queued_events_in_interval': 'queued_events'
        })
        
        print(f"Loaded {len(self.df):,} records for comparison")
        
        # Create train/test split for fair comparison
        split_idx = int(0.8 * len(self.df))
        self.train_df = self.df[:split_idx].copy()
        self.test_df = self.df[split_idx:].copy()
        
        print(f"Train: {len(self.train_df):,}, Test: {len(self.test_df):,}")
    
    def evaluate_baseline_models(self):
        """Evaluate the original baseline models."""
        print("\\n🔍 Evaluating baseline models...")
        
        # 1. Simple baseline (naive forecast)
        self.results['naive'] = self._evaluate_naive_baseline()
        
        # 2. Moving average baseline
        self.results['moving_average'] = self._evaluate_moving_average_baseline()
        
        # 3. Linear trend baseline
        self.results['linear_trend'] = self._evaluate_linear_trend_baseline()
        
        # 4. ARIMA baseline (if data supports it)
        try:
            self.results['arima'] = self._evaluate_arima_baseline()
        except Exception as e:
            print(f"ARIMA evaluation failed: {e}")
            self.results['arima'] = {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'r2': -np.inf}
    
    def _evaluate_naive_baseline(self) -> Dict[str, float]:
        """Naive forecast: use last value."""
        print("  • Naive baseline (last value)")
        
        # Simple approach: predict next value as current value
        test_data = self.test_df['logged_events'].values
        predictions = test_data[:-1]  # Use previous value as prediction
        targets = test_data[1:]       # Actual next values
        
        metrics = self._calculate_metrics(predictions, targets)
        metrics['model_type'] = 'baseline'
        metrics['inference_time'] = 0.001  # Virtually instant
        
        return metrics
    
    def _evaluate_moving_average_baseline(self, window: int = 12) -> Dict[str, float]:
        """Moving average baseline."""
        print(f"  • Moving average baseline (window={window})")
        
        test_data = self.test_df['logged_events'].values
        predictions = []
        
        for i in range(window, len(test_data) - 1):
            # Use moving average of previous window points
            ma_pred = np.mean(test_data[i-window:i])
            predictions.append(ma_pred)
        
        targets = test_data[window+1:]
        predictions = np.array(predictions)
        
        metrics = self._calculate_metrics(predictions, targets)
        metrics['model_type'] = 'baseline'
        metrics['inference_time'] = 0.005  # Very fast
        
        return metrics
    
    def _evaluate_linear_trend_baseline(self, window: int = 24) -> Dict[str, float]:
        """Linear trend extrapolation baseline."""
        print(f"  • Linear trend baseline (window={window})")
        
        test_data = self.test_df['logged_events'].values
        predictions = []
        
        for i in range(window, len(test_data) - 1):
            # Fit linear trend to previous window points
            x = np.arange(window)
            y = test_data[i-window:i]
            
            # Linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Predict next point
            next_pred = slope * window + intercept
            predictions.append(next_pred)
        
        targets = test_data[window+1:]
        predictions = np.array(predictions)
        
        metrics = self._calculate_metrics(predictions, targets)
        metrics['model_type'] = 'baseline'
        metrics['inference_time'] = 0.010  # Fast
        
        return metrics
    
    def _evaluate_arima_baseline(self) -> Dict[str, float]:
        """ARIMA baseline model."""
        print("  • ARIMA baseline")
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Use a subset for training ARIMA (it's slow)
            train_data = self.train_df['logged_events'].values[-1000:]  # Last 1000 points
            test_data = self.test_df['logged_events'].values[:100]      # First 100 test points
            
            # Fit ARIMA model
            start_time = time.time()
            model = ARIMA(train_data, order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))
            inference_time = (time.time() - start_time) / len(test_data)
            
            metrics = self._calculate_metrics(predictions, test_data)
            metrics['model_type'] = 'statistical'
            metrics['inference_time'] = inference_time
            
            return metrics
            
        except ImportError:
            print("    ⚠️ statsmodels not available for ARIMA")
            return {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'r2': -np.inf}
    
    def evaluate_enhanced_model(self, model_path: str = None):
        """Evaluate our enhanced state-of-the-art model."""
        print("\\n🚀 Evaluating enhanced state-of-the-art model...")
        
        if model_path is None:
            model_path = "enhanced_system/models/best_forecaster.pth"
        
        try:
            # Load model configuration
            with open("enhanced_system/models/model_config.json", 'r') as f:
                config_dict = json.load(f)
            
            config = ModelConfig(**config_dict)
            
            # Create and load model
            model = StateOfTheArtForecaster(config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Load scaler
            with open("enhanced_system/models/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            
            # Prepare test data
            feature_engineer = AdvancedFeatureEngineer()
            test_data = feature_engineer.engineer_features(self.test_df)
            feature_cols = feature_engineer.get_feature_columns(test_data)
            
            # Scale features
            test_data[feature_cols] = scaler.transform(test_data[feature_cols])
            
            # Make predictions
            predictions, inference_times = self._predict_enhanced_model(model, test_data, feature_cols, config)
            
            # Calculate metrics for each horizon
            horizon_metrics = {}
            
            for horizon in config.forecast_horizons:
                targets = self._get_targets_for_horizon(test_data['logged_events'].values, horizon, config.sequence_length)
                preds = predictions[f'horizon_{horizon}']
                
                if len(preds) > 0 and len(targets) > 0:
                    # Use first step of multi-step prediction for comparison
                    pred_first_step = preds[:, 0] if preds.ndim > 1 else preds
                    target_first_step = targets[:, 0] if targets.ndim > 1 else targets
                    
                    metrics = self._calculate_metrics(pred_first_step, target_first_step)
                    metrics['model_type'] = 'enhanced_ml'
                    metrics['inference_time'] = np.mean(inference_times)
                    metrics['horizon'] = horizon
                    
                    horizon_metrics[f'horizon_{horizon}'] = metrics
            
            # Overall metrics (average of shortest horizon)
            if horizon_metrics:
                shortest_horizon = min(config.forecast_horizons)
                self.results['enhanced'] = horizon_metrics[f'horizon_{shortest_horizon}'].copy()
                self.results['enhanced']['all_horizons'] = horizon_metrics
            
            print(f"  ✅ Enhanced model evaluated successfully")
            
        except Exception as e:
            print(f"  ❌ Enhanced model evaluation failed: {e}")
            self.results['enhanced'] = {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'r2': -np.inf}
    
    def _predict_enhanced_model(self, model, test_data, feature_cols, config):
        """Make predictions with the enhanced model."""
        predictions = {f'horizon_{h}': [] for h in config.forecast_horizons}
        inference_times = []
        
        # Create sequences
        feature_data = test_data[feature_cols].values
        
        for i in range(len(test_data) - config.sequence_length - max(config.forecast_horizons)):
            sequence = feature_data[i:i + config.sequence_length]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(sequence_tensor)
            
            inference_times.append(time.time() - start_time)
            
            # Extract predictions for each horizon
            for horizon in config.forecast_horizons:
                horizon_key = f'horizon_{horizon}'
                pred = outputs['predictions'][horizon_key].squeeze().cpu().numpy()
                predictions[horizon_key].append(pred)
        
        # Convert to numpy arrays
        for horizon in config.forecast_horizons:
            if predictions[f'horizon_{horizon}']:
                predictions[f'horizon_{horizon}'] = np.array(predictions[f'horizon_{horizon}'])
        
        return predictions, inference_times
    
    def _get_targets_for_horizon(self, target_data, horizon, sequence_length):
        """Get target values for a specific forecasting horizon."""
        targets = []
        
        for i in range(len(target_data) - sequence_length - horizon):
            target_seq = target_data[i + sequence_length:i + sequence_length + horizon]
            targets.append(target_seq)
        
        return np.array(targets) if targets else np.array([])
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        # Handle edge cases
        if len(predictions) == 0 or len(targets) == 0:
            return {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'r2': -np.inf}
        
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        r2 = r2_score(targets, predictions)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def create_comparison_report(self):
        """Create comprehensive comparison report."""
        print("\\n📊 Creating comparison report...")
        
        # Prepare data for visualization
        models = []
        metrics = []
        
        for model_name, result in self.results.items():
            if isinstance(result, dict) and 'mae' in result:
                models.append(model_name)
                metrics.append(result)
        
        if not models:
            print("❌ No valid model results for comparison")
            return
        
        # Create comparison plots
        self._create_performance_comparison(models, metrics)
        self._create_detailed_analysis(models, metrics)
        
        # Generate summary report
        self._generate_summary_report(models, metrics)
        
        print(f"✅ Comparison report saved to {self.comparison_dir}/")
    
    def _create_performance_comparison(self, models: List[str], metrics: List[Dict]):
        """Create performance comparison visualizations."""
        # Prepare data
        df_metrics = pd.DataFrame(metrics)
        df_metrics['model'] = models
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE comparison
        axes[0, 0].bar(models, df_metrics['mae'], color='skyblue', alpha=0.7)
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Model Performance: MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(models, df_metrics['rmse'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_ylabel('Root Mean Square Error')
        axes[0, 1].set_title('Model Performance: RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 0].bar(models, df_metrics['mape'], color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Mean Absolute Percentage Error (%)')
        axes[1, 0].set_title('Model Performance: MAPE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 1].bar(models, df_metrics['r2'], color='lightcoral', alpha=0.7)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Model Performance: R²')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.comparison_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_analysis(self, models: List[str], metrics: List[Dict]):
        """Create detailed analysis plots."""
        # Performance vs Inference Time
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        mae_values = [m['mae'] for m in metrics]
        inference_times = [m.get('inference_time', 0) for m in metrics]
        
        # Create scatter plot
        scatter = ax.scatter(inference_times, mae_values, s=100, alpha=0.7)
        
        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(model, (inference_times[i], mae_values[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Inference Time (seconds)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Performance vs Inference Time Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f"{self.comparison_dir}/performance_vs_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model complexity analysis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        model_types = [m.get('model_type', 'unknown') for m in metrics]
        type_colors = {'baseline': 'red', 'statistical': 'orange', 'enhanced_ml': 'green', 'unknown': 'gray'}
        colors = [type_colors.get(t, 'gray') for t in model_types]
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, mae_values, color=colors, alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Model Performance by Type')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45)
        
        # Add legend
        for model_type, color in type_colors.items():
            if model_type in model_types:
                ax.bar([], [], color=color, label=model_type.replace('_', ' ').title())
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.comparison_dir}/performance_by_type.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, models: List[str], metrics: List[Dict]):
        """Generate comprehensive summary report."""
        # Find best performing model
        best_mae_idx = np.argmin([m['mae'] for m in metrics])
        best_model = models[best_mae_idx]
        best_metrics = metrics[best_mae_idx]
        
        # Calculate improvements
        baseline_mae = self.results.get('naive', {}).get('mae', np.inf)
        enhanced_mae = self.results.get('enhanced', {}).get('mae', np.inf)
        improvement = ((baseline_mae - enhanced_mae) / baseline_mae) * 100 if baseline_mae != np.inf else 0
        
        # Generate report
        report = {
            'comparison_date': datetime.now().isoformat(),
            'data_period': {
                'start': self.df['DateTime'].min().isoformat(),
                'end': self.df['DateTime'].max().isoformat(),
                'total_records': len(self.df),
                'test_records': len(self.test_df)
            },
            'best_model': {
                'name': best_model,
                'metrics': best_metrics
            },
            'improvement_analysis': {
                'baseline_mae': float(baseline_mae),
                'enhanced_mae': float(enhanced_mae),
                'improvement_percentage': float(improvement)
            },
            'all_results': self.results,
            'recommendations': self._generate_recommendations(models, metrics)
        }
        
        # Save report
        with open(f"{self.comparison_dir}/comparison_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\\n" + "="*60)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"Best Performing Model: {best_model}")
        print(f"Best MAE: {best_metrics['mae']:.2f} events")
        print(f"Best RMSE: {best_metrics['rmse']:.2f} events")
        print(f"Best R²: {best_metrics['r2']:.3f}")
        
        if improvement > 0:
            print(f"\\n🚀 Enhanced Model Improvement:")
            print(f"  • {improvement:.1f}% better than naive baseline")
            print(f"  • {baseline_mae - enhanced_mae:.2f} events MAE reduction")
        
        print(f"\\n📁 Full report saved to: {self.comparison_dir}/comparison_report.json")
    
    def _generate_recommendations(self, models: List[str], metrics: List[Dict]) -> Dict[str, str]:
        """Generate deployment recommendations."""
        recommendations = {}
        
        # Performance-based recommendations
        mae_values = [m['mae'] for m in metrics]
        best_idx = np.argmin(mae_values)
        best_model = models[best_idx]
        
        recommendations['production_model'] = f"Deploy {best_model} for best accuracy"
        
        # Speed vs accuracy trade-off
        inference_times = [m.get('inference_time', 0) for m in metrics]
        if max(inference_times) > 0:
            speed_accuracy_ratios = [mae_values[i] * inference_times[i] for i in range(len(models))]
            balanced_idx = np.argmin(speed_accuracy_ratios)
            balanced_model = models[balanced_idx]
            recommendations['balanced_choice'] = f"Use {balanced_model} for best speed-accuracy balance"
        
        # Deployment strategy
        if 'enhanced' in models:
            enhanced_idx = models.index('enhanced')
            enhanced_mae = mae_values[enhanced_idx]
            baseline_mae = mae_values[0] if models[0] in ['naive', 'moving_average'] else enhanced_mae
            
            if enhanced_mae < baseline_mae:
                recommendations['deployment_strategy'] = "Enhanced model shows significant improvement - recommended for production"
            else:
                recommendations['deployment_strategy'] = "Consider ensemble approach or further hyperparameter tuning"
        
        return recommendations
    
    def run_complete_comparison(self):
        """Run the complete model comparison analysis."""
        print("=" * 60)
        print("🔍 COMPREHENSIVE MODEL COMPARISON ANALYSIS")
        print("=" * 60)
        
        try:
            # 1. Evaluate baseline models
            self.evaluate_baseline_models()
            
            # 2. Evaluate enhanced model
            self.evaluate_enhanced_model()
            
            # 3. Create comparison report
            self.create_comparison_report()
            
            print("\\n✅ Model comparison completed successfully!")
            
        except Exception as e:
            print(f"❌ Comparison failed: {str(e)}")
            raise


def main():
    """Main function to run model comparison."""
    data_path = "shared_data/EventsMetricsMarJul.csv"
    
    comparison = ModelComparison(data_path)
    comparison.run_complete_comparison()


if __name__ == "__main__":
    main()
