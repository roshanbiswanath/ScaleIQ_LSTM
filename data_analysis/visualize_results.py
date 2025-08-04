import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(improved=True):
    """Load model results and scaler"""
    prefix = 'improved_' if improved else ''
    
    try:
        predictions = np.load(f'{prefix}predictions.npy')
        targets = np.load(f'{prefix}targets.npy')
        uncertainties = np.load(f'{prefix}uncertainties.npy')
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load training history if available
        try:
            train_losses = np.load(f'{prefix}train_losses.npy')
            val_losses = np.load(f'{prefix}val_losses.npy')
        except FileNotFoundError:
            train_losses = val_losses = None
        
        return predictions, targets, uncertainties, scaler, train_losses, val_losses
    
    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
        print("Please run the training script first to generate results.")
        return None, None, None, None, None, None

def calculate_comprehensive_metrics(predictions, targets, uncertainties=None):
    """Calculate comprehensive evaluation metrics"""
    
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Overall metrics
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
    mape = np.mean(np.abs((target_flat - pred_flat) / (target_flat + 1e-8))) * 100
    
    # Correlation
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # R-squared
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Directional accuracy (for time series)
    target_diff = np.diff(target_flat)
    pred_diff = np.diff(pred_flat)
    directional_accuracy = np.mean(np.sign(target_diff) == np.sign(pred_diff)) * 100
    
    # Per-step metrics
    step_metrics = []
    for i in range(predictions.shape[1]):
        step_mae = mean_absolute_error(targets[:, i], predictions[:, i])
        step_rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
        step_mape = np.mean(np.abs((targets[:, i] - predictions[:, i]) / (targets[:, i] + 1e-8))) * 100
        step_corr = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
        
        step_metrics.append({
            'step': i + 1,
            'mae': step_mae,
            'rmse': step_rmse,
            'mape': step_mape,
            'correlation': step_corr
        })
    
    return {
        'overall': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        },
        'per_step': step_metrics
    }

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    if train_losses is None or val_losses is None:
        print("Training history not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = np.array(val_losses) - np.array(train_losses)
    axes[1].plot(loss_diff, label='Val - Train Loss', linewidth=2, color='orange')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Difference')
    axes[1].set_title('Overfitting Monitor')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_sample_forecasts(predictions, targets, uncertainties, n_samples=12, start_idx=0):
    """Plot multiple sample forecasts for better visualization"""
    
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    time_steps = range(1, predictions.shape[1] + 1)
    
    for i in range(n_samples):
        if start_idx + i >= len(predictions):
            break
            
        sample_idx = start_idx + i
        ax = axes[i]
        
        # Plot actual vs predicted
        ax.plot(time_steps, targets[sample_idx], 'b-', label='Actual', 
                linewidth=3, marker='o', markersize=6)
        ax.plot(time_steps, predictions[sample_idx], 'r--', label='Predicted', 
                linewidth=3, marker='s', markersize=6)
        
        # Add uncertainty bands if available
        if uncertainties is not None:
            ax.fill_between(time_steps, 
                           predictions[sample_idx] - uncertainties[sample_idx],
                           predictions[sample_idx] + uncertainties[sample_idx],
                           alpha=0.2, color='red', label='Uncertainty')
        
        # Calculate sample metrics
        sample_mae = mean_absolute_error(targets[sample_idx], predictions[sample_idx])
        sample_mape = np.mean(np.abs((targets[sample_idx] - predictions[sample_idx]) / 
                                   (targets[sample_idx] + 1e-8))) * 100
        
        ax.set_xlabel('Time Steps (2-min intervals ahead)')
        ax.set_ylabel('Event Count')
        ax.set_title(f'Sample {sample_idx + 1}: MAE={sample_mae:.0f}, MAPE={sample_mape:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to show full range
        y_min = min(np.min(targets[sample_idx]), np.min(predictions[sample_idx]))
        y_max = max(np.max(targets[sample_idx]), np.max(predictions[sample_idx]))
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_error_analysis(predictions, targets):
    """Comprehensive error analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Calculate errors
    errors = predictions - targets
    relative_errors = errors / (targets + 1e-8) * 100
    
    # 1. Error distribution
    axes[0, 0].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Prediction Error (events)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Relative error distribution
    axes[0, 1].hist(relative_errors.flatten(), bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Relative Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted scatter
    sample_size = min(10000, len(predictions.flatten()))
    idx = np.random.choice(len(predictions.flatten()), sample_size, replace=False)
    
    axes[0, 2].scatter(targets.flatten()[idx], predictions.flatten()[idx], 
                      alpha=0.5, s=1)
    
    # Perfect prediction line
    min_val = min(np.min(targets.flatten()[idx]), np.min(predictions.flatten()[idx]))
    max_val = max(np.max(targets.flatten()[idx]), np.max(predictions.flatten()[idx]))
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    axes[0, 2].set_xlabel('Actual Events')
    axes[0, 2].set_ylabel('Predicted Events')
    axes[0, 2].set_title('Actual vs Predicted')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Error vs actual values
    axes[1, 0].scatter(targets.flatten()[idx], errors.flatten()[idx], alpha=0.5, s=1)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Actual Events')
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Error vs Actual Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Error by forecast step
    step_errors = [np.abs(errors[:, i]) for i in range(errors.shape[1])]
    axes[1, 1].boxplot(step_errors, labels=[f'{i+1}' for i in range(len(step_errors))])
    axes[1, 1].set_xlabel('Forecast Step (2-min intervals)')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error by Forecast Horizon')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Q-Q plot for normality
    sample_errors = errors.flatten()[idx]
    stats.probplot(sample_errors, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot: Error Normality')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_time_series_analysis(predictions, targets, sample_length=500):
    """Plot time series analysis for continuous sequences"""
    
    # Take a continuous sequence
    if len(predictions) > sample_length:
        start_idx = np.random.randint(0, len(predictions) - sample_length)
        pred_sample = predictions[start_idx:start_idx + sample_length]
        target_sample = targets[start_idx:start_idx + sample_length]
    else:
        pred_sample = predictions
        target_sample = targets
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    
    # 1. Full time series comparison
    time_indices = range(len(pred_sample))
    
    # Plot first forecast step only for clarity
    axes[0].plot(time_indices, target_sample[:, 0], 'b-', label='Actual (1-step ahead)', 
                linewidth=2, alpha=0.8)
    axes[0].plot(time_indices, pred_sample[:, 0], 'r--', label='Predicted (1-step ahead)', 
                linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Event Count')
    axes[0].set_title('Time Series: Actual vs Predicted (1-step ahead)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals over time
    residuals = target_sample[:, 0] - pred_sample[:, 0]
    axes[1].plot(time_indices, residuals, 'g-', linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].fill_between(time_indices, residuals, alpha=0.3, color='green')
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('Residual (Actual - Predicted)')
    axes[1].set_title('Residuals Over Time')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Rolling metrics
    window_size = 50
    rolling_mae = []
    rolling_mape = []
    
    for i in range(window_size, len(pred_sample)):
        window_target = target_sample[i-window_size:i, 0]
        window_pred = pred_sample[i-window_size:i, 0]
        
        mae = mean_absolute_error(window_target, window_pred)
        mape = np.mean(np.abs((window_target - window_pred) / (window_target + 1e-8))) * 100
        
        rolling_mae.append(mae)
        rolling_mape.append(mape)
    
    rolling_indices = range(window_size, len(pred_sample))
    
    ax3_twin = axes[2].twinx()
    
    line1 = axes[2].plot(rolling_indices, rolling_mae, 'b-', linewidth=2, label='Rolling MAE')
    line2 = ax3_twin.plot(rolling_indices, rolling_mape, 'r-', linewidth=2, label='Rolling MAPE (%)')
    
    axes[2].set_xlabel('Time Index')
    axes[2].set_ylabel('MAE', color='blue')
    ax3_twin.set_ylabel('MAPE (%)', color='red')
    axes[2].set_title(f'Rolling Performance Metrics (window={window_size})')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='upper right')
    
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_forecast_horizon_analysis(predictions, targets):
    """Analyze performance across different forecast horizons"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    n_steps = predictions.shape[1]
    step_labels = [f'{i+1} ({(i+1)*2} min)' for i in range(n_steps)]
    
    # Calculate metrics for each step
    step_maes = []
    step_rmses = []
    step_mapes = []
    step_corrs = []
    
    for i in range(n_steps):
        mae = mean_absolute_error(targets[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
        mape = np.mean(np.abs((targets[:, i] - predictions[:, i]) / (targets[:, i] + 1e-8))) * 100
        corr = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
        
        step_maes.append(mae)
        step_rmses.append(rmse)
        step_mapes.append(mape)
        step_corrs.append(corr)
    
    # 1. MAE by step
    axes[0, 0].bar(range(n_steps), step_maes, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Forecast Step')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('MAE by Forecast Horizon')
    axes[0, 0].set_xticks(range(n_steps))
    axes[0, 0].set_xticklabels(step_labels, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE by step
    axes[0, 1].bar(range(n_steps), step_rmses, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Forecast Step')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE by Forecast Horizon')
    axes[0, 1].set_xticks(range(n_steps))
    axes[0, 1].set_xticklabels(step_labels, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MAPE by step
    axes[1, 0].bar(range(n_steps), step_mapes, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Forecast Step')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].set_title('MAPE by Forecast Horizon')
    axes[1, 0].set_xticks(range(n_steps))
    axes[1, 0].set_xticklabels(step_labels, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Correlation by step
    axes[1, 1].bar(range(n_steps), step_corrs, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('Forecast Step')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation by Forecast Horizon')
    axes[1, 1].set_xticks(range(n_steps))
    axes[1, 1].set_xticklabels(step_labels, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_comprehensive_report(improved=True):
    """Create comprehensive visualization report"""
    
    print("=== LOADING RESULTS ===")
    predictions, targets, uncertainties, scaler, train_losses, val_losses = load_results(improved)
    
    if predictions is None:
        return
    
    print(f"Loaded {len(predictions):,} test samples")
    print(f"Forecast horizon: {predictions.shape[1]} steps")
    print(f"Uncertainty estimates: {'Available' if uncertainties is not None else 'Not available'}")
    
    # Calculate comprehensive metrics
    print("\n=== CALCULATING METRICS ===")
    metrics = calculate_comprehensive_metrics(predictions, targets, uncertainties)
    
    # Print metrics
    print("\n=== COMPREHENSIVE EVALUATION REPORT ===")
    print(f"Overall Performance:")
    print(f"  MAE: {metrics['overall']['mae']:.1f} events")
    print(f"  RMSE: {metrics['overall']['rmse']:.1f} events")
    print(f"  MAPE: {metrics['overall']['mape']:.1f}%")
    print(f"  Correlation: {metrics['overall']['correlation']:.3f}")
    print(f"  R²: {metrics['overall']['r2']:.3f}")
    print(f"  Directional Accuracy: {metrics['overall']['directional_accuracy']:.1f}%")
    
    print(f"\nPer-Step Performance:")
    for step_metric in metrics['per_step']:
        print(f"  Step {step_metric['step']} ({step_metric['step']*2} min): "
              f"MAE={step_metric['mae']:.1f}, "
              f"RMSE={step_metric['rmse']:.1f}, "
              f"MAPE={step_metric['mape']:.1f}%, "
              f"Corr={step_metric['correlation']:.3f}")
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    
    figures = []
    
    # 1. Training history
    if train_losses is not None:
        print("Creating training history plots...")
        fig1 = plot_training_history(train_losses, val_losses)
        figures.append(('training_history', fig1))
    
    # 2. Sample forecasts (larger sample)
    print("Creating sample forecast plots...")
    fig2 = plot_sample_forecasts(predictions, targets, uncertainties, n_samples=12, start_idx=0)
    figures.append(('sample_forecasts_1', fig2))
    
    # Additional sample sets
    if len(predictions) > 50:
        fig3 = plot_sample_forecasts(predictions, targets, uncertainties, n_samples=12, start_idx=25)
        figures.append(('sample_forecasts_2', fig3))
    
    # 3. Error analysis
    print("Creating error analysis plots...")
    fig4 = plot_error_analysis(predictions, targets)
    figures.append(('error_analysis', fig4))
    
    # 4. Time series analysis
    print("Creating time series analysis...")
    fig5 = plot_time_series_analysis(predictions, targets)
    figures.append(('time_series_analysis', fig5))
    
    # 5. Forecast horizon analysis
    print("Creating forecast horizon analysis...")
    fig6 = plot_forecast_horizon_analysis(predictions, targets)
    figures.append(('forecast_horizon_analysis', fig6))
    
    # Save all figures
    print("\n=== SAVING FIGURES ===")
    prefix = 'improved_' if improved else ''
    
    for name, fig in figures:
        filename = f'{prefix}{name}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Show plots
    plt.show()
    
    print(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"✅ {len(figures)} visualization sets created")
    print(f"✅ All plots saved with '{prefix}' prefix")
    
    return metrics, figures

if __name__ == "__main__":
    print("=== EVENT FORECASTING MODEL VISUALIZATION ===")
    print()
    
    # Check which results are available
    improved_available = False
    basic_available = False
    
    try:
        np.load('improved_predictions.npy')
        improved_available = True
        print("✅ Improved model results found")
    except FileNotFoundError:
        print("❌ Improved model results not found")
    
    try:
        np.load('predictions.npy')
        basic_available = True
        print("✅ Basic model results found")
    except FileNotFoundError:
        print("❌ Basic model results not found")
    
    if not improved_available and not basic_available:
        print("\n❌ No model results found. Please run a training script first.")
        exit(1)
    
    # Choose which model to visualize
    if improved_available:
        print("\n📊 Creating visualization report for IMPROVED model...")
        create_comprehensive_report(improved=True)
    elif basic_available:
        print("\n📊 Creating visualization report for BASIC model...")
        create_comprehensive_report(improved=False)
