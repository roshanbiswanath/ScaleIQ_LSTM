# Basic LSTM Forecasting Model

## Overview

This folder contains the foundational LSTM-based forecasting model for event processing optimization. It serves as the baseline implementation using a straightforward architecture to predict queue processing requirements.

## 📁 Files

- `train_forecasting_model.py` - Main training script with LSTM implementation
- `best_forecasting_model.pth` - Trained model weights
- `README.md` - This documentation

## 🏗️ Architecture

### EventQueueForecaster
- **Input Layer**: Configurable sequence length (default: 30 time steps)
- **LSTM Layers**: 2 layers with 128 hidden units each
- **Output Layer**: Linear layer predicting next time step
- **Activation**: ReLU for hidden layers, linear for output

### Key Features
- **Sequence Learning**: Learns patterns from 30 previous time steps
- **Dropout Regularization**: 20% dropout to prevent overfitting
- **Adam Optimizer**: Learning rate 0.001 with weight decay
- **MSE Loss**: Standard regression loss function

## 📊 Data Processing

### EventDataset Class
- Handles sliding window approach for sequence generation
- Automatically scales features using StandardScaler
- Splits data into train/validation/test sets (70/15/15)

### Preprocessing Pipeline
1. Load raw event data from CSV
2. Create datetime features (hour, day of week, month)
3. Apply standard scaling to normalize features
4. Generate sequences for LSTM input

## 🚀 Usage

### Training
```python
python train_forecasting_model.py
```

### Key Parameters
- `sequence_length = 30` - Number of time steps to look back
- `hidden_size = 128` - LSTM hidden dimension
- `num_layers = 2` - Number of LSTM layers
- `batch_size = 64` - Training batch size
- `epochs = 50` - Training epochs

## 📈 Performance

### Training Results
- **Final Training Loss**: ~0.15-0.20 (scaled values)
- **Validation Loss**: ~0.18-0.25 (scaled values)
- **Training Time**: ~10-15 minutes on GPU
- **Model Size**: ~500K parameters

### Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Visual comparison plots

## 🔍 Model Behavior

### Strengths
- ✅ Stable training and convergence
- ✅ Good baseline performance
- ✅ Handles general trends well
- ✅ Simple and interpretable

### Limitations
- ❌ Struggles with sudden spikes/peaks
- ❌ Limited context window (30 steps)
- ❌ No attention mechanism
- ❌ Single-scale temporal modeling

## 🛠️ Technical Details

### Dependencies
```python
torch>=2.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
```

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 4GB+ VRAM
- **Storage**: ~100MB for model and data

## 🔄 Next Steps

This basic model serves as the foundation for more advanced approaches:

1. **02_improved_model**: Enhanced architecture with attention
2. **03_tiered_approach**: Multi-scale forecasting
3. **04_lightning_framework**: Production-ready implementation

## 🐛 Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch_size to 32 or 16
- **Slow training**: Ensure GPU is available and utilized
- **Poor convergence**: Check data preprocessing and scaling

### Model Loading
```python
import torch
model = torch.load('best_forecasting_model.pth')
model.eval()
```

## 📊 Results Analysis

The model generates predictions that can be visualized and analyzed. Key outputs include:
- Predicted vs actual queue lengths
- Training/validation loss curves
- Error distribution analysis
- Time series prediction plots

For detailed analysis, see the visualization tools in `../data_analysis/`.
