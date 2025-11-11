# Game Matchup Estimator

A machine learning project that predicts Clash Royale game outcomes using neural networks.

## Project Overview
Predicting Clash Royale match outcomes with machine learning.

### Features
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Layers**: Prevents overfitting (varying rates per layer)
- **LeakyReLU Activation**: Prevents dying ReLU problem
- **L2 Regularization**: Weight decay of 1e-5
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor 0.5

## Training Details

### Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (adaptive)
- **Batch Size**: 64
- **Epochs**: 100 (with early stopping)
- **Loss Function**: Binary Cross Entropy (BCELoss)
- **Early Stopping**: Patience of 15 epochs

### Data Split
- **Training Set**: 80% (120,000 samples)
- **Validation Set**: 10% (15,000 samples)
- **Test Set**: 10% (15,000 samples)

### Performance Metrics
- **Accuracy**: 74%
- **AUC Score**: 0.81
- **Precision**: 0.73
- **Recall**: 0.75
- **F1-Score**: 0.74

## Feature Engineering

### Original Features
- Player and opponent trophy counts
- Tower health points
- Crown counts
- Deck elixir costs

### Engineered Features
- Trophy difference
- Total tower health
- Tower health difference
- Elixir advantage

### Preprocessing
- **Missing Values**: Dropped (minimal in dataset)
- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Data Split**: Stratified to maintain class distribution

## Model Selection

We experimented with three architectures:

1. **Basic Model**: 3 hidden layers, constant dropout
2. **Advanced Model**: 4 hidden layers, varying dropout (selected)
3. **Residual Model**: Residual connections (overkill for this problem)

The Advanced Model provided the best balance of performance and training efficiency.

## Regularization Techniques

1. **Dropout**: Layer-wise dropout rates (0.4 → 0.1)
2. **L2 Regularization**: Weight decay in optimizer
3. **Batch Normalization**: Per-layer normalization
4. **Early Stopping**: Prevents overfitting

## Deployment

The model is deployed as:
- Python package for local use
- Flask web application for API access
- Pre-trained weights for immediate inference



### Setup

   ```bash
   git clone https://github.com/game-matchup-estimator.git
   cd game-matchup-estimator
   
   pip install -r requirements.txt
   cp .env.example .env
   ```



