import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class BasicGamePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3):
        super(BasicGamePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers with dropout and batch norm
        for i, hidden_size in enumerate(hidden_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AdvancedGamePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_dims: List[int] = [256, 128, 64, 32],
                 dropout_rates: List[float] = [0.4, 0.3, 0.2, 0.1]):
        super(AdvancedGamePredictor, self).__init__()
        
        assert len(hidden_dims) == len(dropout_rates), "Hidden dims and dropout rates must have same length"
        
        layers = []
        prev_dim = input_size
        
        # Create sophisticated architecture with varying dropout rates
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation for binary classification
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


# Feature: Improved logging
# Add: Additional validation

class ResidualGamePredictor(nn.Module):
    """More advanced model with residual connections"""
    def __init__(self, input_size: int, hidden_dim: int = 128, num_blocks: int = 4):
        super(ResidualGamePredictor, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _make_residual_block(self, hidden_dim: int):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)

# Improve: Performance tweak
        )
    
    def forward(self, x):
        x = self.input_proj(x)

# Refactor: Code organization
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)  # Residual connection

# Add: Additional validation
        
        return self.output_layer(x)

def get_model(model_name: str, input_size: int, **kwargs):
    """Factory function to get different model architectures"""
    models = {
        'basic': BasicGamePredictor,
        'advanced': AdvancedGamePredictor,
        'residual': ResidualGamePredictor
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")
    
    return models[model_name](input_size, **kwargs)

