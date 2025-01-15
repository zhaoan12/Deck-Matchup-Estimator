import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import BasicGamePredictor, AdvancedGamePredictor

class TestModels:
    def test_basic_model_forward_pass(self):
        """Test basic model forward pass"""
        model = BasicGamePredictor(input_size=15)
        x = torch.randn(32, 15)  # Batch of 32 samples
        output = model(x)
        
        assert output.shape == (32, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_advanced_model_forward_pass(self):
        """Test advanced model forward pass"""
        model = AdvancedGamePredictor(input_size=15)
        x = torch.randn(32, 15)
        output = model(x)
        
        assert output.shape == (32, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_model_parameters(self):
        """Test that models have trainable parameters"""
        basic_model = BasicGamePredictor(input_size=15)
        advanced_model = AdvancedGamePredictor(input_size=15)
        
        # Both models should have parameters
        assert len(list(basic_model.parameters())) > 0
        assert len(list(advanced_model.parameters())) > 0
        
        # Advanced model should have more parameters (deeper)
        basic_params = sum(p.numel() for p in basic_model.parameters())
        advanced_params = sum(p.numel() for p in advanced_model.parameters())
        assert advanced_params > basic_params

        