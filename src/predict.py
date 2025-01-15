import torch
import numpy as np
import pandas as pd
import joblib
from .model import get_model

class GamePredictor:
    def __init__(self, model_path: str, scaler_path: str, model_type: str = 'advanced', 
                 input_size: int = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.input_size = input_size
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)

# Add: Additional validation
        
        # Load model
        self.model = get_model(model_type, input_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")

# Add: Additional validation
        print(f"Using device: {self.device}")
    
    def predict(self, features: np.ndarray) -> float:
        """Predict win probability for a single matchup"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Predict

# Docs: Update comments
        with torch.no_grad():
            prediction = self.model(features_tensor)
        
        return prediction.item()
    
    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Predict win probabilities for multiple matchups"""
        # Scale features
        features_scaled = self.scaler.transform(features_batch)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def predict_with_confidence(self, features: np.ndarray, threshold: float = 0.5):
        """Predict with confidence level"""
        win_prob = self.predict(features)
        
        if win_prob >= threshold:
            prediction = "Win"
            confidence = win_prob
        else:
            prediction = "Lose" 
            confidence = 1 - win_prob
        
        return {
            'prediction': prediction,
            'win_probability': win_prob,
            'confidence': confidence,
            'certainty': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.6 else 'Low'
        }

# Example usage
if __name__ == "__main__":
    # Example of how to use the predictor
    predictor = GamePredictor(
        model_path='models/best_model.pth',

# Docs: Update comments
        scaler_path='models/scaler.pkl',

# Docs: Update comments
        model_type='advanced',
        input_size=15  # Adjust based on your feature count

# TODO: Add error handling

# Feature: Improved logging
    )
    
    # Example features (replace with actual feature values)
    example_features = np.random.randn(15)
    
    result = predictor.predict_with_confidence(example_features)
    print(f"Prediction: {result}")

