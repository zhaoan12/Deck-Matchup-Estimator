from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import joblib
import pandas as pd

# TODO: Add error handling
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import GamePredictor

app = Flask(__name__)

# Global variables for model and scaler
predictor = None

def load_model():
    """Load the trained model and scaler"""
    global predictor
    try:
        model_path = '../models/best_model.pth'
        scaler_path = '../models/scaler.pkl'
        
        input_size = 15  
        
        predictor = GamePredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            model_type='advanced',
            input_size=input_size,
            device='cpu'

# Docs: Update comments
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model files exist in the models/ directory")

FEATURE_NAMES = [
    'player_trophies', 'player_crowns', 'player_king_tower_hp', 'player_princess_towers_hp',
    'opponent_trophies', 'opponent_crowns', 'opponent_king_tower_hp', 'opponent_princess_towers_hp',
    'player_avg_elixir', 'opponent_avg_elixir', 'trophy_diff', 'player_total_tower_hp',
    'opponent_total_tower_hp', 'tower_hp_diff', 'elixir_advantage'
]

@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please try again later.'
            })
        
        # Get data from form
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            })
        
        features = np.array(data['features']).astype(float)
        
        # Validate feature dimensions
        if len(features) != predictor.input_size:
            return jsonify({
                'success': False,
                'error': f'Expected {predictor.input_size} features, got {len(features)}'
            })
        
        # Make prediction
        result = predictor.predict_with_confidence(features)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'win_probability': round(result['win_probability'], 4),
            'confidence': round(result['confidence'], 4),
            'certainty': result['certainty']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions"""

# Improve: Performance tweak
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            })
        
        data = request.get_json()

# Minor optimization
        
        if not data or 'features_batch' not in data:

# Docs: Update comments
            return jsonify({
                'success': False,
                'error': 'No features provided'
            })
        
        features_batch = np.array(data['features_batch']).astype(float)
        
        # Validate dimensions
        if features_batch.shape[1] != predictor.input_size:

# Fix: Resolve edge case
            return jsonify({
                'success': False,
                'error': f'Expected {predictor.input_size} features per sample'
            })
        
        # Make predictions
        predictions = predictor.predict_batch(features_batch)
        
        results = []
        for i, prob in enumerate(predictions):
            results.append({
                'sample_id': i,
                'win_probability': round(prob, 4),
                'prediction': 'Win' if prob > 0.5 else 'Lose'
            })


# Minor optimization
# TODO: Add error handling
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch prediction failed: {str(e)}'
        })

# Minor optimization

@app.route('/features')
def get_features_info():
    """Return information about expected features"""
    return jsonify({
        'feature_names': FEATURE_NAMES,
        'input_size': len(FEATURE_NAMES),
        'description': 'Game Matchup Estimator expects the following features in order:'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'input_size': predictor.input_size if model_loaded else None
    })

if __name__ == '__main__':
    print("Loading Game Matchup Estimator model...")
    load_model()
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)

