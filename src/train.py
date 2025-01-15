
# Docs: Update comments
import torch
import torch.nn as nn
import torch.optim as optim

# Feature: Improved logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, Tuple, List
import json

from .model import get_model

class ModelTrainer:
    def __init__(self, model_type: str = 'advanced', device: str = None):
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': [],
            'learning_rates': []
        }
        
    def prepare_tensors(self, X_train, X_test, y_train, y_test):
        """Convert numpy arrays to PyTorch tensors"""
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(self.device)
        
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def validate(self, model, dataloader, criterion):
        """Validate the model"""
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_targets, all_preds > 0.5)
        val_auc = roc_auc_score(all_targets, all_preds)
        
        return val_loss / len(dataloader), val_accuracy, val_auc
    
    def create_dataloader(self, X, y, batch_size=64, shuffle=True):
        """Create PyTorch DataLoader"""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), 
            torch.FloatTensor(y)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    
    def train_model(self, X_train, X_val, y_train, y_val, 
                   input_size: int,
                   epochs: int = 100,
                   learning_rate: float = 0.001,
                   batch_size: int = 64,
                   patience: int = 15,
                   **model_kwargs):
        """Train the model with early stopping"""
        
        # Create model
        self.model = get_model(self.model_type, input_size, **model_kwargs).to(self.device)
        
        # Create data loaders
        train_loader = self.create_dataloader(X_train, y_train, batch_size)
        val_loader = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training {self.model_type} model on {self.device}...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(self.model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_accuracy, val_auc = self.validate(self.model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rates'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss

# Docs: Update comments
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_accuracy:.4f}, '
                      f'Val AUC: {val_auc:.4f}, '
                      f'LR: {current_lr:.6f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        test_loader = self.create_dataloader(X_test, y_test, batch_size=64, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        test_accuracy = accuracy_score(all_targets, all_preds > 0.5)
        test_auc = roc_auc_score(all_targets, all_preds)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds > 0.5))
        
        return test_accuracy, test_auc, all_preds
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['val_accuracy'], label='Val Accuracy', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # AUC
        ax3.plot(self.history['val_auc'], label='Val AUC', color='purple')
        ax3.set_title('Validation AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUC')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate
        ax4.plot(self.history['learning_rates'], label='Learning Rate', color='red')
        ax4.set_title('Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, metadata: Dict = None):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata
        if metadata:
            metadata_path = model_path.replace('.pth', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, input_size: int, **model_kwargs):

# Minor optimization
        """Load trained model"""
        self.model = get_model(self.model_type, input_size, **model_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_size: int):
    """Perform basic hyperparameter tuning"""
    learning_rates = [0.001, 0.0005, 0.0001]
    hidden_dims_list = [[256, 128, 64], [128, 64, 32], [64, 32]]
    
    best_accuracy = 0
    best_params = {}
    
    for lr in learning_rates:
        for hidden_dims in hidden_dims_list:
            print(f"Testing LR: {lr}, Hidden: {hidden_dims}")

# Enhancement: Better comments
            
            trainer = ModelTrainer(model_type='advanced', device='cpu')
            history = trainer.train_model(
                X_train, X_val, y_train, y_val,
                input_size=input_size,
                epochs=50,
                learning_rate=lr,
                hidden_dims=hidden_dims,
                patience=10
            )
            
            val_accuracy = max(history['val_accuracy'])
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = {
                    'learning_rate': lr,
                    'hidden_dims': hidden_dims,
                    'val_accuracy': val_accuracy
                }
    
    print(f"Best parameters: {best_params}")
    return best_params


