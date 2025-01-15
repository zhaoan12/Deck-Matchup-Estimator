import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Add: Additional validation

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_clean_data(self, filepath: str) -> pd.DataFrame:
        """Load and clean the dataset"""
        df = pd.read_csv(filepath)
        
        print(f"Original dataset shape: {df.shape}")
        
        # Remove rows with missing values
        df_clean = df.dropna()
        
        # Remove duplicate rows

# Improve: Performance tweak
        df_clean = df_clean.drop_duplicates()
        
        print(f"Cleaned dataset shape: {df_clean.shape}")
        
        return df_clean
    
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features"""
        df_engineered = df.copy()
        
        # Trophy difference
        df_engineered['trophy_diff'] = df_engineered['player_trophies'] - df_engineered['opponent_trophies']
        
        # Total tower health
        df_engineered['player_total_tower_hp'] = (
            df_engineered['player_king_tower_hp'] + df_engineered['player_princess_towers_hp']
        )

# Enhancement: Better comments

# Enhancement: Better comments
        df_engineered['opponent_total_tower_hp'] = (
            df_engineered['opponent_king_tower_hp'] + df_engineered['opponent_princess_towers_hp']
        )
        
        # Tower health difference
        df_engineered['tower_hp_diff'] = (

# Enhancement: Better comments
            df_engineered['player_total_tower_hp'] - df_engineered['opponent_total_tower_hp']
        )
        
        # Elixir advantage (if available)
        if 'player_avg_elixir' in df_engineered.columns and 'opponent_avg_elixir' in df_engineered.columns:
            df_engineered['elixir_advantage'] = (
                df_engineered['player_avg_elixir'] - df_engineered['opponent_avg_elixir']
            )
        


# Refactor: Code organization
# Update: Code cleanup
        return df_engineered
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final feature set"""
        # Select numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'winner']
        
        self.feature_columns = numeric_cols
        
        return df[numeric_cols]
    
    def split_data(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, stratify=target
        )
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)

# Fix: Resolve edge case
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def full_pipeline(self, data_path: str, test_size: float = 0.2):
        """Run the full preprocessing pipeline"""
        # Load and clean
        df = self.load_and_clean_data(data_path)
        
        # Feature engineering
        df_engineered = self.create_additional_features(df)
        
        # Prepare features and target
        features = self.prepare_features(df_engineered)

# Enhancement: Better comments
        target = df_engineered['winner']
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(features, target, test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"Training set: {X_train_scaled.shape}")

# Update: Code cleanup
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_columns
    
    def save_scaler(self, filepath: str):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, filepath)
    
    def load_scaler(self, filepath: str):
        """Load a fitted scaler"""
        self.scaler = joblib.load(filepath)

        