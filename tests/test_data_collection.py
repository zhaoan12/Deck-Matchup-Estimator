import pytest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection import ClashRoyaleDataCollector

class TestDataCollection:
    def test_collector_initialization(self):
        """Test that data collector initializes correctly"""
        collector = ClashRoyaleDataCollector(api_key="test_key")
        assert collector.api_key == "test_key"
        assert collector.base_url == "https://api.clashroyale.com/v1"
    
    def test_feature_extraction_structure(self, sample_battle_data):
        """Test feature extraction returns correct structure"""
        collector = ClashRoyaleDataCollector(api_key="test_key")
        features = collector.extract_features_from_battle(sample_battle_data)
        
        if features is not None:
            expected_features = [
                'player_trophies', 'player_crowns', 'player_king_tower_hp',
                'player_princess_towers_hp', 'opponent_trophies', 'opponent_crowns',
                'opponent_king_tower_hp', 'opponent_princess_towers_hp', 'winner'
            ]
            
            for feature in expected_features:
                assert feature in features

@pytest.fixture
def sample_battle_data():
    """Sample battle data for testing"""
    return {
        'type': 'PvP',
        'gameMode': {'name': 'Ladder'},
        'team': [{
            'startingTrophies': 5000,
            'crowns': 2,
            'kingTowerHitPoints': 6000,
            'princessTowers': [{'hitPoints': 2000}, {'hitPoints': 2000}],
            'cards': [{'elixir': 3}, {'elixir': 4}]
        }],
        'opponent': [{
            'startingTrophies': 4950,
            'crowns': 1,
            'kingTowerHitPoints': 6000,
            'princessTowers': [{'hitPoints': 2000}, {'hitPoints': 2000}],
            'cards': [{'elixir': 4}, {'elixir': 4}]
        }]
    }