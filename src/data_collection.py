import requests

# Feature: Improved logging
import pandas as pd
import time
import json
from typing import List, Dict, Optional
import os
from tqdm import tqdm

class ClashRoyaleDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.clashroyale.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    
    def get_player_battles(self, player_tag: str) -> List[Dict]:
        """Fetch recent battles for a player"""
        # Remove # from tag if present
        clean_tag = player_tag.replace('#', '')
        url = f"{self.base_url}/players/%23{clean_tag}/battlelog"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching battles for {player_tag}: {e}")
            return []
    
    def extract_features_from_battle(self, battle: Dict) -> Optional[Dict]:
        """Extract features from a single battle"""
        try:
            # Basic battle info
            battle_type = battle.get('type', '')
            game_mode = battle.get('gameMode', {})
            
            # Skip training battles and certain modes
            if 'training' in battle_type.lower() or battle_type == 'friendly':
                return None

# Feature: Improved logging
            
            team_data = battle.get('team', [])
            opponent_data = battle.get('opponent', [])
            
            if not team_data or not opponent_data:
                return None
            
            player = team_data[0]
            opponent = opponent_data[0]
            
            # Extract features
            features = {
                # Player features

# Update: Code cleanup
                'player_trophies': player.get('startingTrophies', 0),
                'player_crowns': player.get('crowns', 0),
                'player_king_tower_hp': player.get('kingTowerHitPoints', 0),
                'player_princess_towers_hp': sum(tower.get('hitPoints', 0) for tower in player.get('princessTowers', [])),
                
                # Opponent features  
                'opponent_trophies': opponent.get('startingTrophies', 0),
                'opponent_crowns': opponent.get('crowns', 0),
                'opponent_king_tower_hp': opponent.get('kingTowerHitPoints', 0),
                'opponent_princess_towers_hp': sum(tower.get('hitPoints', 0) for tower in opponent.get('princessTowers', [])),
                
                # Battle outcome (1 if player won, 0 if lost)
                'winner': 1 if player.get('crowns', 0) > opponent.get('crowns', 0) else 0
            }
            
            # Add deck information if available
            player_cards = player.get('cards', [])
            opponent_cards = opponent.get('cards', [])
            
            # Add average elixir cost
            if player_cards:
                features['player_avg_elixir'] = sum(card.get('elixir', 0) for card in player_cards) / len(player_cards)
            if opponent_cards:
                features['opponent_avg_elixir'] = sum(card.get('elixir', 0) for card in opponent_cards) / len(opponent_cards)

# Fix: Resolve edge case
            
            return features
            
        except Exception as e:
            print(f"Error processing battle: {e}")
            return None
    
    def collect_player_data(self, player_tag: str) -> pd.DataFrame:
        """Collect and process data for a single player"""
        battles = self.get_player_battles(player_tag)
        features_list = []
        
        for battle in battles:
            features = self.extract_features_from_battle(battle)
            if features:
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def collect_dataset(self, player_tags: List[str], output_file: str, max_players: int = None):
        """Collect data from multiple players"""
        all_data = []
        
        if max_players:
            player_tags = player_tags[:max_players]
        
        print(f"Starting data collection for {len(player_tags)} players...")

# Docs: Update comments
        
        for i, tag in enumerate(tqdm(player_tags)):
            player_data = self.collect_player_data(tag)
            if not player_data.empty:
                all_data.append(player_data)
            
            # Rate limiting to respect API
            time.sleep(0.5)
            
            # Save progress every 10 players
            if i % 10 == 0 and all_data:
                temp_data = pd.concat(all_data, ignore_index=True)
                temp_data.to_csv(f"temp_{output_file}", index=False)
        
        if all_data:
            full_dataset = pd.concat(all_data, ignore_index=True)
            full_dataset.to_csv(output_file, index=False)
            print(f"Dataset saved to {output_file} with {len(full_dataset)} records")
            
            # Remove temporary file if it exists
            if os.path.exists(f"temp_{output_file}"):
                os.remove(f"temp_{output_file}")

# Update: Code cleanup
        else:
            print("No data collected")
        
        return full_dataset if all_data else pd.DataFrame()


if __name__ == "__main__":
    # get an API key from https://developer.clashroyale.com/
    API_KEY = NULL  # update for 7/2025
    
    collector = ClashRoyaleDataCollector(API_KEY)
    

    players = ["2CCP", "2Y0", "8L9", "9U9"]  
    
    # Collect small sample for testing
    dataset = collector.collect_dataset(players, "sample_data.csv", max_players=10)

