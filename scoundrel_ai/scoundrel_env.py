"""
Custom Gymnasium Environment for Scoundrel Game
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scoundrel.scoundrel import Scoundrel, UI, Card, GameState


class ScoundrelEnv(gym.Env):
    """Custom Environment that follows gym interface for Scoundrel game"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.game = None
        
        # Define action space
        # Actions: 0-3 = interact with card 1-4, 4 = avoid room
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        # We need to encode the game state:
        # - HP (0-20)
        # - Weapon level (0-14)
        # - Weapon max monster level (0-15)
        # - Cards remaining (0-44)
        # - Can avoid (0 or 1)
        # - Can heal (0 or 1)
        # - Current room (4 cards): each card has suit class (3 values), symbol, and value
        # 
        # For simplicity, we'll encode current room as 4 * 3 = 12 values:
        # [card1_class, card1_value, card1_suit_type, card2_class, ...]
        # Where: class encoding: 0=weapon, 1=health, 2=monster
        #        suit_type: 0=diamond, 1=heart, 2=spade, 3=club
        
        self.observation_space = spaces.Box(
            low=0,
            high=44,  # Max value in observation
            shape=(18,),  # HP, weapon_lvl, weapon_max, cards_left, can_avoid, can_heal + 4*3 for cards
            dtype=np.float32
        )
        
    def _get_obs(self):
        """Convert game state to observation vector"""
        # Use the game's current_game_state method
        game_state = self.game.current_game_state()
        
        obs = np.zeros(18, dtype=np.float32)
        
        # Basic state from GameState
        obs[0] = game_state['hp']
        obs[1] = game_state['weapon_level']
        obs[2] = game_state['weapon_max_monster_level']
        obs[3] = game_state['num_cards_remaining']
        obs[4] = 1.0 if game_state['can_avoid'] else 0.0
        obs[5] = 1.0 if game_state['can_heal'] else 0.0
        
        # Encode current room (up to 4 cards)
        class_encoding = {'weapon': 0, 'health': 1, 'monster': 2}
        suit_encoding = {'diamond': 0, 'heart': 1, 'spade': 2, 'club': 3}
        
        for i, card in enumerate(game_state['current_room'][:4]):
            base_idx = 6 + (i * 3)
            obs[base_idx] = class_encoding[card.suit['class']]
            obs[base_idx + 1] = card.val
            obs[base_idx + 2] = suit_encoding[card.suit['name']]
        
        return obs
    
    def _get_info(self):
        """Return auxiliary information"""
        game_state = self.game.current_game_state()
        return {
            "score": game_state['score'],
            "hp": game_state['hp'],
            "cards_remaining": game_state['num_cards_remaining'],
            "weapon_level": game_state['weapon_level'],
            "is_active": game_state['is_active']
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize a new game
        self.game = Scoundrel(ui=UI.API)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Get previous state using current_game_state
        prev_state = self.game.current_game_state()
        previous_score = prev_state['score']
        previous_hp = prev_state['hp']
        previous_cards = prev_state['num_cards_remaining']
        
        terminated = False
        truncated = False
        reward = 0
        
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        
        # Map numeric action to string action for take_action method
        # Actions: 0-3 = interact with card 1-4, 4 = avoid room
        action_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'a'}
        action_str = action_map.get(action)
        
        if action_str is None:
            reward = -5  # Invalid action
        else:
            # Use the game's take_action method with error handling
            try:
                self.game.take_action(action=action_str)
                # Small penalty for avoiding
                if action_str == 'a':
                    reward = -0.5
            except ValueError as e:
                # Invalid action (e.g., card doesn't exist, can't avoid)
                reward = -5
            except Exception as e:
                # Unexpected error
                reward = -5
        
        # Get new state after action
        new_state = self.game.current_game_state()
        
        # Calculate reward based on game state changes
        hp_change = new_state['hp'] - previous_hp
        score_change = new_state['score'] - previous_score
        cards_change = previous_cards - new_state['num_cards_remaining']
        
        # Reward shaping
        reward += hp_change * 0.5  # Gaining HP is good
        reward += score_change * 0.1  # Score improvement is good
        reward += cards_change * 0.2  # Progress through dungeon
        
        # Check if game is over using is_active flag
        if not new_state['is_active']:
            terminated = True
            if new_state['hp'] <= 0:
                reward -= 20  # Big penalty for dying
            elif new_state['num_cards_remaining'] == 0:
                if new_state['score'] > 0:
                    reward += 50  # Big reward for winning
                else:
                    reward += 10  # Smaller reward for surviving
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            game_state = self.game.current_game_state()
            print(f"\n{'='*50}")
            print(f"HP: {game_state['hp']} | Score: {game_state['score']} | Weapon: {game_state['weapon_level']} ({game_state['weapon_max_monster_level']})")
            print(f"Cards remaining: {game_state['num_cards_remaining']}")
            print(f"Can avoid: {game_state['can_avoid']} | Can heal: {game_state['can_heal']}")
            print(f"\nCurrent Room:")
            for i, card in enumerate(game_state['current_room']):
                print(f"  [{i}] {card.symbol}{card.suit['symbol']} (val: {card.val}, class: {card.suit['class']})")
            print(f"{'='*50}\n")
    
    def close(self):
        """Cleanup"""
        pass


# Register the environment
gym.register(
    id='Scoundrel-v0',
    entry_point='scoundrel_ai.scoundrel_env:ScoundrelEnv',
    max_episode_steps=500,
)
