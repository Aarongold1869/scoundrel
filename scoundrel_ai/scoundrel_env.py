"""
Custom Gymnasium Environment for Scoundrel Game
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scoundrel.scoundrel import Scoundrel, UI, GameState


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
        # Basic stats (11 features):
        # - HP (0-20)
        # - HP ratio (0-1) normalized
        # - Weapon level (0-14)
        # - Weapon max monster level (0-15)
        # - Cards remaining (0-44)
        # - Progress ratio (0-1) - how far through dungeon
        # - Can avoid (0 or 1)
        # - Can heal (0 or 1)
        # - Total monster values remaining in deck (sum of card values)
        # - Total weapon values remaining in deck (sum of card values)
        # - Total health potion values remaining in deck (sum of card values)
        # 
        # Room summary (5 features):
        # - Num monsters in room (0-4)
        # - Num weapons in room (0-4)
        # - Num health potions in room (0-4)
        # - Max monster value in room (0-14)
        # - Total healing available (0-40)
        # 
        # Current room cards (4 cards x 6 features = 24):
        #   * class (0=weapon, 1=health, 2=monster)
        #   * value (2-14)
        #   * suit type (0=diamond, 1=heart, 2=spade, 3=club)
        #   * is_valid_action (1 if can interact, 0 if not)
        #   * can_defeat_with_weapon (1 if weapon can defeat without damage)
        #   * weapon_efficiency (normalized: how good this monster is for weapon use)
        # 
        # Total: 11 + 5 + 24 = 40 features
        
        self.observation_space = spaces.Box(
            low=0,
            high=300,  # Max value to accommodate sum of card values (monsters ~208, weapons/health ~54 each)
            shape=(40,),  # Extended observation space
            dtype=np.float32
        )
        
    def _get_obs(self):
        """Convert game state to observation vector"""
        # Use the game's current_game_state method
        game_state: GameState = self.game.current_game_state()
        
        obs = np.zeros(40, dtype=np.float32)
        
        # Basic state from GameState (11 features)
        obs[0] = game_state['hp']
        obs[1] = game_state['hp'] / 20.0  # Normalized HP ratio
        obs[2] = game_state['weapon_level']
        obs[3] = game_state['weapon_max_monster_level']
        obs[4] = game_state['num_cards_remaining']
        obs[5] = (44 - game_state['num_cards_remaining']) / 44.0  # Progress ratio
        obs[6] = 1.0 if game_state['can_avoid'] else 0.0
        obs[7] = 1.0 if game_state['can_heal'] else 0.0
        
        # Count remaining cards in deck by type (sum of values)
        monster_values_in_deck = 0
        weapon_values_in_deck = 0
        health_values_in_deck = 0
        
        for card in self.game.dungeon.cards:
            if card.suit['class'] == 'monster':
                monster_values_in_deck += card.val
            elif card.suit['class'] == 'weapon':
                weapon_values_in_deck += card.val
            elif card.suit['class'] == 'health':
                health_values_in_deck += card.val
        
        obs[8] = monster_values_in_deck
        obs[9] = weapon_values_in_deck
        obs[10] = health_values_in_deck
        
        # Analyze current room (5 features)
        current_room = game_state['current_room']
        num_monsters = 0
        num_weapons = 0
        num_health = 0
        max_monster_val = 0
        total_healing = 0
        
        for card in current_room:
            if card.suit['class'] == 'monster':
                num_monsters += 1
                max_monster_val = max(max_monster_val, card.val)
            elif card.suit['class'] == 'weapon':
                num_weapons += 1
            elif card.suit['class'] == 'health':
                num_health += 1
                if game_state['can_heal']:
                    total_healing += card.val
        
        obs[11] = num_monsters
        obs[12] = num_weapons
        obs[13] = num_health
        obs[14] = max_monster_val
        obs[15] = total_healing
        
        # Encode individual cards with weapon efficiency (24 features)
        class_encoding = {'weapon': 0, 'health': 1, 'monster': 2}
        suit_encoding = {'diamond': 0, 'heart': 1, 'spade': 2, 'club': 3}
        
        weapon_level = game_state['weapon_level']
        weapon_max = game_state['weapon_max_monster_level']
        
        for i in range(4):
            base_idx = 16 + (i * 6)
            if i < len(current_room):
                card = current_room[i]
                obs[base_idx] = class_encoding[card.suit['class']]
                obs[base_idx + 1] = card.val
                obs[base_idx + 2] = suit_encoding[card.suit['name']]
                obs[base_idx + 3] = 1.0  # Valid action
                
                # Weapon efficiency features for monsters
                if card.suit['class'] == 'monster' and weapon_level > 0:
                    # Can defeat without damage if weapon level >= monster value
                    can_defeat = 1.0 if card.val <= weapon_level else 0.0
                    obs[base_idx + 4] = can_defeat
                    
                    # Efficiency: higher value = better monster to use weapon on
                    # Prefer monsters close to weapon_max to minimize degradation
                    # Formula: monster_value / weapon_max (higher is better)
                    if weapon_max > 0:
                        efficiency = min(card.val / weapon_max, 1.0)
                    else:
                        efficiency = 0.0
                    obs[base_idx + 5] = efficiency
                else:
                    obs[base_idx + 4] = 0.0
                    obs[base_idx + 5] = 0.0
            else:
                # No card at this position
                obs[base_idx:base_idx + 6] = 0.0
        
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
        
        # Track state for reward shaping
        self.potions_used_in_room = 0
        self.last_room_id = id(self.game.dungeon.current_room)
        
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
        previous_weapon_level = prev_state['weapon_level']
        previous_weapon_max = prev_state['weapon_max_monster_level']
        
        # Check if we're in a new room (reset potion counter)
        current_room_id = id(self.game.dungeon.current_room)
        if current_room_id != self.last_room_id:
            self.potions_used_in_room = 0
            self.last_room_id = current_room_id
        
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
        
        action_succeeded = False
        
        if action_str is None:
            reward = -5  # Invalid action
            truncated = True  # End episode on invalid action format
        else:
            # Use the game's take_action method with error handling
            try:
                self.game.take_action(action=action_str)
                action_succeeded = True
                # Small penalty for avoiding
                if action_str == 'a':
                    reward = -0.5
            except ValueError as e:
                # Invalid action (e.g., card doesn't exist, can't avoid)
                # This is a critical failure - agent chose an illegal move
                reward = -10
                truncated = True  # End episode when agent makes invalid move
            except Exception as e:
                # Unexpected error - force terminate to prevent hanging
                reward = -15
                terminated = True
        
        # Get new state after action
        new_state = self.game.current_game_state()
        
        # Calculate reward based on game state changes
        hp_change = new_state['hp'] - previous_hp
        score_change = new_state['score'] - previous_score
        cards_change = previous_cards - new_state['num_cards_remaining']
        weapon_change = new_state['weapon_level'] - previous_weapon_level
        
        # Determine what type of card was interacted with
        card_interacted = None
        if cards_change > 0 and action_str != 'a':
            card_idx = int(action_str) - 1
            if card_idx < len(prev_state['current_room']):
                card_interacted = prev_state['current_room'][card_idx]
        
        # ===== REWARD SHAPING =====
        
        # 1. HP management
        if hp_change > 0:
            # Check if this is a potion
            if card_interacted and card_interacted.suit['class'] == 'health':
                self.potions_used_in_room += 1
                
                # PENALIZE WASTING POTIONS: Only 1 potion per room has effect
                if self.potions_used_in_room == 1:
                    reward += hp_change * 2.0  # First potion is good
                    # Bonus for healing when low HP
                    if previous_hp < 10:
                        reward += 2.0
                else:
                    # Additional potions are wasted!
                    reward -= 5.0  # Big penalty for wasting potion
            else:
                reward += hp_change * 2.0
        elif hp_change < 0:
            reward += hp_change * 1.5  # Taking damage is bad
            # Extra penalty for taking damage when low HP
            if previous_hp < 8:
                reward -= 2.0
        
        # 2. Weapon management - strategic value and waste prevention
        if weapon_change > 0:
            # Picking up a new weapon
            new_weapon_level = new_state['weapon_level']
            
            # PENALIZE WASTING WEAPONS
            if previous_weapon_level > 0:
                # Had a weapon before - check if it was used optimally
                # Weapon is "wasted" if we didn't maximize its degradation potential
                degradation_ratio = previous_weapon_max / 15.0  # How degraded was it? (15 is max)
                
                if degradation_ratio > 0.8:  # Weapon still very strong (not degraded much)
                    reward -= 4.0  # Big penalty for replacing unused weapon
                elif degradation_ratio > 0.6:
                    reward -= 2.0  # Moderate penalty
                else:
                    reward += 1.0  # Good, weapon was used before replacing
            
            # Base reward for getting weapon
            reward += new_weapon_level * 5.0  # Getting weapons is critical
            # Bonus for picking up weapons early
            if previous_cards > 30:
                reward += 3.0  # Early weapon pickup is excellent
        
        # 3. Monster fighting efficiency and weapon optimization
        if card_interacted and card_interacted.suit['class'] == 'monster':
            monster_val = card_interacted.val
            had_weapon = previous_weapon_level > 0
            weapon_max_before = previous_weapon_max
            weapon_max_after = new_state['weapon_max_monster_level']
            
            # Reward efficient monster fighting (weapons auto-used when equipped)
            if hp_change == 0:  # No damage taken
                reward += 3.0
                
                # OPTIMIZATION: Reward using weapon on HIGH-level monsters
                if had_weapon and weapon_max_after < weapon_max_before:
                    # Weapon degraded - evaluate if it was a good choice
                    degradation_ratio = monster_val / weapon_max_before if weapon_max_before > 0 else 0
                    
                    if degradation_ratio >= 0.8:  # Used on high-level monster (80%+ of max)
                        reward += 5.0  # Excellent weapon usage!
                    elif degradation_ratio >= 0.6:
                        reward += 3.0  # Good weapon usage
                    elif degradation_ratio >= 0.4:
                        reward += 1.0  # Okay weapon usage
                    else:
                        reward -= 2.0  # Bad! Wasted weapon on weak monster
                        
            elif hp_change > -3:  # Minimal damage
                reward += 1.0
        
        # 4. Strategic avoidance
        if action_str == 'a':
            # Penalty reduced if it was strategic (low HP and dangerous room)
            if previous_hp < 8:
                reward += 1.0  # Good decision to avoid when low HP
        
        # 5. Progress rewards
        if cards_change > 0:
            reward += 1.5  # Making progress is good
            # Bonus for maintaining high HP while progressing
            if new_state['hp'] >= previous_hp:
                reward += 0.5
        
        # 6. Survival bonus - reward maintaining healthy HP
        if new_state['hp'] >= 15:
            reward += 0.3  # Small constant bonus for good health
        
        # 7. PRIMARY GOAL: Maximize score (most important)
        reward += score_change * 2.0  # Increased from 0.3 to focus on score
        
        # Check if game is over - FOCUS ON SCORE, NOT SURVIVAL
        if not new_state['is_active']:
            terminated = True
            # Just use final score as primary signal
            # Positive score = good, negative score = bad
            if new_state['score'] > 0:
                reward += new_state['score'] * 1.0  # Scale final score
            else:
                reward += new_state['score'] * 0.5  # Still penalize negative score but less
        
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
