"""
Custom Gymnasium Environment for Scoundrel Game
"""
from scoundrel.scoundrel import Scoundrel, UI, GameState, Card

import gymnasium as gym
from gymnasium import spaces

import numpy as np
from typing import List


class ScoundrelEnv(gym.Env):
    """Custom Environment that follows gym interface for Scoundrel game"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, eval: bool = False):
        super().__init__()
        
        self.render_mode = render_mode
        self.eval = eval
        self.game = None
        
        # Define action space
        # Actions: 0-3 = interact with card 1-4, 4 = avoid room
        self.action_space = spaces.Discrete(5)

        player_space = spaces.Dict({
            "hp": spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32),
            "weapon_level": spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            "weapon_max_monster_level": spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),
        })

        card_space = spaces.Box(
            low=np.tile([0, 0], (4, 1)).astype(np.float32),
            high=np.tile([3, 14], (4, 1)).astype(np.float32),
            dtype=np.float32
        )
        
        room_context_space = spaces.Dict({
            "cards_remaining": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
            "can_heal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "can_avoid": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "monster_threat_ratio": spaces.Box(low=0, high=54, shape=(1,), dtype=np.float32),
            "hp_after_room": spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32),
            "can_survive_room": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        deck_context_space = spaces.Dict({
            "cards_remaining": spaces.Box(low=0, high=44, shape=(1,), dtype=np.float32),
            "monsters_remaining": spaces.Box(low=0, high=26, shape=(1,), dtype=np.float32),
            "monster_strength_remaining": spaces.Box(low=0, high=208, shape=(1,), dtype=np.float32),
            "weapons_remaining": spaces.Box(low=0, high=13, shape=(1,), dtype=np.float32),
            "weapon_strength_remaining": spaces.Box(low=0, high=54, shape=(1,), dtype=np.float32),
            "potions_remaining": spaces.Box(low=0, high=13, shape=(1,), dtype=np.float32),
            "potion_strength_remaining": spaces.Box(low=0, high=54, shape=(1,), dtype=np.float32),
        })

        self.observation_space = spaces.Dict({
            "player": player_space,
            "room_cards": card_space,
            "room": room_context_space,
            "deck": deck_context_space,
        })


    def _encode_room_cards(self, cards: List[Card]) -> np.ndarray:
        """
        Returns a (4, 2) array:
        [
        [type_id, value],
        ...
        ]
        type_id:
        0 = empty / resolved
        1 = monster
        2 = weapon
        3 = potion
        """
        CARD_TYPE_MAP = {
            "monster": 1,
            "weapon":  2,
            "health":  3,
        }

        encoded = np.zeros((4, 2), dtype=np.float32)

        for i in range(4):
            if i >= len(cards):
                # Should not happen often, but keeps shape stable
                continue

            card = cards[i]
            encoded[i, 0] = CARD_TYPE_MAP[card.suit['class']]
            encoded[i, 1] = card.val

        return encoded
        
    def _get_obs(self):
        """Convert game state to observation space"""
        # Use the game's current_game_state method
        game_state: GameState = self.game.current_game_state()
        hp = game_state['player_state']['hp']

        # Threat Ratio
        room_cards = game_state['room_state']['cards']
        monster_cards = [card.val for card in room_cards if card.suit['class'] == 'monster']
        monster_strength_room = sum(monster_cards)
        threat_ratio = 0
        if hp > 0:
            threat_ratio = monster_strength_room / hp

        # hp after // survivability
        monster_cards.sort()
        weakest_monsters_room = monster_cards[:3]
        if len(weakest_monsters_room) == 0:
            hp_after_room = hp
            can_survive_room = 1 
        else:
            equipped_weapon_level = game_state['player_state']['weapon_level']
            equipped_weapon_max_monster_level = game_state['player_state']['weapon_max_monster_level']
            
            strongest_weapon_level = 0
            weapon_cards = [card.val for card in room_cards if card.suit['class'] == 'weapon']
            if len(weapon_cards) > 0:
                strongest_weapon_level = max(weapon_cards)

            if equipped_weapon_max_monster_level > max(weakest_monsters_room):
                strongest_weapon_level = max(strongest_weapon_level, equipped_weapon_level)

            monster_damage = sum([mc_val - strongest_weapon_level for mc_val in weakest_monsters_room])
            
            health_potions = [card.val for card in room_cards if card.suit['class'] == 'health']
            health_potions.sort(reverse=True)
            available_health = sum(health_potions[:2])

            hp_after_room = monster_damage - available_health - hp
            hp_after_room = max(0, hp_after_room)

            can_survive_room = 1 if hp_after_room > 0 else 0

        return {
            "player": {
                "hp": np.array([hp], dtype=np.float32),
                "weapon_level": np.array([game_state['player_state']['weapon_level']], dtype=np.float32),
                "weapon_max_monster_level": np.array([game_state['player_state']['weapon_max_monster_level']], dtype=np.float32),
            },
            "room_cards": self._encode_room_cards(cards=game_state['room_state']['cards']),
            "room": {
                "cards_remaining": np.array([game_state['room_state']['cards_remaining']], dtype=np.float32),
                "can_avoid": np.array([game_state['room_state']['can_avoid']], dtype=np.float32),
                "can_heal": np.array([game_state['room_state']['can_heal']], dtype=np.float32),
                "monster_threat_ratio": np.array([threat_ratio], dtype=np.float32),
                "hp_after_room": np.array([hp_after_room], dtype=np.float32),
                "can_survive_room": np.array([can_survive_room], dtype=np.float32),
            },
            "deck": {
                "cards_remaining": np.array([game_state['dungeon_state']['cards_remaining']], dtype=np.float32),
                "monsters_remaining": np.array([game_state['dungeon_state']['monsters_remaining']], dtype=np.float32),
                "monster_strength_remaining": np.array([game_state['dungeon_state']['monster_strength_remaining']], dtype=np.float32),
                "weapons_remaining": np.array([game_state['dungeon_state']['weapons_remaining']], dtype=np.float32),
                "weapon_strength_remaining": np.array([game_state['dungeon_state']['weapon_strength_remaining']], dtype=np.float32),
                "potions_remaining": np.array([game_state['dungeon_state']['potions_remaining']], dtype=np.float32),
                "potion_strength_remaining": np.array([game_state['dungeon_state']['potion_strength_remaining']], dtype=np.float32)
            },
        }

    def _get_info(self):
        """Return auxiliary information"""
        game_state = self.game.current_game_state()
        return game_state
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize a new game
       
        self.game = Scoundrel(ui=UI.API)
        
        # Track state for reward shaping
        # self.last_room_id = id(self.game.dungeon.current_room)
        # self.prev_room_cards = None  # Track previous room cards for carryover analysis
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def action_masks(self):
        """Return action mask for current state. Required by ActionMasker wrapper."""
        game_state = self.game.current_game_state()
        room_cards_remaining = game_state['room_state']['cards_remaining']
        can_avoid = game_state['room_state']['can_avoid']
        
        masks = np.array([
            room_cards_remaining > 0,
            room_cards_remaining > 1,
            room_cards_remaining > 2,
            room_cards_remaining > 3,
            can_avoid,
        ], dtype=np.bool_)

        return masks
 
    def step(self, action):
        """Execute one step in the environment
        
        Args:
            action (int): Action to take (0-3: interact with cards 1-4, 4: avoid room)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
            
        Reward Shaping:
            The reward function is designed to maximize the game score while encouraging
            efficient resource management. Strategies are organized by complexity level:

        """
        reward = 0
        
        # Get previous state using current_game_state
        prev_state = self.game.current_game_state()
        
        terminated = False
        truncated = False
        
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()

        # 1. Compute valid actions BEFORE mutating state
        # valid_mask = self.action_masks()

        # if not valid_mask[action]:
        #     # Invalid action slipped through
        #     reward = -0.7   # small penalty
        #     terminated = False
        #     truncated = False
        #     info = {"invalid_action": True}
        #     return self._get_obs(), reward, terminated, truncated, info
        
        # Map numeric action to string action for take_action method
        # Actions: 0-3 = interact with card 1-4, 4 = avoid room
        # Note: Action masking is handled by ActionMasker wrapper when used with MaskablePPO
        
        action_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'a'}
        action_str = action_map.get(action)
        
        # Execute the action (action masking wrapper ensures only valid actions reach here)
        if action_str is None:
            reward = -5  # Invalid action
            truncated = True  # End episode on invalid action format
        else:
            # Use the game's take_action method with error handling
            try:
                self.game.take_action(action=action_str)

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

        # Determine what type of card was interacted with
        cards_change = prev_state['dungeon_state']['cards_remaining'] - new_state['dungeon_state']['cards_remaining']
        card_interacted = None
        if cards_change > 0 and action_str != 'a':
            reward += 3.0  
            card_idx = int(action_str) - 1
            if card_idx < new_state['room_state']['cards_remaining']:
                card_interacted = prev_state['room_state']['cards'][card_idx]
        
        # ===== REWARD SHAPING =====
        # Sharp, impactful rewards without normalization:
        # 1. Taking damage (direct penalty, higher when at low HP)
        # 2. Healing (direct bonus, penalize waste)
        # 3. Equipping weapons (small bonus)
        # 4. Clearing rooms (bonus per card)
        # 5. Ending the game (large bonus based on final score)
        
        
        # 1. DAMAGE: Penalty scaled by current HP (survival risk)
        # Reduced so damage doesn't overshadow necessary progress through dungeon
        hp_change = new_state['player_state']['hp'] - prev_state['player_state']['hp']
        if hp_change < 0:
            damage = abs(hp_change)
            # Higher penalty when at low HP (exponential scaling)
            if prev_state['player_state']['hp'] < 5:
                reward -= damage * 2.5  # Critical: -2.5 to -12.5 per damage
            elif prev_state['player_state']['hp'] < 10:
                reward -= damage * 1.5  # Dangerous: -1.5 to -7.5 per damage
            else:
                reward -= damage * 1.0  # Safe: -1.0 to -5.0 per damage
        
        # 2. HEALING: Bonus scaled by HP deficit (healing when low HP is more valuable)
        if hp_change > 0 and card_interacted and card_interacted.suit['class'] == 'health':
            potion_value = card_interacted.val
            wasted = max(0, potion_value - hp_change)
            
            healing_multiplier = 0.5
            # # Scale healing reward by how much HP was needed (prev_hp)
            if prev_state['player_state']['hp'] < 5:
                healing_multiplier = 3.0  # Critical healing is very valuable
            elif prev_state['player_state']['hp'] < 10:
                healing_multiplier = 2.0  # Dangerous healing is valuable
            else:
                healing_multiplier = 1.0  # Safe healing is neutral
            
            reward += hp_change * healing_multiplier
            reward -= wasted * 1.0  # Light penalty for waste
        
        # 3. WEAPON EQUIP: Bonus based on weapon strength, penalty for downgrading
        weapon_change = new_state['player_state']['weapon_level'] - prev_state['player_state']['weapon_level']
        if weapon_change > 0 and card_interacted and card_interacted.suit['class'] == 'weapon':
            new_weapon = new_state['player_state']['weapon_level']
            reward += new_weapon * 0.5  # +0.5 to +5.0 based on weapon level

        # 4. Fight MONSTERS:
        if card_interacted and card_interacted.suit['class'] == 'monster':
            monster_strength = card_interacted.val
            reward += monster_strength * 0.5
            if hp_change < monster_strength:
                # bonus if you reduced damage with a weapon 
                reward += max(1.0, monster_strength - new_state['player_state']['weapon_level']) * 2.0

            # weapon_degredation = prev_state['player_state']['weapon_max_monster_level'] - new_state['player_state']['weapon_max_monster_level']
            # weapon_degredation_ratio = weapon_degredation / prev_state['player_state']['weapon_max_monster_level']
            # current_weapon = new_state['player_state']['weapon_level']
            # if weapon_degredation_ratio < 0.4:
            #     reward += current_weapon / (1.0 + weapon_degredation_ratio)

        
        # 6. GAME END: Bonus for winning (reaching positive/zero score)
        if not new_state['is_active']:
            terminated = True
            final_score = new_state['score']
            
            reward += final_score * 0.1 # Penalize dying before end of dungeon porportionally to performance

            # Win bonus: if final_score >= 0, the player won
            if final_score >= 0:
                reward += 30.0  # Large bonus for winning

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            game_state = self.game.current_game_state()
            print(f"\n{'='*50}")
            print(f"HP: {game_state['player_state']['hp']} | Score: {game_state['score']} | Weapon: {game_state['player_state']['weapon_level']} ({game_state['player_state']['weapon_max_monster_level']})")
            print(f"Cards remaining: {game_state['dungeon_state']['cards_remaining']}")
            print(f"Can avoid: {game_state['room_state']['can_avoid']} | Can heal: {game_state['room_state']['can_heal']}")
            print(f"\nCurrent Room:")
            for i, card in enumerate(game_state['room_state']['cards']):
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
