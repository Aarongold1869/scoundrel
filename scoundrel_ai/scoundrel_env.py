"""
Custom Gymnasium Environment for Scoundrel Game
"""
from scoundrel.scoundrel import Scoundrel, UI, GameState, Card

import gymnasium as gym
from gymnasium import spaces

import numpy as np
from enum import IntEnum
from typing import List


class StrategyLevel(IntEnum):
    """Reward shaping strategy complexity levels"""
    BASIC = 1         # Fundamental game mechanics
    INTERMEDIATE = 2  # Resource management
    ADVANCED = 3      # Strategic optimization
    EXPERT = 4        # Complex situational play


class ScoundrelEnv(gym.Env):
    """Custom Environment that follows gym interface for Scoundrel game"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, strategy_level=StrategyLevel.EXPERT):
        super().__init__()
        
        self.render_mode = render_mode
        self.game = None
        
        # Convert to StrategyLevel enum if integer provided
        if isinstance(strategy_level, int):
            try:
                self.strategy_level = StrategyLevel(strategy_level)
            except ValueError:
                raise ValueError(f"strategy_level must be 1-4, got {strategy_level}")
        elif isinstance(strategy_level, StrategyLevel):
            self.strategy_level = strategy_level
        else:
            raise ValueError(f"strategy_level must be int or StrategyLevel, got {type(strategy_level)}")
        
        # Define action space
        # Actions: 0-3 = interact with card 1-4, 4 = avoid room
        self.action_space = spaces.Discrete(5)
        
        '''
        Define observation space
        We need to encode the game state:
        Basic stats (11 features):
        - HP (0-20)
        - HP ratio (0-1) normalized
        - Weapon level (0-10)
        - Weapon max monster level (0-15)
        - Cards remaining (0-44)
        - Progress ratio (0-1) - how far through dungeon
        - Can avoid (0 or 1)
        - Can heal (0 or 1)
        - Total monster values remaining in deck (0-208)
        - Total weapon values remaining in deck (0-54)
        - Total health potion values remaining in deck (0-54)
        
        Room summary (3 features):
        - sum_monster_strength: (0-54)
        - sum_weapon_strength (0-34)
        - Total healing available (0-19)
        
        Current room cards (4 cards x 6 features = 24):
          * class (0=weapon, 1=health, 2=monster)
          * value (2-14)
          * suit type (0=diamond, 1=heart, 2=spade, 3=club)
          * is_valid_action (1 if can interact, 0 if not)
          * can_defeat_with_weapon (1 if weapon can defeat with minmal damage)
          * weapon_efficiency (normalized: how good this monster is for weapon use. factors weapon lvl and monster lvl)
        
        Survivability Metrics (3 features):
        - Can survive room (binary)
        - Minimum HP after optimal clear (0-20)
        # - Danger level (max_monster - weapon_level, can be negative)
        
        Threat Assessment (1 feature):
        - Monster threat ratio (max_monster / current_HP, or 0)
        
        Temporal Features (3 features):
        - Game phase discrete (0=early 0-33%, 0.5=mid 33-66%, 1=late 66-100%)
        - Game progress continuous (0.0 to 1.0 ratio of cards played)
        - Resource scarcity (ratio of remaining resources to total remaining cards)
        
        Total: 11 + 3 + 24 + 3 + 1 + 3 = 45 features
        
        # Set up observation space bounds
        # Most features are non-negative, but some can be negative:
        # - obs[41]: Danger level (can be negative when weapon > monster)
        '''

        low_bounds = np.zeros(45, dtype=np.float32)
        # low_bounds[41] = -10  # Danger level: min when weapon 10, monster 0
        
        self.observation_space = spaces.Box(
            low=low_bounds,
            high=300,  # Max value to accommodate sum of card values (monsters ~208, weapons/health ~54 each)
            shape=(45,),  # Extended observation space with survivability, threat, and temporal metrics
            dtype=np.float32
        )
        
    def _get_obs(self):
        """Convert game state to observation vector"""
        # Use the game's current_game_state method
        game_state: GameState = self.game.current_game_state()
        
        obs = np.zeros(45, dtype=np.float32)
        
        # Basic state from GameState (11 features)
        obs[0] = game_state['hp']
        obs[1] = game_state['hp'] / 20.0  # Normalized HP ratio
        obs[2] = game_state['weapon_level']
        obs[3] = game_state['weapon_max_monster_level']
        obs[4] = game_state['num_cards_remaining']
        obs[5] = (44 - game_state['num_cards_remaining']) / 44.0  # Progress ratio
        obs[6] = 1.0 if game_state['can_avoid'] else 0.0
        obs[7] = 1.0 if game_state['can_heal'] else 0.0
        
        # Remaining cards in deck by type (sum of values) - from GameState
        obs[8] = game_state['remaining_monster_sum']
        obs[9] = game_state['remaining_weapon_sum']
        obs[10] = game_state['remaining_health_potion_sum']
        
        # Analyze current room (3 features)
        current_room: List[Card] = game_state['current_room']
        
        monster_card_vals: List[int] = []
        total_monster_strength = 0
        max_monster_strength = 0
        weapon_cards: List[Card] = []
        # weapon_card_vals: List[int] = []
        total_weapon_strength = 0
        health_potion_values: List[int] = []
        for card in current_room:
            if card.suit['class'] == 'monster':
                monster_card_vals.append(card.val)
                total_monster_strength += card.val
                max_monster_strength = max(max_monster_strength, card.val)
            elif card.suit['class'] == 'weapon':
                weapon_cards.append(card)
                # weapon_card_vals.append(card.val)
                total_weapon_strength += card.val
            elif card.suit['class'] == ' health':
                health_potion_values.append(card.val)

        health_potion_values.sort(reverse=True)
        total_healing = sum(health_potion_values[:2])
        
        obs[11] = total_monster_strength
        obs[12] = total_weapon_strength
        obs[13] = total_healing
        
        # Encode individual cards with weapon efficiency (24 features)
        class_encoding = {'weapon': 0, 'health': 1, 'monster': 2}
        suit_encoding = {'diamond': 0, 'heart': 1, 'spade': 2, 'club': 3}
        
        weapon_level = game_state['weapon_level']
        weapon_max = game_state['weapon_max_monster_level']

        def weapon_efficency_ratio(weapon_level, weapon_max, monster_level):
            # Efficiency: combine both weapon_level and weapon_max
            # - weapon_level: how well monster matches original weapon tier (prefer high-value monsters for high-level weapons)
            # - weapon_max: how much degradation potential remains (prefer monsters close to max to minimize waste)

            if weapon_max <= card.val:
                return 0
            # Strength match: how well monster value matches original weapon level
            strength_match = monster_level / weapon_level
            # Degradation efficiency: how close monster is to current weapon max
            degradation_efficiency = monster_level / weapon_max
            # Combine both factors (weighted average favoring strength match slightly)
            efficiency = min((strength_match * 0.6 + degradation_efficiency * 0.4), 1.0)
            return efficiency
        
        for i in range(4):
            base_idx = 14 + (i * 6)
            if i < len(current_room):
                card = current_room[i]
                obs[base_idx] = class_encoding[card.suit['class']]
                obs[base_idx + 1] = card.val
                obs[base_idx + 2] = suit_encoding[card.suit['name']]
                obs[base_idx + 3] = 1.0  # Valid action
                
                # Weapon efficiency features for monsters
                if card.suit['class'] == 'monster' and weapon_level > 0:
                    
                    # Can defeat with minimal damage if current max monster level >= monster value
                    # weapon_max tracks degradation (starts at weapon_level, decreases with use)
                    can_defeat = 1.0 if card.val - weapon_level >= -4 and weapon_max > card.val else 0.0
                    obs[base_idx + 4] = can_defeat
                    
                    obs[base_idx + 5] = weapon_efficency_ratio(weapon_level=weapon_level,
                                                               weapon_max=weapon_max,
                                                               monster_level=card.val)
                else:
                    obs[base_idx + 4] = 0.0
                    obs[base_idx + 5] = 0.0
            else:
                # No card at this position
                obs[base_idx:base_idx + 6] = 0.0
        
        # Survivability Metrics (3 features) - indices 40-42
        hp = game_state['hp']

        monster_card_vals_less_equipped_weapon = [mc_val - weapon_level if mc_val < weapon_max else mc_val for mc_val in monster_card_vals]
        monster_strength_less_equipped_weapon = sum(monster_card_vals_less_equipped_weapon)
        try:
            most_powerful_available_weapon = max([c.val for c in weapon_cards])
        except ValueError:
            most_powerful_available_weapon = 0

        min_hp_after = hp + total_healing + most_powerful_available_weapon - monster_strength_less_equipped_weapon
        min_hp_after = min(min_hp_after, 20)
        
        obs[39] = 1.0 if min_hp_after > 0 else 0.0  # Can survive room
        obs[40] = max(min_hp_after, 0.0)  # Minimum HP after clear

        # danger_current_weapon = max_monster_strength
        # if weapon_max > max_monster_strength:
        #     danger_current_weapon -= weapon_level

        # danger_available_weapon = max_monster_strength
        # if len(weapon_cards) > 0:
        #     for card in weapon_cards:
        #         danger_available_weapon = min(danger_available_weapon, max_monster_strength - card.val)

        # danger_level = min(danger_current_weapon, danger_available_weapon)

        # obs[41] = danger_level # Danger level (can be negative)
        
        # Strategic Context (2 features) - indices 43-44
        # HP buffer: current HP minus strongest remaining monster
        # remaining_monsters = [c for c in self.game.dungeon.cards if c.suit['class'] == 'monster']
        # max_remaining_monster = max([c.val for c in remaining_monsters], default=0)
        # obs[43] = hp - max_remaining_monster  # Can be negative
        
        # Weapon degradation forecast (from optimal burndown calculation)
        # obs[44] = weapon_after_room  # Weapon level after clearing this room optimally
        
        # Threat Assessment (1 feature) - index 45
        # Monster threat ratio
        if hp > 0 and total_monster_strength > 0:
            obs[41] = total_monster_strength / hp
        else:
            obs[41] = 0.0
        
        # Temporal Features (3 features) - indices 46-48
        # Game phase discrete (early/mid/late)
        progress = (44 - game_state['num_cards_remaining']) / 44.0
        if progress < 0.33:
            game_phase = 0.0  # Early game
        elif progress < 0.66:
            game_phase = 0.5  # Mid game
        else:
            game_phase = 1.0  # Late game
        obs[42] = game_phase
        # Game progress continuous (0.0 to 1.0)
        obs[43] = progress
        
        # Resource scarcity (ratio of remaining resources to total cards)
        remaining_resource_value = game_state['remaining_weapon_sum'] + game_state['remaining_health_potion_sum']
        total_remaining_value = (game_state['remaining_monster_sum'] + 
                                game_state['remaining_weapon_sum'] + 
                                game_state['remaining_health_potion_sum'])
        obs[44] = remaining_resource_value / max(total_remaining_value, 1)
        
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
        self.last_room_id = id(self.game.dungeon.current_room)
        self.prev_room_cards = None  # Track previous room cards for carryover analysis
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def basic_strategy(
            self,
            action_str: str,
            card_interacted: Card | None,
            prev_state: GameState,
            new_state: GameState
            ) -> int:
        
        '''
        === BASIC STRATEGIES (Fundamental Game Mechanics) ===
            
        Score Optimization (PRIMARY GOAL):
            - Score increase per step: +score_change * 10.0
        
        HP Changes:
            - Gaining HP: +hp_change * 2.0
            - Taking damage: +hp_change * 1.5 (negative value)
            - Taking damage when low HP (<8): -2.0 extra penalty
        
        Progress & Actions:
            - Playing any card: +1.5
            - Progressing without HP loss: +0.5 bonus
            - Maintaining healthy HP (≥15): +0.3 per step
        '''
        reward = 0

        # Score Optimization (PRIMARY GOAL)
        score_change = new_state['score'] - prev_state['score']
        reward += score_change * 10.0
        
        # HP Changes
        hp_change = new_state['hp'] - prev_state['hp']
        if hp_change > 0:
            reward += hp_change * 2.0

        elif hp_change < 0:
            reward += hp_change * 1.5  # Taking damage is bad
            # Extra penalty for taking damage when low HP
            if prev_state['hp'] < 8:
                reward -= 5.0
        
        # Survival bonus - reward maintaining healthy HP
        if new_state['hp'] >= 15:
            reward += 0.3  # Small constant bonus for good health

        # Progress & Actions
        cards_change = prev_state['num_cards_remaining'] - new_state['num_cards_remaining']
        if cards_change > 0:
            reward += 1.5  # Making progress is good

        return reward

    def intermediate_strategy(
            self,
            action_str: str,
            card_interacted: Card | None,
            prev_state: GameState,
            new_state: GameState
            ) -> int:
        
        '''
        === INTERMEDIATE STRATEGIES (Resource Management) ===
            
            Weapon Acquisition:
                - Acquiring weapon: +weapon_level
            
            Basic Potion Usage:
                - First potion in room: +hp_change * 2.0 + bonus(+2.0 if HP < 10)
                - Additional potions in room: -potion_value * 2.0 (waste penalty)
                - Over-healing penalty: -wasted_healing * 1.5
                    - Example: Using level 10 potion at 18/20 HP wastes 8 healing → -12.0 penalty
            
            Monster Fighting:
                - Minimal damage taken (< 5 HP): +1.0
        '''
        reward = 0

        #  Weapon Acquisition
        if card_interacted and card_interacted.suit['class'] == 'weapon':
            # Simple reward for equipping weapons
            new_weapon_level = new_state['weapon_level']
            reward += new_weapon_level  # Getting weapons is critical
            # Bonus for picking up weapons early
            if prev_state['num_cards_remaining'] > 30:
                reward += 5.0  # Early weapon pickup is excellent

        # Basic Potion Usage
        hp_change = new_state['hp'] - prev_state['hp']
        if card_interacted and card_interacted.suit['class'] == 'health':

            if prev_state['can_heal']:
                potion_value = card_interacted.val
                hp_needed = 20 - prev_state['hp']  # Max HP is 20
                
                reward += hp_change * 2.0  # First potion is good
                
                # Bonus for healing when low HP
                if prev_state['hp'] < 10:
                    reward += 2.0

                # Potion value conservation
                wasted_healing = max(0, potion_value - hp_needed)
                if wasted_healing > 0:
                    # Penalize proportional to wasted healing
                    reward -= wasted_healing * 1.5
            
            # PENALIZE WASTING POTIONS: Only 1 potion per room has effect
            else:
                reward -= card_interacted.val * 2.0  # Penalty proportional to wasted healing 

        # Monster Fightinge 
        if card_interacted and card_interacted.suit['class'] == 'monster':
            hp_change = new_state['hp'] - prev_state['hp']
            if hp_change >= -4:  # Minimal damage
                reward += 3.0
                
        return reward

    def advnaced_strategy(self,
            action_str: str,
            card_interacted: Card | None,
            prev_state: GameState,
            new_state: GameState
            ) -> int:
        
        '''
        === ADVANCED STRATEGIES (Strategic Optimization) ===

            Weapon Waste Prevention:
                - Replacing weapon with high strength (>80%): -old_weapon_level * 2.0
                - Replacing weapon with medium strength (>60%): -old_weapon_level * 1.0
                - Example: Discarding unused level 10 weapon → -20.0 penalty
            
            Weapon Efficiency Optimization:
                - Weapon used on high-level monster (80%+ of weapon max): +10.0
                - Weapon used on medium-level monster (60-80%): +5.0
                - Weapon used on low-level monster (40-60%): +1.0
                - Weapon wasted on weak monster (<40%): -5.0
            
            # Early Game Conservation:
            #     - Fighting monster without weapon in early game: +4.0 (weapon conservation) 
            #         - Acceptable HP trade (damage ≤ monster_value): +2.0
            #         - Poor HP trade (damage > monster_value): -3.0
            #         - Strategy: Save weapons for late game by using HP as resource
        '''
        reward = 0

        # Weapon waste prevention
        if card_interacted and card_interacted.suit['class'] == 'weapon':

            if prev_state['weapon_level'] > 0:
                # Had a weapon before - check if it was used optimally
                # Weapon is "wasted" if we didn't maximize its degradation potential
                degradation_ratio = prev_state['weapon_max_monster_level'] / 15.0  # How degraded was it? (15 is max)
                
                if degradation_ratio > 0.8:  # Weapon still very strong (not degraded much)
                    reward -= prev_state['weapon_level'] * 2.0 # Severe penalty proportional to weapon level
                elif degradation_ratio > 0.6:
                    reward -= prev_state['weapon_level'] * 1.5  # Moderate penalty proportional to weapon level
                else:
                    reward += prev_state['weapon_level'] * 2.0  # Reward proportional to weapon level 
    
        def weapon_efficency_ratio(weapon_level, weapon_max, monster_level):
            '''
            Efficiency: combine both weapon_level and weapon_max
                - weapon_level: how well monster matches original weapon tier (prefer high-value monsters for high-level weapons)
                - weapon_max: how much degradation potential remains (prefer monsters close to max to minimize waste)
            '''
            
            if weapon_max <= monster_level:
                return 0
            # Strength match: how well monster value matches original weapon level
            strength_match = monster_level / weapon_level
            # Degradation efficiency: how close monster is to current weapon max
            degradation_efficiency = monster_level / weapon_max
            # Combine both factors (weighted average favoring strength match slightly)
            efficiency = min((strength_match * 0.6 + degradation_efficiency * 0.4), 1.0)
            return efficiency
        
        # Weapon efficiency optimization
        if card_interacted and card_interacted.suit['class'] == 'monster':
            had_weapon = prev_state['weapon_level'] > 0

            if had_weapon:
                monster_level = card_interacted.val

                # Weapon degraded - evaluate if it was a good choice
                weapon_efficiency = weapon_efficency_ratio(prev_state['weapon_level'],
                                                           weapon_max=prev_state['weapon_max_monster_level'],
                                                           monster_level=monster_level)
                
                if weapon_efficiency >= 0.8:  # Used on high-level monster (80%+ of max)
                    reward += 10.0  # Excellent weapon usage!
                elif weapon_efficiency >= 0.6:
                    reward += 5.0  # Good weapon usage
                elif weapon_efficiency >= 0.4:
                    reward += 1.0  # Okay weapon usage
                elif weapon_efficiency >= 0.3:
                    reward -= 1.0  # Not good weapon useage
                else:
                    reward -= 5.0  # Bad! Wasted weapon on weak monster
            
        #     # Early game weapon conservation strategy
        #     if not had_weapon:
        #         if prev_state['num_cards_remaining'] >= 40:
        #             # Fighting without weapon is good strategy early game
        #             reward += 4.0  # Bonus for weapon conservation
                    
        #             # Evaluate the HP trade-off by checking if health potion is available in room
        #             # Check if there's a health potion in the current room that can offset damage
        #             has_health_potion_available = False
        #             potential_healing = 0
        #             for card in new_state['current_room']:
        #                 if card.suit['class'] == 'health' and new_state['can_heal']:
        #                     has_health_potion_available = True
        #                     potential_healing = max(potential_healing, card.val)
        #                     break
                    
        #             if has_health_potion_available and potential_healing >= abs(hp_change):
        #                 # Can recover the HP loss in this room with available potion
        #                 reward += 2.0  # Good trade - damage can be offset
        #             elif abs(hp_change) <= monster_val:  # Acceptable HP trade without healing
        #                 reward += 1.0  # Okay trade - saving weapon for later
        #             else:  # Taking more damage than monster value without way to recover
        #                 reward -= 3.0  # Poor trade-off
        #         else: 
        #             reward -= 3.0 # fighting with no weapon after first 2 rooms is bad

        return reward

    def expert_strategy(
            self,
            action_str: str,
            card_interacted: Card | None,
            prev_state: GameState,
            new_state: GameState
            ) -> int:
        '''
        === EXPERT STRATEGIES (Complex Situational Play) ===
            
            Strategic Avoidance - Survival:
                - Avoiding unbeatable room (net HP after room ≤ 0): +8.0
                - Avoiding risky room (low HP or near-death): +2.0
                - Avoiding survivable room (wasteful): -3.0
            
            Strategic Avoidance - Endgame Preservation:
                - Avoiding resource-heavy room early game: +5.0
                    * Conditions: cards>25, HP≥12, has weapon
                    * Resource-heavy: no monsters OR 1 weak monster (val 2-4)
                - Avoiding resource-heavy room mid game: +2.0
                    * Conditions: cards>15, HP≥10
                - Strategy: Save safe resource rooms for critical late-game moments
            
            Carryover Optimization:
                - Health potion carryover when low HP: +3.0 to +5.0 (based on healing value)
                - Weapon carryover to avoid degradation: +2.0 to +4.0 (based on weapon level)
                - High-level monster carryover when weapon weak: +3.0 to +5.0
                - Low-level monster carryover with strong weapon: +2.0 to +4.0 (preserve weapon)
                - Poor carryover choice: -2.0 to -4.0
        '''
        reward = 0

        if action_str == 'a':
            # Analyze the room composition
            room_cards = prev_state['current_room']
            total_monster_damage = 0
            health_potion_values = []  # Collect all health potion values
            num_monsters = 0
            num_resources = 0  # weapons + health
            monsters_defeateable_by_weapon = 0
            
            for card in room_cards:
                if card.suit['class'] == 'monster':
                    num_monsters += 1
                    # Check if weapon can defeat without damage
                    if prev_state['weapon_level'] > 0 and card.val <= prev_state['weapon_level']:
                        monsters_defeateable_by_weapon += 1
                    else:
                        # Would take damage equal to monster value
                        total_monster_damage += card.val
                elif card.suit['class'] == 'health':
                    num_resources += 1
                    if prev_state['can_heal']:
                        health_potion_values.append(card.val)
                elif card.suit['class'] == 'weapon':
                    num_resources += 1
            
            # Only top 2 potions are useful (1 for current room, 1 carries to next room)
            # Additional potions beyond 2 will be wasted
            health_potion_values.sort(reverse=True)
            total_healing = sum(health_potion_values[:2])
            
            # ENDGAME STRATEGY: Reward avoiding resource-heavy rooms early to save for later
            # Resource-heavy = no monsters OR only 1 low-level monster (2-4 strength)
            is_resource_heavy = False
            if num_monsters == 0 and num_resources > 0:
                is_resource_heavy = True
            elif num_monsters == 1 and num_resources > 0:
                # Check if the single monster is low-level (2-4 strength)
                for card in room_cards:
                    if card.suit['class'] == 'monster' and card.val <= 4:
                        is_resource_heavy = True
                        break
            
            if is_resource_heavy:
                # Room is resource-heavy - valuable to save for endgame
                if prev_state['num_cards_remaining'] > 25 and prev_state['hp'] >= 12 and prev_state['weapon_level'] > 0:
                    # Early game + good condition = excellent strategic avoidance
                    reward += 5.0  # Save this room for endgame!
                elif prev_state['num_cards_remaining'] > 15 and prev_state['hp'] >= 10:
                    # Mid game + decent condition = reasonable strategy
                    reward += 2.0
                # If low HP or late game without resources, don't reward avoiding

            if num_monsters > 0:
                # Calculate if room is survivable (for monster rooms)
                hp = prev_state['hp']

                monster_card_vals = [mc.val for mc in prev_state['current_room'] if mc.suit['class'] == 'monster']
                monster_card_vals_less_equipped_weapon = [mc_val - prev_state['weapon_level'] if mc_val < prev_state['weapon_max_monster_level'] else mc_val for mc_val in monster_card_vals]
                monster_strength_less_equipped_weapon = sum(monster_card_vals_less_equipped_weapon)
                try:
                    most_powerful_available_weapon = max([wc.val for wc in prev_state['current_room'] if wc.suit['class'] == 'weapon'])
                except ValueError:
                    most_powerful_available_weapon = 0

                min_hp_after = hp + total_healing + most_powerful_available_weapon - monster_strength_less_equipped_weapon
                min_hp_after = min(min_hp_after, 20)
                
                if min_hp_after <= 0:
                    # UNBEATABLE ROOM - cannot survive even with optimal play
                    reward += 8.0  # Excellent decision to avoid!
                elif prev_state['hp'] < 8 or min_hp_after < 5:
                    # Survivable but risky - avoiding is reasonable
                    reward += 2.0
                else:
                    # Room is clearly survivable - avoiding is wasteful
                    reward -= 3.0

        # Evaluate carryover card choice from previous room
        # current_room_id = id(self.game.dungeon.current_room)
        # entered_new_room = current_room_id != self.last_room_id
        # if entered_new_room:
        #     if self.prev_room_cards is not None:
        #         # Carried over card is always at index 0 when entering new room with can_avoid=True
        #         carried_over_card = None
        #         current_room_cards = prev_state['current_room']
                
        #         if prev_state['can_avoid'] and len(current_room_cards) > 0:
        #             carried_over_card = current_room_cards[0]
                
        #         if carried_over_card:
        #             card_class = carried_over_card.suit['class']
        #             card_val = carried_over_card.val
                    
        #             # Evaluate carryover decision
        #             if card_class == 'health':
        #                 # Good if HP is low and potion is high-value
        #                 if prev_state['hp'] < 12:
        #                     reward += min(card_val / 2.0, 5.0)  # Up to +5.0 for high-value potion when low HP
        #                 elif prev_state['hp'] >= 18:
        #                     reward -= 2.0  # Bad - don't need healing
                    
        #             elif card_class == 'weapon':
        #                 # Good if avoiding degradation or weapon waste
        #                 has_weapon = prev_state['weapon_level'] > 0
                        
        #                 if has_weapon and card_val > prev_state['weapon_level']:
        #                     # Carrying over better weapon - good if current weapon is degraded
        #                     degradation_ratio = prev_state['weapon_max_monster_level'] / 15.0
        #                     if degradation_ratio < 0.5:  # Current weapon well-used
        #                         reward += card_val / 3.0  # Up to +4.7 for high weapon
        #                 elif has_weapon and card_val < prev_state['weapon_level']:
        #                     # Carrying over weaker weapon - bad (forced downgrade later)
        #                     reward -= 3.0
        #                 elif not has_weapon and prev_state['remaining_monster_sum'] > 50:
        #                     # Carrying over weapon when unarmed and monsters remain - good
        #                     reward += card_val / 2.5  # Up to +5.6
                    
        #             elif card_class == 'monster':
        #                 has_weapon = prev_state['weapon_level'] > 0
                        
        #                 # High-level monster carryover
        #                 if card_val >= 10:
        #                     if not has_weapon or card_val > prev_state['weapon_level']:
        #                         # Good - avoiding damage or weapon degradation
        #                         if prev_state['remaining_weapon_sum'] > 20:  # Weapons available
        #                             reward += 3.0 + (card_val - 10) * 0.5  # Up to +5.0
        #                     else:
        #                         reward -= 3.0  # Bad - could defeat with weapon
                        
        #                 # Low-level monster carryover
        #                 elif card_val <= 4:
        #                     if has_weapon and prev_state['weapon_level'] >= 8:
        #                         # Good - preserving strong weapon for tougher monsters
        #                         if prev_state['remaining_monster_sum'] > 80:  # Many monsters remain
        #                             degradation_ratio = prev_state['weapon_max_monster_level'] / 15.0
        #                             if degradation_ratio > 0.7:  # Weapon not very degraded
        #                                 reward += 2.0 + (prev_state['weapon_level'] / 5.0)  # Up to +4.8 

        return reward

    def step(self, action):
        """Execute one step in the environment
        
        Args:
            action (int): Action to take (0-3: interact with cards 1-4, 4: avoid room)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
            
        Strategy Level:
            The environment can be configured with different strategy levels (StrategyLevel enum or int 1-4):
            - StrategyLevel.BASIC (1): Only fundamental game mechanics (score, HP, progress)
            - StrategyLevel.INTERMEDIATE (2): Adds resource management (weapons, potions)
            - StrategyLevel.ADVANCED (3): Adds strategic optimization (efficiency, conservation)
            - StrategyLevel.EXPERT (4): Adds complex situational play (endgame preservation, avoidance)
            
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
        
        # Map numeric action to string action for take_action method
        # Actions: 0-3 = interact with card 1-4, 4 = avoid room
        action_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'a'}
        action_str = action_map.get(action)
        
        if action_str is None:
            reward = -5  # Invalid action
            truncated = True  # End episode on invalid action format
        else:
            # Use the game's take_action method with error handling
            try:
                self.game.take_action(action=action_str)
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

        # Determine what type of card was interacted with
        cards_change = prev_state['num_cards_remaining'] - new_state['num_cards_remaining']
        card_interacted = None
        if cards_change > 0 and action_str != 'a':
            card_idx = int(action_str) - 1
            if card_idx < len(prev_state['current_room']):
                card_interacted = prev_state['current_room'][card_idx]
        
        # ===== REWARD SHAPING =====

        reward += self.basic_strategy(action_str=action_str,
                                      card_interacted=card_interacted,
                                      prev_state=prev_state,
                                      new_state=new_state)
        
        if self.strategy_level >= StrategyLevel.INTERMEDIATE:
            reward += self.intermediate_strategy(action_str=action_str,
                                                card_interacted=card_interacted,
                                                prev_state=prev_state,
                                                new_state=new_state)

        if self.strategy_level >= StrategyLevel.ADVANCED:
            reward += self.advnaced_strategy(action_str=action_str,
                                            card_interacted=card_interacted,
                                            prev_state=prev_state,
                                            new_state=new_state)
            
        if self.strategy_level == StrategyLevel.EXPERT:
             reward += self.expert_strategy(action_str=action_str,
                                            card_interacted=card_interacted,
                                            prev_state=prev_state,
                                            new_state=new_state)
      
        current_room_id = id(self.game.dungeon.current_room)
        self.last_room_id = current_room_id

        # Check if game is over - FOCUS ON SCORE, NOT SURVIVAL
        if not new_state['is_active']:
            terminated = True
            # Normalize final score proportionally: -188 (worst) to +30 (best)
            # Map to 0-100 scale to reward higher scores even if negative
            min_score = -188  # Worst possible score
            max_score = 30    # Best possible score
            normalized_score = (new_state['score'] - min_score) / (max_score - min_score)
            # Scale to 0-100 range for meaningful reward signal
            reward += normalized_score * 100.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        # Store current room cards for carryover analysis in next step
        if self.strategy_level == StrategyLevel.EXPERT:
            self.prev_room_cards = new_state['current_room'].copy() if new_state['current_room'] else None
        
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
