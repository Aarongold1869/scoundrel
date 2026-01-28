   
    # def basic_strategy(
    #         self,
    #         action_str: str,
    #         card_interacted: Card | None,
    #         prev_state: GameState,
    #         new_state: GameState
    #         ) -> int:
        
    #     '''
    #     === BASIC STRATEGIES (Fundamental Game Mechanics) ===
            
    #     Score Optimization (PRIMARY GOAL):
    #         - Score increase per step: +score_change * 10.0
        
    #     HP Changes:
    #         - Gaining HP: +hp_change * 2.0
    #         - Taking damage: +hp_change * 1.5 (negative value)
    #         - Taking damage when low HP (<8): -2.0 extra penalty
        
    #     Progress & Actions:
    #         - Playing any card: +1.5
    #         - Progressing without HP loss: +0.5 bonus
    #         - Maintaining healthy HP (≥15): +0.3 per step
    #     '''
    #     reward = 0

    #     # Score Optimization (PRIMARY GOAL)
    #     score_change = new_state['score'] - prev_state['score']
    #     reward += score_change * 10.0
        
    #     # HP Changes
    #     hp_change = new_state['player_state']['hp'] - prev_state['player_state']['hp']
    #     if hp_change > 0:
    #         reward += hp_change * 2.0

    #     elif hp_change < 0:
    #         reward += hp_change * 1.5  # Taking damage is bad
    #         # Extra penalty for taking damage when low HP
    #         if prev_state['player_state']['hp'] < 8:
    #             reward -= 5.0
        
    #     # Survival bonus - reward maintaining healthy HP
    #     if new_state['player_state']['hp'] >= 15:
    #         reward += 1.0  # Small constant bonus for good health

    #     # Progress & Actions
    #     cards_change = prev_state['dungeon_state']['cards_remaining'] - new_state['dungeon_state']['cards_remaining']
    #     if cards_change > 0:
    #         reward += 1.5  # Making progress is good

    #     return reward

    # def intermediate_strategy(
    #         self,
    #         action_str: str,
    #         card_interacted: Card | None,
    #         prev_state: GameState,
    #         new_state: GameState
    #         ) -> int:
        
    #     '''
    #     === INTERMEDIATE STRATEGIES (Resource Management) ===
            
    #     Weapon Acquisition:
    #         - Acquiring weapon: +weapon_level
        
    #     Basic Potion Usage:
    #         - First potion in room: +hp_change * 2.0 + bonus(+2.0 if HP < 10)
    #         - Additional potions in room: -potion_value * 2.0 (waste penalty)
    #         - Over-healing penalty: -wasted_healing * 1.5
    #             - Example: Using level 10 potion at 18/20 HP wastes 8 healing → -12.0 penalty
        
    #     Monster Fighting:
    #         - Minimal damage taken (< 5 HP): +1.0
    #     '''
    #     reward = 0

    #     #  Weapon Acquisition
    #     if card_interacted and card_interacted.suit['class'] == 'weapon':
    #         # Simple reward for equipping weapons
    #         new_weapon_level = new_state['weapon_level']
    #         reward += new_weapon_level  # Getting weapons is critical
    #         # Bonus for picking up weapons early
    #         if prev_state['dungeon_state']['cards_remaining'] > 30:
    #             reward += 5.0  # Early weapon pickup is excellent

    #     # Basic Potion Usage
    #     hp_change = new_state['hp'] - prev_state['hp']
    #     if card_interacted and card_interacted.suit['class'] == 'health':

    #         if prev_state['can_heal']:
    #             potion_value = card_interacted.val
    #             hp_needed = 20 - prev_state['hp']  # Max HP is 20
                
    #             reward += hp_change * 2.0  # First potion is good
                
    #             # Bonus for healing when low HP
    #             if prev_state['hp'] < 10:
    #                 reward += 2.0

    #             # Potion value conservation
    #             wasted_healing = max(0, potion_value - hp_needed)
    #             if wasted_healing > 0:
    #                 # Penalize proportional to wasted healing
    #                 reward -= wasted_healing * 1.5
            
    #         # PENALIZE WASTING POTIONS: Only 1 potion per room has effect
    #         else:
    #             reward -= card_interacted.val * 2.0  # Penalty proportional to wasted healing 

    #     # Monster Fightinge 
    #     if card_interacted and card_interacted.suit['class'] == 'monster':
    #         hp_change = new_state['hp'] - prev_state['hp']
    #         if hp_change >= -4:  # Minimal damage
    #             reward += 3.0
                
    #     return reward

    # def advnaced_strategy(self,
    #         action_str: str,
    #         card_interacted: Card | None,
    #         prev_state: GameState,
    #         new_state: GameState
    #         ) -> int:
        
    #     '''
    #     === ADVANCED STRATEGIES (Strategic Optimization) ===

    #     Weapon Waste Prevention:
    #         - Replacing weapon with high strength (>80%): -old_weapon_level * 2.0
    #         - Replacing weapon with medium strength (>60%): -old_weapon_level * 1.0
    #         - Example: Discarding unused level 10 weapon → -20.0 penalty
        
    #     Weapon Efficiency Optimization:
    #         - Weapon used on high-level monster (80%+ of weapon max): +10.0
    #         - Weapon used on medium-level monster (60-80%): +5.0
    #         - Weapon used on low-level monster (40-60%): +1.0
    #         - Weapon wasted on weak monster (<40%): -5.0
        
    #     # Early Game Conservation:
    #     #     - Fighting monster without weapon in early game: +4.0 (weapon conservation) 
    #     #         - Acceptable HP trade (damage ≤ monster_value): +2.0
    #     #         - Poor HP trade (damage > monster_value): -3.0
    #     #         - Strategy: Save weapons for late game by using HP as resource
    #     '''
    #     reward = 0

    #     # Weapon waste prevention
    #     if card_interacted and card_interacted.suit['class'] == 'weapon':

    #         if prev_state['weapon_level'] > 0:
    #             # Had a weapon before - check if it was used optimally
    #             # Weapon is "wasted" if we didn't maximize its degradation potential
    #             degradation_ratio = prev_state['weapon_max_monster_level'] / 15.0  # How degraded was it? (15 is max)
                
    #             if degradation_ratio > 0.8:  # Weapon still very strong (not degraded much)
    #                 reward -= prev_state['weapon_level'] * 2.0 # Severe penalty proportional to weapon level
    #             elif degradation_ratio > 0.6:
    #                 reward -= prev_state['weapon_level'] * 1.5  # Moderate penalty proportional to weapon level
    #             else:
    #                 reward += prev_state['weapon_level'] * 2.0  # Reward proportional to weapon level 
    
    #     def weapon_efficency_ratio(weapon_level, weapon_max, monster_level):
    #         '''
    #         Efficiency: combine both weapon_level and weapon_max
    #             - weapon_level: how well monster matches original weapon tier (prefer high-value monsters for high-level weapons)
    #             - weapon_max: how much degradation potential remains (prefer monsters close to max to minimize waste)
    #         '''
            
    #         if weapon_max <= monster_level:
    #             return 0
    #         # Strength match: how well monster value matches original weapon level
    #         strength_match = monster_level / weapon_level
    #         # Degradation efficiency: how close monster is to current weapon max
    #         degradation_efficiency = monster_level / weapon_max
    #         # Combine both factors (weighted average favoring strength match slightly)
    #         efficiency = min((strength_match * 0.6 + degradation_efficiency * 0.4), 1.0)
    #         return efficiency
        
    #     # Weapon efficiency optimization
    #     if card_interacted and card_interacted.suit['class'] == 'monster':
    #         had_weapon = prev_state['weapon_level'] > 0

    #         if had_weapon:
    #             monster_level = card_interacted.val

    #             # Weapon degraded - evaluate if it was a good choice
    #             weapon_efficiency = weapon_efficency_ratio(prev_state['weapon_level'],
    #                                                        weapon_max=prev_state['weapon_max_monster_level'],
    #                                                        monster_level=monster_level)
                
    #             if weapon_efficiency >= 0.8:  # Used on high-level monster (80%+ of max)
    #                 reward += 10.0  # Excellent weapon usage!
    #             elif weapon_efficiency >= 0.6:
    #                 reward += 5.0  # Good weapon usage
    #             elif weapon_efficiency >= 0.4:
    #                 reward += 1.0  # Okay weapon usage
    #             elif weapon_efficiency >= 0.3:
    #                 reward -= 1.0  # Not good weapon useage
    #             else:
    #                 reward -= 5.0  # Bad! Wasted weapon on weak monster
            
    #     #     # Early game weapon conservation strategy
    #     #     if not had_weapon:
    #     #         if prev_state['dungeon_state']['cards_remaining'] >= 40:
    #     #             # Fighting without weapon is good strategy early game
    #     #             reward += 4.0  # Bonus for weapon conservation
                    
    #     #             # Evaluate the HP trade-off by checking if health potion is available in room
    #     #             # Check if there's a health potion in the current room that can offset damage
    #     #             has_health_potion_available = False
    #     #             potential_healing = 0
    #     #             for card in new_state['current_room']:
    #     #                 if card.suit['class'] == 'health' and new_state['can_heal']:
    #     #                     has_health_potion_available = True
    #     #                     potential_healing = max(potential_healing, card.val)
    #     #                     break
                    
    #     #             if has_health_potion_available and potential_healing >= abs(hp_change):
    #     #                 # Can recover the HP loss in this room with available potion
    #     #                 reward += 2.0  # Good trade - damage can be offset
    #     #             elif abs(hp_change) <= monster_val:  # Acceptable HP trade without healing
    #     #                 reward += 1.0  # Okay trade - saving weapon for later
    #     #             else:  # Taking more damage than monster value without way to recover
    #     #                 reward -= 3.0  # Poor trade-off
    #     #         else: 
    #     #             reward -= 3.0 # fighting with no weapon after first 2 rooms is bad

    #     return reward

    # def expert_strategy(
    #         self,
    #         action_str: str,
    #         card_interacted: Card | None,
    #         prev_state: GameState,
    #         new_state: GameState
    #         ) -> int:
    #     '''
    #     === EXPERT STRATEGIES (Complex Situational Play) ===
            
    #     Strategic Avoidance - Survival:
    #         - Avoiding unbeatable room (net HP after room ≤ 0): +8.0
    #         - Avoiding risky room (low HP or near-death): +2.0
    #         - Avoiding survivable room (wasteful): -3.0
        
    #     Strategic Avoidance - Endgame Preservation:
    #         - Avoiding resource-heavy room early game: +5.0
    #             * Conditions: cards>25, HP≥12, has weapon
    #             * Resource-heavy: no monsters OR 1 weak monster (val 2-4)
    #         - Avoiding resource-heavy room mid game: +2.0
    #             * Conditions: cards>15, HP≥10
    #         - Strategy: Save safe resource rooms for critical late-game moments
        
    #     # Carryover Optimization:
    #     #     - Health potion carryover when low HP: +3.0 to +5.0 (based on healing value)
    #     #     - Weapon carryover to avoid degradation: +2.0 to +4.0 (based on weapon level)
    #     #     - High-level monster carryover when weapon weak: +3.0 to +5.0
    #     #     - Low-level monster carryover with strong weapon: +2.0 to +4.0 (preserve weapon)
    #     #     - Poor carryover choice: -2.0 to -4.0
    #     '''
    #     reward = 0

    #     if action_str == 'a':
    #         # Analyze the room composition
    #         room_cards = prev_state['current_room']
    #         total_monster_damage = 0
    #         health_potion_values = []  # Collect all health potion values
    #         num_monsters = 0
    #         num_resources = 0  # weapons + health
    #         monsters_defeateable_by_weapon = 0
            
    #         for card in room_cards:
    #             if card.suit['class'] == 'monster':
    #                 num_monsters += 1
    #                 # Check if weapon can defeat without damage
    #                 if prev_state['weapon_level'] > 0 and card.val <= prev_state['weapon_level']:
    #                     monsters_defeateable_by_weapon += 1
    #                 else:
    #                     # Would take damage equal to monster value
    #                     total_monster_damage += card.val
    #             elif card.suit['class'] == 'health':
    #                 num_resources += 1
    #                 if prev_state['can_heal']:
    #                     health_potion_values.append(card.val)
    #             elif card.suit['class'] == 'weapon':
    #                 num_resources += 1
            
    #         # Only top 2 potions are useful (1 for current room, 1 carries to next room)
    #         # Additional potions beyond 2 will be wasted
    #         health_potion_values.sort(reverse=True)
    #         total_healing = sum(health_potion_values[:2])
            
    #         # ENDGAME STRATEGY: Reward avoiding resource-heavy rooms early to save for later
    #         # Resource-heavy = no monsters OR only 1 low-level monster (2-4 strength)
    #         is_resource_heavy = False
    #         if num_monsters == 0 and num_resources > 0:
    #             is_resource_heavy = True
    #         elif num_monsters == 1 and num_resources > 0:
    #             # Check if the single monster is low-level (2-4 strength)
    #             for card in room_cards:
    #                 if card.suit['class'] == 'monster' and card.val <= 4:
    #                     is_resource_heavy = True
    #                     break
            
    #         if is_resource_heavy:
    #             # Room is resource-heavy - valuable to save for endgame
    #             if prev_state['dungeon_state']['cards_remaining'] > 25 and prev_state['hp'] >= 12 and prev_state['weapon_level'] > 0:
    #                 # Early game + good condition = excellent strategic avoidance
    #                 reward += 5.0  # Save this room for endgame!
    #             elif prev_state['dungeon_state']['cards_remaining'] > 15 and prev_state['hp'] >= 10:
    #                 # Mid game + decent condition = reasonable strategy
    #                 reward += 2.0
    #             # If low HP or late game without resources, don't reward avoiding

    #         if num_monsters > 0:
    #             # Calculate if room is survivable (for monster rooms)
    #             hp = prev_state['hp']

    #             monster_card_vals = [mc.val for mc in prev_state['current_room'] if mc.suit['class'] == 'monster']
    #             monster_card_vals_less_equipped_weapon = [mc_val - prev_state['weapon_level'] if mc_val < prev_state['weapon_max_monster_level'] else mc_val for mc_val in monster_card_vals]
    #             monster_strength_less_equipped_weapon = sum(monster_card_vals_less_equipped_weapon)
    #             try:
    #                 most_powerful_available_weapon = max([wc.val for wc in prev_state['current_room'] if wc.suit['class'] == 'weapon'])
    #             except ValueError:
    #                 most_powerful_available_weapon = 0

    #             min_hp_after = hp + total_healing + most_powerful_available_weapon - monster_strength_less_equipped_weapon
    #             min_hp_after = min(min_hp_after, 20)
                
    #             if min_hp_after <= 0:
    #                 # UNBEATABLE ROOM - cannot survive even with optimal play
    #                 reward += 8.0  # Excellent decision to avoid!
    #             elif prev_state['hp'] < 8 or min_hp_after < 5:
    #                 # Survivable but risky - avoiding is reasonable
    #                 reward += 2.0
    #             else:
    #                 # Room is clearly survivable - avoiding is wasteful
    #                 reward -= 3.0

    #     # Evaluate carryover card choice from previous room
    #     # current_room_id = id(self.game.dungeon.current_room)
    #     # entered_new_room = current_room_id != self.last_room_id
    #     # if entered_new_room:
    #     #     if self.prev_room_cards is not None:
    #     #         # Carried over card is always at index 0 when entering new room with can_avoid=True
    #     #         carried_over_card = None
    #     #         current_room_cards = prev_state['current_room']
                
    #     #         if prev_state['can_avoid'] and len(current_room_cards) > 0:
    #     #             carried_over_card = current_room_cards[0]
                
    #     #         if carried_over_card:
    #     #             card_class = carried_over_card.suit['class']
    #     #             card_val = carried_over_card.val
                    
    #     #             # Evaluate carryover decision
    #     #             if card_class == 'health':
    #     #                 # Good if HP is low and potion is high-value
    #     #                 if prev_state['hp'] < 12:
    #     #                     reward += min(card_val / 2.0, 5.0)  # Up to +5.0 for high-value potion when low HP
    #     #                 elif prev_state['hp'] >= 18:
    #     #                     reward -= 2.0  # Bad - don't need healing
                    
    #     #             elif card_class == 'weapon':
    #     #                 # Good if avoiding degradation or weapon waste
    #     #                 has_weapon = prev_state['weapon_level'] > 0
                        
    #     #                 if has_weapon and card_val > prev_state['weapon_level']:
    #     #                     # Carrying over better weapon - good if current weapon is degraded
    #     #                     degradation_ratio = prev_state['weapon_max_monster_level'] / 15.0
    #     #                     if degradation_ratio < 0.5:  # Current weapon well-used
    #     #                         reward += card_val / 3.0  # Up to +4.7 for high weapon
    #     #                 elif has_weapon and card_val < prev_state['weapon_level']:
    #     #                     # Carrying over weaker weapon - bad (forced downgrade later)
    #     #                     reward -= 3.0
    #     #                 elif not has_weapon and prev_state['remaining_monster_sum'] > 50:
    #     #                     # Carrying over weapon when unarmed and monsters remain - good
    #     #                     reward += card_val / 2.5  # Up to +5.6
                    
    #     #             elif card_class == 'monster':
    #     #                 has_weapon = prev_state['weapon_level'] > 0
                        
    #     #                 # High-level monster carryover
    #     #                 if card_val >= 10:
    #     #                     if not has_weapon or card_val > prev_state['weapon_level']:
    #     #                         # Good - avoiding damage or weapon degradation
    #     #                         if prev_state['remaining_weapon_sum'] > 20:  # Weapons available
    #     #                             reward += 3.0 + (card_val - 10) * 0.5  # Up to +5.0
    #     #                     else:
    #     #                         reward -= 3.0  # Bad - could defeat with weapon
                        
    #     #                 # Low-level monster carryover
    #     #                 elif card_val <= 4:
    #     #                     if has_weapon and prev_state['weapon_level'] >= 8:
    #     #                         # Good - preserving strong weapon for tougher monsters
    #     #                         if prev_state['remaining_monster_sum'] > 80:  # Many monsters remain
    #     #                             degradation_ratio = prev_state['weapon_max_monster_level'] / 15.0
    #     #                             if degradation_ratio > 0.7:  # Weapon not very degraded
    #     #                                 reward += 2.0 + (prev_state['weapon_level'] / 5.0)  # Up to +4.8 

    #     return reward
