"""
Scoundrel Deck Winnability Analyzer and Solvable Deck Generator

This module analyzes whether a Scoundrel deck is winnable based on:
1. Proximity of weapons and potions to threats
2. Even distribution of resources throughout deck
3. Availability of tools (weapons/potions) relative to monster damage
4. Depth-first search to prove winnability
"""

from typing import List, Dict, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass
from scoundrel.scoundrel import Card, Dungeon
import random


@dataclass
class DeckAnalysis:
    """Results of analyzing deck winnability"""
    is_winnable: bool
    winnability_score: float  # 0.0 to 1.0
    critical_issues: List[str]
    warnings: List[str]
    
    # Detailed metrics
    total_monster_damage: int
    total_available_healing: int
    total_weapon_coverage: int
    weapon_availability: float  # avg distance to weapon when facing monsters
    potion_availability: float  # avg distance to potion when facing monsters
    distribution_balance: float  # how evenly resources are spread (0=clustered, 1=perfect)
    monster_concentration: float  # how clustered monsters are
    

class DeckAnalyzer:
    """Analyzes Scoundrel deck composition for winnability"""
    
    def __init__(self, dungeon: Dungeon):
        """
        Args:
            dungeon: A Scoundrel Dungeon object
        """
        self.dungeon = dungeon
        self.cards = dungeon.cards  # Full deck in order
        self.card_positions = self._build_position_map()
    
    def _build_position_map(self) -> Dict[int, int]:
        """Map card IDs to their position in deck"""
        return {card.id: idx for idx, card in enumerate(self.cards)}
    
    def _get_card_type(self, card: Card) -> str:
        """Return 'monster', 'weapon', or 'potion'"""
        return card.suit['class']
    
    def _effective_distance(self, raw_distance: int) -> float:
        """
        Calculate effective distance accounting for avoid mechanic.
        
        Cards in the next room (4-7 cards away) are more accessible via avoid.
        Returns weighted distance that reflects actual accessibility.
        
        Args:
            raw_distance: Linear card distance
            
        Returns:
            Effective distance (lower = more accessible)
        """
        if raw_distance == float('inf'):
            return float('inf')
        
        # Room boundaries: every 4 cards is a room
        rooms_away = raw_distance // 4
        position_in_room = raw_distance % 4
        
        if rooms_away == 0:
            # Same room: actual distance
            return float(raw_distance)
        elif rooms_away == 1:
            # Next room: accessible via 1 avoid (lower effective distance)
            # Weight as if it's ~2-3 cards away instead of 4-7
            return 2.0 + (position_in_room * 0.25)
        else:
            # 2+ rooms away: need to clear/avoid multiple rooms
            # Penalize additional rooms but not as severely as raw distance
            return 4.0 + (rooms_away - 1) * 3.0 + (position_in_room * 0.25)
    
    def _distance_to_next(self, position: int, card_type: str) -> int:
        """
        Distance from position to next card of given type.
        Returns large number if not found.
        """
        for idx in range(position + 1, len(self.cards)):
            if self._get_card_type(self.cards[idx]) == card_type:
                return idx - position
        return float('inf')
    
    def _distance_to_previous(self, position: int, card_type: str) -> int:
        """Distance from position back to previous card of given type"""
        for idx in range(position - 1, -1, -1):
            if self._get_card_type(self.cards[idx]) == card_type:
                return position - idx
        return float('inf')
    
    def analyze(self) -> DeckAnalysis:
        """Perform complete winnability analysis"""
        critical_issues = []
        warnings = []
        
        # Extract card lists
        monsters = [c for c in self.cards if self._get_card_type(c) == 'monster']
        weapons = [c for c in self.cards if self._get_card_type(c) == 'weapon']
        potions = [c for c in self.cards if self._get_card_type(c) == 'health']
        
        total_monster_damage = sum(m.val for m in monsters)
        total_available_healing = sum(p.val for p in potions)
        total_weapon_coverage = sum(w.val for w in weapons)
        
        # 1. Check if sufficient resources exist
        if len(weapons) == 0:
            critical_issues.append("No weapons in deck - impossible to win")
        
        # Note: Removed strict healing check since graduated scoring handles this
        # Decks with marginal healing are difficult but not impossible
        
        # 2. Analyze weapon availability relative to monsters
        weapon_availability = self._analyze_weapon_availability(monsters)
        if weapon_availability > 8:  # Average effective distance to weapon when facing monsters
            warnings.append(f"Weapons are far from monsters (avg effective distance: {weapon_availability:.1f})")
        
        # 3. Analyze potion availability
        potion_availability = self._analyze_potion_availability(monsters)
        if potion_availability > 6:  # Average effective distance to potion
            warnings.append(f"Potions are far from monsters (avg effective distance: {potion_availability:.1f})")
        
        # 4. Check distribution balance
        distribution_balance = self._analyze_distribution_balance()
        if distribution_balance < 0.4:
            warnings.append(f"Resources are clustered (balance score: {distribution_balance:.2f})")
        
        # 5. Check monster concentration
        monster_concentration = self._analyze_monster_concentration()
        if monster_concentration > 0.7:
            warnings.append(f"Monsters are heavily clustered (concentration: {monster_concentration:.2f})")
        
        # 6. Early game survivability
        first_monsters = monsters[:len(monsters)//3]  # First third of monsters
        early_weapons = [w for w in weapons if self.card_positions[w.id] < len(self.cards)//3]
        if not early_weapons and first_monsters:
            critical_issues.append("No weapons available in first third of deck for early monsters")
        
        # Calculate winnability score
        is_winnable = len(critical_issues) == 0
        winnability_score = self._calculate_winnability_score(
            is_winnable, weapon_availability, potion_availability, 
            distribution_balance, monster_concentration, 
            total_monster_damage, total_available_healing
        )
        
        return DeckAnalysis(
            is_winnable=is_winnable,
            winnability_score=winnability_score,
            critical_issues=critical_issues,
            warnings=warnings,
            total_monster_damage=total_monster_damage,
            total_available_healing=total_available_healing,
            total_weapon_coverage=total_weapon_coverage,
            weapon_availability=weapon_availability,
            potion_availability=potion_availability,
            distribution_balance=distribution_balance,
            monster_concentration=monster_concentration,
        )
    
    def _analyze_weapon_availability(self, monsters: List[Card]) -> float:
        """
        Average effective distance to a weapon when facing each monster.
        Accounts for avoid mechanic - cards in next room are more accessible.
        Lower is better.
        """
        distances = []
        for monster in monsters:
            monster_pos = self.card_positions[monster.id]
            
            # Distance to previous/next weapon
            dist_prev = self._distance_to_previous(monster_pos, 'weapon')
            dist_next = self._distance_to_next(monster_pos, 'weapon')
            
            # Use effective distance accounting for avoid mechanic
            eff_prev = self._effective_distance(dist_prev)
            eff_next = self._effective_distance(dist_next)
            
            min_distance = min(eff_prev, eff_next)
            if min_distance != float('inf'):
                distances.append(min_distance)
        
        return sum(distances) / len(distances) if distances else float('inf')
    
    def _analyze_potion_availability(self, monsters: List[Card]) -> float:
        """
        Average effective distance to a potion when facing each monster.
        Accounts for avoid mechanic - cards in next room are more accessible.
        Lower is better (potions needed for healing during fights).
        """
        distances = []
        for monster in monsters:
            monster_pos = self.card_positions[monster.id]
            
            dist_prev = self._distance_to_previous(monster_pos, 'health')
            dist_next = self._distance_to_next(monster_pos, 'health')
            
            # Use effective distance accounting for avoid mechanic
            eff_prev = self._effective_distance(dist_prev)
            eff_next = self._effective_distance(dist_next)
            
            min_distance = min(eff_prev, eff_next)
            if min_distance != float('inf'):
                distances.append(min_distance)
        
        return sum(distances) / len(distances) if distances else float('inf')
    
    def _analyze_distribution_balance(self) -> float:
        """
        Score how evenly distributed resources are across the deck.
        Returns 0 (highly clustered) to 1 (perfectly distributed).
        
        Uses variance of resource positions. Lower variance = better distribution.
        """
        # Split deck into thirds and count resources in each
        third_size = len(self.cards) // 3
        if third_size == 0:
            return 1.0
        
        monsters = [c for c in self.cards if self._get_card_type(c) == 'monster']
        weapons = [c for c in self.cards if self._get_card_type(c) == 'weapon']
        potions = [c for c in self.cards if self._get_card_type(c) == 'health']
        
        # Count in each third
        def count_in_thirds(card_list):
            thirds = [0, 0, 0]
            for card in card_list:
                third = min(2, self.card_positions[card.id] // third_size)
                thirds[third] += 1
            return thirds
        
        monster_thirds = count_in_thirds(monsters)
        weapon_thirds = count_in_thirds(weapons)
        potion_thirds = count_in_thirds(potions)
        
        # Calculate variance for each resource type
        def variance_score(thirds):
            """Lower variance = more even distribution = higher score"""
            if sum(thirds) == 0:
                return 1.0  # No cards = perfect (no bias)
            mean = sum(thirds) / 3
            var = sum((x - mean) ** 2 for x in thirds) / 3
            # Normalize: max variance is when all in one third
            max_var = (sum(thirds) ** 2) / 3
            return 1.0 - (var / max_var) if max_var > 0 else 1.0
        
        scores = [variance_score(monster_thirds), variance_score(weapon_thirds), variance_score(potion_thirds)]
        return sum(scores) / len(scores)
    
    def _analyze_monster_concentration(self) -> float:
        """
        Score how clustered monsters are in the deck.
        Returns 0 (spread out) to 1 (heavily clustered).
        """
        monsters = [c for c in self.cards if self._get_card_type(c) == 'monster']
        if len(monsters) < 2:
            return 0.0
        
        positions = [self.card_positions[m.id] for m in monsters]
        positions.sort()
        
        # Calculate gaps between consecutive monsters
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        # Clustering score: low gaps = clustered
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        
        # If average gap is small, monsters are clustered
        # Normalize by deck size
        clustering = 1.0 - min(1.0, avg_gap / (len(self.cards) / len(monsters)))
        return clustering
    
    def _calculate_winnability_score(
        self, 
        is_winnable: bool,
        weapon_availability: float,
        potion_availability: float,
        distribution_balance: float,
        monster_concentration: float,
        total_monster_damage: int,
        total_available_healing: int,
    ) -> float:
        """
        Calculate overall winnability score 0.0-1.0
        
        Uses graduated severity instead of binary critical/pass.
        Even decks with issues can have non-zero winnability.
        """
        # Start with base score based on resources
        # Healing vs damage ratio is the most important factor
        healing_ratio = total_available_healing / max(1, total_monster_damage)
        
        if healing_ratio < 0.3:
            # Very insufficient healing
            base_score = 0.05
        elif healing_ratio < 0.5:
            # Insufficient healing - very hard but possible
            base_score = 0.15
        elif healing_ratio < 0.75:
            # Marginal healing - difficult
            base_score = 0.30
        elif healing_ratio < 1.0:
            # Adequate healing - moderate difficulty
            base_score = 0.45
        else:
            # Good healing - easy
            base_score = 0.60
        
        # Weapon availability (lower effective distance is better)
        if weapon_availability == float('inf'):
            weapon_score = 0.0  # No weapons
        else:
            # Adjusted for effective distance scale (max ~10 instead of 20)
            weapon_score = max(0, 1.0 - (weapon_availability / 10))
        base_score += weapon_score * 0.15
        
        # Potion availability
        if potion_availability == float('inf'):
            potion_score = 0.0  # No potions
        else:
            # Adjusted for effective distance scale (max ~8 instead of 15)
            potion_score = max(0, 1.0 - (potion_availability / 8))
        base_score += potion_score * 0.1
        
        # Distribution balance
        base_score += distribution_balance * 0.1
        
        # Monster concentration (lower clustering is better)
        concentration_score = 1.0 - monster_concentration
        base_score += concentration_score * 0.1
        
        return min(1.0, base_score)


class SolvableDeckGenerator:
    """Generates Scoundrel decks with guaranteed or high winnability"""
    
    @staticmethod
    def generate_solvable_deck(
        difficulty: str = "medium",
        target_winnability: float = 0.8,
        max_attempts: int = 100
    ) -> Tuple[Dungeon, DeckAnalysis]:
        """
        Generate a solvable deck that meets winnability targets.
        
        Uses deterministic construction to ensure balanced resource distribution
        throughout the deck, rather than random ordering which often clusters
        resources at the beginning and leaves monsters at the end.
        
        Args:
            difficulty: "easy", "medium", or "hard"
            target_winnability: Target winnability score (0.0-1.0)
            max_attempts: Unused - kept for API compatibility
        
        Returns:
            (dungeon, analysis) tuple
        """
        assert 0.0 <= target_winnability <= 1.0
        
        # Always use deterministic construction for balanced decks
        return SolvableDeckGenerator._construct_deck(difficulty, target_winnability)
    
    @staticmethod
    def _construct_deck(difficulty: str, target_winnability: float) -> Tuple[Dungeon, DeckAnalysis]:
        """
        Deterministically construct a solvable deck by careful placement.
        Uses difficulty and target_winnability to adjust resource distribution.
        
        Accounts for weapon degradation mechanic:
        - Weapons degrade each time they're used, only beating monsters weaker than last
        - Strong monsters early with strong weapons allows optimal degradation
        
        Distributes weapons and potions evenly throughout the deck by interleaving
        them with monsters, preventing long stretches of monsters at the end.
        
        Difficulty affects monster placement and resource density:
        - "easy": Strong monsters early (descending), 1 weapon/potion per 3 monsters
        - "medium": Strong monsters early (descending), 1 weapon/potion per 2 monsters
        - "hard": Weak monsters first (ascending), 1 weapon/potion per 2 monsters (scarce resources)
        """
        dungeon = Dungeon(shuffle_deck=False) # shuffle deck has cards descending by suit
        
        # Extract cards by type
        cards = dungeon.cards
        # When shuffle_deck=False: weapons (descending), potions (descending), monsters (spades then clubs, descending)
        weapons = list(cards[:9])  # Already sorted descending (A, K, Q, J, 10, 9, 8, 7, 6)
        potions = list(cards[9:18])  # Already sorted descending
        monsters = list(cards[18:])  # Already sorted descending within suits

        # Helper function to add variability by shuffling within chunks
        def shuffle_in_chunks(card_list: List[Card], chunk_size: int = 3) -> List[Card]:
            """
            Partition sorted cards into chunks and shuffle within each chunk.
            Maintains overall trend while adding local variability.
            """
            result = []
            for i in range(0, len(card_list), chunk_size):
                chunk = card_list[i:i+chunk_size]
                random.shuffle(chunk)
                result.extend(chunk)
            return result
        
        potions_shuffled = shuffle_in_chunks(potions, chunk_size=3)
        weapons_shuffled = shuffle_in_chunks(weapons, chunk_size=3)

        # For difficulty-based placement
        if difficulty == "easy" or target_winnability > 0.75:
            # Easy: High weapons + descending monsters (optimal degradation path)
            # Natural order from shuffle_deck=False is perfect: A♠️, K♠️, Q♠️, J♠️, ...
            monsters_to_use = shuffle_in_chunks(monsters, chunk_size=4)  # Larger chunks for easier
            resources = weapons_shuffled + potions_shuffled  # Weapons first for easier access
            monsters_per_resource = 3  # 1 resource per 3 monsters (abundant)
        elif difficulty == "hard" or target_winnability < 0.5:
            # Hard: Weak monsters first (ascending order), scarce resources distributed
            # Reverse the descending list to get ascending (2, 3, 4... J, Q, K, A)
            monsters_ascending = list(reversed(monsters))  # Reverse descending to get ascending
            monsters_to_use = shuffle_in_chunks(monsters_ascending, chunk_size=3)  # Smaller chunks for variety
            resources = potions_shuffled + weapons_shuffled  # Potions first (might help early)
            monsters_per_resource = 4  # 1 resource per 4 monsters (scarce)
        else:
            # Medium: Descending monsters, moderate resource availability
            monsters_to_use = shuffle_in_chunks(monsters, chunk_size=3)  # Medium chunks
            resources = weapons_shuffled + potions_shuffled
            monsters_per_resource = 2  # 1 resource per 2 monsters
        
        # Build deck by interleaving monsters with resources
        new_deck = []
        resource_idx = 0

        # Add monsters and resources together - resources only added while processing monsters
        while monsters_to_use:
            # Add N monsters
            for _ in range(monsters_per_resource):
                if monsters_to_use:
                    new_deck.append(monsters_to_use.pop(0))
            
            # Add next resource if available (only during main loop)
            if resource_idx < len(resources):
                new_deck.append(resources[resource_idx])
                resource_idx += 1

        # Insert any remaining resources at random positions throughout the deck
        while resource_idx < len(resources):
            remaining_resource = resources[resource_idx]
            # Insert at random position in existing deck
            insert_pos = random.randint(0, len(new_deck))
            new_deck.insert(insert_pos, remaining_resource)
            resource_idx += 1
        
        # Replace dungeon cards with new ordering
        dungeon.cards = new_deck
        dungeon.current_room = dungeon.cards[:4]
        
        analyzer = DeckAnalyzer(dungeon)
        analysis = analyzer.analyze()
        
        return dungeon, analysis


@dataclass(frozen=True)
class GameState:
    """Immutable game state for DFS solver"""
    hp: int
    weapon_level: int
    weapon_max_monster_level: int
    deck_position: int  # Index in cards array
    room_cards_taken: FrozenSet[int]  # IDs of cards taken from current room
    can_avoid: bool
    can_heal: bool
    
    def __hash__(self):
        return hash((
            self.hp,
            self.weapon_level,
            self.weapon_max_monster_level,
            self.deck_position,
            self.room_cards_taken,
            self.can_avoid,
            self.can_heal
        ))


class DFSSolver:
    """
    Depth-first search solver to definitively prove if a deck is winnable.
    
    More rigorous than heuristic analysis - explores all possible action sequences
    to find at least one winning path.
    """
    
    def __init__(self, dungeon: Dungeon, initial_hp: int = 20, max_states: int = 1000000):
        """
        Args:
            dungeon: Scoundrel dungeon with ordered deck
            initial_hp: Starting HP (default 20)
            max_states: Maximum states to explore before giving up (prevents infinite loops)
        """
        self.cards = dungeon.cards
        self.initial_hp = initial_hp
        self.max_states = max_states
        self.visited: Set[GameState] = set()
        self.states_explored = 0
        self.winning_path: Optional[List[Tuple[GameState, str]]] = None
        
    def solve(self) -> Tuple[bool, Optional[List[Tuple[GameState, str]]], int]:
        """
        Determine if the deck is winnable and return winning path if found.
        
        Returns:
            (is_winnable, winning_path, states_explored)
            - is_winnable: True if at least one winning path exists
            - winning_path: List of (state, action) pairs showing how to win, or None
            - states_explored: Number of unique states examined
        """
        self.visited.clear()
        self.states_explored = 0
        self.winning_path = None
        
        initial_state = GameState(
            hp=self.initial_hp,
            weapon_level=0,
            weapon_max_monster_level=15,
            deck_position=0,
            room_cards_taken=frozenset(),
            can_avoid=True,
            can_heal=True
        )
        
        path = []
        result = self._dfs(initial_state, path)
        
        return result, self.winning_path, self.states_explored
    
    def _get_current_room(self, state: GameState) -> List[Card]:
        """Get the 4 cards in current room based on deck position"""
        room_start = state.deck_position
        room_end = min(room_start + 4, len(self.cards))
        room_cards = self.cards[room_start:room_end]
        
        # Filter out cards already taken
        available_cards = [c for c in room_cards if c.id not in state.room_cards_taken]
        return available_cards
    
    def _count_remaining_monsters(self, state: GameState) -> int:
        """Count how many monsters are left in the deck (not yet taken)"""
        count = 0
        for card in self.cards[state.deck_position:]:
            if card.suit['class'] == 'monster' and card.id not in state.room_cards_taken:
                count += 1
        return count
    
    def _is_terminal(self, state: GameState) -> Tuple[bool, bool]:
        """
        Check if state is terminal.
        
        Returns:
            (is_terminal, is_win)
        """
        # Win condition: All monsters defeated (even if HP is 0)
        # This handles the rule that killing the last monster wins even at 0 HP
        remaining_monsters = self._count_remaining_monsters(state)
        if remaining_monsters == 0:
            return True, True
        
        # Loss: HP depleted (but only if monsters remain)
        if state.hp <= 0:
            return True, False
        
        # Win: All cards cleared and still alive
        if state.deck_position >= len(self.cards):
            return True, True
        
        # Check if current room has any cards left
        room = self._get_current_room(state)
        if len(room) == 0 and state.deck_position < len(self.cards):
            # Room is empty but deck has more cards - should advance room
            return False, False
        
        return False, False
    
    def _apply_action(self, state: GameState, action: str) -> Optional[GameState]:
        """
        Apply action to state and return new state, or None if invalid.
        
        Actions:
            "card_0", "card_1", "card_2", "card_3": interact with room card
            "avoid": avoid current room
        """
        room = self._get_current_room(state)
        
        if action == "avoid":
            if not state.can_avoid:
                return None
            
            # Avoid: skip current room, move to next 4 cards
            new_position = state.deck_position + len(room)
            return GameState(
                hp=state.hp,
                weapon_level=state.weapon_level,
                weapon_max_monster_level=state.weapon_max_monster_level,
                deck_position=new_position,
                room_cards_taken=frozenset(),
                can_avoid=False,  # Can't avoid next room
                can_heal=state.can_heal
            )
        
        # Parse card action
        if not action.startswith("card_"):
            return None
        
        try:
            card_idx = int(action.split("_")[1])
        except (IndexError, ValueError):
            return None
        
        if card_idx >= len(room):
            return None
        
        card = room[card_idx]
        card_type = card.suit['class']
        
        new_hp = state.hp
        new_weapon_level = state.weapon_level
        new_weapon_max = state.weapon_max_monster_level
        new_can_heal = state.can_heal
        new_room_cards_taken = set(state.room_cards_taken)
        new_room_cards_taken.add(card.id)
        
        # Apply card effect
        if card_type == 'monster':
            # Fight monster
            if state.weapon_level > 0 and card.val < state.weapon_max_monster_level:
                # Use weapon
                damage = max(0, card.val - state.weapon_level)
                new_hp = state.hp - damage
                # Weapon degrades
                new_weapon_max = card.val
            else:
                # Fight with hands
                new_hp = state.hp - card.val
        
        elif card_type == 'health':
            # Heal (only if can_heal is True)
            if state.can_heal:
                new_hp = min(20, state.hp + card.val)
            new_can_heal = False
        
        elif card_type == 'weapon':
            # Equip weapon
            new_weapon_level = card.val
            new_weapon_max = 15  # Reset max to Ace
        
        # Check if room should advance
        new_position = state.deck_position
        new_can_avoid = state.can_avoid
        
        # If only 1 card left in room after taking this one, advance to next room
        if len(room) - 1 == 1:
            new_position = state.deck_position + 4
            new_room_cards_taken = set()
            new_can_avoid = True
            new_can_heal = True
        
        return GameState(
            hp=new_hp,
            weapon_level=new_weapon_level,
            weapon_max_monster_level=new_weapon_max,
            deck_position=new_position,
            room_cards_taken=frozenset(new_room_cards_taken),
            can_avoid=new_can_avoid,
            can_heal=new_can_heal
        )
    
    def _get_valid_actions(self, state: GameState) -> List[str]:
        """Get all valid actions for current state"""
        actions = []
        room = self._get_current_room(state)
        
        # Can interact with any card in room
        for i in range(len(room)):
            actions.append(f"card_{i}")
        
        # Can avoid if flag is set
        if state.can_avoid:
            actions.append("avoid")
        
        return actions
    
    def _dfs(self, state: GameState, path: List[Tuple[GameState, str]]) -> bool:
        """
        Depth-first search from given state.
        
        Returns True if winning path found from this state.
        """
        # Check termination limits
        self.states_explored += 1
        if self.states_explored > self.max_states:
            return False
        
        # Check if already visited
        if state in self.visited:
            return False
        self.visited.add(state)
        
        # Check terminal conditions
        is_terminal, is_win = self._is_terminal(state)
        if is_terminal:
            if is_win:
                self.winning_path = list(path)
                return True
            return False
        
        # Try all valid actions
        actions = self._get_valid_actions(state)
        for action in actions:
            next_state = self._apply_action(state, action)
            
            if next_state is None:
                continue
            
            # Recurse
            path.append((state, action))
            if self._dfs(next_state, path):
                return True
            path.pop()
        
        return False
    
    def get_solution_summary(self) -> str:
        """Get human-readable summary of solution"""
        if self.winning_path is None:
            return "No winning path found"
        
        summary = f"Winning path found in {len(self.winning_path)} steps:\n"
        for i, (state, action) in enumerate(self.winning_path):
            summary += f"Step {i+1}: HP={state.hp} Weapon={state.weapon_level}({state.weapon_max_monster_level}) -> {action}\n"
        
        return summary



# from scoundrel_ai.deck_analysis import DeckAnalyzer, SolvableDeckGenerator    
# gen = SolvableDeckGenerator
# easy_dun = gen.generate_solvable_deck(difficulty='easy')
# hard_dun = gen.generate_solvable_deck(difficulty='hard')
# anal_easy = DeckAnalyzer(dungeon=easy_dun[0])
# anal_hard = DeckAnalyzer(dungeon=hard_dun[0])
# anal_easy.analyze()
# anal_hard.analyze()
