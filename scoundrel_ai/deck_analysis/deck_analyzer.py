"""
Scoundrel Deck Winnability Analyzer and Solvable Deck Generator

This module analyzes whether a Scoundrel deck is winnable based on:
1. Proximity of weapons and potions to threats
2. Even distribution of resources throughout deck
3. Availability of tools (weapons/potions) relative to monster damage
"""

from typing import List, Dict, Tuple
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
        if weapon_availability > 15:  # Average distance to weapon when facing monsters
            warnings.append(f"Weapons are far from monsters (avg distance: {weapon_availability:.1f})")
        
        # 3. Analyze potion availability
        potion_availability = self._analyze_potion_availability(monsters)
        if potion_availability > 10:
            warnings.append(f"Potions are far from monsters (avg distance: {potion_availability:.1f})")
        
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
        Average distance to a weapon when facing each monster.
        Lower is better.
        """
        distances = []
        for monster in monsters:
            monster_pos = self.card_positions[monster.id]
            
            # Distance to previous/next weapon
            dist_prev = self._distance_to_previous(monster_pos, 'weapon')
            dist_next = self._distance_to_next(monster_pos, 'weapon')
            
            min_distance = min(dist_prev, dist_next)
            if min_distance != float('inf'):
                distances.append(min_distance)
        
        return sum(distances) / len(distances) if distances else float('inf')
    
    def _analyze_potion_availability(self, monsters: List[Card]) -> float:
        """
        Average distance to a potion when facing each monster.
        Lower is better (potions needed for healing during fights).
        """
        distances = []
        for monster in monsters:
            monster_pos = self.card_positions[monster.id]
            
            dist_prev = self._distance_to_previous(monster_pos, 'health')
            dist_next = self._distance_to_next(monster_pos, 'health')
            
            min_distance = min(dist_prev, dist_next)
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
        
        # Weapon availability (lower distance is better)
        if weapon_availability == float('inf'):
            weapon_score = 0.0  # No weapons
        else:
            weapon_score = max(0, 1.0 - (weapon_availability / 20))
        base_score += weapon_score * 0.15
        
        # Potion availability
        if potion_availability == float('inf'):
            potion_score = 0.0  # No potions
        else:
            potion_score = max(0, 1.0 - (potion_availability / 15))
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
        
        Args:
            difficulty: "easy", "medium", or "hard"
            target_winnability: Minimum winnability score (0.0-1.0)
            max_attempts: How many random attempts before giving up
        
        Returns:
            (dungeon, analysis) tuple
        """
        assert 0.0 <= target_winnability <= 1.0
        
        for attempt in range(max_attempts):
            dungeon = Dungeon()
            analyzer = DeckAnalyzer(dungeon)
            analysis = analyzer.analyze()
            
            if analysis.winnability_score >= target_winnability:
                return dungeon, analysis
        
        # If random generation fails, construct deck deterministically
        return SolvableDeckGenerator._construct_deck(difficulty, target_winnability)
    
    @staticmethod
    def _construct_deck(difficulty: str, target_winnability: float) -> Tuple[Dungeon, DeckAnalysis]:
        """
        Deterministically construct a solvable deck by careful placement.
        """
        dungeon = Dungeon()
        
        # Extract cards by type
        cards = dungeon.cards
        monsters = [c for c in cards if c.suit['class'] == 'monster']
        weapons = [c for c in cards if c.suit['class'] == 'weapon']
        potions = [c for c in cards if c.suit['class'] == 'health']
        
        # Build new deck with careful ordering
        new_deck = []
        
        # Strategy: Distribute resources evenly
        # Every 4-5 monsters, place a weapon and potion
        cards_per_section = len(cards) // 4
        
        monster_idx = 0
        weapon_idx = 0
        potion_idx = 0
        
        for section in range(4):
            section_cards = []
            
            # Add some monsters
            for _ in range(len(monsters) // 4):
                if monster_idx < len(monsters):
                    section_cards.append(monsters[monster_idx])
                    monster_idx += 1
            
            # Add weapon and potion
            if weapon_idx < len(weapons):
                section_cards.append(weapons[weapon_idx])
                weapon_idx += 1
            
            if potion_idx < len(potions):
                section_cards.append(potions[potion_idx])
                potion_idx += 1
            
            # Shuffle within section but keep structure
            random.shuffle(section_cards)
            new_deck.extend(section_cards)
        
        # Add remaining cards
        while monster_idx < len(monsters):
            new_deck.append(monsters[monster_idx])
            monster_idx += 1
        while weapon_idx < len(weapons):
            new_deck.append(weapons[weapon_idx])
            weapon_idx += 1
        while potion_idx < len(potions):
            new_deck.append(potions[potion_idx])
            potion_idx += 1
        
        # Replace dungeon cards with new ordering
        dungeon.cards = new_deck
        dungeon.current_room = dungeon.cards[:4]
        
        analyzer = DeckAnalyzer(dungeon)
        analysis = analyzer.analyze()
        
        return dungeon, analysis
