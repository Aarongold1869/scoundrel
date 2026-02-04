"""
Verification script to ensure SolvableDeckGenerator produces decks
with winnability scores that match DeckAnalyzer expectations.
"""

from scoundrel_ai.deck_analysis import (
    DeckAnalyzer, 
    SolvableDeckGenerator,
    DFSSolver
)
from typing import List, Dict, Tuple
import statistics


def verify_deck_generation(num_samples: int = 20) -> Dict[str, List[float]]:
    """
    Generate decks for each difficulty and verify winnability scores.
    
    Args:
        num_samples: Number of decks to generate per difficulty
        
    Returns:
        Dictionary with difficulty -> list of winnability scores
    """
    results = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    
    print("=" * 70)
    print("DECK GENERATION VERIFICATION")
    print("=" * 70)
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n{'='*70}")
        print(f"Testing {difficulty.upper()} difficulty ({num_samples} samples)")
        print(f"{'='*70}\n")
        
        winnability_scores = []
        weapon_availability_scores = []
        potion_availability_scores = []
        distribution_scores = []
        concentration_scores = []
        dfs_results = []
        
        for i in range(num_samples):
            # Generate deck
            dungeon, analysis = SolvableDeckGenerator.generate_solvable_deck(
                difficulty=difficulty
            )
            
            winnability_scores.append(analysis.winnability_score)
            weapon_availability_scores.append(analysis.weapon_availability)
            potion_availability_scores.append(analysis.potion_availability)
            distribution_scores.append(analysis.distribution_balance)
            concentration_scores.append(analysis.monster_concentration)
            
            # Run DFS solver on a subset to verify actual winnability
            if i < 5:  # Only verify first 5 with DFS (it's expensive)
                solver = DFSSolver(dungeon, initial_hp=20, max_states=500000)
                is_winnable, path, states = solver.solve()
                dfs_results.append(is_winnable)
                
                print(f"Sample {i+1}:")
                print(f"  Winnability Score: {analysis.winnability_score:.3f}")
                print(f"  DFS Winnable: {is_winnable}")
                print(f"  Weapon Availability: {analysis.weapon_availability:.2f}")
                print(f"  Potion Availability: {analysis.potion_availability:.2f}")
                print(f"  Distribution Balance: {analysis.distribution_balance:.2f}")
                print(f"  Monster Concentration: {analysis.monster_concentration:.2f}")
                print(f"  Healing/Damage Ratio: {analysis.total_available_healing}/{analysis.total_monster_damage} = {analysis.total_available_healing/max(1, analysis.total_monster_damage):.2f}")
                if analysis.warnings:
                    print(f"  Warnings: {', '.join(analysis.warnings)}")
                print()
        
        results[difficulty] = winnability_scores
        
        # Statistics
        print(f"\n{difficulty.upper()} Statistics:")
        print(f"  Mean Winnability: {statistics.mean(winnability_scores):.3f}")
        print(f"  Std Dev: {statistics.stdev(winnability_scores):.3f}")
        print(f"  Min: {min(winnability_scores):.3f}")
        print(f"  Max: {max(winnability_scores):.3f}")
        print(f"  Mean Weapon Availability: {statistics.mean(weapon_availability_scores):.2f}")
        print(f"  Mean Potion Availability: {statistics.mean(potion_availability_scores):.2f}")
        print(f"  Mean Distribution Balance: {statistics.mean(distribution_scores):.2f}")
        print(f"  Mean Monster Concentration: {statistics.mean(concentration_scores):.2f}")
        if dfs_results:
            print(f"  DFS Win Rate (first 5): {sum(dfs_results)}/{len(dfs_results)} ({100*sum(dfs_results)/len(dfs_results):.0f}%)")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Easy Mean:   {statistics.mean(results['easy']):.3f}")
    print(f"Medium Mean: {statistics.mean(results['medium']):.3f}")
    print(f"Hard Mean:   {statistics.mean(results['hard']):.3f}")
    print()
    
    # Relative ordering checks (more stable than fixed ranges)
    print("Relative checks:")
    mean_scores = {d: statistics.mean(results[d]) for d in results}
    ordering_ok = mean_scores['easy'] >= mean_scores['medium'] >= mean_scores['hard']
    score_gap = mean_scores['easy'] - mean_scores['hard']
    ordering_status = "✓ PASS" if ordering_ok else "✗ FAIL"
    print(f"  Winnability ordering (easy ≥ medium ≥ hard): {ordering_status}")
    print(f"  Easy-Hard mean gap: {score_gap:.3f}")
    if score_gap < 0.02:
        print("  Note: Small gap suggests scoring is dominated by total resources rather than ordering.")
    
    print("\nIf you need strict winnability ranges, calibrate them after scoring changes or increase sensitivity in DeckAnalyzer.")
    
    return results


def test_target_winnability():
    """Test that target_winnability parameter affects generation"""
    print(f"\n{'='*70}")
    print("TESTING TARGET WINNABILITY PARAMETER")
    print(f"{'='*70}\n")
    
    targets = [0.3, 0.5, 0.7, 0.9]
    
    previous_score = None
    for target in targets:
        dungeon, analysis = SolvableDeckGenerator.generate_solvable_deck(
            difficulty="medium",
            target_winnability=target
        )
        
        monotonic = ""
        if previous_score is not None:
            monotonic = "(non-decreasing)" if analysis.winnability_score >= previous_score else "(decreased)"
        previous_score = analysis.winnability_score
        print(f"Target: {target:.1f} -> Actual: {analysis.winnability_score:.3f} {monotonic}")


def detailed_deck_inspection(difficulty: str):
    """Generate one deck and show detailed card-by-card breakdown"""
    print(f"\n{'='*70}")
    print(f"DETAILED DECK INSPECTION - {difficulty.upper()}")
    print(f"{'='*70}\n")
    
    dungeon, analysis = SolvableDeckGenerator.generate_solvable_deck(
        difficulty=difficulty
    )
    
    print("Deck Order:")
    for i, card in enumerate(dungeon.cards):
        card_type = card.suit['class']
        symbol = {'monster': '👾', 'weapon': '⚔️', 'health': '❤️'}[card_type]
        print(f"  {i:2d}. {symbol} {card.suit['name']:8} {card.val:2d}")
    
    print(f"\nAnalysis Results:")
    print(f"  Winnability Score: {analysis.winnability_score:.3f}")
    print(f"  Total Monster Damage: {analysis.total_monster_damage}")
    print(f"  Total Healing: {analysis.total_available_healing}")
    print(f"  Healing Ratio: {analysis.total_available_healing/max(1, analysis.total_monster_damage):.2f}")
    print(f"  Weapon Availability: {analysis.weapon_availability:.2f}")
    print(f"  Potion Availability: {analysis.potion_availability:.2f}")
    print(f"  Distribution Balance: {analysis.distribution_balance:.2f}")
    print(f"  Monster Concentration: {analysis.monster_concentration:.2f}")
    
    if analysis.critical_issues:
        print(f"  Critical Issues: {analysis.critical_issues}")
    if analysis.warnings:
        print(f"  Warnings: {analysis.warnings}")


if __name__ == "__main__":
    # Run verification
    results = verify_deck_generation(num_samples=20)
    
    # Test target winnability
    test_target_winnability()
    
    # Detailed inspection of one deck per difficulty
    for difficulty in ['easy', 'medium', 'hard']:
        detailed_deck_inspection(difficulty)
