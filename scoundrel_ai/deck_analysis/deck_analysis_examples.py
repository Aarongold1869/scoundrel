"""
Examples and tests for deck analysis and generation

Usage:
    python scoundrel_ai/deck_analysis_examples.py
"""

from scoundrel.scoundrel import Dungeon
from deck_analyzer import DeckAnalyzer, SolvableDeckGenerator
from partial_observability import create_partial_observability_env


def print_analysis(analysis, title="Deck Analysis"):
    """Pretty-print deck analysis results"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Winnability: {analysis.winnability_score:.2%}")
    print(f"Is Winnable: {analysis.is_winnable}")
    print(f"\nResources:")
    print(f"  Monster Damage: {analysis.total_monster_damage}")
    print(f"  Available Healing: {analysis.total_available_healing}")
    print(f"  Weapon Coverage: {analysis.total_weapon_coverage}")
    print(f"\nAvailability Metrics:")
    print(f"  Avg Distance to Weapon: {analysis.weapon_availability:.1f}")
    print(f"  Avg Distance to Potion: {analysis.potion_availability:.1f}")
    print(f"  Distribution Balance: {analysis.distribution_balance:.2%}")
    print(f"  Monster Concentration: {analysis.monster_concentration:.2%}")
    
    if analysis.critical_issues:
        print(f"\nCritical Issues:")
        for issue in analysis.critical_issues:
            print(f"  ❌ {issue}")
    
    if analysis.warnings:
        print(f"\nWarnings:")
        for warning in analysis.warnings:
            print(f"  ⚠️  {warning}")


def analyze_random_decks(num_decks: int = 10):
    """Analyze random decks and show statistics"""
    print(f"\n{'='*60}")
    print(f"Analyzing {num_decks} Random Decks")
    print(f"{'='*60}")
    
    winnability_scores = []
    
    for i in range(num_decks):
        dungeon = Dungeon()
        analyzer = DeckAnalyzer(dungeon)
        analysis = analyzer.analyze()
        winnability_scores.append(analysis.winnability_score)
        
        status = "✓ WINNABLE" if analysis.is_winnable else "✗ UNWINNABLE"
        print(f"Deck {i+1:2d}: {analysis.winnability_score:.2%} - {status}")
    
    avg_score = sum(winnability_scores) / len(winnability_scores)
    max_score = max(winnability_scores)
    min_score = min(winnability_scores)
    
    print(f"\nStatistics:")
    print(f"  Average Winnability: {avg_score:.2%}")
    print(f"  Best Deck: {max_score:.2%}")
    print(f"  Worst Deck: {min_score:.2%}")
    print(f"  Winnable Decks: {sum(1 for s in winnability_scores if s > 0)}/{num_decks}")


def generate_solvable_decks():
    """Generate and analyze guaranteed-solvable decks"""
    difficulties = {
        "easy": (0.9, 0.3),
        "medium": (0.7, 0.5),
        "hard": (0.5, 0.8),
    }
    
    for difficulty, (target_win, target_conc) in difficulties.items():
        print(f"\n{'='*60}")
        print(f"Generating {difficulty.upper()} Deck")
        print(f"{'='*60}")
        
        dungeon, analysis = SolvableDeckGenerator.generate_solvable_deck(
            difficulty=difficulty,
            target_winnability=target_win
        )
        
        print_analysis(analysis, f"Generated {difficulty.capitalize()} Deck")


def test_partial_observability():
    """Test partial observability modes"""
    print(f"\n{'='*60}")
    print("Testing Partial Observability Modes")
    print(f"{'='*60}")
    
    modes = ["none", "partial", "partial_memory"]
    
    for mode in modes:
        print(f"\nMode: {mode}")
        env = create_partial_observability_env(mode=mode, memory_size=12)
        obs, info = env.reset()
        
        print(f"  Observation keys: {obs.keys()}")
        for key, value in obs.items():
            if isinstance(value, dict):
                print(f"    {key}: {value.keys()}")
            else:
                print(f"    {key}: shape {value.shape if hasattr(value, 'shape') else 'unknown'}")
        
        # Take a few steps
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        env.close()


def compare_deck_quality():
    """Compare random vs generated decks"""
    print(f"\n{'='*60}")
    print("Comparing Random vs Generated Decks")
    print(f"{'='*60}")
    
    # Random decks
    print(f"\nRandom Decks (n=20):")
    random_scores = []
    for _ in range(20):
        dungeon = Dungeon()
        analyzer = DeckAnalyzer(dungeon)
        analysis = analyzer.analyze()
        random_scores.append(analysis.winnability_score)
    
    print(f"  Average: {sum(random_scores)/len(random_scores):.2%}")
    print(f"  Min: {min(random_scores):.2%}, Max: {max(random_scores):.2%}")
    
    # Generated decks
    print(f"\nGenerated Decks (n=20):")
    generated_scores = []
    for _ in range(20):
        dungeon, analysis = SolvableDeckGenerator.generate_solvable_deck(
            target_winnability=0.7,
            max_attempts=5  # Quick generation
        )
        generated_scores.append(analysis.winnability_score)
    
    print(f"  Average: {sum(generated_scores)/len(generated_scores):.2%}")
    print(f"  Min: {min(generated_scores):.2%}, Max: {max(generated_scores):.2%}")


if __name__ == "__main__":
    # Run all examples
    analyze_random_decks(num_decks=20)
    compare_deck_quality()
    generate_solvable_decks()
    test_partial_observability()
    
    print(f"\n{'='*60}")
    print("Examples Complete!")
    print(f"{'='*60}")
