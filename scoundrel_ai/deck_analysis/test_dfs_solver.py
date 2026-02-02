"""
Test and demonstrate DFS solver for exact winnability determination

Usage:
    python -m scoundrel_ai.deck_analysis.test_dfs_solver
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoundrel.scoundrel import Dungeon
from scoundrel_ai.deck_analysis import DFSSolver, DeckAnalyzer


def test_single_deck():
    """Test DFS solver on a single random deck"""
    print("\n" + "="*60)
    print("Testing DFS Solver on Random Deck")
    print("="*60)
    
    dungeon = Dungeon()
    
    # First, run heuristic analysis
    print("\n1. Heuristic Analysis:")
    analyzer = DeckAnalyzer(dungeon)
    analysis = analyzer.analyze()
    print(f"   Winnability Score: {analysis.winnability_score:.3f}")
    print(f"   Is Winnable (heuristic): {analysis.is_winnable}")
    if analysis.warnings:
        print(f"   Warnings: {len(analysis.warnings)}")
    
    # Now run DFS solver
    print("\n2. DFS Exact Solver:")
    solver = DFSSolver(dungeon, initial_hp=20, max_states=500000)
    is_winnable, path, states_explored = solver.solve()
    
    print(f"   Proven Winnable: {is_winnable}")
    print(f"   States Explored: {states_explored:,}")
    
    if is_winnable and path:
        print(f"   Solution Length: {len(path)} steps")
        print(f"\n   First 5 steps of winning path:")
        for i, (state, action) in enumerate(path[:5]):
            print(f"     Step {i+1}: HP={state.hp:2d} Weapon={state.weapon_level}({state.weapon_max_monster_level:2d}) -> {action}")
    
    return is_winnable, analysis.winnability_score, states_explored


def test_multiple_decks(num_decks=20):
    """Test DFS solver on multiple decks and compare with heuristic"""
    print("\n" + "="*60)
    print(f"Comparing Heuristic vs DFS on {num_decks} Random Decks")
    print("="*60)
    
    results = []
    
    for i in range(num_decks):
        dungeon = Dungeon()
        
        # Heuristic
        analyzer = DeckAnalyzer(dungeon)
        analysis = analyzer.analyze()
        
        # DFS (with limited search)
        solver = DFSSolver(dungeon, max_states=100000)
        is_winnable, path, states = solver.solve()
        
        results.append({
            'heuristic_score': analysis.winnability_score,
            'heuristic_winnable': analysis.is_winnable,
            'dfs_winnable': is_winnable,
            'dfs_states': states,
        })
        
        status = "✓ Win" if is_winnable else "✗ Loss"
        agreement = "✓" if (is_winnable == analysis.is_winnable) else "✗"
        
        print(f"Deck {i+1:2d}: Heur={analysis.winnability_score:.2f} DFS={status} Agreement={agreement} States={states:,}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    heuristic_winnable = sum(1 for r in results if r['heuristic_winnable'])
    dfs_winnable = sum(1 for r in results if r['dfs_winnable'])
    agreement = sum(1 for r in results if r['heuristic_winnable'] == r['dfs_winnable'])
    
    avg_heuristic = sum(r['heuristic_score'] for r in results) / len(results)
    avg_states = sum(r['dfs_states'] for r in results) / len(results)
    
    print(f"Heuristic Predicted Winnable: {heuristic_winnable}/{num_decks} ({heuristic_winnable/num_decks*100:.1f}%)")
    print(f"DFS Proven Winnable: {dfs_winnable}/{num_decks} ({dfs_winnable/num_decks*100:.1f}%)")
    print(f"Agreement Rate: {agreement}/{num_decks} ({agreement/num_decks*100:.1f}%)")
    print(f"Average Heuristic Score: {avg_heuristic:.3f}")
    print(f"Average States Explored: {avg_states:,.0f}")
    
    # Disagreement analysis
    disagreements = [r for r in results if r['heuristic_winnable'] != r['dfs_winnable']]
    if disagreements:
        print(f"\nDisagreements: {len(disagreements)} cases")
        false_positives = sum(1 for r in disagreements if r['heuristic_winnable'] and not r['dfs_winnable'])
        false_negatives = sum(1 for r in disagreements if not r['heuristic_winnable'] and r['dfs_winnable'])
        print(f"  False Positives (heuristic said yes, DFS said no): {false_positives}")
        print(f"  False Negatives (heuristic said no, DFS said yes): {false_negatives}")


def test_edge_cases():
    """Test solver on known edge cases"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    # Test case 1: Deck with no weapons (should be unwinnable)
    print("\n1. Deck with no weapons:")
    dungeon = Dungeon()
    dungeon.cards = [c for c in dungeon.cards if c.suit['class'] != 'weapon']
    dungeon.current_room = dungeon.cards[:4]
    
    solver = DFSSolver(dungeon, max_states=50000)
    is_winnable, path, states = solver.solve()
    print(f"   Winnable: {is_winnable} (should be False)")
    print(f"   States explored: {states:,}")
    
    # Test case 2: Deck with only potions (should be unwinnable due to monsters)
    print("\n2. Deck with excessive monsters, minimal weapons:")
    dungeon = Dungeon()
    # Keep only weak weapons
    dungeon.cards = [c for c in dungeon.cards if c.suit['class'] != 'weapon' or c.val <= 3]
    dungeon.current_room = dungeon.cards[:4]
    
    solver = DFSSolver(dungeon, max_states=50000)
    is_winnable, path, states = solver.solve()
    print(f"   Winnable: {is_winnable}")
    print(f"   States explored: {states:,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DFS Solver")
    parser.add_argument("--mode", type=str, default="single", 
                       choices=["single", "multiple", "edge"],
                       help="Test mode")
    parser.add_argument("--num-decks", type=int, default=20,
                       help="Number of decks for multiple mode")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        test_single_deck()
    elif args.mode == "multiple":
        test_multiple_decks(args.num_decks)
    elif args.mode == "edge":
        test_edge_cases()
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
