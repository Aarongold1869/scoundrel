"""
Scoundrel Deck Analysis Module

Provides tools for analyzing Scoundrel deck winnability and generating
solvable decks for curriculum learning and analysis.

Key Components:
- DeckAnalyzer: Heuristic-based deck winnability analysis
- DFSSolver: Depth-first search to prove exact winnability
- SolvableDeckGenerator: Creates decks guaranteed to be solvable
- PartialObservabilityWrapper: Hides deck information for realistic training

Example Usage:
    from scoundrel_ai.deck_analysis import DeckAnalyzer, DFSSolver
    from scoundrel.scoundrel import Dungeon
    
    # Heuristic analysis
    dungeon = Dungeon()
    analyzer = DeckAnalyzer(dungeon)
    analysis = analyzer.analyze()
    print(f"Winnability: {analysis.winnability_score:.2f}")
    
    # Exact DFS solver
    solver = DFSSolver(dungeon)
    is_winnable, path, states = solver.solve()
    print(f"Proven winnable: {is_winnable}")
    print(f"States explored: {states}")
"""

from .deck_analyzer import (
    DeckAnalyzer,
    DeckAnalysis,
    SolvableDeckGenerator,
    DFSSolver,
    GameState,
)

from .partial_observability import (
    PartialObservabilityWrapper,
    PartialObservabilityWithMemory,
    create_partial_observability_env,
)

__all__ = [
    "DeckAnalyzer",
    "DeckAnalysis",
    "SolvableDeckGenerator",
    "DFSSolver",
    "GameState",
    "PartialObservabilityWrapper",
    "PartialObservabilityWithMemory",
    "create_partial_observability_env",
]

__version__ = "0.2.0"
