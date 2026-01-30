"""
Partial Observability Wrapper for Scoundrel Environment

This module wraps the ScoundrelEnv to implement partial observability:
- Agent only sees current room (4 cards)
- Agent doesn't know deck composition beyond current room
- Agent learns probability distributions of future cards through experience
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scoundrel_ai.scoundrel_env import ScoundrelEnv
from typing import Dict, Any


class PartialObservabilityWrapper(gym.Wrapper):
    """
    Wraps ScoundrelEnv to remove full deck visibility.
    
    Modifications:
    - Remove all dungeon_state information (cards_remaining, monsters_remaining, etc.)
    - Keep only current room, player state, and room state
    - Agent must learn resource distribution through experience
    """
    
    def __init__(self, env: ScoundrelEnv):
        super().__init__(env)
        
        # Modify observation space to remove deck context
        # Keep: player, room_cards, room (but remove some unnecessary fields)
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
        
        # No deck context - agent must infer from experience
        self.observation_space = spaces.Dict({
            "player": player_space,
            "room_cards": card_space,
            "room": room_context_space,
        })
    
    def reset(self, seed=None, options=None):
        """Reset and return partial observation"""
        obs, info = self.env.reset(seed=seed, options=options)
        return self._make_partial_obs(obs), info
    
    def step(self, action):
        """Step and return partial observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        partial_obs = self._make_partial_obs(obs)
        return partial_obs, reward, terminated, truncated, info
    
    def _make_partial_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove deck information from observation"""
        return {
            "player": obs["player"],
            "room_cards": obs["room_cards"],
            "room": obs["room"],
            # deck information is NOT included - agent can't see future cards
        }


class PartialObservabilityWithMemory(PartialObservabilityWrapper):
    """
    Partial observability wrapper that includes limited memory.
    
    Agent sees:
    - Current room
    - Last N cards encountered (for learning distributions)
    - Player state
    
    This allows the agent to learn patterns without full deck visibility.
    """
    
    def __init__(self, env: ScoundrelEnv, memory_size: int = 12):
        super().__init__(env)
        self.memory_size = memory_size
        self.card_memory = []  # Recent cards encountered
        
        # Modify observation space to include memory
        memory_space = spaces.Box(
            low=np.tile([0, 0], (memory_size, 1)).astype(np.float32),
            high=np.tile([3, 14], (memory_size, 1)).astype(np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "player": self.observation_space["player"],
            "room_cards": self.observation_space["room_cards"],
            "room": self.observation_space["room"],
            "card_memory": memory_space,  # Last N cards seen
        })
    
    def reset(self, seed=None, options=None):
        """Reset and return partial observation with empty memory"""
        self.card_memory = []
        obs, info = self.env.reset(seed=seed, options=options)
        return self._make_partial_obs_with_memory(obs), info
    
    def step(self, action):
        """Step and return partial observation with updated memory"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update memory with current room cards
        current_room = obs["room_cards"]
        for card in current_room:
            if np.any(card != 0):  # Non-empty card
                self.card_memory.append(card.copy())
        
        # Keep only recent memory
        if len(self.card_memory) > self.memory_size:
            self.card_memory = self.card_memory[-self.memory_size:]
        
        partial_obs = self._make_partial_obs_with_memory(obs)
        return partial_obs, reward, terminated, truncated, info
    
    def _make_partial_obs_with_memory(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove deck info but include card memory"""
        # Build memory array
        memory_array = np.zeros((self.memory_size, 2), dtype=np.float32)
        for i, card in enumerate(self.card_memory[-self.memory_size:]):
            memory_array[i] = card
        
        return {
            "player": obs["player"],
            "room_cards": obs["room_cards"],
            "room": obs["room"],
            "card_memory": memory_array,
        }


def create_partial_observability_env(
    mode: str = "none",
    memory_size: int = 12
) -> gym.Env:
    """
    Factory function to create ScoundrelEnv with specified observability.
    
    Args:
        mode: 
            "none" - Full observability (standard env)
            "partial" - No deck information visible
            "partial_memory" - No deck info but agent remembers recent cards
        memory_size: Size of card memory for "partial_memory" mode
    
    Returns:
        Wrapped ScoundrelEnv
    """
    env = ScoundrelEnv()
    
    if mode == "none":
        return env
    elif mode == "partial":
        return PartialObservabilityWrapper(env)
    elif mode == "partial_memory":
        return PartialObservabilityWithMemory(env, memory_size=memory_size)
    else:
        raise ValueError(f"Unknown observability mode: {mode}")
