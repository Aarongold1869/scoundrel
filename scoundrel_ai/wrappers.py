"""
Gym wrappers
"""
from scoundrel_ai.deck_analysis.deck_analyzer import DeckAnalyzer, SolvableDeckGenerator, DFSSolver, GameState

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import QRDQN

import gymnasium as gym
import numpy as np
import random
import torch as th


def get_base_env(env):
    base_env = env
    if hasattr(base_env, "venv"):
        base_env = base_env.venv
    if hasattr(base_env, "envs"):
        base_env = base_env.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def find_wrapper(env, wrapper_type):
    base_env = env
    if hasattr(base_env, "venv"):
        base_env = base_env.venv
    if hasattr(base_env, "envs"):
        base_env = base_env.envs[0]
    while True:
        if isinstance(base_env, wrapper_type):
            return base_env
        if hasattr(base_env, "env"):
            base_env = base_env.env
        else:
            return None


class ActionMaskingDQNWrapper(gym.Wrapper):
    """
    Wrapper that applies action masking for DQN by modifying Q-values.
    Invalid actions get Q-value of -inf, preventing them from being selected.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def get_action_mask(self):
        """Get the action mask from the base ScoundrelEnv"""
        # Unwrap to get base environment
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env.action_masks()
    
    def action_masks(self):
        """Compatibility method for ActionMasker-style interface"""
        return self.get_action_mask()


def mask_q_values(q_values, action_mask):
    """
    Mask Q-values by setting invalid actions to -inf
    
    Args:
        q_values: Tensor of Q-values for each action
        action_mask: Boolean array where True = valid action
    
    Returns:
        Masked Q-values tensor
    """
    masked_q_values = q_values.clone()
    masked_q_values[~action_mask] = -float('inf')
    return masked_q_values


class MaskedDQN(DQN):
    """
    Custom DQN that applies action masking by setting invalid action Q-values to -inf
    Overrides both training and prediction action selection
    """
    
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """
        Override action sampling to apply action masking during training
        """
        # Get action mask from environment
        if hasattr(self.env, 'get_action_mask'):
            action_mask = self.env.get_action_mask()
        elif hasattr(self.env, 'action_masks'):
            action_mask = self.env.action_masks()
        else:
            # No masking, use default behavior
            return super()._sample_action(learning_starts, action_noise, n_envs)
        
        # Select action with epsilon-greedy and masking
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Random action from valid actions only
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if len(valid_actions) == 0:
                # Fallback if no valid actions (shouldn't happen)
                return np.array([self.action_space.sample()])
            action = np.array([np.random.choice(valid_actions)])
        else:
            # Epsilon-greedy with masking
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
            action = unscaled_action
        
        return action
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Override predict to apply action masking before selecting actions
        """
        # Get the base environment by unwrapping the VecEnv stack
        base_env = self.env
        
        # Unwrap VecNormalize, DummyVecEnv, FlattenObservation, Monitor to get to ActionMaskingDQNWrapper
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Get action mask from ActionMaskingDQNWrapper
        action_mask = None
        if hasattr(base_env, 'get_action_mask'):
            action_mask = base_env.get_action_mask()
        elif hasattr(base_env, 'action_masks'):
            action_mask = base_env.action_masks()
        
        # Get Q-values
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        
        with th.no_grad():
            q_values = self.policy.q_net(obs_tensor)
            
            # Apply action mask if available
            if action_mask is not None:
                action_mask_tensor = th.as_tensor(action_mask, dtype=th.bool, device=self.device)
                q_values_masked = q_values.clone()
                q_values_masked[~action_mask_tensor] = -float('inf')
                q_values = q_values_masked
            
            # Select action
            action = th.argmax(q_values, dim=1).reshape(-1)
        
        # Convert to numpy
        action = action.cpu().numpy()
        
        if not vectorized_env:
            action = action[0]
        
        return action, state


class MaskedDQNPolicy(DQNPolicy):
    """
    Custom DQN Policy that supports action masking.
    Masks invalid actions during action selection (predict).
    """
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Override predict to apply action masking
        """
        # Get Q-values from the network
        obs_tensor = th.as_tensor(observation, device=self.device)
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
        
        # Get action mask from environment wrapper
        # Note: This assumes the environment is wrapped with ActionMaskingDQNWrapper
        if hasattr(self, '_last_obs_wrapper'):
            action_mask = self._last_obs_wrapper.get_action_mask()
            action_mask_tensor = th.as_tensor(action_mask, dtype=th.bool, device=self.device)
            q_values = mask_q_values(q_values, action_mask_tensor)
        
        # Select action
        if deterministic:
            action = q_values.argmax(dim=1).cpu().numpy()
        else:
            # Epsilon-greedy with masking
            action = q_values.argmax(dim=1).cpu().numpy()
        
        return action, state


class DeckAnalysisWrapper(gym.Wrapper):
    """
    Adds DeckAnalyzer signals to info and optional reward shaping.
    Can also early-terminate hopeless decks based on winnability score.
    """
    def __init__(self, env, margin_reward_scale=0.0, early_terminate_threshold=None):
        super().__init__(env)
        self.margin_reward_scale = margin_reward_scale
        self.early_terminate_threshold = early_terminate_threshold
        self._last_margin = None
        self._force_done = False
        self._last_obs = None
        self._last_info = None

    def _compute_analysis(self):
        base_env = get_base_env(self.env)
        if base_env.game is None:
            return None
        analyzer = DeckAnalyzer(base_env.game.dungeon)
        return analyzer.analyze()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = dict(info)
        analysis = self._compute_analysis()
        self._last_margin = None
        self._force_done = False

        if analysis is not None:
            self._last_margin = analysis.winnability_score
            info["deck_margin"] = analysis.winnability_score
            info["deck_winnability"] = analysis.winnability_score
            info["deck_warnings"] = analysis.warnings
            info["deck_critical_issues"] = analysis.critical_issues

            if self.early_terminate_threshold is not None:
                if analysis.winnability_score < self.early_terminate_threshold:
                    self._force_done = True
                    info["early_terminated"] = True

        self._last_info = dict(info)
        return obs, info

    def step(self, action):
        if self._force_done:
            info = dict(self._last_info or {})
            info["early_terminated"] = True
            return self._last_obs, 0.0, True, False, info

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        analysis = self._compute_analysis()
        if analysis is not None:
            margin = analysis.winnability_score
            margin_delta = 0.0
            if self._last_margin is not None:
                margin_delta = margin - self._last_margin
            shaped_reward = self.margin_reward_scale * margin_delta
            reward = reward + shaped_reward

            info["deck_margin"] = margin
            info["deck_margin_delta"] = margin_delta
            info["deck_winnability"] = analysis.winnability_score
            info["deck_warnings"] = analysis.warnings
            info["deck_critical_issues"] = analysis.critical_issues

            self._last_margin = margin

        return obs, reward, terminated, truncated, info


class DFSEarlyTerminationWrapper(gym.Wrapper):
    """
    Uses DFS solver to prove when a game state is unwinnable and terminates early.
    More accurate than heuristic-based early termination.
    """
    def __init__(self, env, max_states=50000, check_frequency=1):
        super().__init__(env)
        self.max_states = max_states
        self.check_frequency = check_frequency
        self._step_count = 0
        self._last_obs = None
        self._last_info = None

    def _is_winnable_from_current_state(self) -> bool:
        """Check if game is still winnable from current state using DFS"""
        base_env = get_base_env(self.env)
        if base_env.game is None:
            return True
        
        game = base_env.game
        game_state = game.current_game_state()
        
        # Create a game state for DFS solver
        current_state = GameState(
            hp=game_state['player_state']['hp'],
            weapon_level=game_state['player_state']['weapon_level'],
            weapon_max_monster_level=game_state['player_state']['weapon_max_monster_level'],
            deck_position=len(game.dungeon.cards) - game_state['dungeon_state']['cards_remaining'],
            room_cards_taken=frozenset(),  # Current room state
            can_avoid=bool(game_state['room_state']['can_avoid']),
            can_heal=bool(game_state['room_state']['can_heal'])
        )
        
        # Run DFS solver from current state
        solver = DFSSolver(game.dungeon, max_states=self.max_states)
        # Manually check from current state instead of initial
        solver.visited.clear()
        solver.states_explored = 0
        path = []
        is_winnable = solver._dfs(current_state, path)
        
        return is_winnable

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = dict(info)
        self._step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._step_count += 1
        
        # Check winnability periodically (not every step for performance)
        if not terminated and not truncated and self._step_count % self.check_frequency == 0:
            if not self._is_winnable_from_current_state():
                # Game is proven unwinnable - terminate early
                info["dfs_unwinnable"] = True
                info["dfs_early_terminated"] = True
                terminated = True
                reward = reward - 5.0  # Small penalty for unwinnable state
        
        self._last_info = dict(info)
        return obs, reward, terminated, truncated, info


class DeckGeneratorWrapper(gym.Wrapper):
    """
    Replaces the dungeon deck on reset using SolvableDeckGenerator.
    Supports dynamic target winnability for curriculum learning.
    Can optionally sample real shuffled decks some of the time.
    """
    def __init__(
        self,
        env,
        difficulty="medium",
        target_winnability=0.7,
        max_attempts=100,
        real_deck_prob=0.0,
    ):
        super().__init__(env)
        self.difficulty = difficulty
        self.target_winnability = target_winnability
        self.max_attempts = max_attempts
        self.real_deck_prob = real_deck_prob

    def set_target_winnability(self, target_winnability):
        self.target_winnability = target_winnability

    def set_real_deck_prob(self, real_deck_prob):
        self.real_deck_prob = real_deck_prob

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_env = get_base_env(self.env)
        if base_env.game is not None:
            use_real_deck = random.random() < self.real_deck_prob
            if use_real_deck:
                # Use the already-shuffled dungeon from the base environment
                obs = base_env._get_obs()
                info = base_env._get_info()
                info["deck_generated"] = False
                info["deck_winnability"] = None
            else:
                dungeon, analysis = SolvableDeckGenerator.generate_solvable_deck(
                    difficulty=self.difficulty,
                    target_winnability=self.target_winnability,
                    max_attempts=self.max_attempts
                )
                base_env.game.dungeon = dungeon
                base_env.game.update_score()
                obs = base_env._get_obs()
                info = base_env._get_info()
                info["deck_generated"] = True
                info["deck_winnability"] = analysis.winnability_score
        return obs, info


class DeckCurriculumCallback(BaseCallback):
    """
    Updates DeckGeneratorWrapper target over time for curriculum learning.
    schedule: list of (progress, target_winnability)
    """
    def __init__(self, schedule, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(1, self.total_timesteps)
        target = self.schedule[-1][1]
        for p, t in self.schedule:
            if progress <= p:
                target = t
                break

        wrapper = find_wrapper(self.training_env, DeckGeneratorWrapper)
        if wrapper is not None:
            wrapper.set_target_winnability(target)

        return True

