"""
Train a DQN (Deep Q-Network) agent to play Scoundrel using Stable-Baselines3

Requirements:
pip install stable-baselines3[extra]
"""
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
import torch as th
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from scoundrel_ai.scoundrel_env import ScoundrelEnv
from scoundrel_ai.deck_analysis.deck_analyzer import DeckAnalyzer, SolvableDeckGenerator, DFSSolver, GameState
import os
import glob
import json
import random
from datetime import datetime
import numpy as np
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
    """
    def __init__(self, env, difficulty="medium", target_winnability=0.7, max_attempts=100):
        super().__init__(env)
        self.difficulty = difficulty
        self.target_winnability = target_winnability
        self.max_attempts = max_attempts

    def set_target_winnability(self, target_winnability):
        self.target_winnability = target_winnability

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_env = get_base_env(self.env)
        if base_env.game is not None:
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


def get_tensorboard_log_name(algorithm, timesteps):
    """Generate tensorboard log directory name with iteration number
    
    Format: {ALGORITHM}_{timesteps}k_{iteration}
    Example: DQN_100k_1
    """
    # Format timesteps (100000 -> 100k, 1000000 -> 1000k)
    timesteps_k = f"{timesteps // 1000}k"
    
    # Find next iteration number
    pattern = f"./scoundrel_ai/tensorboard_logs/{algorithm.upper()}_{timesteps_k}_*"
    existing_dirs = glob.glob(pattern)
    
    if not existing_dirs:
        iteration = 1
    else:
        # Extract iteration numbers and find max
        iterations = []
        for dir_path in existing_dirs:
            try:
                iter_num = int(dir_path.split('_')[-1])
                iterations.append(iter_num)
            except (ValueError, IndexError):
                pass
        iteration = max(iterations) + 1 if iterations else 1
    
    return f"./scoundrel_ai/tensorboard_logs/{algorithm.upper()}_{timesteps_k}_{iteration}"


def train_dqn(
    total_timesteps=100000,
    save_path="scoundrel_ai/models/scoundrel_dqn",
    use_deck_analysis=False,
    early_terminate_threshold=None,
    margin_reward_scale=0.0,
    use_curriculum=False,
    curriculum_schedule=None,
    use_dfs_termination=False,
    dfs_max_states=50000,
    dfs_check_frequency=5,
):
    """Train a DQN agent on Scoundrel"""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)
    
    # Create the environment
    env = ScoundrelEnv()
    if use_curriculum:
        env = DeckGeneratorWrapper(env, target_winnability=0.8)
    env = ActionMaskingDQNWrapper(env)  # Add action masking wrapper
    if use_dfs_termination:
        env = DFSEarlyTerminationWrapper(env, max_states=dfs_max_states, check_frequency=dfs_check_frequency)
    if use_deck_analysis:
        env = DeckAnalysisWrapper(
            env,
            margin_reward_scale=margin_reward_scale,
            early_terminate_threshold=early_terminate_threshold,
        )
    env = Monitor(env, "scoundrel_ai/logs", info_keywords=())
    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Validate the environment
    print("Checking environment...")
    test_env = ScoundrelEnv()
    test_env = Monitor(test_env, info_keywords=())
    test_env = FlattenObservation(test_env)
    check_env(test_env, warn=True)
    print("Environment check passed!")
    
    # Create evaluation environment
    eval_env = ScoundrelEnv()
    if use_curriculum:
        eval_env = DeckGeneratorWrapper(eval_env, target_winnability=0.8)
    eval_env = ActionMaskingDQNWrapper(eval_env)  # Add action masking wrapper
    if use_dfs_termination:
        eval_env = DFSEarlyTerminationWrapper(eval_env, max_states=dfs_max_states, check_frequency=dfs_check_frequency)
    if use_deck_analysis:
        eval_env = DeckAnalysisWrapper(
            eval_env,
            margin_reward_scale=0.0,
            early_terminate_threshold=None,
        )
    eval_env = Monitor(eval_env, info_keywords=())
    eval_env = FlattenObservation(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    # Generate tensorboard log directory name
    tensorboard_log_dir = get_tensorboard_log_name("dqn", total_timesteps)
    print(f"Tensorboard logs will be saved to: {tensorboard_log_dir}")
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="scoundrel_ai/logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    callbacks = [eval_callback]

    if use_curriculum:
        if curriculum_schedule is None:
            curriculum_schedule = [
                (0.33, 0.8),
                (0.66, 0.6),
                (1.0, 0.45),
            ]
        callbacks.append(DeckCurriculumCallback(curriculum_schedule, total_timesteps))
    
    # Initialize the DQN agent with action masking
    print("\nInitializing DQN agent with action masking...")
    model = MaskedDQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Lower learning rate for stability
        buffer_size=200000,  # Larger buffer for more experience
        learning_starts=1500,  # Start learning after 1000 steps for better data
        batch_size=64,  # Reasonable batch size
        tau=1.0,
        gamma=0.99,  # Standard discount factor
        train_freq=4,
        gradient_steps=2,  # More gradient steps per update
        target_update_interval=500,  # Update target network every 1000 steps
        exploration_fraction=0.5,  # Explore for 50% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,   # Low final exploration for fine-tuning
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=dict(net_arch=[256, 256])  # Larger network for complex observations
    )
    
    # Train the agent
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )
    
    # Save the final model
    model.save(f"{save_path}/final_model")
    print(f"\nModel saved to {save_path}/final_model")

    # Save VecNormalize statistics
    env.save(f"{save_path}/vec_normalize.pkl")
    print(f"VecNormalize stats saved to {save_path}/vec_normalize.pkl")
    
    # Write tensorboard model name to text file
    tensorboard_name = os.path.basename(tensorboard_log_dir)
    model_name_file = f"{save_path}/model_name.txt"
    with open(model_name_file, 'w') as f:
        f.write(tensorboard_name)
    print(f"Model name saved to {model_name_file}: {tensorboard_name}")
    
    return model


def train_ppo(
    total_timesteps=100000,
    save_path="scoundrel_ai/models/scoundrel_ppo",
    use_deck_analysis=False,
    early_terminate_threshold=None,
    margin_reward_scale=0.0,
    use_curriculum=False,
    curriculum_schedule=None,
    use_dfs_termination=False,
    dfs_max_states=50000,
    dfs_check_frequency=5,
):
    """Train a PPO agent on Scoundrel (alternative algorithm)"""
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)

    def mask_fn(env: ScoundrelEnv):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        masks = base_env.action_masks()
        return masks
    
    def make_env():
        env = ScoundrelEnv()
        if use_curriculum:
            env = DeckGeneratorWrapper(env, target_winnability=0.8)
        env = ActionMasker(env, mask_fn)      # ActionMasker calls mask_fn
        if use_dfs_termination:
            env = DFSEarlyTerminationWrapper(env, max_states=dfs_max_states, check_frequency=dfs_check_frequency)
        if use_deck_analysis:
            env = DeckAnalysisWrapper(
                env,
                margin_reward_scale=margin_reward_scale,
                early_terminate_threshold=early_terminate_threshold,
            )
        env = Monitor(env, "scoundrel_ai/logs", info_keywords=())
        env = FlattenObservation(env)
        return env

    # env = make_env()
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    
    print("Checking environment (non-vectorized version)...")
    test_env = ScoundrelEnv()
    test_env = ActionMasker(test_env, mask_fn)
    test_env = Monitor(test_env, info_keywords=())
    test_env = FlattenObservation(test_env)
    check_env(test_env, warn=True)
    print("Environment check passed!")
    
    eval_env = ScoundrelEnv()
    if use_curriculum:
        eval_env = DeckGeneratorWrapper(eval_env, target_winnability=0.8)
    eval_env = ActionMasker(eval_env, mask_fn)  # Apply ActionMasker first
    if use_dfs_termination:
        eval_env = DFSEarlyTerminationWrapper(eval_env, max_states=dfs_max_states, check_frequency=dfs_check_frequency)
    if use_deck_analysis:
        eval_env = DeckAnalysisWrapper(
            eval_env,
            margin_reward_scale=0.0,
            early_terminate_threshold=None,
        )
    eval_env = Monitor(eval_env, info_keywords=())
    eval_env = FlattenObservation(eval_env)

    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # eval_env = DummyVecEnv([make_env])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    # Generate tensorboard log directory name
    tensorboard_log_dir = get_tensorboard_log_name("ppo", total_timesteps)
    print(f"Tensorboard logs will be saved to: {tensorboard_log_dir}")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="scoundrel_ai/logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    callbacks = [eval_callback]

    if use_curriculum:
        if curriculum_schedule is None:
            curriculum_schedule = [
                (0.33, 0.8),
                (0.66, 0.6),
                (1.0, 0.45),
            ]
        callbacks.append(DeckCurriculumCallback(curriculum_schedule, total_timesteps))
    
    print("\nInitializing PPO agent...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=dict(net_arch=[256, 256])  # Larger network for complex observations
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=1,
        progress_bar=True
    )
    
    model.save(f"{save_path}/final_model")
    print(f"\nModel saved to {save_path}/final_model")
    
    # Save VecNormalize statistics
    env.save(f"{save_path}/vec_normalize.pkl")
    print(f"VecNormalize stats saved to {save_path}/vec_normalize.pkl")
    
    # Write tensorboard model name to text file
    tensorboard_name = os.path.basename(tensorboard_log_dir)
    model_name_file = f"{save_path}/model_name.txt"
    with open(model_name_file, 'w') as f:
        f.write(tensorboard_name)
    print(f"Model name saved to {model_name_file}: {tensorboard_name}")
    
    return model


def evaluate_model(model_path, episodes=10, gameplay_path=None, eval_deck_seed=None):
    """Evaluate a trained model and save best gameplay replay"""
    # Detect algorithm and apply appropriate wrappers
    is_ppo = "ppo" in model_path.lower()
    is_dqn = "dqn" in model_path.lower()
    
    if eval_deck_seed is not None:
        random.seed(eval_deck_seed)
        np.random.seed(eval_deck_seed)
        th.manual_seed(eval_deck_seed)

    env = ScoundrelEnv(render_mode="human", eval=True)
    
    # Apply wrappers for DQN (requires ActionMaskingDQNWrapper)
    if is_dqn:
        env = ActionMaskingDQNWrapper(env)

    # Apply wrappers for PPO (MaskablePPO requires ActionMasker and FlattenObservation)
    elif is_ppo:
        def mask_fn(env: gym.Env):
            while hasattr(env, 'env'):
                env = env.env
            return env.action_masks()
        env = ActionMasker(env, mask_fn)

    # Consistent wrapper order with training
    env = Monitor(env, info_keywords=())  # Add Monitor like in training
    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize statistics if available
    model_dir = os.path.dirname(model_path)
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vec_normalize_path}")
    else:
        print("Warning: No VecNormalize stats found, running without normalization")
    
    # Load the model - auto-detect algorithm
    if "dqn" in model_path.lower():
        model = MaskedDQN.load(model_path, env=env)
    elif is_ppo:
        model = MaskablePPO.load(model_path, env=env)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, env=env)
    else:
        # Try DQN by default
        try:
            model = MaskedDQN.load(model_path, env=env)
        except:
            model = MaskablePPO.load(model_path, env=env)

    # Try to read model name from model_name.txt file
    model_dir = os.path.dirname(model_path)
    model_name_file = os.path.join(model_dir, "model_name.txt")
    
    if os.path.exists(model_name_file):
        with open(model_name_file, 'r') as f:
            model_name = f.read().strip()
    else:
        # Fall back to basename if file doesn't exist
        model_name = os.path.basename(model_path)
    
    scores_file = "high_scores.json"
    if gameplay_path is None:
        gameplay_path = os.path.join(model_dir, "gameplay.json")
    
    # Load existing scores if file exists
    if os.path.exists(scores_file):
        with open(scores_file, 'r') as f:
            scores_data = json.load(f)
    else:
        scores_data = []
    
    total_rewards = []
    total_scores = []
    wins = 0
    best_score = float("-inf")
    best_gameplay = None
    
    # Environment is always vectorized (wrapped in DummyVecEnv)
    is_vec_env = True
    
    def serialize_card(card):
        return {
            "symbol": card.symbol,
            "suit": card.suit,
            "val": card.val,
            "id": card.id,
        }

    for episode in range(episodes):
        if is_vec_env:
            obs = env.reset()
        else:
            obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = []
        step_index = 0
        
        print(f"\n{'='*60}")
        print(f"Evaluation Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")
        
        while not done:
            if not is_vec_env:
                env.render()
            action, _states = model.predict(obs, deterministic=True)
            base_env = get_base_env(env)
            if base_env.game is not None:
                game_state = base_env.game.current_game_state()
                room_cards = [serialize_card(card) for card in game_state["room_state"]["cards"]]
                player_state = game_state["player_state"]
                room_state = game_state["room_state"]
                dungeon_state = game_state['dungeon_state']
                deck_analysis = DeckAnalyzer(base_env.game.dungeon).analyze()
                deck_margin = deck_analysis.winnability_score
                deck_warnings = deck_analysis.warnings
                deck_critical_issues = deck_analysis.critical_issues
                future_danger = 1.0 - deck_margin
            else:
                room_cards = []
                player_state = {}
                room_state = {}
                dungeon_state = {}
                game_state = {}
                deck_margin = None
                deck_warnings = []
                deck_critical_issues = []
                future_danger = None

            if isinstance(action, (list, tuple, np.ndarray)):
                action_value = int(action[0])
            else:
                action_value = int(action)

            action_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "a"}
            action_str = action_map.get(action_value, str(action_value))

            episode_steps.append({
                "step": step_index,
                "room_cards": room_cards,
                "action": action_value,
                "action_str": action_str,
                "score": game_state.get("score", 0) if base_env.game else 0,
                "hp": player_state.get("hp", 0),
                "weapon_level": player_state.get("weapon_level", 0),
                "weapon_max_monster_level": player_state.get("weapon_max_monster_level", 0),
                "can_avoid": room_state.get("can_avoid", 0),
                "cards_remaining": dungeon_state.get("cards_remaining", 0),
                "deck_margin": deck_margin,
                "deck_warnings": deck_warnings,
                "deck_critical_issues": deck_critical_issues,
                "future_danger": future_danger,
            })
            step_index += 1
            
            if is_vec_env:
                obs, reward, done, info = env.step(action)
                done = done[0]  # VecEnv returns arrays
                reward = reward[0]
                info = info[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        total_scores.append(info['score'])
        if info['score'] >= 0:
            wins += 1

        if info['score'] > best_score:
            best_score = info['score']
            best_gameplay = {
                "model_name": model_name,
                "episode": episode + 1,
                "score": info['score'],
                "steps": episode_steps,
                "timestamp": datetime.now().isoformat(),
            }

        # Append new score entry
        scores_data.append({
            "name": model_name,
            "score": info['score'],
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Score: {info['score']}")
        print(f"  HP: {info['player_state']['hp']}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({episodes} episodes):")
    print(f"  Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"  Average Score: {sum(total_scores)/len(total_scores):.2f}")
    print(f"  Max Score: {max(total_scores):.2f}")
    print(f"  Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"{'='*60}")
    
    # Write back to file
    with open(scores_file, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    print(f"\nScore saved to {scores_file}")

    if best_gameplay is not None:
        with open(gameplay_path, "w") as f:
            json.dump(best_gameplay, f, indent=2)
        print(f"Best gameplay saved to {gameplay_path}")
    
    env.close()


def replay(gameplay_path="gameplay.json"):
    """Replay the best gameplay saved during evaluation"""
    if not os.path.exists(gameplay_path):
        print(f"Gameplay file not found: {gameplay_path}")
        return

    with open(gameplay_path, "r") as f:
        gameplay = json.load(f)

    print(f"\nReplaying model: {gameplay.get('model_name', 'unknown')}")
    print(f"Score: {gameplay.get('score', 'unknown')} | Episode: {gameplay.get('episode', 'unknown')}")
    print("Press Enter to step through moves. Type 'q' to quit.\n")

    for step in gameplay.get("steps", []):
        room_cards = step.get("room_cards", [])
        action_str = step.get("action_str", step.get("action"))
        
        # Extract player stats
        score = step.get("score", 0)
        hp = step.get("hp", 0)
        weapon_level = step.get("weapon_level", 0)
        weapon_max = step.get("weapon_max_monster_level", 0)
        can_avoid = step.get("can_avoid", 0)
        cards_remaining = step.get("cards_remaining", 0)

        print(f"Player State: HP={hp}, Weapon={weapon_level} ({weapon_max}), Score={score}")
        print(f"Dungeon: {cards_remaining} cards remaining | Can Avoid={bool(can_avoid)}")
        
        if room_cards:
            print(f"\nCurrent Room ({len(room_cards)} cards):")
            for idx, card in enumerate(room_cards, 1):
                suit = card.get("suit", {})
                suit_symbol = suit.get("symbol", "?")
                card_symbol = card.get("symbol", "?")
                card_val = card.get("val", "?")
                card_class = suit.get("class", "?")
                print(f"  [{idx}] {card_symbol}{suit_symbol} (val: {card_val}, class: {card_class})")
        else:
            print("Current Room: (no data)")

        print(f"\n{'='*60}")
        print(f"Step {step.get('step')}: Action -> {action_str}")
        print(f"{'='*60}")

        user_input = input("\nNext step (Enter) or 'q' to quit: ")
        if user_input.strip().lower() == "q":
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train, evaluate, or replay Scoundrel RL agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "replay"],
                        help="Mode: train, eval, or replay")
    parser.add_argument("--algorithm", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model for evaluation")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--gameplay-path", type=str, default=None,
                        help="Path to gameplay.json for saving or replaying")
    parser.add_argument("--eval-deck-seed", type=int, default=None,
                        help="Seed for deterministic evaluation decks")
    parser.add_argument("--use-deck-analysis", action="store_true",
                        help="Enable DeckAnalyzer signals for shaping and labeling")
    parser.add_argument("--margin-reward-scale", type=float, default=0.0,
                        help="Reward shaping scale for +Δmargin")
    parser.add_argument("--early-terminate-threshold", type=float, default=-1.0,
                        help="Early terminate if winnability score is below this value")
    parser.add_argument("--use-curriculum", action="store_true",
                        help="Enable curriculum learning with generated decks")
    parser.add_argument("--curriculum-easy", type=float, default=0.8,
                        help="Target winnability for early curriculum phase")
    parser.add_argument("--curriculum-medium", type=float, default=0.6,
                        help="Target winnability for mid curriculum phase")
    parser.add_argument("--curriculum-hard", type=float, default=0.45,
                        help="Target winnability for late curriculum phase")
    parser.add_argument("--use-dfs-termination", action="store_true",
                        help="Enable DFS solver for exact unwinnable detection and early termination")
    parser.add_argument("--dfs-max-states", type=int, default=50000,
                        help="Maximum states for DFS solver to explore per check")
    parser.add_argument("--dfs-check-frequency", type=int, default=5,
                        help="Check winnability every N steps (1=every step)")
    
    args = parser.parse_args()
    early_terminate_threshold = None
    if args.early_terminate_threshold >= 0:
        early_terminate_threshold = args.early_terminate_threshold

    curriculum_schedule = None
    if args.use_curriculum:
        curriculum_schedule = [
            (0.33, args.curriculum_easy),
            (0.66, args.curriculum_medium),
            (1.0, args.curriculum_hard),
        ]
    
    if args.mode == "train":
        if args.algorithm == "dqn":
            train_dqn(
                total_timesteps=args.timesteps,
                use_deck_analysis=args.use_deck_analysis,
                early_terminate_threshold=early_terminate_threshold,
                margin_reward_scale=args.margin_reward_scale,
                use_curriculum=args.use_curriculum,
                curriculum_schedule=curriculum_schedule,
                use_dfs_termination=args.use_dfs_termination,
                dfs_max_states=args.dfs_max_states,
                dfs_check_frequency=args.dfs_check_frequency,
            )
        elif args.algorithm == "ppo":
            train_ppo(
                total_timesteps=args.timesteps,
                use_deck_analysis=args.use_deck_analysis,
                early_terminate_threshold=early_terminate_threshold,
                margin_reward_scale=args.margin_reward_scale,
                use_curriculum=args.use_curriculum,
                curriculum_schedule=curriculum_schedule,
                use_dfs_termination=args.use_dfs_termination,
                dfs_max_states=args.dfs_max_states,
                dfs_check_frequency=args.dfs_check_frequency,
            )
    elif args.mode == "eval":
        if args.model_path is None:
            print("Error: --model-path is required for evaluation mode")
        else:
            evaluate_model(
                args.model_path,
                episodes=args.episodes,
                gameplay_path=args.gameplay_path,
                eval_deck_seed=args.eval_deck_seed,
            )
    else:
        replay(gameplay_path=args.gameplay_path or "gameplay.json")
