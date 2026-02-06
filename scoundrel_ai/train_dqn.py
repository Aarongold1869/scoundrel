"""
Train a DQN (Deep Q-Network) agent to play Scoundrel using Stable-Baselines3

Requirements:
pip install stable-baselines3[extra]
"""
from scoundrel_ai.helpers import get_tensorboard_log_dir
from scoundrel_ai.deck_analysis.deck_analyzer import DeckAnalyzer
from scoundrel_ai.wrappers import (
    ActionMaskingDQNWrapper, 
    DeckAnalysisWrapper, 
    DeckCurriculumCallback, 
    DeckGeneratorWrapper, 
    DFSEarlyTerminationWrapper,
    MaskedDQN, 
    MaskedDQNPolicy,
    get_base_env
)
from scoundrel_ai.scoundrel_env import ScoundrelEnv

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import QRDQN
from sb3_contrib.common.wrappers import ActionMasker

import argparse
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os
import json
import random
from datetime import datetime
import numpy as np
import torch as th


def train_dqn(
    total_timesteps=100000,
    save_path="scoundrel_ai/models/scoundrel_dqn",
    disable_action_masking=False,
    use_deck_analysis=False,
    early_terminate_threshold=None,
    margin_reward_scale=0.0,
    use_curriculum=False,
    curriculum_schedule=None,
    real_deck_prob=0.0,
    use_dfs_termination=False,
    dfs_max_states=50000,
    dfs_check_frequency=5,
    use_qrdqn=False,
    tensorboard_log_name="dqn"
):
    """Train a DQN agent on Scoundrel"""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)
    
    # Create the environment
    env = ScoundrelEnv()
    if use_curriculum:
        env = DeckGeneratorWrapper(env, target_winnability=0.8, real_deck_prob=real_deck_prob)
    if not disable_action_masking:
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
        eval_env = DeckGeneratorWrapper(eval_env, target_winnability=0.8, real_deck_prob=real_deck_prob)
    if not disable_action_masking:
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
    tensorboard_log_dir = get_tensorboard_log_dir(tensorboard_log_name, total_timesteps)
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
    if use_qrdqn:
        model_class = QRDQN
        print('using QRDQN')
        
    else: 
        if not disable_action_masking:
            model_class = MaskedDQN
            # policy = MaskedDQNPolicy
            policy = "MlpPolicy"
            print('using MaskedDQN')
        else: 
            model_class = DQN
            policy = "MlpPolicy"
            print('using DQN')

    model = model_class(
        policy=policy,
        env=env,
        learning_rate=1e-4,  # Lower learning rate for stability
        buffer_size=200000,  # Larger buffer for more experience
        learning_starts=1500,  # Start learning after 1000 steps for better data
        batch_size=64,  # Reasonable batch size
        tau=1.0,
        gamma=0.99,  # Standard discount factor
        train_freq=4,
        gradient_steps=2,  # More gradient steps per update
        target_update_interval=10000,  # Update target network every 1000 steps
        exploration_fraction=0.3,  # Explore for 50% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,   # Low final exploration for fine-tuning
        verbose=0,
        tensorboard_log=tensorboard_log_dir,
        # n_steps=5,
        policy_kwargs=dict(net_arch = [256, 256, 128])  # Larger network for complex observations
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


def evaluate_model(
        model_path, 
        episodes=10, 
        disable_action_masking=False,
        gameplay_path=None, 
        eval_deck_seed=None,
        use_qrdqn=False
        ):
    """Evaluate a trained model and save best gameplay replay"""
    # Detect algorithm and apply appropriate wrappers
    # is_ppo = "ppo" in model_path.lower()
    # is_dqn = "dqn" in model_path.lower()
    
    if eval_deck_seed is not None:
        random.seed(eval_deck_seed)
        np.random.seed(eval_deck_seed)
        th.manual_seed(eval_deck_seed)

    env = ScoundrelEnv(render_mode="human", eval=True)
    
    # Apply wrappers for DQN (requires ActionMaskingDQNWrapper)
    if not disable_action_masking:
        env = ActionMaskingDQNWrapper(env)

    # Apply wrappers for PPO (MaskablePPO requires ActionMasker and FlattenObservation)
    # elif is_ppo:
    #     def mask_fn(env: gym.Env):
    #         while hasattr(env, 'env'):
    #             env = env.env
    #         return env.action_masks()
    #     env = ActionMasker(env, mask_fn)

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
    if use_qrdqn:
        model = QRDQN.load(model_path, env=env)
    else:
        if not disable_action_masking:
            model = MaskedDQN.load(model_path, env=env)
        else:
            model = DQN.load(model_path, env=env)

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
        
        # print(f"\n{'='*60}")
        # print(f"Evaluation Episode {episode + 1}/{episodes}")
        # print(f"{'='*60}")
        
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
        
        # print(f"\nEpisode {episode + 1} Results:")
        # print(f"  Reward: {episode_reward:.2f}")
        # print(f"  Score: {info['score']}")
        # print(f"  HP: {info['player_state']['hp']}")
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, or evaluate Scoundrel RL agent")

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
    parser.add_argument("--use-qrdqn", action="store_true",
                        help="use qrdqn model")
    # parser.add_argument("--use-drdqn", default="store_true",
    #                     help="use drdqn model")
    parser.add_argument("--eval-deck-seed", type=int, default=None,
                        help="Seed for deterministic evaluation decks")
    parser.add_argument("--disable-action-masking", action="store_true",
                        help="Enable Custom DQN Action Masking")
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
    parser.add_argument("--real-deck-prob", type=float, default=0.0,
                        help="Probability of using a real shuffled deck on reset")
    parser.add_argument("--use-dfs-termination", action="store_true",
                        help="Enable DFS solver for exact unwinnable detection and early termination")
    parser.add_argument("--dfs-max-states", type=int, default=50000,
                        help="Maximum states for DFS solver to explore per check")
    parser.add_argument("--dfs-check-frequency", type=int, default=5,
                        help="Check winnability every N steps (1=every step)")
    parser.add_argument("--gameplay-path", type=str, default=None,
                        help="Path to gameplay.json for saving or replaying")
    parser.add_argument("--tensorboard-log-name", type=str, default=None,
                        help="Name of tensorboard log dir")
    
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
                disable_action_masking=args.disable_action_masking,
                use_deck_analysis=args.use_deck_analysis,
                early_terminate_threshold=early_terminate_threshold,
                margin_reward_scale=args.margin_reward_scale,
                use_curriculum=args.use_curriculum,
                curriculum_schedule=curriculum_schedule,
                real_deck_prob=args.real_deck_prob,
                use_dfs_termination=args.use_dfs_termination,
                dfs_max_states=args.dfs_max_states,
                dfs_check_frequency=args.dfs_check_frequency,
                use_qrdqn=args.use_qrdqn,
                tensorboard_log_name=args.tensorboard_log_name
            )
    elif args.mode == "eval":
        if args.model_path is None:
            print("Error: --model-path is required for evaluation mode")
        else:
            evaluate_model(
                args.model_path,
                use_qrdqn=args.use_qrdqn,
                disable_action_masking=args.disable_action_masking,
                episodes=args.episodes,
                gameplay_path=args.gameplay_path,
                eval_deck_seed=args.eval_deck_seed,
            )
