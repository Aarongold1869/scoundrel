"""
Train a DQN (Deep Q-Network) agent to play Scoundrel using Stable-Baselines3

Requirements:
pip install stable-baselines3[extra]
"""
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from scoundrel_ai.scoundrel_env import ScoundrelEnv, StrategyLevel
import os
import glob
import json
from datetime import datetime


def get_tensorboard_log_name(algorithm, strategy_level, timesteps):
    """Generate tensorboard log directory name with iteration number
    
    Format: {ALGORITHM}_{STRATEGY}_{timesteps}k_{iteration}
    Example: DQN_EXPERT_100k_1
    """
    # Convert strategy level to name
    if isinstance(strategy_level, int):
        strategy_level = StrategyLevel(strategy_level)
    strategy_name = strategy_level.name  # BASIC, INTERMEDIATE, ADVANCED, EXPERT
    
    # Format timesteps (100000 -> 100k, 1000000 -> 1000k)
    timesteps_k = f"{timesteps // 1000}k"
    
    # Find next iteration number
    pattern = f"./scoundrel_ai/tensorboard_logs/{algorithm.upper()}_{strategy_name}_{timesteps_k}_*"
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
    
    return f"./scoundrel_ai/tensorboard_logs/{algorithm.upper()}_{strategy_name}_{timesteps_k}_{iteration}"


def train_dqn(total_timesteps=100000, save_path="scoundrel_ai/models/scoundrel_dqn", strategy_level=StrategyLevel.EXPERT):
    """Train a DQN agent on Scoundrel"""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)
    
    # Create the environment
    env = ScoundrelEnv(strategy_level=strategy_level)
    env = Monitor(env, "scoundrel_ai/logs", info_keywords=())
    
    # Validate the environment
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Create evaluation environment
    eval_env = ScoundrelEnv(strategy_level=strategy_level)
    eval_env = Monitor(eval_env, info_keywords=())
    
    # Generate tensorboard log directory name
    tensorboard_log_dir = get_tensorboard_log_name("dqn", strategy_level, total_timesteps)
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
    
    # Initialize the DQN agent
    print("\nInitializing DQN agent...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  # Increased for faster learning
        buffer_size=100000,  # Larger buffer for more experience
        learning_starts=500,  # Start learning sooner
        batch_size=64,  # Larger batch for more stable updates
        tau=1.0,
        gamma=0.95,  # Slightly lower to value immediate rewards more
        train_freq=4,
        gradient_steps=2,  # More gradient steps per update
        target_update_interval=500,  # Update target network more frequently
        exploration_fraction=0.4,  # Explore longer
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # Keep some exploration
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=dict(net_arch=[256, 256])  # Larger network
    )
    
    # Train the agent
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True
    )
    
    # Save the final model
    model.save(f"{save_path}/final_model")
    print(f"\nModel saved to {save_path}/final_model")
    
    # Write tensorboard model name to text file
    tensorboard_name = os.path.basename(tensorboard_log_dir)
    model_name_file = f"{save_path}/model_name.txt"
    with open(model_name_file, 'w') as f:
        f.write(tensorboard_name)
    print(f"Model name saved to {model_name_file}: {tensorboard_name}")
    
    return model


def train_ppo(total_timesteps=100000, save_path="scoundrel_ai/models/scoundrel_ppo", strategy_level=StrategyLevel.EXPERT):
    """Train a PPO agent on Scoundrel (alternative algorithm)"""
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)
    
    env = ScoundrelEnv(strategy_level=strategy_level)
    env = Monitor(env, "scoundrel_ai/logs", info_keywords=())
    
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    eval_env = ScoundrelEnv(strategy_level=strategy_level)
    eval_env = Monitor(eval_env, info_keywords=())
    
    # Generate tensorboard log directory name
    tensorboard_log_dir = get_tensorboard_log_name("ppo", strategy_level, total_timesteps)
    print(f"Tensorboard logs will be saved to: {tensorboard_log_dir}")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="scoundrel_ai/logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    print("\nInitializing PPO agent...")
    model = PPO(
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
        tensorboard_log=tensorboard_log_dir
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=1,
        progress_bar=True
    )
    
    model.save(f"{save_path}/final_model")
    print(f"\nModel saved to {save_path}/final_model")
    
    # Write tensorboard model name to text file
    tensorboard_name = os.path.basename(tensorboard_log_dir)
    model_name_file = f"{save_path}/model_name.txt"
    with open(model_name_file, 'w') as f:
        f.write(tensorboard_name)
    print(f"Model name saved to {model_name_file}: {tensorboard_name}")
    
    return model


def evaluate_model(model_path, episodes=10):
    """Evaluate a trained model"""
    env = ScoundrelEnv(render_mode="human")
    
    # Load the model - auto-detect algorithm
    if "dqn" in model_path.lower():
        model = DQN.load(model_path, env=env)
    elif "ppo" in model_path.lower():
        model = PPO.load(model_path, env=env)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, env=env)
    else:
        # Try DQN by default
        try:
            model = DQN.load(model_path, env=env)
        except:
            model = PPO.load(model_path, env=env)
    
    total_rewards = []
    total_scores = []
    wins = 0
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\n{'='*60}")
        print(f"Evaluation Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")
        
        while not done:
            env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_scores.append(info['score'])
        if info['score'] > 0:
            wins += 1
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Score: {info['score']}")
        print(f"  HP: {info['hp']}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({episodes} episodes):")
    print(f"  Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"  Average Score: {sum(total_scores)/len(total_scores):.2f}")
    print(f"  Max Score: {max(total_scores):.2f}")
    print(f"  Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"{'='*60}")
    
    # Write scores to JSON file
    # Try to read model name from model_name.txt file
    model_dir = os.path.dirname(model_path)
    model_name_file = os.path.join(model_dir, "model_name.txt")
    
    if os.path.exists(model_name_file):
        with open(model_name_file, 'r') as f:
            model_name = f.read().strip()
    else:
        # Fall back to basename if file doesn't exist
        model_name = os.path.basename(model_path)
    
    scores_file = "scoundrel_ai/evaluation_scores.json"
    
    # Load existing scores if file exists
    if os.path.exists(scores_file):
        with open(scores_file, 'r') as f:
            scores_data = json.load(f)
    else:
        scores_data = []
    
    # Append new score entry
    score_entry = {
        "model": model_name,
        "average_reward": sum(total_rewards)/len(total_rewards),
        "average_score": sum(total_scores)/len(total_scores),
        "max_score": max(total_scores),
        "wins": f"{wins}/{episodes} {wins/episodes*100:.1f}",
        "timestamp": datetime.now().isoformat()
    }
    scores_data.append(score_entry)
    
    # Write back to file
    os.makedirs("scoundrel_ai", exist_ok=True)
    with open(scores_file, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    print(f"\nScore saved to {scores_file}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate Scoundrel RL agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--algorithm", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model for evaluation")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--strategy-level", type=int, default=4,
                        help="Strategy level: 1=BASIC, 2=INTERMEDIATE, 3=ADVANCED, 4=EXPERT (default)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        strategy_level = StrategyLevel(args.strategy_level)
        if args.algorithm == "dqn":
            train_dqn(total_timesteps=args.timesteps, strategy_level=strategy_level)
        elif args.algorithm == "ppo":
            train_ppo(total_timesteps=args.timesteps, strategy_level=strategy_level)
    else:
        if args.model_path is None:
            print("Error: --model-path is required for evaluation mode")
        else:
            evaluate_model(args.model_path, episodes=args.episodes)
