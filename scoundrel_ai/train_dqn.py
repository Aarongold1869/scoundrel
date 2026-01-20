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
from scoundrel_ai.scoundrel_env import ScoundrelEnv
import os


def train_dqn(total_timesteps=100000, save_path="scoundrel_ai/models/scoundrel_dqn"):
    """Train a DQN agent on Scoundrel"""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)
    
    # Create the environment
    env = ScoundrelEnv()
    env = Monitor(env, "scoundrel_ai/logs")
    
    # Validate the environment
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Create evaluation environment
    eval_env = ScoundrelEnv()
    eval_env = Monitor(eval_env)
    
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
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./scoundrel_ai/tensorboard_logs/"
    )
    
    # Train the agent
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=100
    )
    
    # Save the final model
    model.save(f"{save_path}/final_model")
    print(f"\nModel saved to {save_path}/final_model")
    
    return model


def train_ppo(total_timesteps=100000, save_path="scoundrel_ai/models/scoundrel_ppo"):
    """Train a PPO agent on Scoundrel (alternative algorithm)"""
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("scoundrel_ai/logs", exist_ok=True)
    
    env = ScoundrelEnv()
    env = Monitor(env, "scoundrel_ai/logs")
    
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    eval_env = ScoundrelEnv()
    eval_env = Monitor(eval_env)
    
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
        tensorboard_log="./scoundrel_ai/tensorboard_logs/"
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=10
    )
    
    model.save(f"{save_path}/final_model")
    print(f"\nModel saved to {save_path}/final_model")
    
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
    print(f"  Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"{'='*60}")
    
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
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.algorithm == "dqn":
            train_dqn(total_timesteps=args.timesteps)
        elif args.algorithm == "ppo":
            train_ppo(total_timesteps=args.timesteps)
    else:
        if args.model_path is None:
            print("Error: --model-path is required for evaluation mode")
        else:
            evaluate_model(args.model_path, episodes=args.episodes)
