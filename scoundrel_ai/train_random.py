"""
Example: Random agent playing Scoundrel
This demonstrates the environment without actual training
"""
import gymnasium as gym
from scoundrel_ai.scoundrel_env import ScoundrelEnv

def test_random_agent(episodes=5):
    """Test the environment with a random agent"""
    env = ScoundrelEnv(render_mode="human")
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n{'#'*60}")
        print(f"Starting Episode {episode + 1}")
        print(f"{'#'*60}")
        
        done = False
        while not done:
            env.render()
            
            # Random action
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            done = terminated or truncated
            
            if done:
                env.render()
                print(f"\n{'='*60}")
                print(f"Episode {episode + 1} finished!")
                print(f"Total Steps: {steps}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Final Score: {info['score']}")
                print(f"Final HP: {info['hp']}")
                print(f"{'='*60}\n")
        
    env.close()


if __name__ == "__main__":
    test_random_agent(episodes=3)
