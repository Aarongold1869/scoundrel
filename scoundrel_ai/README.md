# Scoundrel RL Training

This directory contains code for training reinforcement learning agents to play the Scoundrel card game.

## Setup

1. Install required packages:
```bash
pip install gymnasium stable-baselines3[extra] tensorboard
```

2. Update your requirements.txt:
```bash
pip freeze > scoundrel_ai/requirements.txt
```

## Project Structure

- `scoundrel_env.py` - Custom Gymnasium environment wrapping the Scoundrel game
- `train_random.py` - Test the environment with a random agent
- `train_dqn.py` - Train and evaluate RL agents (DQN, PPO)

## Usage

### Test Environment with Random Agent
```bash
python -m scoundrel_ai.train_random
```

### Train a DQN Agent
```bash
python -m scoundrel_ai.train_dqn --mode train --algorithm dqn --timesteps 100000
```

### Train a PPO Agent
```bash
python -m scoundrel_ai.train_dqn --mode train --algorithm ppo --timesteps 100000
```

### Evaluate a Trained Model
```bash
python -m scoundrel_ai.train_dqn --mode eval --model-path scoundrel_ai/models/scoundrel_dqn/best_model --episodes 10
```

### Monitor Training with TensorBoard
```bash
tensorboard --logdir ./scoundrel_ai/tensorboard_logs/
```

## Environment Details

### Observation Space
The observation is an 18-dimensional vector containing:
- HP (0-20)
- Weapon level (0-14)
- Weapon max monster level (0-15)
- Cards remaining (0-44)
- Can avoid room (0/1)
- Can heal (0/1)
- Current room state (4 cards × 3 features each):
  - Card class (0=weapon, 1=health, 2=monster)
  - Card value (2-14)
  - Suit type (0=diamond, 1=heart, 2=spade, 3=club)

### Action Space
5 discrete actions:
- 0-3: Interact with card 1-4 in current room
- 4: Avoid current room (if allowed)

### Reward Function
The reward is shaped to encourage:
- Gaining HP (+0.5 per HP)
- Improving score (+0.1 per score point)
- Making progress through dungeon (+0.2 per card)
- Winning the game (+50)
- Surviving to the end (+10)
- Avoiding death (-20)
- Avoiding invalid actions (-2 to -5)

## Algorithms

### DQN (Deep Q-Network)
- Value-based method
- Good for discrete action spaces
- Uses experience replay
- Recommended for Scoundrel due to strategic decision-making

### PPO (Proximal Policy Optimization)
- Policy gradient method
- More stable training
- Good general-purpose algorithm
- May require more timesteps

## Tips for Training

1. **Start with fewer timesteps** (10k-50k) to verify training works
2. **Monitor TensorBoard** for reward curves and success rates
3. **Adjust reward shaping** if agent learns unwanted behaviors
4. **Try different algorithms** - DQN and PPO have different strengths
5. **Increase training time** - 500k+ timesteps may be needed for good performance

## Extending the Environment

To improve the RL agent:

1. **Better observation encoding**: Include weapon degradation info, past actions
2. **Smarter reward shaping**: Reward picking up weapons early, avoiding monsters without weapons
3. **Curriculum learning**: Start with easier dungeons, increase difficulty
4. **Ensemble methods**: Train multiple agents and combine their decisions
5. **Add recurrence**: Use LSTM/GRU to handle sequential decision-making better
