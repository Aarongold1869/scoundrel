from scoundrel_ai.scoundrel_env import ScoundrelEnv
import numpy as np

env = ScoundrelEnv()
obs, info = env.reset()

print('Initial state:')
print(f'Cards remaining: {info["room_state"]["cards_remaining"]}')
print(f'Action mask: {env.action_masks()}')
print(f'Can avoid: {info["room_state"]["can_avoid"]}')

# Test different scenarios
for i in range(5):
    mask = env.action_masks()
    print(f'\nStep {i}:')
    print(f'  Cards: {info["room_state"]["cards_remaining"]}')
    print(f'  Mask: {mask}')
    print(f'  Valid actions: {np.where(mask)[0]}')
    
    # Take a valid action
    valid_actions = np.where(mask)[0]
    if len(valid_actions) > 0:
        action = valid_actions[0]
        print(f'  Taking action: {action}')
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    else:
        print('  No valid actions!')
        break
