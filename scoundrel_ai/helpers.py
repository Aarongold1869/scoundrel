import glob


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
