import gym
import numpy as np
import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC, TD3, A2C
# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Reward function to encourage backflip
import numpy as np

def reward_function(obs, action, next_obs):
    # Check if observation vector contains quaternion components
    if len(obs) < 7:
        print("Observation vector does not contain quaternion components.")
        return 0

    # Extract quaternion components from observation
    q_x, q_y, q_z, q_w = obs[3:7]

    # Calculate the rotation matrix from the quaternion components
    rotation_matrix = np.array([
        [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x*q_y - q_z*q_w), 2 * (q_x*q_z + q_y*q_w)],
        [2 * (q_x*q_y + q_z*q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y*q_z - q_x*q_w)],
        [2 * (q_x*q_z - q_y*q_w), 2 * (q_y*q_z + q_x*q_w), 1 - 2 * (q_x**2 + q_y**2)]
    ])

    # Define the upward direction (world up)
    upward_direction = np.array([0, 0, 1])

    # Calculate the dot product of the upward direction and the torso orientation
    dot_product = np.dot(rotation_matrix[2], upward_direction)

    # Encourage the agent to achieve a specific orientation for a backflip
    # Here, we can check if the torso is oriented upside-down
    # We want the dot product to be close to -1 (indicating upside-down)
    upside_down_reward = max(0, -dot_product)  # Reward for being upside-down

    # Scale the upside-down reward based on the weight
    orientation_reward_weight = 0.5
    orientation_reward = orientation_reward_weight * upside_down_reward

    return orientation_reward

# Train function
def train(env, sb3_algo):
    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: env])

    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        obs = env.reset()  # Reset environment
        done = [False]
        while not all(done):
            action, _ = model.predict(obs)
            next_obs, rewards, done, info = env.step(action)
            rewards = reward_function(obs, action, next_obs)  # Apply custom reward function
            model.learn(total_timesteps=1, reset_num_timesteps=False)  # Incremental learning
            obs = next_obs

        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

# Test function
def test(env, sb3_algo, path_to_model):
    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: env])

    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return
    obs = env.reset()
    done = [False]
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if all(done):
            extra_steps -= 1
            if extra_steps < 0:
                break

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        # Create the Gym environment
        env = gym.make(args.gymenv, render_mode=None)
        train(env, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            # Create the Gym environment with render_mode parameter
            env = gym.make(args.gymenv, render_mode='human')
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
