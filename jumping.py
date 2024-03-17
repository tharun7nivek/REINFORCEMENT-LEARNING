import gym
import numpy as np
import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC, TD3, A2C
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# Define your custom environment with the custom reward function
class CustomEnv(gym.Env):
    def __init__(self):
        self.env = gym.make('Humanoid-v4')
        # Initialize any other variables or parameters you need
        
    def step(self, action):
        # Execute the action and obtain the next observation, reward, done flag, and additional info
        next_observation = np.random.randn(10)  # Example: Replace with actual environment dynamics
        reward = self.custom_reward_function(self.observation, action, next_observation)
        done = self.current_step >= self.max_steps
        info = {}  # Additional info, if needed
        
        # Update the current observation
        self.observation = next_observation
        self.current_step += 1
        
        return next_observation, reward, done, info

    def reset(self):
        # Reset the environment to its initial state
        self.observation = np.random.randn(10)  # Example: Replace with actual initial state
        self.current_step = 0
        return self.observation

    def render(self, mode='human'):
        # Render the environment
        pass  # Add rendering code if necessary

    def close(self):
        # Close any resources or connections
        pass

    def custom_reward_function(self, observation, action, next_observation):
        # Define your custom reward function here
        # Example: Compute the reward based on the difference between observations
            jump_reward_weight = 0.5
            stability_reward_weight = 0.3
            fall_penalty_weight = -1.0
            
            # Extract relevant observations
            vertical_velocity = observation[0][24]  # Vertical velocity
            orientation = observation[0][3:7]  # Orientation of the torso (quaternion)
            
            # Calculate jump reward based on vertical velocity
            jump_reward = jump_reward_weight * vertical_velocity
            
            # Calculate stability reward based on orientation deviation
            desired_orientation = np.array([0, 0, 1, 0])  # Desired upright orientation
            orientation_difference = np.dot(orientation, desired_orientation)
            stability_reward = stability_reward_weight * orientation_difference
            
            # Penalize falling or large deviations from desired orientation
            if abs(orientation_difference) < 0.95:  # Threshold for upright orientation
                fall_penalty = 0  # No penalty if close to upright
            else:
                fall_penalty = fall_penalty_weight
            
            # Calculate total reward
            reward = jump_reward + stability_reward + fall_penalty
            
            return reward


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
            rewards = [env.custom_reward_function(obs, action, next_obs)]
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
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
        env = CustomEnv()  # Use your custom environment
        train(env, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            # Create the Gym environment with render_mode parameter
            env = gym.make(args.gymenv, render_mode='human')
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')