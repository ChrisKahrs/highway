import gym
import os
import highway_env
from stable_baselines3 import PPO

TRIAL= 11
ALGO = "PPO"

models_dir = f"models/{ALGO}" + str(TRIAL)
log_dir = f"logs/{ALGO}" + str(TRIAL)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
config = {
  "simulation_frequency": 5,
  "show_trajectories": False,
  "observation": {
    "type": "Kinematics",
    "vehicles_count": 5,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {
      "x": [-100, 100],
      "y": [-100, 100],
      "vx": [-20, 20],
      "vy": [-20, 20]
    },
    "verbose": True,
    "absolute": False,
    "order": "sorted"
  }
}

env = gym.make("highway-fast-v0")
env.config = config

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"{ALGO}{TRIAL}" , reset_num_timesteps=False)
    model.save(f"{models_dir}/run{i}")
    print(f"Run {i} complete")
