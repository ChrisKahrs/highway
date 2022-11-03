import gym
import os
from stable_baselines3 import PPO, DQN, TD3

TRIAL= 2
ALGO = "TD3"

models_dir = f"LLmodels/{ALGO}" + str(TRIAL)
log_dir = f"LLlogs/{ALGO}" + str(TRIAL)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
env = gym.make("LunarLanderContinuous-v2")

model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1,1000):
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"{ALGO}{TRIAL}" , reset_num_timesteps=False)
    model.save(f"{models_dir}/run{i}")
    print(f"Run {i} complete")
