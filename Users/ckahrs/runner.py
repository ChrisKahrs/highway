import gym
import highway_env
# sdfasd

import gym
from gym.wrappers import Monitor
env = Monitor(gym.make('highway-v0'), './video', force=True)
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
env.close()
# # Agent
# from stable_baselines3 import DQN
# from utils import record_videos, show_videos

# env = gym.make("highway-v0")
# env = record_videos(env)
# model = DQN.load("asdf")
# for episode in range(1):
#     obs, done = env.reset(), False
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(int(action))
#         # env.render()
# env.close()
# show_videos()
