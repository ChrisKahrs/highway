import gym
import highway_env
from stable_baselines3 import PPO

env = gym.make("highway-v0")

# Load and test saved model
model = PPO.load("models/PPO10/run6")
reward_sum = 0
for i in range(10):
  done = truncated = False
  obs = env.reset()
  print("reward_sum: ", reward_sum)
  reward_sum = 0
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    print("action: ", action)
    obs, reward, done, info = env.step(action)
    print("reward: ", reward)
    reward_sum += reward
    env.render()