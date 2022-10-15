from tabnanny import verbose
import gym
import highway_env
from stable_baselines3 import PPO

config = {
  "simulation_frequency": 15,
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

env = gym.make("highway-v0")
env.config = config

print("1 - env created")
model = PPO('MlpPolicy', env, tensorboard_log="highway_ppo/", verbose=1)
print("2 - model created")
model.learn(int(100000))
model.save("highway_ppo/model")
print("3 - model saved")


# Load and test saved model
model = PPO.load("highway_ppo/model2")
reward_sum = 0
while True:
  done = truncated = False
  obs = env.reset()
  print("reward_sum: ", reward_sum)
  reward_sum = 0
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    obs, reward, done, info = env.step(action)
    print("reward: ", reward)
    reward_sum += reward
    env.render()