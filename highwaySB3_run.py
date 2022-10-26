import gym
import highway_env
from stable_baselines3 import PPO
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
env.reset(seed=42)
env.configure({"offscreen_rendering": True})
env = gym.wrappers.RecordVideo(env, "recording4", env.metadata)
env.start_video_recorder()

# Load and test saved model
model = PPO.load("models/PPO10/run6")
reward_sum = 0
for i in range(1):
  terminated = truncated = False
  obs, info = env.reset(seed=42)
  print("reward_sum: ", reward_sum)
  reward_sum = 0
  while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    print("action: ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    print("reward: ", reward)
    reward_sum += reward
    print(env.render())