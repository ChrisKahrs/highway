import gym
import highway_env
from stable_baselines3 import PPO, DQN
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

env = gym.make("highway-fast-v0")
env.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
env.metadata['render_fps'] = 10
env.reset()
# env.configure({"offscreen_rendering": True})
# env = gym.wrappers.RecordVideo(env, "recording4", env.metadata)
# env.start_video_recorder()

# Load and test saved model
model = DQN.load("models/DQN9/run1")
reward_sum = 0
for i in range(10):
  terminated = truncated = done = False
  obs = env.reset()
  print("reward_sum: ", reward_sum)
  reward_sum = 0
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    # print("action: ", action)
    obs, reward, done, info = env.step(action)
    # print("reward: ", reward)
    reward_sum += reward
    env.render()