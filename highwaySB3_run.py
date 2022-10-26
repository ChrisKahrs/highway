import gym
import highway_env
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from moviepy.editor import ImageSequenceClip
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
screen = env.render(mode='rgb_array')
plt.imshow(screen)
screens = []
screens.append(screen)
count = 0

# Load and test saved model
model = DQN.load("models/DQN9/run1")
reward_sum = 0
for i in range(3):
  terminated = truncated = done = False
  obs = env.reset()
  print("reward_sum: ", reward_sum)
  if count > 0:
    clip = ImageSequenceClip(list(screens), fps=5)
    clip.write_gif(f'test_{count}.gif', fps=15)
  reward_sum = 0
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    # print("action: ", action)
    obs, reward, done, info = env.step(action)
    # print("reward: ", reward)
    reward_sum += reward
    # env.render()
    screen = env.render(mode='rgb_array')
    # print(obs)
    screens.append(screen)
    plt.imshow(screen)
    count += 1

