from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


import gym
import highway_env
env = gym.make("highway-v0", render_mode="rgb_array")
env.reset()
env.action_space.seed(42)
env = gym.wrappers.RecordVideo(env, "recording6")
env.start_video_recorder()

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()