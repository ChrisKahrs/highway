import gym
import highway_env
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from moviepy.editor import ImageSequenceClip
import pprint
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()


config = {"observation": {
          "type": "Kinematics",
          "vehicles_count": 10,
          "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
          "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
          }}}

config = {'action': {'type': 'DiscreteMetaAction'},
 'centering_position': [0.3, 0.5],
 'collision_reward': -1,
 'controlled_vehicles': 1,
 'duration': 40,
 'ego_spacing': 2,
 'high_speed_reward': 0.4,
 'initial_lane_id': None,
 'lane_change_reward': 0,
 'lanes_count': 5,
 'manual_control': False,
 "observation": {
    "type": "Kinematics",
    # "vehicles_count": 30,
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {
      "x": [-100, 100],
      "y": [-100, 100],
      "vx": [-20, 20],
      "vy": [-20, 20]
    }},
 'offroad_terminal': False,
 'offscreen_rendering': False,
 'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
 'policy_frequency': 1,
 'real_time_rendering': False,
 'render_agent': True,
 'reward_speed_range': [20, 30],
 'right_lane_reward': 0.1,
 'scaling': 5.5,
 'screen_height': 150,
 'screen_width': 600,
 'show_trajectories': False,
 'simulation_frequency': 15,
 'vehicles_count': 50,
 'vehicles_density': 2}

env = gym.make("highway-v0", config=config)
# pprint.pprint(env.config)

env.metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}
env.metadata['render_fps'] = 10

env.reset()
# env.configure({"offscreen_rendering": True})
# env = gym.wrappers.RecordVideo(env, "gifs/recording5")
# env.start_video_recorder()
env = gym.wrappers.RecordEpisodeStatistics(env)
# screen = env.render(mode='rgb_array')
# plt.imshow(screen)
# screens = []
# screens.append(screen)
count = 0

# Load and test saved model
model = DQN.load("models/DQN9/run8")

for i in range(1):
  terminated = truncated = done = False
  obs, info = env.reset(seed=i)
  reward_sum = 0
  screens = []
  screen = env.render(mode='rgb_array')
  # plt.imshow(screen)
  screens.append(screen)
  while (not truncated) and (not terminated):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    # print("action: ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # print("reward: ", reward)
    reward_sum += reward
    # env.render()
    screen = env.render(mode='rgb_array')
    # print(obs)
    screens.append(screen)
    # plt.imshow(screen)
    count += 1
    if terminated or truncated:
      print("reward_sum: ", reward_sum)
      clip = ImageSequenceClip(list(screens), fps=3)
      clip.write_gif(f'gifs/test_{count}.gif', fps=3)

pprint.pprint(env.return_queue)
env.close()


