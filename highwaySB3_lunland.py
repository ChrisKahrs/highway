import gym
import highway_env
from stable_baselines3 import PPO, DQN, TD3
from moviepy.editor import ImageSequenceClip
from gym.wrappers.record_video import capped_cubic_video_schedule
import pprint

# Video Stuff? 
    # from pyvirtualdisplay import Display
    # display = Display(visible=0, size=(1400, 900))
    # display.start()
    # env.metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}
    # env.metadata['render_fps'] = 10
    # env.configure({"offscreen_rendering": True})
    # env = gym.wrappers.RecordVideo(env, "gifs/recording5", metadata?)
    # env.start_video_recorder()

# "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],

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
    "vehicles_count": 5,
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
 'simulation_frequency': 5,
 'vehicles_count': 50,
 'vehicles_density': 1}

env = gym.make("LunarLander-v2", continuous=True, render_mode="human")

env.reset()
env = gym.wrappers.RecordEpisodeStatistics(env)
count = 0
total_sum = 0
range_sum = 30

# Load and test saved model
# model = DQN.load("models/DQN9/run5")
model = TD3.load("LLmodels/TD32/run340")

for i in range(range_sum):
  terminated = truncated = done = False
  obs, info = env.reset(seed=i)
  reward_sum = 0
  screens = []
  screen = env.render()
  screens.append(screen)
  while (not truncated) and (not terminated):
    action, _states = model.predict(obs, deterministic=True)
    action = [float(action[0]),float(action[1])]
    obs, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    screen = env.render()
    screens.append(screen)
    count += 1
    if terminated or truncated:
      print(f"reward_sum{i}: {reward_sum:.0f}")
      total_sum += reward_sum
      # clip = ImageSequenceClip(list(screens), fps=3)
      # clip.write_gif(f'gifs/test_{count}.gif', fps=3)

  
new_avg = total_sum / range_sum
print(f"total_avg: {new_avg:.2f}")

# pprint.pprint(env.return_queue)
env.close()

