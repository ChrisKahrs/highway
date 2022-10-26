from tabnanny import verbose
import gym
import os
import highway_env
from stable_baselines3 import PPO, DQN

TRIAL= 9

models_dir = f"models/dqn" + str(TRIAL)
log_dir = f"logs/dqn" + str(TRIAL)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

config = {'action': {'type': 'DiscreteMetaAction'},
 'absolute': False,
 'order': 'sorted',
 'verbose': True,
 'centering_position': [0.3, 0.5],
 'collision_reward': -0.6,
 'controlled_vehicles': 1,
 'duration': 40,
 'ego_spacing': 2,
 'high_speed_reward': 0.6,
 'initial_lane_id': None,
 'lane_change_reward': 0,
 'lanes_count': 5,
 'manual_control': False,
 'normalize_reward': True,
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
 'offroad_terminal': False,
 'offscreen_rendering': True,
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
 'vehicles_density': 1}
}

env = gym.make("highway-fast-v0")
env.config = config

print("1 - env created")
# model = PPO('MlpPolicy', env, tensorboard_log="highway_ppo3/", verbose=1)
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log=log_dir)
print("2 - model created")

TIMESTEPS = 100000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"DQN{TRIAL}" , reset_num_timesteps=False)
    model.save(f"{models_dir}/run{i}")
    print(f"Run {i} complete")
    lastCount = i

# model.learn(int(100000))
# model.save("highway_dqn2/model2")
print("3 - model saved")


# Load and test saved model
model = DQN.load(f"{models_dir}/run{lastCount}")
reward_sum = 0
for _ in range(100):
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