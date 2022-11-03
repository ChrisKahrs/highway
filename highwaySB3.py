import gym
import highway_env
from stable_baselines3 import DQN, PPO, TD3

env = gym.make("LunarLander-v2", continuous= True)
# print("1 - env created")
# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#               buffer_size=15000,
#               learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               tensorboard_log="highway_dqn/")
# print("2 - model created")
# model.learn(int(100000))

# model.save("highway_dqn/model99")
# print("3 - model saved")


# Load and test saved model
model = TD3.load("LLmodels/TD32/run288")
reward_sum = 0
for i in range(3):
  done = terminated = truncated = False
  obs = env.reset()
  reward_sum = 0
  while not (done):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    print("action: ", action)
    obs, reward, done, info = env.step(action)
    # print("reward: ", reward)
    reward_sum += reward
    env.render()
  print("reward_sum: ", reward_sum)