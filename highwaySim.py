import gym
import highway_env


class highwaySim:
  def __init__(self):
    config = {
      "simulation_frequency": 5,
      "show_trajectories": False,
      "observation": {
          "type": "Kinematics",
          "vehicles_count": 15,
          "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
          "features_range": {
              "x": [-100, 100],
              "y": [-100, 100],
              "vx": [-20, 20],
              "vy": [-20, 20]
          },
          "absolute": False,
          "order": "sorted"
      }
    }
    self.env = gym.make('highway-v0')
    self.env.configure(config)
    self.env.reset()
  
  def reset(self, config):
    # use config to set initial state before reset in gym
    obs = self.env.reset()
    return_dict = { 'sim_halted': False }
    return_dict.update({"vehicle1": obs[0].tolist(), 
                        "vehicle2": obs[1].tolist(), 
                        "vehicle3": obs[2].tolist(),
                        "vehicle4": obs[3].tolist(),
                        "vehicle5": obs[4].tolist()})
        
    return_dict.update({"gym_reward": 0})
    return_dict.update({"gym_terminal": False})
    print("reset: ", return_dict)
    
    return return_dict
  
  def step(self, action):
    
    # done = truncated = False

      specific_action = action['steer']
      obs, reward, done, info = self.env.step(specific_action)
      
      return_dict = { 'sim_halted': False }
      return_dict.update({"vehicle1": obs[0].tolist(), 
                          "vehicle2": obs[1].tolist(), 
                          "vehicle3": obs[2].tolist(),
                          "vehicle4": obs[3].tolist(),
                          "vehicle5": obs[4].tolist()})
        
      return_dict.update({"gym_reward": float(reward)})
      return_dict.update({"gym_terminal": done})

      print("step: ", return_dict)
      
      return return_dict