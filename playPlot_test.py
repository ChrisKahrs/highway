import gym
import numpy as np
from gym.utils.play import play, PlayPlot

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
       return [rew,]
   
plotter = PlayPlot(callback, 150, ["reward"])

play(gym.make("CarRacing-v2", callback=plotter.callback, 
              render_mode="rgb_array", keys_to_action={
                                               "w": np.array([0, 0.7, 0]),
                                               "a": np.array([-1, 0, 0]),
                                               "s": np.array([0, 0, 1]),
                                               "d": np.array([1, 0, 0]),
                                               "wa": np.array([-1, 0.7, 0]),
                                               "dw": np.array([1, 0.7, 0]),
                                               "ds": np.array([1, 0, 1]),
                                               "as": np.array([-1, 0, 1]),
                                              }, noop=np.array([0,0,0])))