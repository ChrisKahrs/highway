import gymnasium as gym
import requests
import json
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human", continuous= True)
obs = env.reset()

# General variables
# url = "http://localhost:5000"
url = "https://ckll.azurewebsites.net"
predictionPath = "/v1/prediction"
headers = {
  "Content-Type": "application/json"
}
endpoint = url + predictionPath
info = {}

for _ in range(3):
  terminated = truncated = done = False
  obs, _ = env.reset()
  reward_sum = 0
  # print("info: ", info)
  while not terminated:
    # print("obs: ", obs)

    requestBody = {
      "x_position": float(obs[0]),
      "y_position": float(obs[1]),
      "x_velocity": float(obs[2]),
      "y_velocity": float(obs[3]),
      "angle": float(obs[4]),
      "rotation": float(obs[5]),
      "left_leg": float(obs[6]),
      "right_leg": float(obs[7])
    }
    
    # print("requestBody: ", requestBody)

# Send the POST request
    response = requests.post(
                endpoint,
                data = json.dumps(requestBody),
                headers = headers
              )

    # Extract the JSON response
    prediction = response.json()
    # print("prediction: ", prediction)
    action = [float(prediction["engine1"]),float(prediction["engine2"])]
    # print("action: ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    # print("reward: ", reward)
    env.render()
  print("reward_sum: ", reward_sum)
    
    