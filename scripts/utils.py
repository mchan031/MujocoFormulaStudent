from collections import deque
import numpy as np
import gymnasium as gym
import cv2
from gymnasium import spaces

class MovingAverageFilter:
    def __init__(self, max_len=5):
        self.value_history = deque(maxlen=max_len)
        
    def compute(self, value):
        self.value_history.append(value)
        return np.mean(self.value_history)
    

class ForceForwardWrapper(gym.Wrapper):

    def __init__(self, env, force_steps=50, force_episodes=5):
        super().__init__(env)
        self.force_steps = force_steps
        self.force_episodes = force_episodes
        self.episode = 0
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        self.episode += 1
        return self.env.reset(**kwargs)

    def step(self, action):

        if self.episode < self.force_episodes and self.t < self.force_steps:
            # action = np.array([0.0, 1.0])  #throttle forward

            action = [0.0]
            throttle = np.random.random()/2 + 0.5
            action.append(throttle)
            action = np.array(action)


        self.t += 1

        # print(f"Steering: {action[0]} Throttle: {action[1]}")
        return self.env.step(action)

class GrayscaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        h, w, _ = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = gray[:, :, None]   # add channel dimension
        return gray.astype(np.uint8)