import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from typing import Optional, Tuple, Dict, Any



# if __name__ == "__main__":  
    
    
class MujocoFormulaStudentEnv(gym.Env):
    def __init__(self, model_path: str):
        super(MujocoFormulaStudentEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self._setup_action_space()
        self._setup_observation_space()
        
    
    def _setup_action_space(self):
        # Define the action space based on the model's control inputs
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        
    def _setup_observation_space(self):
        # Define the observation space based on the model's state variables
        obs_dim = self.model.nq + self.model.nv  # positions + velocities
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        pass
    
    
