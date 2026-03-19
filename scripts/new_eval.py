import os
import gymnasium as gym
from env import MujocoFormulaStudent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from matplotlib.collections import LineCollection
import torch
import cv2
from utils import plot_racing_line, TelemetryStorage

def make_env(cfg):
    """Create a single environment instance"""
    
    
    def _init():
        global track_cones_path

        env = MujocoFormulaStudent(
            render_mode="rgb_array",
            num_checkpoints=cfg.env.num_checkpoints,
            lap_completion_reward=cfg.env.lap_completion_reward,
            max_env_step=cfg.env.max_env_step,
            checkpoint_bonus_step=cfg.env.checkpoint_bonus_step,
            max_throttle=1.0,
            domain_randomization=False,
            track_idx=4
        )
        # env.reset(seed=cfg.seed)
        
        track_cones_path = env.random_track_path
        
        return env
    return _init

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

@hydra.main(config_path=FILE_PATH, config_name="eval", version_base=None)
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() and config.cuda else 'cpu')
    
    model_path = config.ppo.model_dir + "/model.zip"
    vecnormalize_path = config.ppo.model_dir + "/vecnormalize.pkl"
    
    eval_env = make_vec_env(
        make_env(config),
        n_envs=1,
    )
    eval_env = VecTransposeImage(eval_env)  
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False  # Set to evaluation mode
    
    if config.capture_video:
        video_folder = os.path.join("videos", f"eval_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
        eval_env = VecVideoRecorder(
            eval_env,
            video_folder=video_folder,
            record_video_trigger=lambda x: x == 0,  # Record only the first episode
            video_length=config.video_length,  # Max length of the video
            name_prefix=config.video_name
        )
    
    model = PPO.load(path=model_path,
                     env=eval_env,
                     device=device
                     )

    total_reward = 0
    step_count = 0
    done = False
    obs = eval_env.reset()
    num_lap = 0
    lap_time_list = []
    lap_start_time = time.time()
    start_time = time.time()
    telemetry = TelemetryStorage()
    
    if config.plot_racing_line:
        trajectory = []
        speed_profile = []
        
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        step_count += 1

        car_states = info[0]["car_states"]
        current_time = time.time() - start_time
        telemetry.append(step_count, car_states, current_time)
        
        if config.capture_video and step_count >= config.video_length:
            break
        
        if not config.capture_video:
            frame = eval_env.envs[0].render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if config.plot_racing_line:
            car_pos = info[0]["car_pos"]

            x, y, _ = car_pos
            long_vel = car_states[0]
            lat_vel = car_states[1]
            speed = np.sqrt(long_vel**2 + lat_vel**2)
            trajectory.append([x, y])
            speed_profile.append(speed)


        if done:
            print("Colllision!!!!!")
            
        if config.max_step is not None and step_count >= config.max_step:
            print("Max step reached, ending evaluation.")
            break
        
        if info[0]["progress"] - num_lap == 1.0:
            time_now = time.time()
            lap_time = time_now - lap_start_time
            lap_time_list.append(lap_time)
            lap_start_time = time_now
            num_lap += 1
            
            print("-" * 30)
            print("    Lap Time    ")
            print("-" * 30)
            
            for i, item in enumerate(lap_time_list):
                print(f"Lap: {i + 1} Time: {item}")
            
            print()
            print("-" * 30)
            
        if num_lap >= config.num_laps:
            print("Max laps reached, ending evaluation.")
            break
        
    cv2.destroyAllWindows()
    telemetry.plot_telemetry()
    eval_env.close()
    if config.plot_racing_line:
        plot_racing_line(trajectory, speed_profile, track_cones_path, lap_time_list)
        

if __name__ == "__main__":
    main()