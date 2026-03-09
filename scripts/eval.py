import os
import gymnasium as gym
import numpy as np
from env import MujocoFormulaStudent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from utils import ForceForwardWrapper
import cv2

mujoco_path = "mujoco_tracks/sim_env.xml"
centreline_path = "mujoco_tracks/centreline.csv"
seed = 42
device = "cpu"

# model_path = "models/MujocoFormulaStudent__testing__2026-03-08_16-46-30/model.zip"
model_path = "d:/Users/mchan031/Downloads/last_model.zip"
# model_path = "wandb/run-20260308_164630-k5enfe6y/files/model.zip"

def make_env(env_path, centreline_path, seed=42):
    """Create a single environment instance"""
    def _init():
        env = MujocoFormulaStudent(
            model_path=env_path,
            render_mode="rgb_array",
            centreline_file=centreline_path,
            # extra_progress_time=2500
            # lap_completion_reward=2000
        )
        env.reset(seed=seed)
        return env
    return _init

def eval():
    # Get full path
    full_path = os.path.join(os.path.dirname(__file__), os.path.pardir, mujoco_path)
    
    # eval_env = make_vec_env(make_env)
    eval_env = make_vec_env(
        make_env(full_path, centreline_path, seed),
        n_envs=1,
    )
    # Apply the same wrappers as during training
    eval_env = VecTransposeImage(eval_env)  # This transposes (84,84,3) to (3,84,84)
    eval_env = VecNormalize.load("d:/Users/mchan031/Downloads/vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False    
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder="./videos",
        record_video_trigger=lambda step: step == 0,  # start recording immediately
        video_length=2000,  # max steps to record
        name_prefix="fs_unseen_track",
    )
    
    np.random.seed(seed)
    
    # Load the trained model (NOT create a new one)
    model = PPO.load(
        path=model_path,
        env=eval_env,
        device=device
    )
    
    # Evaluation loop
    num_episodes = 1
    for episode in range(num_episodes):
        obs = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            step_count += 1
            # Optional: render is handled automatically with render_mode="human"
            # print(obs)
            # raise
        #             # Show camera
        #     frame = eval_env.envs[0].render()

        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     cv2.imshow("View", frame)
        #     # cv2.imshow("View", cv2.resize(frame, (256, 256)))
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
            
        #     if step_count % 10 == 0:
        #         # print(f"{total_reward = } {info = }")
        #         # print(action)
        #         print(f"{info[0]["progress"] = }")
        
        # # print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {step_count}")
    
    eval_env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    eval()