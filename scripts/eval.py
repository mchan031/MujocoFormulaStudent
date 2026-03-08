import os
import gymnasium as gym
import numpy as np
from env import MujocoFormulaStudent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from utils import ForceForwardWrapper


mujoco_path = "mujoco_tracks/sim_env.xml"
checkpoint_path = "mujoco_tracks/checkpoints.json"
seed = 42
device = "cpu"

model_path = "models/MujocoFormulaStudent__testing__2026-03-08_16-46-30/model.zip"
# model_path = "wandb/run-20260308_164630-k5enfe6y/files/model.zip"

def make_env(env_path, checkpoint_path, seed=42):
    """Create a single environment instance"""
    def _init():
        env = MujocoFormulaStudent(
            model_path=env_path,
            render_mode="human",
            checkpoint_file=checkpoint_path,
            # lap_completion_reward=2000
        )
        env = ForceForwardWrapper(env)
        env.reset(seed=seed)
        return env
    return _init

def eval():
    # Get full path
    full_path = os.path.join(os.path.dirname(__file__), os.path.pardir, mujoco_path)
    
    # # Create environment
    # env = MujocoFormulaStudent(
    #     model_path=full_path,
    #     render_mode="human",
    #     checkpoint_file=checkpoint_path,
    #     lap_completion_reward=2000
    # )
    
        # Create vectorized environment (single env for evaluation)
    # eval_env = DummyVecEnv([make_env(full_path, checkpoint_path, seed)])

    # eval_env = make_vec_env(make_env)
    eval_env = make_vec_env(
        make_env(full_path, checkpoint_path, seed),
        n_envs=1,
    )
    # Apply the same wrappers as during training
    eval_env = VecTransposeImage(eval_env)  # This transposes (84,84,3) to (3,84,84)
    # eval_env = VecNormalize(eval_env, training=False)  # Set training=False for evaluation
        # print(eval_env.observation_space)
    eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    # raise
    np.random.seed(seed)
    
    # Load the trained model (NOT create a new one)
    model = PPO.load(
        path=model_path,
        env=eval_env,
        device=device
    )
    
    # Evaluation loop
    num_episodes = 2000
    for episode in range(num_episodes):
        obs = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (done):
            # Get action from model
            # print(obs)
            # raise
            action, _ = model.predict(obs, deterministic=False)
            
            # print(action[0])
            # print(action[1])
            # action[0][1] *= -1
            # raise
            # Take step in environment
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            step_count += 1
            # Optional: render is handled automatically with render_mode="human"
            # print(obs)
            # raise
            if step_count % 10 == 0:
                # print(f"{total_reward = } {info = }")
                print(action)
        
        # print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {step_count}")
    
    eval_env.close()

if __name__ == "__main__":
    eval()