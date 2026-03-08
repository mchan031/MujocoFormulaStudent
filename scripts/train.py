import gymnasium as gym
import numpy as np
from env import MujocoFormulaStudent
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecVideoRecorder, VecNormalize
import random
import torch
import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    StopTrainingOnRewardThreshold, 
    StopTrainingOnNoModelImprovement, 
    EvalCallback, 
    CallbackList
)
from utils import ForceForwardWrapper

seed = 42
model_dir = "models"
log_dir = "logs"
mujoco_path = "mujoco_tracks/sim_env.xml"
checkpoint_path = "mujoco_tracks/checkpoints_20.json"
# def parse_args():
#     parser = argparse.ArgumentParser()

    # pass


def make_env(model_path, run_name, seed=10, capture_video=False): #, idx, capture_video, run_name):

    def thunk():
        env = MujocoFormulaStudent(
            model_path=model_path,
            render_mode="human",
            checkpoint_file=checkpoint_path
        )
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        if capture_video:
            env = gym.wrappers.RecordVideo(env, 
                                     f"videos/{run_name}", 
                                     episode_trigger=lambda x: x % 10 == 0)
                    
    

        env = ForceForwardWrapper(env)

        return env


    return thunk
    
    

def main():
    # pass
    # args = parse_args()

    run_name = f"MujocoFormulaStudent__{seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_path = os.path.join(os.path.dirname(__file__), os.path.pardir, mujoco_path)
    
    ## make env
    train_env = make_vec_env(
        make_env(full_path, run_name, seed, False),
        n_envs=1,
    )
    train_env = VecTransposeImage(train_env)
    train_env = VecNormalize(train_env)
    
    # eval_env = make_vec_env(
    #     make_env(full_path, run_name, seed, True),
    #     n_envs=1,
    # )
    # eval_env = VecTransposeImage(eval_env)

    model = PPO("CnnPolicy", 
                train_env, 
                # verbose=1, 
                tensorboard_log=log_dir,
                device=device,
                n_steps=1024
                )

    model_run_dir = os.path.join(model_dir, run_name)
    os.makedirs(model_run_dir, exist_ok=True)
    
    
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=400, 
    #                                                  verbose=1)

    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, 
    #                                                        min_evals = 10000, 
    #                                                        verbose=1
    #                                                        )

    # eval_callback = EvalCallback(
    #     eval_env=eval_env,
    #     eval_freq=5000,
    #     callback_on_new_best=callback_on_best,
    #     callback_after_eval=stop_train_callback,
    #     verbose=1,
    #     best_model_save_path=model_run_dir
    # )
    
    # train_env.observation_space
    # print(f"{train_env.observation_space = }")
    # raise
    
    model.learn(total_timesteps=10_000, 
                tb_log_name=run_name, 
                # callback=CallbackList([eval_callback]),
                progress_bar=True
                )
    
    model.save(os.path.join(model_run_dir, 'last_model'))
    
if __name__ == "__main__":
    main()