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
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf

model_dir = "models"
log_dir = "logs"
# mujoco_path = "mujoco_tracks/sim_env.xml"
# checkpoint_path = "mujoco_tracks/checkpoints.json"
# centreline_path = "mujoco_tracks/centreline.csv"

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

def make_env(model_path, cfg): #, idx, capture_video, run_name):
    def thunk():
        env = MujocoFormulaStudent(
            model_path=model_path,
            render_mode="rgb_array",
            centreline_file=cfg.env.centreline_path,
            num_checkpoints=cfg.env.num_checkpoints,
            lap_completion_reward=cfg.env.lap_completion_reward,
            max_env_step=cfg.env.max_env_step,
            checkpoint_bonus_step=cfg.env.checkpoint_bonus_step,
            max_throttle=cfg.env.max_throttle,
            )
        
        env.action_space.seed(cfg.seed)
        env.observation_space.seed(cfg.seed)
        env = ForceForwardWrapper(env)
        return env    
    return thunk
    
    
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(config):
    run_name = f"MujocoFormulaStudent__{config.exp_name}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    full_path = os.path.join(os.path.dirname(__file__), os.path.pardir, config.env.mujoco_path)


    if config.track_exp:
        wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

        wandb.init(
            project="mujoco-racing",
            name=run_name,
            config=wandb_config,
            sync_tensorboard=True,  # automatically logs tensorboard
            monitor_gym=True,       # logs episode rewards
            save_code=True
        )        

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.cuda_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and config.cuda else 'cpu')
    
    ## make train env
    train_env = make_vec_env(
        make_env(full_path, config),
        n_envs=config.env.num_envs,
    )
    train_env.seed(seed=config.seed)
    train_env = VecTransposeImage(train_env)
    train_env = VecNormalize(train_env)

    if config.capture_video:
        train_env = VecVideoRecorder(
            train_env,
            video_folder=f"videos/{run_name}",
            # record_video_trigger=
            record_video_trigger=lambda step: step % 2000 == 0,
            video_length=500,
            name_prefix="ppo-car"
        )
        
    eval_env = make_vec_env(
        make_env(full_path, config),
        n_envs=1,
    )
    eval_env.seed(seed=config.seed)
    eval_env = VecTransposeImage(eval_env)
    # eval_env = VecNormalize(eval_env)
    eval_env = VecNormalize(eval_env, training=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    model = PPO("MultiInputPolicy", 
                train_env, 
                verbose=1, 
                tensorboard_log=log_dir,
                device=device,
                n_steps=config.ppo.n_steps,
                batch_size=config.ppo.batchsize,
                )

    model_run_dir = os.path.join(model_dir, run_name)
    os.makedirs(model_run_dir, exist_ok=True)
    
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=600, 
                                                     verbose=1)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, 
                                                           min_evals = 10000, 
                                                           verbose=1
                                                           )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=2000,
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_train_callback,
        verbose=1,
        best_model_save_path=model_run_dir
    )

    # cb_list = [eval_callback]
    cb_list = []

    if config.track_exp:
        wandb_cb = WandbCallback(
            model_save_path=model_run_dir,
            verbose=2
        )
        cb_list.append(wandb_cb)

    model.learn(total_timesteps=config.ppo.total_timesteps, 
                tb_log_name=run_name, 
                progress_bar=True,
                callback=CallbackList(cb_list)
                )
    
    model.save(os.path.join(model_run_dir, 'last_model'))
    train_env.save(os.path.join(model_run_dir, "vecnormalize.pkl"))

    if config.track_exp:
        wandb.finish(exit_code=0)

if __name__ == "__main__":
    main()