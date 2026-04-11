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
    CallbackList,
    CheckpointCallback
)
from utils import ForceForwardWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf

model_dir = "models"
log_dir = "logs"

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

def make_env(cfg): #, idx, capture_video, run_name):
    def thunk():
        env = MujocoFormulaStudent(
            render_mode="rgb_array",
            num_checkpoints=cfg.env.num_checkpoints,
            lap_completion_reward=cfg.env.lap_completion_reward,
            max_env_step=cfg.env.max_env_step,
            # checkpoint_bonus_step=cfg.env.checkpoint_bonus_step,
            stuck_patience=cfg.env.stuck_patience,
            # max_throttle=cfg.env.max_throttle,
            forward_velocity_reward=cfg.env.forward_velocity_reward,    
            vel_penalty_weight=cfg.env.vel_penalty_weight,
            steer_penalty_weight=cfg.env.steer_penalty_weight,
            domain_randomization=cfg.env.domain_randomization,
            track_idx=cfg.env.track_idx,
            waypoint_mode=cfg.env.waypoint_mode,
            crash_penalty=cfg.env.crash_penalty
            )
        
        env.action_space.seed(cfg.seed)
        env.observation_space.seed(cfg.seed)
        # env = ForceForwardWrapper(env)
        return env    
    return thunk


def resolve_resume_model_path(model_path):
    if os.path.isdir(model_path):
        candidates = ["last.zip", "last_model.zip"]
        for filename in candidates:
            candidate_path = os.path.join(model_path, filename)
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(
            f"No checkpoint found in {model_path}. Expected one of: {candidates}"
        )

    if os.path.isfile(model_path):
        return model_path

    if os.path.isfile(f"{model_path}.zip"):
        return f"{model_path}.zip"

    raise FileNotFoundError(f"Could not find model checkpoint at: {model_path}")
    
    
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(config):
    run_name = f"MujocoFormulaStudent__{config.exp_name}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    if config.track_exp:
        wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

        wandb.init(
            project=config.wandb.project,
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

    resume_model_path = config.ppo.model
    is_resume = bool(resume_model_path)
    
    ## make train env
    train_env = make_vec_env(
        make_env(config),
        n_envs=config.env.num_envs,
    )
    train_env.seed(seed=config.seed)
    train_env = VecTransposeImage(train_env)
    
    norm_keys = ["car_states"]
    if config.env.waypoint_mode != "none":
        norm_keys.append("checkpoints")    

    if is_resume:
        resolved_model_path = resolve_resume_model_path(resume_model_path)
        vecnormalize_path = os.path.join(os.path.dirname(resolved_model_path), "vecnormalize.pkl")
        if not os.path.exists(vecnormalize_path):
            raise FileNotFoundError(f"Could not find VecNormalize stats at: {vecnormalize_path}")

        print(f"Resuming training from model: {resolved_model_path}")
        print(f"Loading VecNormalize stats from: {vecnormalize_path}")
        train_env = VecNormalize.load(vecnormalize_path, train_env)
        train_env.training = True
        
        
    else:

        train_env = VecNormalize(train_env,
                                norm_obs=True,
                                norm_reward=True,
                                norm_obs_keys=norm_keys
                                 )


    if config.capture_video:
        train_env = VecVideoRecorder(
            train_env,
            video_folder=f"videos/{run_name}",
            # record_video_trigger=
            record_video_trigger=lambda step: step % 2000 == 0,
            video_length=500,
            name_prefix="ppo-car"
        )

    if is_resume:
        model = PPO.load(
            resolved_model_path,
            env=train_env,
            device=device,
            tensorboard_log=log_dir,
        )
    else:
        model = PPO("MultiInputPolicy", 
                    train_env, 
                    verbose=1, 
                    tensorboard_log=log_dir,
                    device=device,
                    n_steps=config.ppo.n_steps,
                    batch_size=config.ppo.batchsize,
                    ent_coef=config.ppo.entropy_coef
                    )
        
    eval_env = make_vec_env(
        make_env(config),
        n_envs=1,
    )
    eval_env.seed(seed=config.seed)
    eval_env = VecTransposeImage(eval_env)
    # eval_env = VecNormalize(eval_env, training=False)
    # CORRECT - should exclude image same as train_env
    eval_env = VecNormalize(
        eval_env, 
        training=False,
        norm_obs=True,
        norm_reward=False,  # don't normalise reward at eval
        norm_obs_keys=norm_keys
    )
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
        
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

    checkpoint_callback = CheckpointCallback(save_freq=5000, 
                                             save_path=model_run_dir,
                                             name_prefix="ppo_checkpoint")


    # cb_list = [eval_callback]
    cb_list = []
    cb_list.append(checkpoint_callback)

    if config.track_exp:
        wandb_cb = WandbCallback(
            model_save_path=model_run_dir,
            verbose=2
        )
        cb_list.append(wandb_cb)
 
    try:
        model.learn(total_timesteps=config.ppo.total_timesteps, 
                    tb_log_name=run_name, 
                    progress_bar=True,
                    reset_num_timesteps=not is_resume,
                    callback=CallbackList(cb_list)
                    )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    model.save(os.path.join(model_run_dir, 'last_model'))
    train_env.save(os.path.join(model_run_dir, "vecnormalize.pkl"))

    if config.track_exp:
        wandb.finish(exit_code=0)

if __name__ == "__main__":
    main()