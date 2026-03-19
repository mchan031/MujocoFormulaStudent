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
    BaseCallback,
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
            checkpoint_bonus_step=cfg.env.checkpoint_bonus_step,
            max_throttle=cfg.env.max_throttle,
            vel_penalty_weight=cfg.env.vel_penalty_weight,
            steer_penalty_weight=cfg.env.steer_penalty_weight,
            domain_randomization=cfg.env.domain_randomization,
            track_idx=cfg.env.track_idx,
        )

        env.action_space.seed(cfg.seed)
        env.observation_space.seed(cfg.seed)
        # env = ForceForwardWrapper(env)
        return env

    return thunk


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_cfg, verbose=0):
        super().__init__(verbose=verbose)
        self.stages = [OmegaConf.to_container(stage, resolve=True) for stage in curriculum_cfg]
        self.active_stage_idx = None

    def _select_stage_idx(self, timestep: int) -> int:
        for idx, stage in enumerate(self.stages):
            start = int(stage.get("start_timestep", 0))
            end = stage.get("end_timestep")
            if timestep >= start and (end is None or timestep < int(end)):
                return idx
        return len(self.stages) - 1

    def _apply_stage(self, stage_idx: int) -> None:
        stage = self.stages[stage_idx]
        if self.verbose:
            print(f"Applying curriculum stage {stage_idx}: {stage.get('name', f'stage_{stage_idx}')}")

        self.training_env.env_method("set_track_mode", stage["track_mode"], stage.get("track_indices"))
        self.training_env.env_method("set_domain_randomization", bool(stage["domain_randomization"]))
        self.training_env.env_method("set_max_throttle", float(stage["max_throttle"]))
        self.training_env.env_method("set_episode_budget", int(stage["max_env_step"]), float(stage["checkpoint_bonus_step"]))
        self.training_env.env_method("set_checkpoint_layout", int(stage["num_checkpoints"]))
        self.training_env.set_attr("curriculum_stage_name", stage.get("name", f"stage_{stage_idx}"))

    def _on_training_start(self) -> None:
        initial_stage_idx = self._select_stage_idx(self.num_timesteps)
        self._apply_stage(initial_stage_idx)
        self.active_stage_idx = initial_stage_idx

    def _on_step(self) -> bool:
        stage_idx = self._select_stage_idx(self.num_timesteps)
        if stage_idx != self.active_stage_idx:
            self._apply_stage(stage_idx)
            self.active_stage_idx = stage_idx
        return True


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

    ## make train env
    train_env = make_vec_env(
        make_env(config),
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
        make_env(config),
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
                                                           min_evals=10000,
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

    cb_list = [CurriculumCallback(config.curriculum, verbose=1)]

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
