# new_eval.py
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hydra
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecNormalize
from env import MujocoFormulaStudent
from utils import load_cones

FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config"
)


def make_env(cfg):
    def _init():
        env = MujocoFormulaStudent(
            render_mode="rgb_array",
            num_checkpoints=cfg.env.num_checkpoints,
            lap_completion_reward=cfg.env.lap_completion_reward,
            max_env_step=cfg.env.max_env_step,
            stuck_patience=cfg.env.stuck_patience,
            domain_randomization=False,
            track_idx=cfg.env.track_idx,
            waypoint_mode=cfg.env.get("waypoint_mode", "none"),
        )
        return env
    return _init


def build_eval_env(config, vecnorm_path):
    """Build and return eval env with normalisation loaded."""
    eval_env = make_vec_env(make_env(config), n_envs=1)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def run_episode(model, eval_env):
    """Run one episode deterministically. Returns metrics + telemetry."""
    obs = eval_env.reset()
    done = False

    # get raw env for sim params - unwrap Monitor wrapper
    raw_env = eval_env.venv.envs[0].unwrapped
    sim_dt = raw_env.model.opt.timestep
    frame_skip = raw_env.frame_skip

    ep_reward = 0.0
    step_count = 0
    crashed = False
    positions = []
    long_vels = []
    lat_vels = []
    steerings = []
    throttles = []
    sim_times = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        ep_reward += float(reward[0])
        step_count += 1
        sim_t = step_count * sim_dt * frame_skip

        car_states = info[0].get("car_states", np.zeros(7))
        car_pos = info[0].get("car_pos", np.zeros(3))

        positions.append([car_pos[0], car_pos[1]])
        long_vels.append(float(car_states[0]))
        lat_vels.append(float(car_states[1]))
        steerings.append(float(action[0][0]))
        throttles.append(float(action[0][1]))
        sim_times.append(sim_t)

        if done[0]:
            crashed = bool(info[0].get("crashed", False))

    lap_count = int(info[0].get("lap_count", 0))

    # total checkpoints crossed including previous laps
    checkpoints = (raw_env.current_checkpoint +
                   raw_env.lap_count * raw_env.n_checkpoints)

    lap_time = (sim_times[-1] / lap_count) if lap_count > 0 else None

    metrics = {
        "reward":            ep_reward,
        "steps":             step_count,
        "sim_time":          sim_times[-1] if sim_times else 0,
        "crashed":           crashed,
        "lap_completed":     lap_count > 0,
        "lap_count":         lap_count,
        "checkpoints":       checkpoints,
        "lap_time_sim":      lap_time,
        "mean_speed":        float(np.mean(np.abs(long_vels)))
                             if long_vels else 0,
        "mean_steer_change": float(np.mean(np.abs(np.diff(steerings))))
                             if len(steerings) > 1 else 0,
    }

    telemetry = {
        "positions":  np.array(positions),
        "long_vel":   long_vels,
        "lat_vel":    lat_vels,
        "steerings":  steerings,
        "throttles":  throttles,
        "sim_times":  sim_times,
    }

    return metrics, telemetry


def aggregate(results):
    """Compute mean/std across episodes."""
    rewards   = [r["reward"] for r in results]
    crashes   = [r["crashed"] for r in results]
    laps      = [r["lap_completed"] for r in results]
    ckpts     = [r["checkpoints"] for r in results]
    speeds    = [r["mean_speed"] for r in results]
    lap_times = [r["lap_time_sim"] for r in results
                 if r["lap_time_sim"] is not None]

    return {
        "mean_reward":         round(float(np.mean(rewards)), 2),
        "std_reward":          round(float(np.std(rewards)), 2),
        "crash_rate":          round(float(np.mean(crashes)), 3),
        "lap_completion_rate": round(float(np.mean(laps)), 3),
        "mean_checkpoints":    round(float(np.mean(ckpts)), 1),
        "mean_speed_ms":       round(float(np.mean(speeds)), 2),
        "mean_lap_time_sim":   round(float(np.mean(lap_times)), 2)
                               if lap_times else None,
        "laps_completed":      int(sum(laps)),
        "n_episodes":          len(results),
    }



def save_racing_line(telem, track_idx, output_dir, track_root="mujoco_tracks"):
    """
    Plot racing line coloured by speed (turbo cmap) with track boundaries.
    track_root: root folder containing track_1, track_2, ... subfolders
    """
    pos = telem["positions"]
    speeds = np.sqrt(
        np.array(telem["long_vel"])**2 +
        np.array(telem["lat_vel"])**2
    )

    # load cone boundaries for this track
    track_csv = os.path.join(
        track_root,
        f"track_{track_idx + 1}",
        "random_track.csv"
    )
    blue, yellow = load_cones(track_csv)

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- track boundaries ---
    if len(blue) > 1:
        ax.plot(
            np.append(blue[:, 0], blue[0, 0]),
            np.append(blue[:, 1], blue[0, 1]),
            color="#2166ac", linewidth=1.2,
            linestyle="-", label="Blue boundary"
        )
        ax.scatter(blue[:, 0], blue[:, 1],
                   c="#2166ac", s=6, zorder=3)

    if len(yellow) > 1:
        ax.plot(
            np.append(yellow[:, 0], yellow[0, 0]),
            np.append(yellow[:, 1], yellow[0, 1]),
            color="#d4a017", linewidth=1.2,
            linestyle="-", label="Yellow boundary"
        )
        ax.scatter(yellow[:, 0], yellow[:, 1],
                   c="#d4a017", s=6, zorder=3)

    # --- racing line coloured by speed ---
    norm = plt.Normalize(0, 12)
    cmap = cm.get_cmap("turbo")

    for i in range(len(pos) - 1):
        color = cmap(norm(speeds[i]))
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1],
                color=color, linewidth=2.0, zorder=4)

    # --- start marker ---
    if len(pos) > 0:
        ax.scatter(pos[0, 0], pos[0, 1],
                   c="white", edgecolors="black",
                   s=60, zorder=5, label="Start")

    # --- colourbar ---
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Speed [m/s]", fraction=0.03, pad=0.04)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Racing Line — Track {track_idx + 1}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"racing_line_track{track_idx+1}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    

def save_telemetry(telem, track_idx, output_dir):
    t = telem["sim_times"]
    speeds = np.sqrt(
        np.array(telem["long_vel"])**2 +
        np.array(telem["lat_vel"])**2
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Telemetry — Track {track_idx + 1}")

    axes[0,0].plot(t, telem["long_vel"], label="Longitudinal")
    axes[0,0].plot(t, telem["lat_vel"],  label="Lateral")
    axes[0,0].set_ylabel("Velocity [m/s]")
    axes[0,0].legend()

    axes[0,1].plot(t, telem["steerings"], color="darkred")
    axes[0,1].set_ylabel("Steering [-1,1]")
    axes[0,1].set_ylim(-1.1, 1.1)

    axes[0,2].plot(t, telem["throttles"], color="purple")
    axes[0,2].set_ylabel("Throttle [-1,1]")
    axes[0,2].set_ylim(-1.1, 1.1)

    axes[1,0].plot(t, speeds, color="navy")
    axes[1,0].set_ylabel("Speed [m/s]")

    steer_arr = np.array(telem["steerings"])
    axes[1,1].plot(t, np.abs(np.diff(steer_arr,
                   prepend=steer_arr[0])), color="darkorange")
    axes[1,1].set_ylabel("|Steering Change|")

    thr_arr = np.array(telem["throttles"])
    axes[1,2].plot(t, np.abs(np.diff(thr_arr,
                   prepend=thr_arr[0])), color="teal")
    axes[1,2].set_ylabel("|Throttle Change|")

    for ax in axes.flat:
        ax.set_xlabel("Sim Time [s]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir,
                     f"telemetry_track{track_idx+1}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()


@hydra.main(config_path=FILE_PATH, config_name="eval",
            version_base=None)
def main(config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.cuda
        else "cpu"
    )

    model_path  = os.path.join(config.ppo.model_dir, "model.zip")
    vecnorm_path = os.path.join(config.ppo.model_dir,
                                "vecnormalize.pkl")

    eval_tracks  = list(config.get("eval_tracks", [0]))
    n_episodes   = int(config.get("n_episodes_per_track", 5))
    waypoint_mode = config.env.get("waypoint_mode", "none")

    output_dir = os.path.join(
        "eval_results",
        os.path.basename(config.ppo.model_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model:         {model_path}")
    print(f"Waypoint mode: {waypoint_mode}")
    print(f"Eval tracks:   {eval_tracks}")
    print(f"Episodes/track:{n_episodes}")

    all_results = []

    for track_idx in eval_tracks:
        print(f"\n=== Track {track_idx + 1} "
              f"(folder: track_{track_idx + 1}) ===")

        # update config track index
        OmegaConf.update(config, "env.track_idx", track_idx)

        eval_env = build_eval_env(config, vecnorm_path)
        model = PPO.load(model_path, env=eval_env, device=device)

        track_results = []
        first_telem = None

        for ep in range(n_episodes):
            result, telem = run_episode(model, eval_env)
            result["track_idx"] = track_idx
            result["episode"]   = ep
            track_results.append(result)
            all_results.append(result)

            if first_telem is None:
                first_telem = telem

            print(
                f"  ep{ep:02d}: "
                f"reward={result['reward']:>8.1f}  "
                f"steps={result['steps']:>5}  "
                f"crash={result['crashed']}  "
                f"lap={result['lap_completed']}  "
                f"ckpt={result['checkpoints']:>4}  "
                f"speed={result['mean_speed']:.1f}m/s"
            )

        # per-track summary
        track_metrics = aggregate(track_results)
        print(f"\n  Track {track_idx+1} summary:")
        for k, v in track_metrics.items():
            print(f"    {k}: {v}")

        # save plots for first episode
        if first_telem is not None:
            save_racing_line(first_telem, track_idx, output_dir)
            save_telemetry(first_telem, track_idx, output_dir)

        eval_env.close()

    # overall summary
    overall = aggregate(all_results)
    print("\n=== OVERALL SUMMARY ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # save json
    out = {
        "model":    config.ppo.model_dir,
        "waypoint": waypoint_mode,
        "tracks":   eval_tracks,
        "overall":  overall,
        "episodes": all_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()