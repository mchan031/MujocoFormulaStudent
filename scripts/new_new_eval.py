# new_eval.py
import os
import json
import cv2
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


def run_episode(model, eval_env, visualize=False):
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
    max_checkpoints_seen = 0


    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        if visualize:
            frame = eval_env.render()  # render after step to capture final frame on done=True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
        # track max checkpoints seen during episode
        current_total = int(info[0].get("total_checkpoints",
                            raw_env.current_checkpoint +
                            raw_env.lap_count * raw_env.n_checkpoints))
        max_checkpoints_seen = max(max_checkpoints_seen, current_total)
        
        if done[0]:
            crashed = bool(info[0].get("crashed", False))

    lap_count = int(info[0].get("lap_count", 0))
    lap_time = (sim_times[-1] / lap_count) if lap_count > 0 else None

    if visualize:
        cv2.destroyAllWindows()
    # total checkpoints crossed including previous laps
    # checkpoints = (raw_env.current_checkpoint +
    #                raw_env.lap_count * raw_env.n_checkpoints)
    # checkpoints = int(info[0].get("total_checkpoints", 0))

    # lap_time = (sim_times[-1] / lap_count) if lap_count > 0 else None

    metrics = {
        "reward":            ep_reward,
        "steps":             step_count,
        "sim_time":          sim_times[-1] if sim_times else 0,
        "crashed":           crashed,
        "lap_completed":     lap_count > 0,
        "lap_count":         lap_count,
        "checkpoints":       max_checkpoints_seen,
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



def save_racing_line(telem, metrics, track_idx, output_dir,
                     track_root="mujoco_tracks", episode_label="best"):
    """
    Plot racing line coloured by speed (turbo cmap) with track
    boundaries and episode stats overlay.
    """
    pos = telem["positions"]
    speeds = np.sqrt(
        np.array(telem["long_vel"])**2 +
        np.array(telem["lat_vel"])**2
    )

    # load cone boundaries
    track_csv = os.path.join(
        track_root, f"track_{track_idx + 1}", "random_track.csv"
    )
    blue, yellow = load_cones(track_csv)

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- track boundaries ---
    if len(blue) > 1:
        ax.plot(
            np.append(blue[:, 0], blue[0, 0]),
            np.append(blue[:, 1], blue[0, 1]),
            color="#2166ac", linewidth=1.2, linestyle="-",
            label="Blue boundary", zorder=2
        )
        ax.scatter(blue[:, 0], blue[:, 1],
                   c="#2166ac", s=6, zorder=3)

    if len(yellow) > 1:
        ax.plot(
            np.append(yellow[:, 0], yellow[0, 0]),
            np.append(yellow[:, 1], yellow[0, 1]),
            color="#d4a017", linewidth=1.2, linestyle="-",
            label="Yellow boundary", zorder=2
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
                   s=80, zorder=5, label="Start")

    # --- crash marker ---
    if metrics.get("crashed") and len(pos) > 0:
        ax.scatter(pos[-1, 0], pos[-1, 1],
                   c="red", marker="x", s=120,
                   linewidths=2, zorder=6, label="Crash")

    # --- stats overlay ---
    lap_completed  = metrics.get("lap_completed", False)
    lap_time       = metrics.get("lap_time_sim", None)
    avg_speed      = metrics.get("mean_speed", 0)
    checkpoints    = metrics.get("checkpoints", 0)

    lap_str  = "✓" if lap_completed else "✗"
    time_str = f"{lap_time:.1f} s" if lap_time is not None else "N/A"

    stats_text = (
        f"Lap completed: {lap_str} | "
        f"Lap time:      {time_str} | "
        f"Avg speed:     {avg_speed:.2f} m/s"
        # f"Checkpoints:   {checkpoints}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white",
                  alpha=0.85,
                  edgecolor="gray")
    )

    
    # --- colourbar ---
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Speed [m/s]",
                 fraction=0.03, pad=0.04)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Racing Line — Track {track_idx + 1} ({episode_label})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    # ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir,
                     f"racing_line_track{track_idx+1}_{episode_label}.png"),
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


def save_racing_line_2(telem, metrics, track_idx, output_dir,
                     track_root="mujoco_tracks",
                     episode_label="best"):

    pos    = telem["positions"]
    speeds = np.sqrt(
        np.array(telem["long_vel"])**2 +
        np.array(telem["lat_vel"])**2
    )

    track_csv = os.path.join(
        track_root, f"track_{track_idx + 1}", "random_track.csv"
    )
    blue, yellow = load_cones(track_csv)

    # two panels: track left, stats table right
    fig, (ax, ax_stats) = plt.subplots(
        1, 2,
        figsize=(11, 7),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # --- track boundaries ---
    if len(blue) > 1:
        ax.plot(
            np.append(blue[:, 0], blue[0, 0]),
            np.append(blue[:, 1], blue[0, 1]),
            color="#2166ac", linewidth=1.2,
            linestyle="-", label="Blue boundary", zorder=2
        )
        ax.scatter(blue[:, 0], blue[:, 1],
                   c="#2166ac", s=6, zorder=3)

    if len(yellow) > 1:
        ax.plot(
            np.append(yellow[:, 0], yellow[0, 0]),
            np.append(yellow[:, 1], yellow[0, 1]),
            color="#d4a017", linewidth=1.2,
            linestyle="-", label="Yellow boundary", zorder=2
        )
        ax.scatter(yellow[:, 0], yellow[:, 1],
                   c="#d4a017", s=6, zorder=3)

    # --- racing line ---
    norm = plt.Normalize(0, 12)
    cmap = cm.get_cmap("turbo")
    for i in range(len(pos) - 1):
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1],
                color=cmap(norm(speeds[i])),
                linewidth=2.0, zorder=4)

    # --- start / crash markers ---
    if len(pos) > 0:
        ax.scatter(pos[0, 0], pos[0, 1],
                   c="white", edgecolors="black",
                   s=80, zorder=5, label="Start")
    if metrics.get("crashed") and len(pos) > 0:
        ax.scatter(pos[-1, 0], pos[-1, 1],
                   c="red", marker="x", s=150,
                   linewidths=2.5, zorder=6, label="Crash")

    # --- colourbar ---
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Speed [m/s]",
                 fraction=0.03, pad=0.02)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Track {track_idx + 1}  ({episode_label})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8,
              framealpha=0.9)

    # --- stats panel (right) ---
    ax_stats.axis("off")

    lap_completed = metrics.get("lap_completed", False)
    lap_time      = metrics.get("lap_time_sim", None)
    avg_speed     = metrics.get("mean_speed", 0)
    checkpoints   = metrics.get("checkpoints", 0)
    crashed       = metrics.get("crashed", False)
    steps         = metrics.get("steps", 0)

    rows = [
        ["Lap completed",  "✓" if lap_completed else "✗"],
        ["Lap time",       f"{lap_time:.1f} s"
                           if lap_time else "N/A"],
        ["Avg speed",      f"{avg_speed:.2f} m/s"],
        # ["Checkpoints",    str(checkpoints)],
        ["Crashed",        "Yes" if crashed else "No"],
        # ["Steps",          str(steps)],
    ]

    table = ax_stats.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # colour the header row
    for j in range(2):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # colour lap completed row green/red
    lap_row_color = "#d4edda" if lap_completed else "#f8d7da"
    for j in range(2):
        table[1, j].set_facecolor(lap_row_color)

    # ax_stats.set_title("Episode Stats", fontsize=10,
    #                     fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir,
                     f"racing_line_track{track_idx+1}"
                     f"_{episode_label}.png"),
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
        # first_telem = None
        best_result = None    # ← initialise HERE, inside track loop
        best_telem  = None    # ← so best resets per track

        for ep in range(n_episodes):
            result, telem = run_episode(model, eval_env, visualize=config.visualize)
            result["track_idx"] = track_idx
            result["episode"]   = ep
            track_results.append(result)
            all_results.append(result)

            # if first_telem is None:
            #     first_telem = telem
            # track best episode by reward
            if best_result is None or result["reward"] > best_result["reward"]:
                best_result = result
                best_telem  = telem
                
            print(
                f"  ep{ep:02d}: "
                f"reward={result['reward']:>8.1f}  "
                f"steps={result['steps']:>5}  "
                f"crash={result['crashed']}  "
                f"lap={'✓' if result['lap_completed'] else '✗'}  "
                f"ckpt={result['checkpoints']:>4}  "
                f"speed={result['mean_speed']:.1f}m/s  "
                f"laptime={str(round(result['lap_time_sim'],1))+'s' if result['lap_time_sim'] else 'N/A'}"
            )

        # per-track summary
        track_metrics = aggregate(track_results)
        print(f"\n  Track {track_idx+1} summary:")
        for k, v in track_metrics.items():
            print(f"    {k}: {v}")

        # # save plots for first episode
        # if first_telem is not None:
        #     save_racing_line(first_telem, track_idx, output_dir)
        #     save_telemetry(first_telem, track_idx, output_dir)

        # save best episode plots
        if best_telem is not None:
            save_racing_line(best_telem, best_result, track_idx, output_dir,
                            episode_label=f"best_ep{best_result['episode']}")
            save_telemetry(best_telem, track_idx, output_dir)

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