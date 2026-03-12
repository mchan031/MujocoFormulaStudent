import os
import gymnasium as gym
import numpy as np
from env import MujocoFormulaStudent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from utils import ForceForwardWrapper
import cv2
import matplotlib.pyplot as plt
import time
import os

mujoco_path = "mujoco_tracks/sim_env.xml"
centreline_path = "mujoco_tracks/centreline.csv"
seed = 42
device = "cpu"

# model_path = "models/MujocoFormulaStudent__testing__2026-03-08_16-46-30/model.zip"
model_path = "models/MujocoFormulaStudent__faster_speed_reverse_penalty__2026-03-11_16-06-12/model.zip"
# model_path = "models/MujocoFormulaStudent__testing__2026-03-09_17-55-13/model.zip"

# model_path = "wandb/run-20260308_164630-k5enfe6y/files/model.zip"

def make_env(env_path, centreline_path, seed=42):
    """Create a single environment instance"""
    def _init():
        env = MujocoFormulaStudent(
            model_path=env_path,
            render_mode="rgb_array",
            centreline_file=centreline_path,
            num_checkpoints=20
            # lap_completion_reward=cfg.env.lap_completion_reward,
            # max_env_step=cfg.env.max_env_step,
            # checkpoint_bonus_step=cfg.env.checkpoint_bonus_step
            
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
    eval_env = VecNormalize.load("models/MujocoFormulaStudent__faster_speed_reverse_penalty__2026-03-11_16-06-12/vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False    
    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     video_folder="./videos",
    #     record_video_trigger=lambda step: step == 0,  # start recording immediately
    #     video_length=2000,  # max steps to record
    #     name_prefix="fs_unseen_track",
    # )
    
        # Data storage
    data = {
        'step': [],
        'long_vel': [], 'lat_vel': [],
        'long_acc': [], 'lat_acc': [],
        'yaw_rate': [], 'steering': [], 'throttle': [],
        'g': [],
        'time': []
    }
    
    lap_time_list = []
    
    
    np.random.seed(seed)
    
    # Load the trained model (NOT create a new one)
    model = PPO.load(
        path=model_path,
        env=eval_env,
        device=device
    )
    
    # Evaluation loop
    obs = eval_env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    start_time = time.time()
    lap_start_time = start_time
    acc_prev_long = 0.0
    acc_prev_lat = 0.0
    g_prev = 9.81
    num_lap = 0
    # method = eval_env.env_method("_check_checkpoint_crossing")
    # print(f"{method = }")
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = eval_env.step(action)
        
        
        total_reward += reward
        step_count += 1
        # Optional: render is handled automatically with render_mode="human"
        # print(obs)
        # raise
    #             # Show camera
        frame = eval_env.envs[0].render()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("View", frame)
        # cv2.imshow("View", cv2.resize(frame, (256, 256)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if done:
            print("Colllision!!!!!")
        
        if step_count % 10 == 0:
    #         # print(f"{total_reward = } {info = }")
            print(action)
            # print(f"{info[0]["progress"] = }")
            # Store data
        
        # if info[] > 1:
        #     break
        # print(info[0])    
        if info[0]["lap_count"] - num_lap == 1:
            time_now = time.time()
            lap_time = time_now - lap_start_time
            lap_time_list.append(lap_time)
            lap_start_time = time_now
            num_lap += 1
            
            print("-" * 30)
            print("    Lap Time    ")
            print("-" * 30)
            
            # print(f"Lap Time:")
            for i, item in enumerate(lap_time_list):
                print(f"Lap: {i} Time: {item}")
            
            # print("-" * 50)
        if step_count > 1000:
            print(f"{step_count = }")
            break
            # break
        
        car_states = obs['car_states'][0]
        current_time = time.time() - start_time
        acc_filtered_long = 0.9 * acc_prev_long + 0.1 * car_states[2]
        acc_filtered_lat = 0.9 * acc_prev_lat + 0.1 * car_states[3]
        data['step'].append(step_count)
        data['time'].append(current_time)
        data['long_vel'].append(car_states[0])
        data['lat_vel'].append(car_states[1])
        # data['long_acc'].append(car_states[2])
        data['long_acc'].append(acc_filtered_long)

        data['lat_acc'].append(acc_filtered_lat)
        data['yaw_rate'].append(car_states[4])
        data['steering'].append(car_states[5])
        data['throttle'].append(car_states[6])
        # data['g'].append(g_filtered)

        acc_prev_long = acc_filtered_long
        acc_prev_lat = acc_filtered_lat
    # # print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {step_count}")

    eval_env.close()
    cv2.destroyAllWindows()
    plot_telemetry(data)

def plot_telemetry(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Velocity plot
    axes[0, 0].plot(data['time'], data['long_vel'], 'b-', label='Longitudinal')
    axes[0, 0].plot(data['time'], data['lat_vel'], 'r-', label='Lateral')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_title('Vehicle Velocity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Acceleration plot
    axes[0, 1].plot(data['time'], data['long_acc'], 'g-', label='Longitudinal')
    axes[0, 1].plot(data['time'], data['lat_acc'], 'orange', label='Lateral')
    # axes[0, 1].plot(data['time'], data['g'], 'b-', label='g')
    axes[0, 1].set_ylabel('Acceleration (m/s²)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_title('Vehicle Acceleration')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Yaw rate
    axes[1, 0].plot(data['time'], data['yaw_rate'], 'purple')
    axes[1, 0].plot(data['time'], data['steering'], 'brown')
    axes[1, 0].set_ylabel('Yaw Rate (rad/s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Yaw Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Steering
    axes[1, 1].plot(data['time'], data['steering'], 'brown')
    axes[1, 1].set_ylabel('Steering Angle')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Steering Command')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Throttle
    axes[2, 0].plot(data['time'], data['throttle'], 'pink')
    axes[2, 0].set_ylabel('Throttle')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_title('Throttle Command')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Combined plot
    axes[2, 1].plot(data['time'], data['long_vel'], 'b-', label='Long Vel', alpha=0.7)
    axes[2, 1].plot(data['time'], data['steering'], 'brown', label='Steering', alpha=0.7)
    axes[2, 1].plot(data['time'], data['throttle'], 'pink', label='Throttle', alpha=0.7)
    axes[2, 1].set_ylabel('Values')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title('Combined Telemetry')
    axes[2, 1].legend(loc='upper right', fontsize='small')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Vehicle Telemetry Data', fontsize=14)
    plt.tight_layout()
    # plt.savefig('one_lap.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    eval()