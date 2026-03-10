import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from env import MujocoFormulaStudent
import time
import os

def test_longitudinal_acceleration():
    """Test acceleration and braking"""
    actions = []
    
    # Phase 1: Acceleration (0-100 steps)
    for step in range(500):
        actions.append([0.0, 1.0])  # Full throttle, straight
    
    # Phase 2: Coast (100-150 steps)
    for step in range(50):
        actions.append([0.5, 0.0])  # No throttle
    
    # Phase 3: Braking (150-200 steps)
    for step in range(50):
        actions.append([0.0, -1.0])  # Brake
    
    for step in range(200):
        actions.append([0.0, 1.0])  # Full throttle, straight
    return actions


def test_lateral_acceleration_sinusoidal(steps=500, amplitude=1.0, frequency=0.05):
    """
    Test lateral acceleration with sinusoidal steering
    amplitude: max steering angle (radians, typically 0.38 max)
    frequency: oscillation frequency
    """
    actions = []
    for step in range(steps):
        # Sinusoidal steering, constant throttle
        steering = amplitude * np.sin(2 * np.pi * frequency * step * 0.05)  # 0.05 is dt
        throttle = 0.5  # Constant half throttle
        actions.append([steering, throttle])
    return actions

# def filter_acc(acc_now):
#     acc_filtered = 0.9 * acc_prev + 0.1 * acc_now
#     return acc_filtered

def test_env_with_logging():
    # model_path = "C:/Users/munki/OneDrive - Nanyang Technological University/Y4/new_fyp/models/one_car.xml"
    # model_path = "C:/Users/munki/OneDrive - Nanyang Technological University/Y4/new_fyp/mujoco_tracks/new_scene.xml"
    mujoco_path = "mujoco_tracks/sim_env.xml"
    
    full_path = os.path.join(os.path.dirname(__file__), os.path.pardir, mujoco_path)

    env = MujocoFormulaStudent(
        model_path=full_path,
        render_mode="rgb_array",
        # checkpoint_file="mujoco_tracks/checkpoints.json",
        centreline_file="mujoco_tracks/centreline.csv",
        num_checkpoints=4
    )
    
    actions = test_longitudinal_acceleration()
    # actions = test_lateral_acceleration_sinusoidal()
    
    # Data storage
    data = {
        'step': [],
        'long_vel': [], 'lat_vel': [],
        'long_acc': [], 'lat_acc': [],
        'yaw_rate': [], 'steering': [], 'throttle': [],
        'g': [],
        'time': []
    }
    
    
    # for _ in range(10):
    obs, info = env.reset()
        
    #     states = env._get_car_states()
    
    start_time = time.time()
    
    print("Collecting data...")
    
    acc_prev_long = 0.0
    acc_prev_lat = 0.0
    g_prev = 9.81
    
    # cp = env.checkpoints[0].keys()
    pos = env._get_car_pos()
    # print(f"{pos = }")
    
    dist = env._compute_cp_distances(pos)
    normal = env.checkpoint_normals
    # current_cp = env.che
    # env.
    # print(f"{normal = }")
    # print(f"{dist = }")
    
    total_reward = 0.0
    # raise
    for step in range(len(actions)):
        # action = [0.0, 1.0]  # Constant throttle
        # print(data.)
        action = actions[step]
        obs, reward, terminated, truncated, info = env.step(action)
        # print(info)
        # print(obs["progress"])
        # print(obs["checkpoint_distances"])
        total_reward += reward
        # print(f"{total_reward = }")
        # print(env._get_checkpoint_distances())
        # print(env.current_checkpoint)
        # print(env.progress)
        pos = env._get_car_pos()
        # print(f"{pos[0]}")
        car_states = obs['car_states']
        current_time = time.time() - start_time
        
        acc_filtered_long = 0.9 * acc_prev_long + 0.1 * car_states[2]
        acc_filtered_lat = 0.9 * acc_prev_lat + 0.1 * car_states[3]
        # g_filtered = 0.9 * g_prev + 0.1 * car_states[7]


        # Store data
        data['step'].append(step)
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
        # g_prev = g_filtered
        
        # if step % 10 ==0:
        #     print(data)
        
        # Show camera
        frame = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
        cv2.imshow("View", cv2.resize(frame, (256, 256)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if terminated or truncated:
            print("Terminated!!!!!!")
            obs, info = env.reset()
            total_reward = 0.0
    
    env.close()
    cv2.destroyAllWindows()
    
    # Plot results
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
    # plt.savefig('telemetry.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    test_env_with_logging()