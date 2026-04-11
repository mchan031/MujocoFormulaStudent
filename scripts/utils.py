from collections import deque
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
import os
import abc
from gymnasium import spaces


class TelemetryStorage:
    def __init__(self):
        self.data = {
            "step": [],
            "long_vel": [], "lat_vel": [],
            "long_acc": [], "lat_acc": [],
            "yaw_rate": [], "steering": [], "throttle": [],
            "g": [],
            "time": []
        }
        # self.acc_prev_long = 0.0
        # self.acc_prev_lat = 0.0

    def append(self, step, car_states, current_time):
        # acc_filtered_long = 0.9 * self.acc_prev_long + 0.1 * car_states[2]
        # acc_filtered_lat = 0.9 * self.acc_prev_lat + 0.1 * car_states[3]

        self.data["step"].append(step)
        self.data["time"].append(current_time)
        self.data["long_vel"].append(car_states[0])
        self.data["lat_vel"].append(car_states[1])
        self.data["long_acc"].append(car_states[2])
        self.data["lat_acc"].append(car_states[3])
        self.data["yaw_rate"].append(car_states[4])
        self.data["steering"].append(car_states[5])
        self.data["throttle"].append(car_states[6])

        # self.acc_prev_long = acc_filtered_long
        # self.acc_prev_lat = acc_filtered_lat

    def plot_telemetry(self):
        if len(self.data["time"]) == 0:
            return

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        axes[0, 0].plot(self.data["time"], self.data["long_vel"], "b-", label="Longitudinal")
        axes[0, 0].plot(self.data["time"], self.data["lat_vel"], "r-", label="Lateral")
        axes[0, 0].set_ylabel("Velocity (m/s)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_title("Vehicle Velocity")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.data["time"], self.data["long_acc"], "g-", label="Longitudinal")
        axes[0, 1].plot(self.data["time"], self.data["lat_acc"], "orange", label="Lateral")
        axes[0, 1].set_ylabel("Acceleration (m/s^2)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_title("Vehicle Acceleration")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.data["time"], self.data["yaw_rate"], "purple")
        axes[1, 0].set_ylabel("Yaw Rate (rad/s)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_title("Yaw Rate")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.data["time"], self.data["steering"], "brown")
        axes[1, 1].set_ylabel("Steering Angle")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_title("Steering Command")
        axes[1, 1].grid(True, alpha=0.3)

        axes[2, 0].plot(self.data["time"], self.data["throttle"], "pink")
        axes[2, 0].set_ylabel("Throttle")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_title("Throttle Command")
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(self.data["time"], self.data["long_vel"], "b-", label="Long Vel", alpha=0.7)
        axes[2, 1].plot(self.data["time"], self.data["steering"], "brown", label="Steering", alpha=0.7)
        axes[2, 1].plot(self.data["time"], self.data["throttle"], "pink", label="Throttle", alpha=0.7)
        axes[2, 1].set_ylabel("Values")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_title("Combined Telemetry")
        axes[2, 1].legend(loc="upper right", fontsize="small")
        axes[2, 1].grid(True, alpha=0.3)

        plt.suptitle("Vehicle Telemetry Data", fontsize=14)
        plt.tight_layout()
        # plt.savefig('one_lap.png', dpi=150)
        plt.show()

class MovingAverageFilter:
    def __init__(self, max_len=5):
        self.value_history = deque(maxlen=max_len)
        
    def compute(self, value):
        self.value_history.append(value)
        return np.mean(self.value_history)
    

class ForceForwardWrapper(gym.Wrapper):

    def __init__(self, env, force_steps=50, force_episodes=5):
        super().__init__(env)
        self.force_steps = force_steps
        self.force_episodes = force_episodes
        self.episode = 0
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        self.episode += 1
        return self.env.reset(**kwargs)

    def step(self, action):

        if self.episode < self.force_episodes and self.t < self.force_steps:
            # action = np.array([0.0, 1.0])  #throttle forward

            action = [0.0]
            throttle = np.random.random()/2 + 0.5
            action.append(throttle)
            action = np.array(action)


        self.t += 1

        # print(f"Steering: {action[0]} Throttle: {action[1]}")
        return self.env.step(action)

def load_cones(csv_path):

    blue = []
    yellow = []

    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")

            color = parts[0]
            x = float(parts[1])
            y = float(parts[2])

            if color == "blue":
                blue.append((x,y))
            elif color == "yellow":
                yellow.append((x,y))

    return np.array(blue), np.array(yellow)
    
def plot_racing_line(traj, speed, track_csv, lap_time_list=None):

    traj = np.array(traj)
    speed = np.array(speed)

    blue, yellow = load_cones(track_csv)

    fig, ax = plt.subplots(figsize=(8,8))

    x = traj[:,0]
    y = traj[:,1]
    # Track cones
    if len(blue) > 0:
        # Connect blue cones as a polygon (closed loop)
        ax.plot(
            np.append(blue[:, 0], blue[0, 0]),
            np.append(blue[:, 1], blue[0, 1]),
            c='blue', linewidth=1, linestyle='-', markersize=3, label='Blue Cones'
        )

    if len(yellow) > 0:
        # Connect yellow cones as a polygon (closed loop)
        ax.plot(
            np.append(yellow[:, 0], yellow[0, 0]),
            np.append(yellow[:, 1], yellow[0, 1]),
            c='orange', linewidth=1, linestyle='-', markersize=3, label='Yellow Cones'
        )
        
    if lap_time_list is not None:
        # lap_time_str = "\n".join([f"Lap {i}: {lap_time:.2f} s" for i, lap_time in enumerate(lap_time_list)])
        lap_time_str = "\n".join([
            f"Lap Time: {lap_time:.2f} s | Average Speed: {np.mean(speed):.2f} m/s"
            for i, lap_time in enumerate(lap_time_list)
        ])
        plt.text(0.05, 0.95, lap_time_str, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -------------------------
    # Create line segments
    # -------------------------
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # -------------------------
    # Create colored line
    # -------------------------
    lc = LineCollection(
        segments,
        cmap='turbo',
        norm=plt.Normalize(speed.min(), speed.max())
    )

    lc.set_array(speed)
    lc.set_linewidth(3)

    ax.add_collection(lc)

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Speed [m/s]")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Racing Line (Velocity Colored)")

    ax.axis("equal")
    ax.grid(alpha=0.3)

    plt.show()
    
    
def create_env_xml(track_dir):

    template = open("mujoco_tracks/sim_env.xml").read()

    track_name = os.path.basename(track_dir)
    track_xml = f"{track_name}/track.xml"

    xml = template.replace("{TRACK_FILE}", track_xml)

    with open("mujoco_tracks/sim_env_runtime.xml", "w") as f:
        f.write(xml)

    return "mujoco_tracks/sim_env_runtime.xml"


class Controller:
    def set_path(self, path):
        self.path = path

    @abc.abstractmethod
    def feedback(self, info):
        return NotImplementedError
    
def search_nearest(path, pos):
    min_dist = 99999999
    min_id = -1
    for i in range(path.shape[0]):
        dist = (pos[0] - path[i,0])**2 + (pos[1] - path[i,1])**2
        if dist < min_dist:
            min_dist = dist
            min_id = i
    return min_id, min_dist


class GrayscaleDictObservation(gym.ObservationWrapper):
    """Converts only the 'image' key in a Dict observation space to grayscale."""
    
    def __init__(self, env: gym.Env, image_key: str = "image"):
        super().__init__(env)
        self.image_key = image_key
        
        assert isinstance(env.observation_space, spaces.Dict), \
            "GrayscaleDictObservation requires a Dict observation space"
        assert image_key in env.observation_space.spaces, \
            f"Key '{image_key}' not found in observation space"
        
        img_space = env.observation_space.spaces[image_key]
        assert isinstance(img_space, spaces.Box), \
            f"Observation key '{image_key}' must be a Box space"
        
        # HWC: (H, W, 3) -> (H, W, 1)
        h, w, _ = img_space.shape
        new_img_space = spaces.Box(
            low=0, high=255,
            shape=(h, w, 1),
            dtype=np.uint8
        )
        
        new_spaces = dict(env.observation_space.spaces)
        new_spaces[image_key] = new_img_space
        self.observation_space = spaces.Dict(new_spaces)
    
    def observation(self, obs):
        img = obs[self.image_key]  # (H, W, 3)
        # Rec. 601 luminance weights — same as gymnasium's GrayscaleObservation
        gray = np.dot(img[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.uint8)
        gray = gray[..., np.newaxis]  # (H, W, 1)
        return {**obs, self.image_key: gray}