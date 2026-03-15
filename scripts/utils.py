from collections import deque
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection

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
        lap_time_str = "\n".join([f"Lap {i}: {lap_time:.2f} s" for i, lap_time in enumerate(lap_time_list)])
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