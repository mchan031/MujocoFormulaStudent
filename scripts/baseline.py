import os
import time
import cv2
import numpy as np
from env import MujocoFormulaStudent
from utils import plot_racing_line, TelemetryStorage
from controller import ControllerPIDBicycle



def main():
    env = MujocoFormulaStudent(
        render_mode="rgb_array",
        num_checkpoints=10,
        lap_completion_reward=1000,
        max_env_step=250,
        checkpoint_bonus_step=150,
        max_throttle=3.0,
        domain_randomization=False,
        track_idx=3,  # Set to a specific track index or None for random
    )

    controller = ControllerPIDBicycle(
        kp=0.35,
        ki=0.0001,
        kd=0.18,
        heading_gain=2.5,
        cte_gain=0.45,
        target_speed=4.0,
    )
    controller.set_path(env.centreline[:, :2])

    telemetry = TelemetryStorage()
    trajectory = []
    speed_profile = []

    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        _, info = reset_out
    else:
        info = {}

    max_steps = 4000
    step_count = 0
    total_reward = 0.0
    start_time = time.time()
    lap_time_list = []
    lap_start_time = start_time
    prev_lap_count = 0

    lap_count = 0
    # while step_count < max_steps:
    while lap_count < 1:
        car_pos = info.get("car_pos", env._get_car_pos())
        car_states = info.get("car_states", env._get_car_states())
        yaw = env._get_car_yaw()

        ctrl_info = {
            "x": float(car_pos[0]),
            "y": float(car_pos[1]),
            "yaw": float(yaw),
            "dt": env.dt,
            "long_vel": float(car_states[0]),
            "lat_vel": float(car_states[1]),
        }

        action, _ = controller.feedback(ctrl_info)
        _, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += float(reward)

        car_states = info["car_states"]
        car_pos = info["car_pos"]
        current_time = time.time() - start_time
        telemetry.append(step_count, car_states, current_time)

        trajectory.append([car_pos[0], car_pos[1]])
        speed_profile.append(float(np.sqrt(car_states[0] ** 2 + car_states[1] ** 2)))

        lap_count = int(info.get("lap_count", 0))
        if lap_count > prev_lap_count:
            now = time.time()
            lap_time_list.append(now - lap_start_time)
            lap_start_time = now
            prev_lap_count = lap_count
            print(f"Lap {lap_count} complete. Lap time: {lap_time_list[-1]:.2f}s")

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break


        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting evaluation loop.")
            break

    env.close()

    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")

    if len(trajectory) > 1 and os.path.exists(env.random_track_path):
        plot_racing_line(trajectory, speed_profile, env.random_track_path, lap_time_list)
    telemetry.plot_telemetry()



if __name__ == "__main__":
    main()


