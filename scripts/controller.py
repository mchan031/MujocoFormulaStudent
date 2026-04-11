import numpy as np
from utils import Controller


class ControllerPIDBicycle(Controller):
    def __init__(
        self,
        kp=0.45,
        ki=0.0,
        kd=0.16,
        heading_gain=2.2,
        cte_gain=0.5,
        steer_ff_gain=0.55,
        target_speed=3.8,
        min_target_speed=1.1,
        speed_kp=0.65,
        speed_ki=0.03,
        speed_kd=0.03,
        lookahead_base=5,
        lookahead_gain=1.0,
        curvature_speed_gain=9.0,
        curvature_lookahead_gain=0.65,
        heading_brake_gain=0.55,
        max_integral_cte=3.0,
        max_integral_speed=6.0,
    ):
        self.path = None
        self.path_heading = None
        self.path_curvature = None

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.heading_gain = heading_gain
        self.cte_gain = cte_gain
        self.steer_ff_gain = steer_ff_gain

        self.target_speed = target_speed
        self.min_target_speed = min_target_speed
        self.speed_kp = speed_kp
        self.speed_ki = speed_ki
        self.speed_kd = speed_kd

        self.lookahead_base = lookahead_base
        self.lookahead_gain = lookahead_gain
        self.curvature_speed_gain = curvature_speed_gain
        self.curvature_lookahead_gain = curvature_lookahead_gain
        self.heading_brake_gain = heading_brake_gain

        self.max_integral_cte = max_integral_cte
        self.max_integral_speed = max_integral_speed

        self.acc_ep = 0.0
        self.last_ep = 0.0
        self.acc_speed_error = 0.0
        self.last_speed_error = 0.0
        self._last_nearest_idx = 0   # ADD this


    def set_path(self, path):
        super().set_path(path)
        self._last_nearest_idx = 0   # ADD
        self.acc_ep = 0.0
        self.last_ep = 0.0
        self.acc_speed_error = 0.0
        self.last_speed_error = 0.0

        self.path_heading = self._compute_heading(path)
        self.path_curvature = self._compute_curvature(path, self.path_heading)

    @staticmethod
    def _normalize_angle(rad):
        return (rad + np.pi) % (2 * np.pi) - np.pi

    def _search_nearest(self, pos_xy):
        diffs = self.path - pos_xy
        d2 = np.sum(diffs * diffs, axis=1)
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def _compute_heading(self, path):
        next_pts = np.roll(path, -1, axis=0)
        delta = next_pts - path
        return np.arctan2(delta[:, 1], delta[:, 0])

    def _compute_curvature(self, path, heading):
        prev_heading = np.roll(heading, 1)
        dpsi = self._normalize_angle(heading - prev_heading)

        prev_pts = np.roll(path, 1, axis=0)
        ds = np.linalg.norm(path - prev_pts, axis=1)
        ds = np.maximum(ds, 1e-4)

        return np.abs(dpsi / ds)

    def _lookahead_curvature(self, start_idx, horizon):
        n = self.path.shape[0]
        idxs = (start_idx + np.arange(horizon)) % n
        return float(np.max(self.path_curvature[idxs]))

    def feedback(self, info):
        if self.path is None:
            print("No path !!")
            return np.array([0.0, 0.0], dtype=np.float32), None

        x = float(info["x"])
        y = float(info["y"])
        yaw = self._normalize_angle(float(info["yaw"]))
        dt = max(float(info.get("dt", 0.05)), 1e-3)
        long_vel = float(info.get("long_vel", 0.0))

        # nearest_idx, _ = self._search_nearest(np.array([x, y]))
        nearest_idx, _ = self._search_nearest_forward(np.array([x, y]))


        base_lookahead = int(self.lookahead_base + self.lookahead_gain * max(long_vel, 0.0))
        base_lookahead = max(base_lookahead, 2)

        preview_curvature = self._lookahead_curvature(nearest_idx, base_lookahead)
        lookahead_scale = 1.0 - self.curvature_lookahead_gain * np.clip(preview_curvature, 0.0, 1.0)
        lookahead_steps = int(np.clip(base_lookahead * lookahead_scale, 2, 25))

        target_idx = (nearest_idx + lookahead_steps) % self.path.shape[0]
        target = self.path[target_idx]

        desired_heading = np.arctan2(target[1] - y, target[0] - x)
        heading_error = self._normalize_angle(desired_heading - yaw)

        prev_idx = (nearest_idx - 1) % self.path.shape[0]
        next_idx = (nearest_idx + 1) % self.path.shape[0]
        tangent = self.path[next_idx] - self.path[prev_idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
        rel = np.array([x, y]) - self.path[nearest_idx]
        ep = float(tangent[0] * rel[1] - tangent[1] * rel[0])

        self.acc_ep += dt * ep
        self.acc_ep = float(np.clip(self.acc_ep, -self.max_integral_cte, self.max_integral_cte))
        diff_ep = (ep - self.last_ep) / dt
        cte_pid = self.kp * ep + self.ki * self.acc_ep + self.kd * diff_ep

        curve_sign = np.sign(self._normalize_angle(self.path_heading[target_idx] - self.path_heading[nearest_idx]))
        curvature_ff = curve_sign * preview_curvature

        steer_cmd = cte_pid + self.heading_gain * heading_error
        steer_cmd += self.cte_gain * np.arctan2(ep, max(abs(long_vel), 0.5))
        steer_cmd += self.steer_ff_gain * curvature_ff
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

        self.last_ep = ep

        curve_speed = self.target_speed / (1.0 + self.curvature_speed_gain * preview_curvature)
        heading_speed = self.target_speed * (1.0 - self.heading_brake_gain * min(abs(heading_error) / np.pi, 1.0))
        dynamic_target_speed = min(self.target_speed, curve_speed, heading_speed)
        dynamic_target_speed = max(self.min_target_speed, dynamic_target_speed)

        speed_error = dynamic_target_speed - long_vel
        self.acc_speed_error += speed_error * dt
        self.acc_speed_error = float(np.clip(self.acc_speed_error, -self.max_integral_speed, self.max_integral_speed))
        speed_diff = (speed_error - self.last_speed_error) / dt

        throttle_cmd = (
            self.speed_kp * speed_error
            + self.speed_ki * self.acc_speed_error
            + self.speed_kd * speed_diff
        )

        if abs(steer_cmd) > 0.8 and long_vel > 2.0:
            throttle_cmd -= 0.18

        throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))
        self.last_speed_error = speed_error

        return np.array([steer_cmd, throttle_cmd], dtype=np.float32), target
    
    def _search_nearest_forward(self, pos_xy, search_window=30):
        """
        Only search within a forward window from last known position.
        Prevents jumping to wrong part of track.
        """
        n = self.path.shape[0]
        
        # search only within window ahead of last position
        indices = [(self._last_nearest_idx + i) % n 
                for i in range(-5, search_window)]
        
        candidate_points = self.path[indices]
        diffs = candidate_points - pos_xy
        d2 = np.sum(diffs * diffs, axis=1)
        
        local_idx = int(np.argmin(d2))
        global_idx = indices[local_idx]
        
        # update state
        self._last_nearest_idx = global_idx
        
        return global_idx, float(np.sqrt(d2[local_idx]))