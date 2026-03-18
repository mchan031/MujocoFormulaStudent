import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium import spaces
import mujoco
from typing import Optional, Dict, Any, Tuple, List
import os
import json
from utils import MovingAverageFilter, create_env_xml 
from gymnasium.utils import EzPickle
import random

DISTANCE_MODE = False

class MujocoFormulaStudent(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 20,
    }
    
    def __init__(
        self,
        track_root: str = "mujoco_tracks",
        frame_skip: int = 5,
        observation_width: int = 84,
        observation_height: int = 84,
        render_width: int = 640,
        render_height: int = 480,
        step_penalty: float = 0.1,
        lap_completion_reward: float = 1000.0,
        forward_velocity_reward: float = 0.05,
        crash_penalty: float = 100.0,
        checkpoint_bonus_step: float = 250.0,
        reset_noise_scale: float = 0.01,
        next_n_checkpoint: int = 5,
        max_env_step: int = 1500,
        num_checkpoints: int = 10,
        vel_penalty_weight: int = 1,
        steer_penalty_weight: int = 1,
        max_throttle: float = 2.5,
        track_idx: Optional[int] = None,
        domain_randomization: bool = True,
        **kwargs,
    ):
        
        EzPickle.__init__(
            self,
            frame_skip,
            observation_width,
            observation_height,
            render_width,
            render_height,
            step_penalty,
            lap_completion_reward,
            forward_velocity_reward,
            crash_penalty,
            checkpoint_bonus_step,
            reset_noise_scale,
            next_n_checkpoint,
            max_env_step,
            num_checkpoints,
            vel_penalty_weight,
            steer_penalty_weight,
            max_throttle,
            track_idx,  
            domain_randomization,
            **kwargs,
        )
        
        ## Reward Config
        # self._ctrl_cost_weight = ctrl_cost_weight
        self._step_penalty = step_penalty
        self._lap_completion_reward = lap_completion_reward
        self._forward_velocity_reward = forward_velocity_reward
        self._crash_penalty = crash_penalty
        self._checkpoint_bonus_step = checkpoint_bonus_step
        self.max_steps = max_env_step
        self._vel_penalty_weight = vel_penalty_weight
        self._steer_penalty_weight = steer_penalty_weight
        self._max_throttle = max_throttle

        self.step_count = 0
        self.prev_velocity = None
        self.prev_steering = 0.0
        
        ## Noise Config
        self._reset_noise_scale = reset_noise_scale
    
        ## Camera Config
        self._observation_width = observation_width
        self._observation_height = observation_height
        self._render_width = render_width
        self._render_height = render_height
        
        # ## Checkpoints Config
        self.track_length = 0.0
        self.checkpoint_distances = []
        self.checkpoint_points = None
        self.checkpoint_normals = None
        self.checkpoint_tangents = None
        self.centreline = None
        self.n_checkpoints = num_checkpoints
        self._next_n_checkpoint = next_n_checkpoint

        # Initialize observation space
        self._initialize_observation_space()
        
        # Randomly select a track from the track root directory
        track_dir = self._select_random_track(track_root, track_idx)        
        print("Running on track:", track_dir)
        
        # Load Selected Track Centreline
        self._load_centreline(track_dir)
        self._create_checkpoint()
        
        self.random_track_path = os.path.join(track_dir, "random_track.csv") 

        # Create Runtime XML with selected track
        model_xml_path = create_env_xml(track_dir)
        full_xml_path = os.path.join(os.path.dirname(__file__), os.path.pardir, model_xml_path)
            
        # Initialize Parent Class
        MujocoEnv.__init__(
            self,
            model_path=full_xml_path,
            frame_skip=frame_skip,
            observation_space=self.observation_space,
            width=render_width,
            height=render_height,
            camera_name="third_person",
            **kwargs,
        )
        
        # Overwrite action space (steering and throttle)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Get Car, Sensor Ids, and Floor Id
        self._get_car_joint_ids()
        self._get_sensor_id()
        self._get_floor_id()
        
        # Initialize Agent Camera
        self._initialize_agent_camera()

        # Progress tracker
        self.lap_count = 0
        self.progress = 0.0
        self.prev_cp_distances = np.zeros(self.n_checkpoints)
        self.current_checkpoint = 0
        self.prev_checkpoint = 0
        
        # Collision Detection
        self.car_geoms = self._get_car_geom_ids()

        print(f"Car Geoms: {self.car_geoms}")



        self.cone_geoms = self._get_cone_geom_ids()
        
        # Filter
        self.acc_filter = MovingAverageFilter(10)
        
        # Domain Randomization
        self.domain_randomization = domain_randomization
        self.default_friction = self.model.geom_friction[self.floor_id][0]
        self.wind_strength = None
        self.wind_dir = None

    def _initialize_observation_space(self):
        """
        Initialize mixed observation space:
        - Image (RGB): height x width x 3
        - Car states: velocity, steering angle, progress, etc.
        - Checkpoint distances: distance to next few checkpoints
        """
        # Image space
        image_space = spaces.Box(
            low=0, 
            high=255,
            shape=(self._observation_height, self._observation_width, 3),
            dtype=np.uint8
        )
        
        # Vehicle States (Vel, Acc, Yaw, etc.)
        car_states_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(7,), 
            dtype=np.float32
        )
                
        if DISTANCE_MODE:
            # Relative Distance to the next n (n = 5) checkpoints
            checkpoint_space = spaces.Box(
                low=0.0,
                high=np.inf, 
                shape=(self._next_n_checkpoint,),  # distance to next n checkpoints
                dtype=np.float32)  # distances to next 5
        else:
            # Relative Position to the next n (n = 5) checkpoints
            checkpoint_space = spaces.Box(
                low=-np.inf,
                high=np.inf, 
                shape=(self._next_n_checkpoint, 2),  # relative x, y to next n checkpoints
                dtype=np.float32)  # distances to next 5            

        # Progress 0.0 to 1.0, 1.0 for full lap completion 
        prog_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'image': image_space,
            'car_states': car_states_space,
            'checkpoints': checkpoint_space,
            'progress': prog_space
        })
  
    def _initialize_agent_camera(self):
        # Renderer for agent view (first person)
        self.agent_renderer = MujocoRenderer(
            self.model,
            self.data,
            width=self._observation_width,
            height=self._observation_height,
            camera_name="realsense_rgb"
        )
        
    def step(self, action):      
        
        self.step_count += 1
        
        #1. Process Action and Step Env
        self._apply_wind_force()
        action = self._process_action(action)
        self.do_simulation(action, self.frame_skip)

        #2. Check Checkpoint
        crossed = self._check_checkpoint_crossing()
        crashed = self._check_crash()
        observation = self._get_obs()

        #3. Compute Reward
        reward = self._compute_reward(action, crashed)
        self.prev_checkpoint = self.current_checkpoint

        #4. Check Termination
        terminated = crashed
        ##  max_allow_step =  max_step + (checkpoint_bonus_step * current_checkpoint) * num_of_lap
        # max_allow_step = self.max_steps + (self._checkpoint_bonus_step * (self.current_checkpoint)) * (self.lap_count + 1)
        max_allow_step = self.max_steps + (self.current_checkpoint + self.n_checkpoints * self.lap_count) * self._checkpoint_bonus_step
        truncated = self.step_count >= max_allow_step
        if truncated:
            print(f"Step Count: {self.step_count} >= Max Allowed Step: {max_allow_step}")
        
        # Info dict
        info = {
            'velocity': self.prev_velocity,
            'progress': self.progress,
            'lap_count': self.lap_count,
            'crashed': crashed,
            'car_states':self._get_car_states(),
            'car_pos': self._get_car_pos()
        }        
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # Add noise only to the car's joint
        noise_pos = self.np_random.uniform(low=-self._reset_noise_scale,
                                            high=self._reset_noise_scale,
                                            size=7)   # 7 for free joint        
        qpos[self.car_qpos_start : self.car_qpos_start+7] += noise_pos
        self.set_state(qpos, qvel)

        # Reset internal state
        self.lap_count = 0
        self.progress = 0.0
        self.prev_cp_distances = np.zeros(self.n_checkpoints)
        self.current_checkpoint = 0
        self.prev_checkpoint = 0
        self.step_count = 0
        self.prev_velocity = np.array(self._get_velocimeter())
        self.prev_steering = 0.0
        
        # Domain Randomize
        if self.domain_randomization:
            self._apply_domain_randomization()
        
        return self._get_obs()

    def _get_obs(self):
        if DISTANCE_MODE:
            checkpoint_obs = self._get_checkpoint_distances()
        else:
            checkpoint_obs = self._get_checkpoint_relative_positions()
            
        return {
            'image': self._get_image(),
            'car_states': self._get_car_states(),
            'checkpoints': checkpoint_obs,
            'progress': np.array([self.progress], dtype=np.float32)
        }

    def _get_car_states(self):
        # Sensor Values
        lateral_speed, longitudinal_speed = self._get_velocimeter()
        longitudinal_acc, lateral_acc, _ = self._get_acceleromter()
        yaw_rate = self._get_yaw_rate()
        yaw_rate = self.acc_filter.compute(yaw_rate)
        # Actuator values (Last Inputs)
        steering = self.data.ctrl[0] if self.data.ctrl.size > 0 else 0.0
        throttle = self.data.ctrl[1] if self.data.ctrl.size > 1 else 0.0
        
        return np.array([
            longitudinal_speed, 
            lateral_speed,
            longitudinal_acc,
            lateral_acc,
            yaw_rate,
            steering,
            throttle
        ], dtype=np.float32)

    def _get_car_joint_ids(self):
        """Find the qpos/qvel indices of the car's free joint."""
        self.car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if self.car_body_id < 0:
            raise ValueError("Body 'chassis' not found.")
        self.car_qpos_start = None
        self.car_qvel_start = None
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == self.car_body_id and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.car_qpos_start = self.model.jnt_qposadr[j]
                self.car_qvel_start = self.model.jnt_dofadr[j]
                break
        if self.car_qpos_start is None:
            raise RuntimeError("Car does not have a free joint.")

    def _get_sensor_id(self):
        """Get MuJoCo IDs for sensors."""
        self.vel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ads_dv_velocimeter")
        if self.vel_sensor_id < 0:
            raise ValueError("Body 'Vel Sensor' not found.")

        self.acc_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ads_dv_accelerometer")
        if self.acc_sensor_id < 0:
            raise ValueError("Body 'Acc Sensor' not found.")        
        self.gyro_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ads_dv_gyro")
        if self.gyro_sensor_id < 0:
            raise ValueError("Body 'Gyro Sensor' not found.")   
           
    def _get_floor_id(self):
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_id < 0:
            raise ValueError("Geom 'floor' not found.")
           
    def _get_velocimeter(self):
        vel = self._read_sensor(self.vel_sensor_id)
        forward_speed = vel[0]   #  x forward
        lateral_speed = vel[1]   # y left/right
        return lateral_speed, forward_speed
        
    def _get_acceleromter(self):
        acc = self._read_sensor(self.acc_sensor_id)
        ax, ay, az = acc
        return ax, ay, az
    
    def _get_yaw_rate(self):
        gyro = self._read_sensor(self.gyro_sensor_id)
        yaw_rate = gyro[2]   # z-axis rotation
        return yaw_rate
    
    def _read_sensor(self, sensor_id):
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr:adr+dim].copy()
    
    def _get_image(self) -> np.ndarray:
        return self.agent_renderer.render("rgb_array")

    def _get_car_pos(self):
        return self.data.xpos[self.car_body_id].copy()
    
    def _get_car_yaw(self):
        R = self.data.xmat[self.car_body_id].reshape(3,3)
        yaw = np.arctan2(R[1,0], R[0,0])
        return yaw
    
    def _get_checkpoint_distances(self) -> np.ndarray:
        if self.checkpoint_points is None:
            return np.zeros(self._next_n_checkpoint, dtype=np.float32)

        car_pos = self._get_car_pos()[:2]
        distances = np.zeros(self._next_n_checkpoint, dtype=np.float32)
        cp = self.current_checkpoint
        for i in range(self._next_n_checkpoint):
            idx = (cp + i) % self.n_checkpoints
            cp_pos = self.checkpoint_points[idx]
            distances[i] = np.linalg.norm(cp_pos - car_pos)
        return distances
    
    def _get_checkpoint_relative_positions(self):

        if self.checkpoint_points is None:
            return np.zeros((self._next_n_checkpoint, 2), dtype=np.float32)

        car_pos = self._get_car_pos()[:2]
        yaw = self._get_car_yaw()

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        R = np.array([
            [cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw]
        ])

        rel_positions = np.zeros((self._next_n_checkpoint, 2), dtype=np.float32)
        cp = self.current_checkpoint

        for i in range(self._next_n_checkpoint):
            idx = (cp + i) % self.n_checkpoints
            cp_pos = self.checkpoint_points[idx]

            delta = cp_pos - car_pos
            rel = R @ delta
            rel_positions[i] = rel
            
        return rel_positions

    def _update_progress(self):
        ## Discrete
        self.progress = (self.current_checkpoint) / (self.n_checkpoints) + self.lap_count

    def _compute_reward(self, action, crashed):
        
        curr_vel = self._get_velocimeter()
        lat_vel, long_vel = curr_vel
        reward = 0.0

        # -------------------------------------------------
        # 1 Step penalty (encourages finishing faster)
        # -------------------------------------------------
        reward -= self._step_penalty

        # -------------------------------------------------
        # 2 Checkpoint reward
        # -------------------------------------------------
        if self.current_checkpoint != self.prev_checkpoint:
            reward += self._lap_completion_reward / self.n_checkpoints

        # -------------------------------------------------
        # 3 Forward velocity reward
        # -------------------------------------------------
        reward += self._forward_velocity_reward * max(long_vel, 0.0)

        # # -------------------------------------------------
        # # 4 Velocity Smoothness Reward
        # # -------------------------------------------------
        
        curr_vel = np.array(curr_vel)
        vel_change = np.linalg.norm(curr_vel - self.prev_velocity)
        self.prev_velocity = curr_vel
        # reward -= self._vel_penalty_weight * vel_change
        reward -= (1.5 * vel_change) ** 2


        # # # -------------------------------------------------
        # # # 5 Steering Smoothness Reward
        # # # -------------------------------------------------

        steering = action[0]
        steering_change = abs(steering - self.prev_steering)
        self.prev_steering = steering
        # reward -= self._steer_penalty_weight * steering_change
        reward -= (1.5 * steering_change) ** 2

        throttle = action[1]
        if long_vel <= 0.05 and throttle <= 0:
            # reward -= 1 * abs(throttle)
            reward -= 5
            
        # reward += steering_smooth_penalty        
        # -------------------------------------------------
        # 6 Crash penalty
        # -------------------------------------------------

        if crashed:
            reward -= self._crash_penalty


        return reward

    def _process_action(self, action):
        steer = action[0]
        throttle = action[1]
        
        if abs(throttle) < 0.05:
            throttle = 0
        
        _, long_vel = self.prev_velocity
        if long_vel <= 0.05 and throttle < 0:
            throttle = 0

        throttle *= self._max_throttle        
        return [steer, throttle]
        
    def _compute_cp_distances(self, car_pos):
        diff = car_pos[:2] - self.checkpoint_points
        return np.einsum('ij,ij->i', diff, self.checkpoint_tangents)
    
    def _check_checkpoint_crossing(self):
        car_pos = self._get_car_pos()
        dists = self._compute_cp_distances(car_pos)
        
        crossed = False
        cp = self.current_checkpoint

        if self.prev_cp_distances[cp] < 0 and dists[cp] >= 0:
            print(f"Checkpoint Crossed: {cp}")
            crossed = True
            self.current_checkpoint = (cp + 1) % self.n_checkpoints
            
            if self.current_checkpoint < cp:
                self.lap_count += 1

            self._update_progress()

        self.prev_cp_distances = dists

        return crossed

    def _check_crash(self):
        crashed = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self.car_geoms and c.geom2 in self.cone_geoms) or \
                (c.geom2 in self.car_geoms and c.geom1 in self.cone_geoms):
                crashed = True
                break 
        return crashed

    def _get_car_geom_ids(self):
        """Get all geom IDs that belong to the car."""
        car_geom_ids = []
        car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        
        if car_body_id < 0:
            raise ValueError("Geom 'chassis' not found.")
         
        # Find all geoms that belong to the car body or its children
        for i in range(self.model.ngeom):
            geom_body_id = self.model.geom_bodyid[i]
            # Check if this geom belongs to the car body hierarchy
            body_id = geom_body_id
            while body_id != 0:  # 0 is world body
                if body_id == car_body_id:
                    car_geom_ids.append(i)
                    break
                body_id = self.model.body_parentid[body_id]
        
        return car_geom_ids

    def _get_cone_geom_ids(self):
        """Get all geom IDs that belong to cones."""
        cone_geom_ids = []
        cone_names = ["cone_left", "cone_right", "cone_orange"]
        for cone_prefix in cone_names:
            # Find all bodies with this prefix
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and body_name.startswith(cone_prefix):
                    # Find geoms under this body
                    for j in range(self.model.ngeom):
                        if self.model.geom_bodyid[j] == i:
                            cone_geom_ids.append(j)
        
        return cone_geom_ids
    
    
    def _load_centreline(self, track_dir):
        
        centreline_file = os.path.join(track_dir, "centreline.csv")
        if not os.path.exists(centreline_file):
            raise FileNotFoundError(f"Centreline file not found: {centreline_file}")
        
        self.centreline = np.loadtxt(centreline_file, delimiter=',', skiprows=1)

        
    def _create_checkpoint(self):
        if self.centreline is None:
            raise ValueError("Track has no centreline data")
        
        # Calculate cumulative arc length along centreline
        distances = np.zeros(len(self.centreline))
        for i in range(1, len(self.centreline)):
            distances[i] = distances[i-1] + np.linalg.norm(
                self.centreline[i] - self.centreline[i-1]
            )
        
        total_length = distances[-1]        
        checkpoint_distances = np.linspace(0, total_length, self.n_checkpoints+1)[:-1]

        # Temporary containers
        cp_points = []
        cp_normals = []
        cp_tangents = []
        cp_distances = []
        
        for i, target_dist in enumerate(checkpoint_distances):
            
            # Find the closest point on centreline to target distance
            idx = np.argmin(np.abs(distances - target_dist)) 
            
            # Get point coordinates
            point = self.centreline[idx]
            
            # Get Frenet frame
            tangent, normal = self._get_frenet_frame(idx)
            
            # Store raw arrays
            cp_points.append(point)
            cp_normals.append(normal)
            cp_tangents.append(tangent)
            cp_distances.append(target_dist)
                
        # Convert to numpy arrays
        self.checkpoint_points = np.array(cp_points, dtype=np.float32)
        self.checkpoint_normals = np.array(cp_normals, dtype=np.float32)
        self.checkpoint_tangents = np.array(cp_tangents, dtype=np.float32)
        self.checkpoint_distances = np.array(cp_distances, dtype=np.float32)
        self.track_length = total_length
        # print(f"Generated {self.n_checkpoints} checkpoints")
        # print(f"Track length: {self.track_length:.2f} m")
        
        
    def _get_frenet_frame(self, point_index):
        # Use central differences for interior points, forward/backward for edges
        n_points = len(self.centreline)

        if point_index == 0:
            # Forward difference
            dx = self.centreline[1, 0] - self.centreline[0, 0]
            dy = self.centreline[1, 1] - self.centreline[0, 1]
        elif point_index == n_points - 1:
            # Backward difference
            dx = self.centreline[-1, 0] - self.centreline[-2, 0]
            dy = self.centreline[-1, 1] - self.centreline[-2, 1]
        else:
            # Central difference
            dx = self.centreline[point_index + 1, 0] - self.centreline[point_index - 1, 0]
            dy = self.centreline[point_index + 1, 1] - self.centreline[point_index - 1, 1]
        
        # Tangent vector (derivative)
        tangent = np.array([dx, dy])
        # Normalize
        tangent = tangent / np.linalg.norm(tangent)
        
        # Normal vector (rotate tangent by +90 degrees for left normal)
        # For a 2D curve, normal is perpendicular to tangent
        normal = np.array([tangent[1], -tangent[0]])  # Points left (counter-clockwise)
        
        return tangent, normal
    
    def _set_windy_scenario(self):
        dir = self.np_random.integers(0, 4)
        angle = dir * (np.pi / 2)
        
        self.wind_dir = np.array([
            np.cos(angle),
            np.sin(angle),
            0
        ])
        
        self.wind_strength = self.np_random.uniform(10, 30) 
        print(f"Wind Angle: {angle:.2f} rad, Strength: {self.wind_strength:.2f} N")
        
    def _set_slippery_scenario(self):
        self.road_friction = self.np_random.uniform(0.1, 0.6)
        self.model.geom_friction[self.floor_id][0] = self.road_friction
        print(f"Road Friction: {self.road_friction:.4f}")
        
        
    def _apply_wind_force(self):
        if self.wind_strength is None or self.wind_dir is None:
            return
        wind_force = self.wind_strength * self.wind_dir 
        self.data.xfrc_applied[self.car_body_id][:3] = wind_force
        
    def _select_random_track(self, track_root, track_idx=None):
        
        if not os.path.exists(track_root):
            raise FileNotFoundError(f"Track root directory not found: {track_root}")

        tracks = [
            os.path.join(track_root, d)
            for d in os.listdir(track_root)
            if d.startswith("track")
        ]
        
        if track_idx is not None:
            return tracks[track_idx]
        else:
            return random.choice(tracks)


    def _apply_domain_randomization(self):
        # Reset to default first
        self.model.geom_friction[self.floor_id][0] = self.default_friction
        self.wind_strength = None
        self.wind_dir = None

        # Random slippery condition
        if self.np_random.random() < 0.3:  # 30% chance
            self._set_slippery_scenario()

        # Random wind
        if self.np_random.random() < 0.3:  # 30% chance
            self._set_windy_scenario()