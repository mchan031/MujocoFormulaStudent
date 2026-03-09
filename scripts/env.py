import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium import spaces
import mujoco
from typing import Optional, Dict, Any, Tuple, List
import os
import json
from utils import MovingAverageFilter
from gymnasium.utils import EzPickle


MAX_THROTTLE = 1.5

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
        model_path: str,
        frame_skip: int = 5,
        observation_width: int = 84,
        observation_height: int = 84,
        render_width: int = 640,
        render_height: int = 480,
        step_penalty: float = 0.1,
        lap_completion_reward: float = 1000.0,
        forward_velocity_reward: float = 0.05,
        crash_penalty: float = 100.0,
        extra_progress_time: float = 2000.0,
        reset_noise_scale: float = 0.01,
        centreline_file: str = None,
        next_n_checkpoint: int = 5,
        max_env_step: int = 2000,
        num_checkpoints: int = 10,
        **kwargs,
    ):
        
        EzPickle.__init__(
            self,
            model_path,
            frame_skip,
            observation_width,
            observation_height,
            render_width,
            render_height,
            step_penalty,
            lap_completion_reward,
            forward_velocity_reward,
            crash_penalty,
            extra_progress_time,
            reset_noise_scale,
            next_n_checkpoint,
            max_env_step,
            num_checkpoints,
            **kwargs,
        )
        
        ## Reward Config
        # self._ctrl_cost_weight = ctrl_cost_weight
        self._step_penalty = step_penalty
        self._lap_completion_reward = lap_completion_reward
        self._forward_velocity_reward = forward_velocity_reward
        self._crash_penalty = crash_penalty
        self._extra_progress_time = extra_progress_time
        self.max_steps = max_env_step
        
        self.step_count = 0
        self.longitudinal_vel = 0.0
        
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
        
        self._load_centreline(centreline_file)
        self._create_checkpoint()
                
        # Initialize observation space
        self._initialize_observation_space()
        
        # Initialize Parent Class
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=self.observation_space,
            width=render_width,
            height=render_height,
            camera_name="buddy_third_person",
            **kwargs,
        )
        
        # Overwrite action space (steering and throttle)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Get Car and Sensor Ids
        self._get_car_joint_ids()
        self._get_sensor_id()
        
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
        self.cone_geoms = self._get_cone_geom_ids()
        
        # Filter
        self.acc_filter = MovingAverageFilter(10)
        

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
        
        self.observation_space = image_space
  
    def _initialize_agent_camera(self):
        # Renderer for agent view (first person)
        self.agent_renderer = MujocoRenderer(
            self.model,
            self.data,
            width=self._observation_width,
            height=self._observation_height,
            camera_name="buddy_realsense_d435i"
        )
        
    def step(self, action):      
        
        self.step_count += 1

        #1. Process Action and Step Env
        action = self._process_action(action)
        self.do_simulation(action, self.frame_skip)

        #2. Check Checkpoint
        crossed = self._check_checkpoint_crossing()
        crashed = self._check_crash()
        observation = self._get_obs()

        #3. Compute Reward
        _, self.longitudinal_vel = self._get_velocimeter()
        reward = self._compute_reward(self.longitudinal_vel, action, crashed)
        self.prev_checkpoint = self.current_checkpoint

        #4. Check Termination
        terminated = crashed
        progress_count = int(self.progress * self._extra_progress_time) + self.max_steps 
        truncated = self.step_count >= progress_count
        if truncated:
            print(f"Step Count: {self.step_count} >= Progress Count: {progress_count}")
        
        # Info dict
        info = {
            'longitudinal_vel': self.longitudinal_vel,
            'progress': self.progress,
            'lap_count': self.lap_count,
            'crashed': crashed,
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
        self.longitudinal_vel = 0.0
        self.step_count = 0
        
        return self._get_obs()

    def _get_obs(self):
        # return {
        #     'image': self._get_image(),
        #     'car_states': self._get_car_states(),
        #     'checkpoint_distances': self._get_checkpoint_distances(),
        #     'progress': np.array([self.progress], dtype=np.float32)
        # }
        return self._get_image()

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
        self.car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "buddy")
        if self.car_body_id < 0:
            raise ValueError("Body 'buddy' not found.")
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
        self.vel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "buddy_velocimeter")
        if self.vel_sensor_id < 0:
            raise ValueError("Body 'Vel Sensor' not found.")

        self.acc_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "buddy_accelerometer")
        if self.acc_sensor_id < 0:
            raise ValueError("Body 'Acc Sensor' not found.")        
        self.gyro_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "buddy_gyro")
        if self.gyro_sensor_id < 0:
            raise ValueError("Body 'Gyro Sensor' not found.")   
           
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

    def _update_progress(self):
        ## Discrete
        self.progress = (self.current_checkpoint - 1) / (self.n_checkpoints) + self.lap_count

    def _compute_reward(self, velocity, action, crashed):
        
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
        reward += self._forward_velocity_reward * max(velocity, 0.0)

        if velocity < 0.1:
            reward -= 0.1

        # # -------------------------------------------------
        # # 4 Penalize sideways motion
        # # -------------------------------------------------
        # lateral_speed = self._get_velocimeter()[0]

        # reward -= 0.02 * abs(lateral_speed)


        # # -------------------------------------------------
        # # 5 Steering penalty (prevents zig-zag)
        # # -------------------------------------------------

        # steering = action[0]
        # reward -= 0.01 * abs(steering)


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
            
        if self.longitudinal_vel <= 0.05 and throttle < 0:
            throttle = 0

        throttle *= MAX_THROTTLE        
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

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.car_geoms and c.geom2 in self.cone_geoms:
                return True
            if c.geom2 in self.car_geoms and c.geom1 in self.cone_geoms:
                return True
        return False

    def _get_car_geom_ids(self):
        """Get all geom IDs that belong to the car."""
        car_geom_ids = []
        car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "buddy")
        
        if car_body_id < 0:
            raise ValueError("Geom 'buddy' not found.")
         
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
    
    
    def _load_centreline(self, centreline_file):
        if not os.path.exists(centreline_file):
            raise FileNotFoundError(f"Centreline file not found: {centreline_file}")
        
        self.centreline = np.loadtxt(centreline_file, delimiter=',', skiprows=1)
        # print(self.centreline[0])
        # print(centreline_file)

        
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
        print(f"Generated {self.n_checkpoints} checkpoints")
        print(f"Track length: {self.track_length:.2f} m")
        
        
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