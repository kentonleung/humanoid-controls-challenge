import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import mujoco

from utils.filters import EMAFilter
from utils.trajectory import TrajectoryGenerator

class FrankaTrackingEnv(gym.Env):
    """
    Gymnasium environment for 3D end-effector tracking with a Franka Emika Panda robot.
    Control rate: 100 Hz (Physics rate: 500 Hz, decimation = 5 steps).
    Observation space: 45-dimensional (filtered current state + noiseless look-ahead target trajectory).
    Action space: 7-dimensional normalized torque residuals on top of gravity compensation.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Safety bounds & Limits
    JOINT_POS_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    JOINT_POS_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
    JOINT_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
    
    WORKSPACE_X_LIMITS = np.array([-0.8, 0.8], dtype=np.float32)
    WORKSPACE_Y_LIMITS = np.array([-0.8, 0.8], dtype=np.float32)
    WORKSPACE_Z_LIMITS = np.array([0.1, 1.2], dtype=np.float32)
    
    TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)

    def __init__(self):
        super().__init__()

        # Resolve paths dynamically
        env_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(env_dir)
        xml_path = os.path.join(workspace_dir, "mujoco_menagerie", "franka_emika_panda", "scene.xml")

        # Load MuJoCo model and data
        if not os.path.exists(xml_path):
            xml_path = os.path.abspath(os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"))

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.002  # 500 Hz physics
        
        # Convert first 7 actuators (arm joints) to pure torque actuators (motors)
        for i in range(7):
            self.model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_gainprm[i, :3] = [1.0, 0.0, 0.0]
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            self.model.actuator_biasprm[i, :3] = [0.0, 0.0, 0.0]
            self.model.actuator_ctrlrange[i] = [-self.TORQUE_LIMITS[i], self.TORQUE_LIMITS[i]]
            
        # Programmatically set joint damping for stability (natural motor and gearbox damping)
        self.model.dof_damping[:7] = 30.0
            
        self.data = mujoco.MjData(self.model)

        self.dt = 0.002 * 5  # = 0.01 s per env step (100 Hz control rate)

        # Dynamic end-effector body identification (using hand or link7)
        self.ee_body_id = -1
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name in ["hand", "link7"]:
                self.ee_body_id = i
                break
        if self.ee_body_id == -1:
            raise ValueError("Could not find hand or link7 in the model")
            
        # Store nominal dynamics for Phase 2 domain randomization
        self.nominal_masses = self.model.body_mass.copy()
        self.nominal_damping = self.model.dof_damping.copy()

        # Dynamic floor geom identification for collision detection
        self.floor_geom_id = self.model.ngeom
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name.lower():
                self.floor_geom_id = i
                break

        # Define Gym spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(68,), dtype=np.float32)

        # Trajectory Generator
        self.traj = TrajectoryGenerator()

        # Step tracking
        self.step_count = 0
        
        self.orientation_weight = 0.0

        # Phase 2: 50% Safety Torque Limits (to prevent motor burnout)
        self.torque_limits = np.array([43.5, 43.5, 43.5, 43.5, 6.0, 6.0, 6.0])

        # Nominal qpos for Phase 2 warm-start
        self.NOMINAL_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def _randomise_dynamics(self):
        # Phase 2: Domain randomization for sim-to-real robustness
        for i in range(self.model.nbody):
            nominal_mass = self.nominal_masses[i]
            self.model.body_mass[i] = nominal_mass * self.np_random.uniform(0.9, 1.1)
            
        for i in range(self.model.njnt):
            nominal_damp = self.nominal_damping[i]
            self.model.dof_damping[i] = nominal_damp * self.np_random.uniform(0.8, 1.2)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize buffers, history, and filters
        self.action_buffer = deque([np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)], maxlen=2)
        self.ema_q = EMAFilter(alpha=0.3)
        self.ema_qdot = EMAFilter(alpha=0.3)
        self.ema_pee = EMAFilter(alpha=0.3)
        self.ema_quat_ee = EMAFilter(alpha=0.3)
        self.ema_omega_ee = EMAFilter(alpha=0.3)
        self.ee_history = deque(maxlen=50)
        self.prev_action = np.zeros(7, dtype=np.float32)
        
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        
        # self._randomise_dynamics()

        # Phase 2: Warm-start near the figure-8 trajectory
        self.data.qpos[:7] = self.NOMINAL_QPOS + self.np_random.uniform(-0.05, 0.05, size=7)
        self.data.qvel[:7] = self.np_random.uniform(-0.01, 0.01, size=7)

        mujoco.mj_forward(self.model, self.data)

        # Get initial noiseless states for priming EMA filters
        init_q = self.data.qpos[:7].copy()
        init_qdot = self.data.qvel[:7].copy()
        init_pee = self.data.xpos[self.ee_body_id].copy()

        quat_ee = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_ee, self.data.xmat[self.ee_body_id].copy().flatten())
        init_quat_ee = quat_ee.copy()
        
        # Read angular velocity dynamically
        vel_out = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_id, vel_out, 0)
        init_omega_ee = vel_out[:3].copy()

        # Prime EMA Filters
        self.ema_q.update(init_q)
        self.ema_qdot.update(init_qdot)
        self.ema_pee.update(init_pee)
        self.ema_quat_ee.update(init_quat_ee)
        self.ema_omega_ee.update(init_omega_ee)

        # Record starting position in history
        self.ee_history.append(init_pee)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        # Apply normalization clipping on input action
        action_clipped = np.clip(action, -1.0, 1.0)
        
        # 2-step delayed action buffer
        applied_action = self.action_buffer[0].copy()
        self.action_buffer.append(action_clipped)

        # Control decimation loop (5 physics steps * 0.002s = 10ms control period)
        for _ in range(5):
            tau_gravity_comp = self.data.qfrc_bias[:7].copy()
            # Scale action to actual torque limits (Phase 2: 50% limits)
            tau_residual = (applied_action * self.torque_limits).astype(np.float64)
            
            # 1. Joint velocity safety governor (smoothly scale down and apply active damping)
            for idx in range(7):
                vel = self.data.qvel[idx]
                limit = self.JOINT_VEL_LIMITS[idx]
                margin = 0.7 * limit
                if vel > margin:
                    if tau_residual[idx] > 0:
                        tau_residual[idx] = 0.0
                    tau_residual[idx] -= 3.0 * (vel - margin) / (limit - margin) * self.TORQUE_LIMITS[idx]
                elif vel < -margin:
                    if tau_residual[idx] < 0:
                        tau_residual[idx] = 0.0
                    tau_residual[idx] -= 3.0 * (vel + margin) / (limit - margin) * self.TORQUE_LIMITS[idx]

            # 2. Joint position safety governor (smoothly scale down and apply virtual spring return force)
            for idx in range(7):
                pos = self.data.qpos[idx]
                low = self.JOINT_POS_LIMITS_LOWER[idx]
                high = self.JOINT_POS_LIMITS_UPPER[idx]
                buffer = 0.05 * (high - low)
                if pos > high - buffer:
                    if tau_residual[idx] > 0:
                        tau_residual[idx] = 0.0
                    tau_residual[idx] -= 3.0 * (pos - (high - buffer)) / buffer * self.TORQUE_LIMITS[idx]
                elif pos < low + buffer:
                    if tau_residual[idx] < 0:
                        tau_residual[idx] = 0.0
                    tau_residual[idx] += 3.0 * ((low + buffer) - pos) / buffer * self.TORQUE_LIMITS[idx]

            # Torque safety clamping
            tau_total = np.clip(tau_gravity_comp + tau_residual, -self.TORQUE_LIMITS, self.TORQUE_LIMITS)
            
            self.data.ctrl[:7] = tau_total
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Simulate noisy sensor measurements
        noisy_q = self.data.qpos[:7].copy() + np.random.normal(0, 0.01, size=7)
        noisy_qdot = self.data.qvel[:7].copy() + np.random.normal(0, 0.05, size=7)
        noisy_pee = self.data.xpos[self.ee_body_id].copy() + np.random.normal(0, 0.002, size=3)
        
        # Geodesic end-effector orientation and IMU drift noise
        quat_ee = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_ee, self.data.xmat[self.ee_body_id].copy().flatten())
        noisy_quat_ee = quat_ee + np.random.normal(0, 0.005, size=4)
        noisy_quat_ee /= np.linalg.norm(noisy_quat_ee)

        # Angular velocity noise (world frame)
        vel_out = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_id, vel_out, 0)
        noisy_omega = vel_out[:3].copy() + np.random.normal(0, 0.01, size=3)

        # Smooth readings through EMA filters
        ema_q = self.ema_q.update(noisy_q)
        ema_qdot = self.ema_qdot.update(noisy_qdot)
        ema_pee = self.ema_pee.update(noisy_pee)
        
        ema_quat_ee = self.ema_quat_ee.update(noisy_quat_ee)
        ema_quat_ee /= np.linalg.norm(ema_quat_ee)  # keep normalized
        
        ema_omega_ee = self.ema_omega_ee.update(noisy_omega)

        # Record end-effector history for stuck detection
        curr_pee = self.data.xpos[self.ee_body_id].copy()
        self.ee_history.append(curr_pee)

        # Safety checking backstops
        terminated_oob, terminated_stuck, terminated_collision = self._check_limits()
        terminated = terminated_oob or terminated_stuck or terminated_collision
        
        # Max episode length is 1250 steps (12.5 seconds) - exact period of the figure 8
        truncated = (self.step_count >= 1250)

        # Targets (at t, t+5, t+10 steps ahead)
        t = self.step_count * self.dt
        p_target = self.traj.get_target(t)
        q_target = self.traj.get_target_orientation(t)

        reward = self._compute_reward(
            ema_pee, ema_quat_ee, p_target, q_target, action, self.data.qacc[:7].copy()
        )

        # Overwrite penalty rewards if safety limits breached
        if terminated_oob:
            reward = -10.0
        elif terminated_collision:
            reward = -10.0
        elif terminated_stuck:
            reward = -5.0

        self.prev_action = action.copy()

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        t = self.step_count * self.dt
        
        # Look-ahead trajectory computations (no EMA)
        p_target_t = self.traj.get_target(t)
        p_target_t5 = self.traj.get_target(t + 5 * self.dt)
        p_target_t10 = self.traj.get_target(t + 10 * self.dt)

        q_target_t = self.traj.get_target_orientation(t)
        q_target_t5 = self.traj.get_target_orientation(t + 5 * self.dt)
        q_target_t10 = self.traj.get_target_orientation(t + 10 * self.dt)

        obs = np.concatenate([
            self.ema_q.state,
            self.ema_qdot.state,
            self.ema_pee.state,
            self.ema_quat_ee.state,
            self.ema_omega_ee.state,
            p_target_t,
            p_target_t5,
            p_target_t10,
            q_target_t,
            q_target_t5,
            q_target_t10,
            p_target_t - self.ema_pee.state,
            p_target_t5 - self.ema_pee.state,
            p_target_t10 - self.ema_pee.state,
            self.action_buffer[0],
            self.action_buffer[1],
        ]).astype(np.float32)

        return obs

    def _get_info(self):
        t = self.step_count * self.dt
        quat_ee = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_ee, self.data.xmat[self.ee_body_id].copy().flatten())
        return {
            "p_target": self.traj.get_target(t),
            "q_target": self.traj.get_target_orientation(t),
            "p_ee": self.data.xpos[self.ee_body_id].copy(),
            "quat_ee": quat_ee,
            "qacc": self.data.qacc[:7].copy(),
        }

    def _check_limits(self):
        # 1. Joint position bounds check (no clamping, terminate on breach)
        q = self.data.qpos[:7]
        pos_breach = np.any(q < self.JOINT_POS_LIMITS_LOWER) or np.any(q > self.JOINT_POS_LIMITS_UPPER)

        # 2. Joint velocity bounds check
        qvel = self.data.qvel[:7]
        vel_breach = np.any(np.abs(qvel) > self.JOINT_VEL_LIMITS)

        # 3. Cartesian workspace bounds check
        pee = self.data.xpos[self.ee_body_id]
        cartesian_breach = (
            pee[0] < self.WORKSPACE_X_LIMITS[0] or pee[0] > self.WORKSPACE_X_LIMITS[1] or
            pee[1] < self.WORKSPACE_Y_LIMITS[0] or pee[1] > self.WORKSPACE_Y_LIMITS[1] or
            pee[2] < self.WORKSPACE_Z_LIMITS[0] or pee[2] > self.WORKSPACE_Z_LIMITS[1]
        )

        terminated_oob = bool(pos_breach or vel_breach or cartesian_breach)

        # 4. Self-collision detection (geom IDs < floor geom ID)
        terminated_collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 < self.floor_geom_id and contact.geom2 < self.floor_geom_id:
                terminated_collision = True
                break

        # 5. Stuck detection (std over 50 past steps < 0.001 m)
        terminated_stuck = False
        if len(self.ee_history) >= 50:
            arr = np.array(self.ee_history)
            std = np.std(arr, axis=0)
            if np.all(std < 0.001):
                terminated_stuck = True

        return terminated_oob, terminated_stuck, terminated_collision

    def _compute_manipulability(self) -> float:
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.ee_body_id)
        J = jac_pos[:, :7]
        JJT = J @ J.T
        return float(np.sqrt(max(np.linalg.det(JJT), 0.0)))

    def _compute_reward(self, 
                        p_ee: np.ndarray, 
                        q_ee: np.ndarray, 
                        p_target: np.ndarray, 
                        q_target: np.ndarray,
                        action: np.ndarray,
                        qacc: np.ndarray) -> float:
        
        pos_err = np.linalg.norm(p_target - p_ee)
        # Blended reward: Broad basin (5.0) to guide it in, sharp peak (50.0) for high precision
        r_pos = 0.5 * np.exp(-5.0 * pos_err) + 0.5 * np.exp(-50.0 * pos_err)

        dot_val = np.clip(np.abs(np.einsum("i,i->", q_ee, q_target)), 0.0, 1.0)
        orient_err = 2.0 * np.arccos(dot_val)
        # Linear broad slope to prevent vanishing gradient + sharp peak for precision
        r_ori = 0.5 * (1.0 - (orient_err / np.pi)) + 0.5 * np.exp(-10.0 * orient_err)
        
        # Phase 2: Singularity penalty (Barrier function)
        manip = self._compute_manipulability()
        if manip < 0.02:
            r_singularity = -0.01 / (manip + 1e-3)
        else:
            r_singularity = 0.0
        
        w = self.orientation_weight
        r_tracking = (1.0 - w) * r_pos + w * (0.5 * r_pos + 0.5 * r_ori)

        r_jerk = -0.0001 * np.sum(np.square(qacc))
        r_effort = -0.0001 * np.sum(np.square(action))

        return float(r_tracking + r_jerk + r_effort + r_singularity)

    def close(self):
        pass

    def set_orientation_weight(self, w):
        self.orientation_weight = w
