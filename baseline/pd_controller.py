import os
import sys
if sys.platform != "win32":
    os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import mujoco
from env.franka_tracking_env import FrankaTrackingEnv

# Operational-space PD gains — hand-tuned for the figure-eight at 100 Hz
KP = np.diag([200.0, 200.0, 200.0])   # position stiffness (N/m)
KD = np.diag([ 20.0,  20.0,  20.0])   # velocity damping  (N·s/m)

def run_pd_episode(env: FrankaTrackingEnv) -> dict:
    """
    Run one episode using a Cartesian PD controller with gravity compensation.

    Torque law:
      tau = J^T (Kp * e_pos - Kd * xdot) + qfrc_bias

    where:
      e_pos  = p_target - p_ee          (Cartesian position error, 3D)
      xdot   = J * qdot                 (Cartesian velocity, 3D)
      J      = 3x7 translational Jacobian (from mj_jacSite)
    """
    obs, _ = env.reset()
    targets, positions, actions, qaccs = [], [], [], []
    done = False
    
    # We need ee_site_id instead of ee_body_id for Jacobian, but let's use the body xpos for now
    # or get the site if it exists. Wait, FrankaTrackingEnv defines ee_body_id.
    # The prompt says `env.ee_site_id` but the env only has `env.ee_body_id`.
    # I'll use `env.ee_body_id`. But mj_jacSite needs a site. mj_jacBody needs a body.
    # Let's use mj_jacBody.

    while not done:
        t = env.step_count * env.dt
        p_target = env.traj.get_target(t)
        p_ee     = env.data.xpos[env.ee_body_id].copy()
        e_pos    = p_target - p_ee

        # Translational Jacobian (3xnv), slice to 7 actuated joints
        jac_pos = np.zeros((3, env.model.nv))
        jac_rot = np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data,
                          jac_pos, jac_rot, env.ee_body_id)
        J    = jac_pos[:, :7]           # 3x7
        qdot = env.data.qvel[:7]
        xdot = J @ qdot                 # Cartesian velocity (3D)

        # Operational-space PD command
        F_cart = KP @ e_pos - KD @ xdot

        # Map to joint torques via Jacobian transpose
        tau_pd = J.T @ F_cart

        # Add gravity + Coriolis compensation
        tau_gc    = env.data.qfrc_bias[:7].copy()
        tau_total = np.clip(tau_pd + tau_gc,
                            -env.TORQUE_LIMITS, env.TORQUE_LIMITS)

        # Apply directly (no action buffer — PD has no delay)
        env.data.ctrl[:7] = tau_total
        for _ in range(5):
            mujoco.mj_step(env.model, env.data)
        env.step_count += 1

        info = env._get_info()
        terminated_oob, terminated_stuck, terminated_collision = env._check_limits()
        terminated = terminated_oob or terminated_stuck or terminated_collision
        truncated = (env.step_count >= 1250)
        
        done = terminated or truncated
        targets.append(info["p_target"])
        positions.append(info["p_ee"])
        actions.append(tau_total / env.TORQUE_LIMITS)  # normalise for FFT parity
        qaccs.append(info["qacc"])

    return {
        "targets":   np.array(targets),
        "positions": np.array(positions),
        "actions":   np.array(actions),
        "qaccs":     np.array(qaccs),
    }

if __name__ == "__main__":
    env = FrankaTrackingEnv()
    result = run_pd_episode(env)
    pos_err = np.linalg.norm(result["targets"] - result["positions"], axis=1)
    print(f"PD Baseline - Position MSE: {np.mean(pos_err**2):.6f} m²  "
          f"Max: {np.max(pos_err)*1000:.1f} mm")
    env.close()
