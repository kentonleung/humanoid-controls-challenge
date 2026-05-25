import os
import sys
if sys.platform != "win32":
    os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.franka_tracking_env import FrankaTrackingEnv
from baseline.pd_controller import run_pd_episode
import imageio

def run_rl_episode(model, env, vec_norm=None, deterministic=True):
    obs, _ = env.reset()
    env.set_orientation_weight(0.5)   # always evaluate with full SE(3)
    done = False
    targets_pos, targets_quat = [], []
    positions, quats, actions, qaccs = [], [], [], []
    
    while not done:
        norm_obs = vec_norm.normalize_obs(np.array([obs])) if vec_norm else obs
        action, _ = model.predict(norm_obs, deterministic=deterministic)
        
        obs, reward, terminated, truncated, info = env.step(action[0] if vec_norm else action)
        done = terminated or truncated
        
        targets_pos.append(info["p_target"])
        targets_quat.append(info["q_target"])
        positions.append(info["p_ee"])
        quats.append(info["quat_ee"])
        actions.append(action[0] if vec_norm else action)
        qaccs.append(info["qacc"])
        
    return (np.array(targets_pos), np.array(targets_quat),
            np.array(positions),   np.array(quats),
            np.array(actions),     np.array(qaccs))

def quat_geodesic_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(np.einsum("ij,ij->i", q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dots)

def compute_metrics(targets_pos, positions, quats=None,
                    targets_quat=None, qaccs=None, dt=0.01):
    pos_err = np.linalg.norm(targets_pos - positions, axis=1) * 100.0
    jerk    = np.diff(qaccs, axis=0) / dt if qaccs is not None else None
    out = {
        "Position MSE (cm^2)":    float(np.mean(pos_err ** 2)),
        "Max Position Error (cm)": float(np.max(pos_err)),
        "Jerk RMS":               float(np.sqrt(np.mean(jerk ** 2))) if jerk is not None else float("nan"),
    }
    if quats is not None and targets_quat is not None:
        oe = quat_geodesic_error(quats, targets_quat)
        out["Orientation MSE (rad^2)"] = float(np.mean(oe ** 2))
        out["Max Orient Error (rad)"]  = float(np.max(oe))
    return out

def print_comparison(pd_metrics: dict, rl_metrics: dict):
    print("\n=== Tracking Performance Comparison ===")
    print(f"{'Metric':<30} {'PD Baseline':>14} {'SAC Policy':>12} {'Improvement':>12}")
    print("-" * 72)
    for k in pd_metrics:
        pd_v = pd_metrics[k]
        rl_v = rl_metrics.get(k, float("nan"))
        if not np.isnan(pd_v) and not np.isnan(rl_v) and pd_v > 0:
            impr = f"{(pd_v - rl_v) / pd_v * 100:+.1f}%"
        else:
            impr = "—"
        print(f"{k:<30} {pd_v:>14.4f} {rl_v:>12.4f} {impr:>12}")
    # Orientation — RL only
    for k in rl_metrics:
        if k not in pd_metrics:
            print(f"{k:<30} {'N/A':>14} {rl_metrics[k]:>12.4f} {'—':>12}")

def plot_comparison(pd_res, rl_tp, rl_pos, rl_quats, rl_tq,
                    rl_actions, dt, out_dir):
    t_pd = np.arange(len(pd_res["positions"])) * dt
    t_rl = np.arange(len(rl_pos))              * dt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 3D trajectory
    ax = fig.add_subplot(131, projection="3d")
    ax.plot(*(pd_res["targets"].T * 100.0),   label="Target",  color="royalblue", lw=2)
    ax.plot(*(rl_pos.T * 100.0),              label="SAC",     color="tomato",    lw=1.5, ls=":")
    
    # Mark Start and Finish
    ax.scatter(*(rl_pos[0] * 100.0), color="green", marker="o", s=60, label="SAC Start")
    ax.scatter(*(rl_pos[-1] * 100.0), color="red", marker="x", s=60, label="SAC Finish")
    ax.text(*(rl_pos[0] * 100.0), " Start", color="green", fontsize=9, fontweight="bold")
    ax.text(*(rl_pos[-1] * 100.0), " Finish", color="red", fontsize=9, fontweight="bold")
    ax.set_title("3D Position Trajectory")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.legend(fontsize=8)
    fig.delaxes(axes[0])

    axes[1].plot(t_rl,
                 np.linalg.norm(rl_tp - rl_pos, axis=1) * 100.0,
                 color="tomato", label="SAC", lw=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position Error (cm)")
    axes[1].set_title("Position Tracking Error")
    axes[1].legend()
    axes[1].grid(True)

    orient_err = quat_geodesic_error(rl_quats, rl_tq)
    axes[2].plot(t_rl, np.degrees(orient_err), color="darkorange", lw=1.5)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Orientation Error (deg)")
    axes[2].set_title("SAC Orientation Error")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison.png", dpi=150)
    plt.close()

    # FFT comparison
    fig, a2 = plt.subplots(1, 1, figsize=(6, 3))
    for actions, label, color, ax in [
        (rl_actions,        "SAC","tomato", a2),
    ]:
        N     = len(actions)
        freqs = fftfreq(N, d=dt)[:N // 2]
        mag   = np.abs(fft(actions, axis=0))[:N // 2].mean(axis=1)
        ax.plot(freqs, mag, color=color)
        ax.set_title(f"{label} Action Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fft_comparison.png", dpi=150)
    plt.close()


def record_video(model, out_path, vec_norm=None, n_frames=1250):
    import mujoco
    env = FrankaTrackingEnv()
    env.set_orientation_weight(0.5)
    obs, _ = env.reset()
    frames = []
    print("Recording video using programmatic custom workspace camera (height=480, width=640)...")
    
    # Make robot geoms translucent
    for i in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if not (geom_name and "floor" in geom_name.lower()):
            env.model.geom_rgba[i, 3] = 0.5
            
    trail_actual = []
    
    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
        for _ in range(n_frames):
            norm_obs = vec_norm.normalize_obs(np.array([obs])) if vec_norm else obs
            action, _ = model.predict(norm_obs, deterministic=True)
            
            obs, _, terminated, truncated, info = env.step(action[0] if vec_norm else action)
            
            trail_actual.append(info["p_ee"].copy())
            
            # Simple custom camera looking down at the workspace
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.45, 0.0, 0.5]
            cam.distance = 1.8
            cam.azimuth = 135
            cam.elevation = -30
            
            renderer.update_scene(env.data, camera=cam)
            
            # Draw trails
            for i, p in enumerate(trail_actual):
                if renderer.scene.ngeom >= renderer.scene.maxgeom: break
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[renderer.scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.005, 0.0, 0.0]),
                    pos=p,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 0, 0, 0.8]) # Red for robot
                )
                renderer.scene.ngeom += 1
            
            frames.append(renderer.render())
            
            if terminated or truncated:
                obs, _ = env.reset()
                trail_actual.clear()
                
    print(f"Saving video to {out_path}...")
    imageio.mimsave(out_path, frames, fps=50)
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC model on Franka 3D Tracking")
    parser.add_argument("--model", type=str, default="models/sac_franka_se3_high_precision_final", help="Path to trained model")
    parser.add_argument("--n-episodes", type=int, default=3, help="Number of episodes to evaluate")
    args = parser.parse_args()

    print(f"Loading SAC policy from: {args.model}")
    model = SAC.load(args.model, device="cpu")
    env   = FrankaTrackingEnv()
    dt    = env.dt

    stats_path = args.model.replace("_final", "_vec_normalize.pkl").replace("best_model", "sac_franka_se3_high_precision_vec_normalize.pkl")
    if os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from {stats_path}")
        vec_norm = VecNormalize.load(stats_path, DummyVecEnv([lambda: FrankaTrackingEnv()]))
        vec_norm.training = False
        vec_norm.norm_reward = False
    else:
        print("No VecNormalize stats found, using raw observations.")
        vec_norm = None

    pd_all_metrics, rl_all_metrics = [], []
    
    # Setup directories for outputs
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    for ep in range(args.n_episodes):
        # RL episode
        tp, tq, pos, quat, act, qacc = run_rl_episode(model, env, vec_norm)
        rl_m = compute_metrics(tp, pos, quat, tq, qacc, dt)
        rl_all_metrics.append(rl_m)

        # PD baseline episode
        pd_res = run_pd_episode(FrankaTrackingEnv())
        pd_m   = compute_metrics(pd_res["targets"], pd_res["positions"],
                                 qaccs=pd_res["qaccs"], dt=dt)
        pd_all_metrics.append(pd_m)

        if ep == 0:
            print("Generating evaluation tracking and action FFT plots...")
            plot_comparison(pd_res, tp, pos, quat, tq, act, dt, "results/plots")
            record_video(model, "results/videos/tracking.mp4", vec_norm)

    # Aggregate and print comparison
    def mean_metrics(ms):
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    print_comparison(mean_metrics(pd_all_metrics), mean_metrics(rl_all_metrics))
    env.close()

if __name__ == "__main__":
    main()
