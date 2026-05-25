import os
import sys
# Force software rendering for WSL2 headless rendering compatibility before mujoco import
if sys.platform != "win32":
    os.environ["MUJOCO_GL"] = "osmesa"

import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from env.franka_tracking_env import FrankaTrackingEnv

def make_env(rank, seed=0):
    def _init():
        env = FrankaTrackingEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

class CurriculumCallback(BaseCallback):
    def __init__(self, start_step: int, end_step: int, verbose=0):
        super().__init__(verbose)
        self.start_step = start_step
        self.end_step = end_step

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_step:
            w = 0.0
        elif self.num_timesteps > self.end_step:
            w = 1.0
        else:
            w = (self.num_timesteps - self.start_step) / (self.end_step - self.start_step)
            
        self.training_env.env_method("set_orientation_weight", float(w))
        return True

def main():
    parser = argparse.ArgumentParser(description="Train SAC on Franka End-Effector 3D Tracking Env")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total timesteps to train")
    parser.add_argument("--run-name", type=str, default="sac_franka_se3_high_precision", help="Name of the training run")
    args = parser.parse_args()

    n_envs    = 6
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env  = SubprocVecEnv([make_env(99)])
    eval_env  = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # Setup directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = SAC(
        "MlpPolicy",
        train_env,
        device="cpu",  # Strict CPU-only execution constraint
        verbose=1,
        batch_size=256,
        buffer_size=300_000,
        learning_rate=3e-4,
        learning_starts=0,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[512, 512, 512], log_std_init=-1.0),
        tensorboard_log=f"logs/{args.run_name}",
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path="checkpoints/",
            name_prefix=args.run_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"models",
            log_path=f"logs",
            eval_freq=20_000,
            n_eval_episodes=3,
            deterministic=True,
        ),
        CurriculumCallback(start_step=args.timesteps // 2, end_step=args.timesteps)
    ]

    print(f"Starting SAC training on CPU for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    
    # Save the final trained policy and normalization stats
    model.save(f"models/{args.run_name}_final")
    train_env.save(f"models/{args.run_name}_vec_normalize.pkl")
    
    train_env.close()
    eval_env.close()
    print("Training finished and model saved.")

if __name__ == "__main__":
    main()
