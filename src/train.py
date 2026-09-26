from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.common import ensure_directories, set_global_seed, write_json
from src.plot_results import make_training_plot


def train(total_timesteps: int, seed: int, root: Path) -> Path:
    paths = ensure_directories(root)
    train_log = paths["logs"] / "train"
    eval_log = paths["logs"] / "eval"
    train_log.mkdir(parents=True, exist_ok=True)
    eval_log.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    train_env = Monitor(gym.make("CartPole-v1"), str(train_log / "monitor.csv"))
    eval_env = Monitor(gym.make("CartPole-v1"), str(eval_log / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1,
        device="cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["models"] / "best"),
        log_path=str(eval_log),
        eval_freq=max(1_000, total_timesteps // 10),
        n_eval_episodes=10,
        deterministic=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model_path = paths["models"] / "ppo_cartpole"
    model.save(model_path)

    write_json(
        paths["outputs"] / "training_config.json",
        {
            "environment": "CartPole-v1",
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "seed": seed,
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
    )
    make_training_plot(train_log, paths["outputs"] / "training_curve.png")
    train_env.close()
    eval_env.close()
    return model_path.with_suffix(".zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huan luyen PPO tren CartPole-v1.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Da luu mo hinh tai: {train(args.timesteps, args.seed, args.root)}")
