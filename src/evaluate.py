from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from src.common import ensure_directories, write_json


def evaluate(model_path: Path, episodes: int, seed: int, root: Path, video: bool) -> dict:
    paths = ensure_directories(root)
    render_mode = "rgb_array" if video else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if video:
        env = RecordVideo(
            env,
            video_folder=str(paths["videos"]),
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix="ppo-cartpole",
        )

    model = PPO.load(model_path, device="cpu")
    rows: list[dict[str, float | int]] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
        rows.append(
            {
                "episode": episode + 1,
                "seed": seed + episode,
                "return": episode_return,
                "length": episode_length,
            }
        )
    env.close()

    csv_path = paths["outputs"] / "evaluation_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    returns = np.array([row["return"] for row in rows], dtype=float)
    summary = {
        "environment": "CartPole-v1",
        "episodes": episodes,
        "seed_start": seed,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)) if episodes > 1 else 0.0,
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "episode_csv": str(csv_path),
        "video_requested": video,
    }
    write_json(paths["outputs"] / "evaluation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PPO đã huấn luyện.")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_cartpole.zip"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1_000)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.model, args.episodes, args.seed, args.root, args.video)
    for key, value in result.items():
        print(f"{key}: {value}")

