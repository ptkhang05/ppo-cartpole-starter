from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.monitor import load_results


def make_training_plot(log_dir: Path, output_path: Path, window: int = 20) -> Path:
    data = load_results(str(log_dir)).sort_values("t")
    if data.empty:
        raise ValueError(f"Khong tim thay episode hoan tat trong {log_dir}")

    data["episode"] = range(1, len(data) + 1)
    data["rolling_return"] = data["r"].rolling(window, min_periods=1).mean()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["episode"], data["r"], alpha=0.28, label="Episode return")
    ax.plot(
        data["episode"],
        data["rolling_return"],
        linewidth=2,
        label=f"Moving average ({window} episodes)",
    )
    ax.set(xlabel="Episode", ylabel="Return", title="PPO on CartPole-v1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    data[["episode", "r", "l", "t", "rolling_return"]].to_csv(
        output_path.with_suffix(".csv"), index=False
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ve duong cong huan luyen PPO.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_curve.png"))
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(make_training_plot(args.log_dir, args.output, args.window))
