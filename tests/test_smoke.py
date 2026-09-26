from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from src.common import ensure_directories


def test_directories_are_created(tmp_path: Path) -> None:
    paths = ensure_directories(tmp_path)
    assert all(path.is_dir() for path in paths.values())


def test_ppo_can_learn_and_predict() -> None:
    env = gym.make("CartPole-v1")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        seed=7,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=64)
    observation, _ = env.reset(seed=7)
    action, _ = model.predict(observation, deterministic=True)
    assert env.action_space.contains(action)
    env.close()

