from __future__ import annotations
import os, json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

from src.envs.detection_env import DetectionEnv

def train_dqn(X: np.ndarray, y: np.ndarray, k_classes: int, out_path: str, seed: int = 0, total_steps: int = 20000, **dqn_kwargs):
    set_random_seed(seed)
    env = DetectionEnv(X, y, k_classes, sampling_mode="balanced")
    check_env(env, warn=True)
    env = Monitor(env)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        **dqn_kwargs,
    )
    model.learn(total_timesteps=int(total_steps))
    model.save(out_path)
    return out_path

