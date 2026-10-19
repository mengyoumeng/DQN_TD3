from __future__ import annotations
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

from src.envs.response_env import ResponseEnv

def train_td3(X: np.ndarray, y: np.ndarray, attack_embed: np.ndarray, out_path: str, seed: int = 0, total_steps: int = 20000, action_noise_sigma: float = 0.1, **td3_kwargs):
    set_random_seed(seed)
    env = ResponseEnv(X, y, attack_embed)
    check_env(env, warn=True)
    env = Monitor(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions))

    model = TD3(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        action_noise=action_noise,
        **td3_kwargs,
    )
    model.learn(total_timesteps=int(total_steps))
    model.save(out_path)
    return out_path

