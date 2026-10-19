"""
TD3 响应模型训练脚本（二分类版本）

使用与 DQN(binary) 相同的无泄漏二分类数据
"""
from __future__ import annotations
import os
import argparse
import json
import numpy as np
import yaml
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.envs.response_env import ResponseEnv
from src.data.io import load_npz_with_kg


def train_td3_binary(
    X_num: np.ndarray,
    X_kg: np.ndarray,
    y: np.ndarray,
    out_path: str,
    seed: int = 0,
    total_steps: int = 30000,
    action_noise_sigma: float = 0.1,
    learning_rate: float = 1e-3,
    buffer_size: int = 50000,
    learning_starts: int = 1000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,
    policy_delay: int = 2,
    verbose: int = 1
) -> str:
    """
    训练 TD3 响应模型

    Args:
        X_num: 数值特征
        X_kg: KG 嵌入（作为 attack_embed）
        y: 标签
        out_path: 输出路径
        seed: 随机种子
        total_steps: 总训练步数
        action_noise_sigma: 动作噪声标准差
        其他参数: TD3 超参数

    Returns:
        str: 模型保存路径
    """
    set_random_seed(seed)

    # 创建环境
    env = ResponseEnv(X_num, y, X_kg)
    env = Monitor(env)

    # 动作噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=action_noise_sigma * np.ones(n_actions)
    )

    # 创建模型
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        policy_delay=policy_delay,
        action_noise=action_noise,
        verbose=verbose,
        seed=seed,
    )

    # 训练
    print(f"\n{'='*60}")
    print(f"Training TD3 Response Model (Binary)")
    print(f"{'='*60}")
    print(f"Data: X_num={X_num.shape}, X_kg={X_kg.shape}, y={y.shape}")
    print(f"Total steps: {total_steps}")
    print(f"Output: {out_path}")
    print(f"{'='*60}\n")

    model.learn(total_timesteps=total_steps)

    # 保存模型
    model.save(out_path)
    print(f"\nModel saved to {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train TD3 response model (binary)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="runs/formal_exp_binary")
    parser.add_argument("--config", type=str, default="src/configs/default.yaml")
    parser.add_argument("--train_file", type=str, default="train_binary.npz")
    args = parser.parse_args()

    # 加载配置
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据
    train_path = os.path.join(args.data_dir, args.train_file)
    print(f"Loading training data: {train_path}")
    X_num, X_kg, y = load_npz_with_kg(train_path)
    print(f"X_num: {X_num.shape}, X_kg: {X_kg.shape}, y: {y.shape}")
    print(f"Classes: {np.unique(y)}, Distribution: {np.bincount(y)}")

    # 训练
    td3_path = os.path.join(args.out_dir, "td3_resp_binary")
    train_td3_binary(
        X_num, X_kg, y,
        out_path=td3_path,
        seed=seed,
        total_steps=int(cfg.get("train", {}).get("td3_steps", 30000)),
        action_noise_sigma=float(cfg.get("td3", {}).get("action_noise_sigma", 0.1)),
        learning_rate=float(cfg.get("td3", {}).get("learning_rate", 1e-3)),
        buffer_size=int(cfg.get("td3", {}).get("buffer_size", 50000)),
        learning_starts=int(cfg.get("td3", {}).get("learning_starts", 1000)),
        batch_size=int(cfg.get("td3", {}).get("batch_size", 256)),
        gamma=float(cfg.get("td3", {}).get("gamma", 0.99)),
        tau=float(cfg.get("td3", {}).get("tau", 0.005)),
        train_freq=int(cfg.get("td3", {}).get("train_freq", 1)),
        policy_delay=int(cfg.get("td3", {}).get("policy_delay", 2)),
    )

    print("\n=== TD3 Training Complete ===")


if __name__ == "__main__":
    main()

