"""
B1方案：阈值约束奖励函数响应环境
核心改进：将奖励与评估阈值指标直接对齐
- 攻击样本：intensity >= 0.5 → 高奖励，否则 → 惩罚
- 正常样本：intensity < 0.5 → 高奖励，否则 → 惩罚
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ResponseEnvThresholdReward(gym.Env):
    """
    阈值约束奖励响应环境：
    - 观测: [数值特征; 攻击embedding]
    - 动作: [delta_threshold, response_intensity] in [-1,1]
    - 奖励: 直接与阈值评估指标对齐

    奖励设计：
    - 攻击样本(label=1): intensity >= 0.5 → r=+2.0 (达标奖励)
                       intensity < 0.5 → r=-1.0 (未达标惩罚)
    - 正常样本(label=0): intensity < 0.5 → r=+1.0 (温和奖励)
                       intensity >= 0.5 → r=-0.5 (过度惩罚)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_embed: np.ndarray,
        threshold: float = 0.5,  # 响应强度阈值
        attack_success_reward: float = 2.0,  # 攻击响应达标奖励
        attack_fail_penalty: float = -1.0,   # 攻击响应未达标惩罚
        benign_success_reward: float = 1.0,  # 正常温和响应奖励
        benign_fail_penalty: float = -0.5,   # 正常过度响应惩罚
        small_cost_coefficient: float = 0.01,  # 小幅动作成本（避免极端值）
    ):
        super().__init__()
        assert len(X) == len(y)
        self.X = X.astype("float32")
        self.y = y.astype("int64")
        self.attack_embed = attack_embed.astype("float32")
        self.n = len(X)

        # 阈值约束奖励参数
        self.threshold = threshold
        self.attack_success_reward = attack_success_reward
        self.attack_fail_penalty = attack_fail_penalty
        self.benign_success_reward = benign_success_reward
        self.benign_fail_penalty = benign_fail_penalty
        self.small_cost_coefficient = small_cost_coefficient

        obs_dim = self.X.shape[1] + self.attack_embed.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._idx = 0
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def _obs(self, i: int):
        obs = np.concatenate([self.X[i], self.attack_embed[i]], axis=0)
        return self._sanitize(obs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._idx = int(self._rng.integers(0, self.n))
        obs = self._obs(self._idx)
        info = {"label": int(self.y[self._idx])}
        return obs, info

    def step(self, action: np.ndarray):
        label = int(self.y[self._idx])

        a = np.asarray(action, dtype="float32")
        a = self._sanitize(a)
        a = np.clip(a, -1.0, 1.0)

        delta_thr = float(a[0])
        intensity = float(a[1])

        # 阈值约束奖励（核心改进）
        if label == 1:  # 攻击样本
            if intensity >= self.threshold:
                # 响应达标：高奖励
                r_task = self.attack_success_reward
            else:
                # 响应未达标：惩罚
                r_task = self.attack_fail_penalty
        else:  # 正常样本
            if intensity < self.threshold:
                # 温和响应：奖励
                r_task = self.benign_success_reward
            else:
                # 过度响应：轻微惩罚
                r_task = self.benign_fail_penalty

        # 小幅动作成本（鼓励中等强度响应，避免极端值）
        cost = self.small_cost_coefficient * (abs(delta_thr) + abs(intensity) ** 2)

        r = r_task - cost

        terminated = True
        truncated = False
        info = {
            "label": label,
            "delta_thr": delta_thr,
            "intensity": intensity,
            "cost": cost,
            "threshold_met": int(intensity >= self.threshold) if label == 1 else int(intensity < self.threshold),
            "raw_reward": r_task,
        }
        obs = self._obs(self._idx)
        return obs, float(r), terminated, truncated, info


def evaluate_threshold_reward_model(model_path, X, y, attack_embed, threshold=0.5):
    """
    评估模型在阈值约束下的表现
    返回与评估指标对齐的结果
    """
    import torch
    from stable_baselines3 import TD3

    model = TD3.load(model_path)

    n_benign = sum(1 for l in y if l == 0)
    n_attack = sum(1 for l in y if l == 1)

    intensities = []
    benign_intensities = []
    attack_intensities = []
    threshold_met_count = 0
    attack_threshold_met = 0
    benign_threshold_met = 0

    for i in range(len(X)):
        X_num = X[i][:78]  # 只取数值特征
        obs = np.concatenate([X_num, attack_embed[i]], axis=0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(model.device)
            action = model.actor(obs_tensor).cpu().numpy()[0]

        intensity = float(np.clip(action[1], -1.0, 1.0))
        intensities.append(intensity)

        label = int(y[i])
        if label == 1:
            attack_intensities.append(intensity)
            if intensity >= threshold:
                attack_threshold_met += 1
        else:
            benign_intensities.append(intensity)
            if intensity < threshold:
                benign_threshold_met += 1

        if (label == 1 and intensity >= threshold) or (label == 0 and intensity < threshold):
            threshold_met_count += 1

    # 计算评估指标
    avg_intensity = float(np.mean(intensities))
    avg_benign_intensity = float(np.mean(benign_intensities))
    avg_attack_intensity = float(np.mean(attack_intensities))

    fp_cost = float((n_benign - benign_threshold_met) / n_benign)  # 正常样本过度响应率
    fn_cost = float((n_attack - attack_threshold_met) / n_attack)  # 攻击样本响应不足率
    total_cost = fp_cost + fn_cost

    threshold_success_rate = float(threshold_met_count / len(y))
    attack_threshold_met_rate = float(attack_threshold_met / n_attack)
    benign_threshold_met_rate = float(benign_threshold_met / n_benign)

    # 计算期望奖励（基于阈值约束奖励）
    expected_reward = 0.0
    for i in range(len(X)):
        label = int(y[i])
        intensity = intensities[i]
        if label == 1:
            if intensity >= threshold:
                expected_reward += 2.0
            else:
                expected_reward += -1.0
        else:
            if intensity < threshold:
                expected_reward += 1.0
            else:
                expected_reward += -0.5
    avg_expected_reward = expected_reward / len(y)

    intensity_dist = {
        "low": sum(1 for int_val in intensities if int_val < 0.3),
        "medium": sum(1 for int_val in intensities if 0.3 <= int_val < 0.7),
        "high": sum(1 for int_val in intensities if int_val >= 0.7),
    }

    return {
        "avg_intensity": avg_intensity,
        "avg_benign_intensity": avg_benign_intensity,
        "avg_attack_intensity": avg_attack_intensity,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "total_cost": total_cost,
        "threshold_success_rate": threshold_success_rate,
        "attack_threshold_met_rate": attack_threshold_met_rate,
        "benign_threshold_met_rate": benign_threshold_met_rate,
        "avg_expected_reward": avg_expected_reward,
        "intensity_distribution": intensity_dist,
        "attack_threshold_met": attack_threshold_met,
        "benign_threshold_met": benign_threshold_met,
    }


if __name__ == "__main__":
    print("阈值约束奖励环境已定义")
    print("核心参数：")
    print("  - threshold = 0.5")
    print("  - attack_success_reward = 2.0")
    print("  - attack_fail_penalty = -1.0")
    print("  - benign_success_reward = 1.0")
    print("  - benign_fail_penalty = -0.5")
