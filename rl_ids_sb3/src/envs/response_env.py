from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ResponseEnv(gym.Env):
    """
    连续响应环境：
      - 观测: [数值特征; 攻击embedding]
      - 动作: [delta_threshold, response_intensity] in [-1,1]
    为了工程稳健：对 obs/action 做 nan/inf 清洗 + action clip

    预训练增强实验参数（Phase E-2）：
      - attack_reward_weight: 攻击样本奖励权重（默认1.0）
      - cost_coefficient: 代价系数（默认0.1）
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_embed: np.ndarray,
        attack_reward_weight: float = 1.0,  # 主线默认值
        cost_coefficient: float = 0.1,       # 主线默认值
    ):
        super().__init__()
        assert len(X) == len(y)
        self.X = X.astype("float32")
        self.y = y.astype("int64")
        self.attack_embed = attack_embed.astype("float32")
        self.n = len(X)

        # 预训练增强实验参数
        self.attack_reward_weight = attack_reward_weight
        self.cost_coefficient = cost_coefficient

        obs_dim = self.X.shape[1] + self.attack_embed.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._idx = 0
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        # 把 nan/inf 变成有限值，避免网络输出/环境奖励出现 NaN
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

        # 代价项：动作越大成本越高
        # 【预训练增强实验修正】：代价系数可参数化
        cost = self.cost_coefficient * (abs(delta_thr) + abs(intensity))

        # 任务项：攻击希望强响应，良性希望温和
        if label == 0:
            r_task = 1.0 - max(0.0, intensity)
        else:
            # 【预训练增强实验修正】：攻击样本奖励权重可参数化
            r_task = self.attack_reward_weight * max(0.0, intensity)

        r = r_task - cost

        terminated = True
        truncated = False
        info = {"label": label, "delta_thr": delta_thr, "intensity": intensity, "cost": cost}
        obs = self._obs(self._idx)
        return obs, float(r), terminated, truncated, info

