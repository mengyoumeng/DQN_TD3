from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DetectionEnv(gym.Env):
    """Single-step detection environment for benign/attack classification."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_classes: int,
        sampling_mode: str = "random",
    ):
        super().__init__()
        assert len(X) == len(y)
        self.X = X.astype("float32")
        self.y = y.astype("int64")
        self.k = int(k_classes)
        self.n = len(X)
        self.sampling_mode = str(sampling_mode)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.X.shape[1],),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.k)

        self._idx = 0
        self._rng = np.random.default_rng(0)
        self._classes = np.unique(self.y)
        self._class_indices = {
            cls: np.flatnonzero(self.y == cls) for cls in self._classes
        }

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _sample_index(self) -> int:
        if self.sampling_mode == "balanced":
            cls = int(self._rng.choice(self._classes))
            cls_indices = self._class_indices[cls]
            return int(self._rng.choice(cls_indices))
        return int(self._rng.integers(0, self.n))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._idx = self._sample_index()
        obs = self.X[self._idx]
        info = {"label": int(self.y[self._idx])}
        return obs, info

    def step(self, action: int):
        label = int(self.y[self._idx])
        action = int(action)

        if action == label:
            reward = 1.0
        else:
            reward = -1.0
            if label != 0 and action == 0:
                # Penalize false negatives more heavily to push recall upward.
                reward -= 2.0

        terminated = True
        truncated = False
        info = {"label": label, "pred": action, "correct": int(action == label)}
        obs = self.X[self._idx]
        return obs, float(reward), terminated, truncated, info

