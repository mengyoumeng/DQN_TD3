from __future__ import annotations
import numpy as np

def simple_feature_engineering(X_num: np.ndarray) -> np.ndarray:
    """
    一个“毕业设计可解释”的粗浅特征工程示例：
    - 原始数值特征保持不变
    - 追加 2 个统计特征：全局均值、全局标准差（按样本向量）
    你可以在这里扩展：连接频率、包长方差、时间间隔序列统计等。
    """
    mu = X_num.mean(axis=1, keepdims=True)
    sd = X_num.std(axis=1, keepdims=True)
    return np.concatenate([X_num, mu, sd], axis=1).astype("float32")

