from __future__ import annotations
import os
import numpy as np

def load_npz(path: str):
    """
    加载 NPZ 数据文件，支持多种格式：
    1. X_num + X_kg 格式（二分类无泄漏数据）
    2. X_num 格式
    3. X 格式
    """
    data = np.load(path, allow_pickle=True)

    # 优先使用 X_num + X_kg 拼接（二分类无泄漏数据）
    if "X_num" in data and "X_kg" in data:
        X_num = data["X_num"].astype("float32")
        X_kg = data["X_kg"].astype("float32")
        X = np.concatenate([X_num, X_kg], axis=1)
    elif "X_num" in data:
        X = data["X_num"].astype("float32")
    elif "X" in data:
        X = data["X"].astype("float32")
    else:
        raise KeyError(f"npz 里找不到 X_num/X_kg 或 X: keys={list(data.keys())}")

    # 标签
    if "y" in data:
        y = data["y"].astype("int64")
    elif "Y" in data:
        y = data["Y"].astype("int64")
    else:
        raise KeyError(f"npz 里找不到 y 或 Y: keys={list(data.keys())}")

    return X, y

def load_npz_with_kg(path: str):
    """
    加载 NPZ 数据文件，分别返回 X_num, X_kg, y
    用于需要单独处理 KG 嵌入的场景
    """
    data = np.load(path, allow_pickle=True)

    if "X_num" in data:
        X_num = data["X_num"].astype("float32")
    else:
        raise KeyError(f"npz 里找不到 X_num: keys={list(data.keys())}")

    X_kg = data.get("X_kg", None)
    if X_kg is not None:
        X_kg = X_kg.astype("float32")

    if "y" in data:
        y = data["y"].astype("int64")
    else:
        raise KeyError(f"npz 里找不到 y: keys={list(data.keys())}")

    return X_num, X_kg, y

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

