from __future__ import annotations
import numpy as np
import torch
from stable_baselines3 import DQN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from src.envs.detection_env import DetectionEnv

def eval_detection(dqn_path: str, X: np.ndarray, y: np.ndarray, k_classes: int, n_samples: int = 0):
    """
    评估 DQN 检测模型

    Args:
        dqn_path: DQN 模型路径
        X: 测试数据特征
        y: 测试数据标签
        k_classes: 类别数
        n_samples: 评估样本数，0 表示全部

    Returns:
        dict: 包含 Accuracy, Precision, Recall, F1, AUC 等指标
    """
    model = DQN.load(dqn_path)

    n = len(X) if n_samples == 0 else min(n_samples, len(X))
    X_eval = X[:n]
    y_eval = y[:n]

    # 预测
    preds = []
    probs = []  # 用于计算 AUC

    for i in range(n):
        obs = X_eval[i]
        # 获取 Q 值
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(model.device)
            q_values = model.q_net(obs_tensor)
            # 获取预测概率
            prob = torch.softmax(q_values, dim=1)[0]
            probs.append(prob.cpu().numpy())
            pred = torch.argmax(prob).item()
            preds.append(pred)

    y_pred = np.array(preds, dtype=int)
    probs = np.array(probs)

    # 计算指标
    acc = float(accuracy_score(y_eval, y_pred))
    precision = float(precision_score(y_eval, y_pred, average='macro', zero_division=0))
    recall = float(recall_score(y_eval, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, average='macro', zero_division=0))

    # 计算 AUC（二分类）
    if k_classes == 2:
        prob_attack = probs[:, 1]  # 类别 1 的概率
        auc = float(roc_auc_score(y_eval, prob_attack))
    else:
        auc = None

    # 打印详细结果
    print(f"\n=== DQN Detection Results (n={n}) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    if auc is not None:
        print(f"AUC:       {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_eval, y_pred))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "auc": auc,
        "n_samples": int(n)
    }

def eval_response(td3_path: str, X: np.ndarray, y: np.ndarray, attack_embed: np.ndarray, n_samples: int = 2000):
    """评估 TD3 响应模型"""
    from stable_baselines3 import TD3
    from src.envs.response_env import ResponseEnv

    model = TD3.load(td3_path)
    env = ResponseEnv(X, y, attack_embed)

    rewards = []
    n = min(n_samples, len(X))
    for i in range(n):
        env._idx = i
        obs = env._obs(i)
        a, _ = model.predict(obs, deterministic=True)
        _, r, _, _, _ = env.step(a)
        rewards.append(float(r))

    rewards = np.asarray(rewards, dtype="float32")
    nan_cnt = int(np.isnan(rewards).sum())
    avg = float(np.nanmean(rewards))

    return {"avg_reward": avg, "nan_rewards": nan_cnt, "n_eval": int(n)}

