"""
TD3 响应模块独立评估脚本

评估指标：
- avg_reward: 平均奖励
- avg_intensity: 平均响应强度
- intensity_dist: 响应强度分布
- fn_cost: 对攻击样本的响应不足代价
- fp_cost: 对正常样本的过度响应代价
- total_cost: 总代价
"""
from __future__ import annotations
import argparse
import numpy as np
from stable_baselines3 import TD3
from src.envs.response_env import ResponseEnv
from src.data.io import load_npz_with_kg


def eval_td3_response(
    td3_path: str,
    X_num: np.ndarray,
    X_kg: np.ndarray,
    y: np.ndarray,
    n_samples: int = 0,
    verbose: bool = True
) -> dict:
    """
    评估 TD3 响应模型

    Args:
        td3_path: TD3 模型路径
        X_num: 数值特征
        X_kg: KG 嵌入（作为 attack_embed）
        y: 标签
        n_samples: 评估样本数，0 表示全部
        verbose: 是否打印详细结果

    Returns:
        dict: 评估指标
    """
    model = TD3.load(td3_path)

    # 数据准备
    n = len(y) if n_samples == 0 else min(n_samples, len(y))
    X_eval = X_num[:n]
    kg_eval = X_kg[:n]
    y_eval = y[:n]

    # 创建环境
    env = ResponseEnv(X_eval, y_eval, kg_eval)

    # 收集评估数据
    rewards = []
    intensities = []
    delta_thrs = []
    costs = []

    benign_intensities = []
    attack_intensities = []
    benign_rewards = []
    attack_rewards = []

    for i in range(n):
        env._idx = i
        obs = env._obs(i)
        action, _ = model.predict(obs, deterministic=True)

        # 执行动作并获取详细反馈
        _, reward, _, _, info = env.step(action)

        rewards.append(reward)
        intensities.append(info["intensity"])
        delta_thrs.append(info["delta_thr"])
        costs.append(info["cost"])

        # 按标签分组
        label = info["label"]
        if label == 0:  # Benign
            benign_intensities.append(info["intensity"])
            benign_rewards.append(reward)
        else:  # Attack
            attack_intensities.append(info["intensity"])
            attack_rewards.append(reward)

    # 转换为数组
    rewards = np.array(rewards)
    intensities = np.array(intensities)
    costs = np.array(costs)
    y_eval = np.array(y_eval)

    # 计算指标
    n_benign = np.sum(y_eval == 0)
    n_attack = np.sum(y_eval == 1)

    # 核心指标
    avg_reward = float(np.nanmean(rewards))
    avg_intensity = float(np.nanmean(intensities))
    avg_cost = float(np.nanmean(costs))

    # 分组指标
    avg_benign_intensity = float(np.mean(benign_intensities)) if benign_intensities else 0.0
    avg_attack_intensity = float(np.mean(attack_intensities)) if attack_intensities else 0.0
    avg_benign_reward = float(np.mean(benign_rewards)) if benign_rewards else 0.0
    avg_attack_reward = float(np.mean(attack_rewards)) if attack_rewards else 0.0

    # 代价分析
    # FP_cost: 对正常样本响应过强（强度 > 0.5 视为过度响应）
    fp_cases = [i for i in range(n) if y_eval[i] == 0 and intensities[i] > 0.5]
    fp_cost = float(len(fp_cases) / n_benign) if n_benign > 0 else 0.0

    # FN_cost: 对攻击样本响应不足（强度 < 0.5 视为响应不足）
    fn_cases = [i for i in range(n) if y_eval[i] == 1 and intensities[i] < 0.5]
    fn_cost = float(len(fn_cases) / n_attack) if n_attack > 0 else 0.0

    # 总代价（归一化）
    total_cost = fp_cost + fn_cost

    # 响应强度分布
    intensity_bins = {
        "low": int(np.sum(intensities < 0.3)),      # < 0.3
        "medium": int(np.sum((intensities >= 0.3) & (intensities < 0.7))),  # 0.3-0.7
        "high": int(np.sum(intensities >= 0.7))     # >= 0.7
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"TD3 Response Evaluation Results (n={n})")
        print(f"{'='*60}")
        print(f"\n[Core Metrics]")
        print(f"  Average Reward:     {avg_reward:.4f}")
        print(f"  Average Intensity:  {avg_intensity:.4f}")
        print(f"  Average Cost:       {avg_cost:.4f}")

        print(f"\n[By Class]")
        print(f"  Benign (n={n_benign}):")
        print(f"    - Avg Intensity:  {avg_benign_intensity:.4f}")
        print(f"    - Avg Reward:     {avg_benign_reward:.4f}")
        print(f"  Attack (n={n_attack}):")
        print(f"    - Avg Intensity:  {avg_attack_intensity:.4f}")
        print(f"    - Avg Reward:     {avg_attack_reward:.4f}")

        print(f"\n[Cost Analysis]")
        print(f"  FP Rate (over-response to benign): {fp_cost:.4f}")
        print(f"  FN Rate (under-response to attack): {fn_cost:.4f}")
        print(f"  Total Cost:                       {total_cost:.4f}")

        print(f"\n[Intensity Distribution]")
        print(f"  Low    (< 0.3):    {intensity_bins['low']:5d} ({100*intensity_bins['low']/n:.1f}%)")
        print(f"  Medium (0.3-0.7):  {intensity_bins['medium']:5d} ({100*intensity_bins['medium']/n:.1f}%)")
        print(f"  High   (>= 0.7):   {intensity_bins['high']:5d} ({100*intensity_bins['high']/n:.1f}%)")
        print(f"{'='*60}\n")

    return {
        "avg_reward": avg_reward,
        "avg_intensity": avg_intensity,
        "avg_cost": avg_cost,
        "avg_benign_intensity": avg_benign_intensity,
        "avg_attack_intensity": avg_attack_intensity,
        "avg_benign_reward": avg_benign_reward,
        "avg_attack_reward": avg_attack_reward,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "total_cost": total_cost,
        "intensity_distribution": intensity_bins,
        "n_samples": int(n),
        "n_benign": int(n_benign),
        "n_attack": int(n_attack)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TD3 response model")
    parser.add_argument("--model", type=str, required=True, help="Path to TD3 model (.zip)")
    parser.add_argument("--data", type=str, required=True, help="Path to test data (.npz)")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples to evaluate (0=all)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # 加载数据
    print(f"Loading data from {args.data}...")
    X_num, X_kg, y = load_npz_with_kg(args.data)
    print(f"X_num: {X_num.shape}, X_kg: {X_kg.shape}, y: {y.shape}")

    # 评估
    results = eval_td3_response(
        args.model,
        X_num,
        X_kg,
        y,
        n_samples=args.n_samples,
        verbose=True
    )

    # 保存结果
    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.out}")

    return results


if __name__ == "__main__":
    main()

