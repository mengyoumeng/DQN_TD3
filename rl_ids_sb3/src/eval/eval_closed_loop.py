from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from stable_baselines3 import DQN, TD3

from src.data.io import load_npz_with_kg
from src.envs.response_env import ResponseEnv


def eval_closed_loop(
    dqn_path: str,
    td3_path: str,
    X_num: np.ndarray,
    X_kg: np.ndarray,
    y: np.ndarray,
    n_samples: int = 0,
    response_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    dqn = DQN.load(dqn_path)
    td3 = TD3.load(td3_path)

    n = len(y) if n_samples == 0 else min(n_samples, len(y))
    X_num_eval = X_num[:n]
    X_kg_eval = X_kg[:n]
    y_eval = y[:n].astype(int)
    X_state = np.concatenate([X_num_eval, X_kg_eval], axis=1).astype(np.float32)

    env = ResponseEnv(X_num_eval, y_eval, X_kg_eval)

    preds = []
    probs = []
    rewards = []
    intensities = []
    delta_thrs = []
    costs = []

    triggered_count = 0
    tp_trigger = 0
    fp_trigger = 0
    fn_missed = 0
    tn_silent = 0
    attack_mitigated = 0
    attack_under_response_detected = 0
    benign_over_response = 0

    for i in range(n):
        state = X_state[i]
        with torch.no_grad():
            obs_tensor = torch.from_numpy(state).unsqueeze(0).float().to(dqn.device)
            q_values = dqn.q_net(obs_tensor)
            prob = torch.softmax(q_values, dim=1)[0].cpu().numpy()

        pred = int(np.argmax(prob))
        probs.append(prob)
        preds.append(pred)

        if pred == 1:
            action, _ = td3.predict(state, deterministic=True)
            triggered_count += 1
        else:
            action = np.zeros(2, dtype=np.float32)

        env._idx = i
        _, reward, _, _, info = env.step(action)

        rewards.append(float(reward))
        intensities.append(float(info["intensity"]))
        delta_thrs.append(float(info["delta_thr"]))
        costs.append(float(info["cost"]))

        label = int(y_eval[i])
        intensity = float(info["intensity"])
        is_strong_response = intensity >= response_threshold

        if label == 1 and pred == 1:
            tp_trigger += 1
            if is_strong_response:
                attack_mitigated += 1
            else:
                attack_under_response_detected += 1
        elif label == 0 and pred == 1:
            fp_trigger += 1
            if is_strong_response:
                benign_over_response += 1
        elif label == 1 and pred == 0:
            fn_missed += 1
        else:
            tn_silent += 1

    preds = np.asarray(preds, dtype=int)
    probs = np.asarray(probs, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32)
    delta_thrs = np.asarray(delta_thrs, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float32)

    n_benign = int(np.sum(y_eval == 0))
    n_attack = int(np.sum(y_eval == 1))

    det_acc = float(accuracy_score(y_eval, preds))
    det_precision = float(precision_score(y_eval, preds, average="macro", zero_division=0))
    det_recall = float(recall_score(y_eval, preds, average="macro", zero_division=0))
    det_f1 = float(f1_score(y_eval, preds, average="macro", zero_division=0))
    det_auc = float(roc_auc_score(y_eval, probs[:, 1])) if probs.shape[1] == 2 else None
    det_report = classification_report(y_eval, preds, digits=4, zero_division=0, output_dict=True)
    det_confusion = confusion_matrix(y_eval, preds).tolist()

    benign_mask = y_eval == 0
    attack_mask = y_eval == 1

    avg_reward = float(np.nanmean(rewards))
    avg_intensity = float(np.nanmean(intensities))
    avg_cost = float(np.nanmean(costs))
    avg_delta_thr = float(np.nanmean(delta_thrs))

    avg_benign_intensity = float(np.mean(intensities[benign_mask])) if n_benign > 0 else 0.0
    avg_attack_intensity = float(np.mean(intensities[attack_mask])) if n_attack > 0 else 0.0
    avg_benign_reward = float(np.mean(rewards[benign_mask])) if n_benign > 0 else 0.0
    avg_attack_reward = float(np.mean(rewards[attack_mask])) if n_attack > 0 else 0.0

    fp_cost = float(np.sum(benign_mask & (intensities >= response_threshold)) / n_benign) if n_benign > 0 else 0.0
    fn_cost = float(np.sum(attack_mask & (intensities < response_threshold)) / n_attack) if n_attack > 0 else 0.0
    total_cost = fp_cost + fn_cost

    attack_mitigation_rate = float(attack_mitigated / n_attack) if n_attack > 0 else 0.0
    attack_under_response_detected_rate = float(attack_under_response_detected / n_attack) if n_attack > 0 else 0.0
    missed_attack_rate = float(fn_missed / n_attack) if n_attack > 0 else 0.0
    benign_stability_rate = float(np.sum(benign_mask & (intensities < response_threshold)) / n_benign) if n_benign > 0 else 0.0
    false_alarm_trigger_rate = float(fp_trigger / n_benign) if n_benign > 0 else 0.0
    response_trigger_rate = float(triggered_count / n)
    silent_rate = float((n - triggered_count) / n)
    closed_loop_success_rate = float((tn_silent + attack_mitigated) / n)
    attack_response_given_detected = float(attack_mitigated / tp_trigger) if tp_trigger > 0 else 0.0

    intensity_distribution = {
        "low": int(np.sum(intensities < 0.3)),
        "medium": int(np.sum((intensities >= 0.3) & (intensities < response_threshold))),
        "high": int(np.sum(intensities >= response_threshold)),
    }

    results = {
        "detection": {
            "accuracy": det_acc,
            "precision_macro": det_precision,
            "recall_macro": det_recall,
            "f1_macro": det_f1,
            "auc": det_auc,
            "confusion_matrix": det_confusion,
            "class_0_precision": float(det_report["0"]["precision"]),
            "class_0_recall": float(det_report["0"]["recall"]),
            "class_1_precision": float(det_report["1"]["precision"]),
            "class_1_recall": float(det_report["1"]["recall"]),
            "class_1_f1": float(det_report["1"]["f1-score"]),
        },
        "response": {
            "avg_reward": avg_reward,
            "avg_intensity": avg_intensity,
            "avg_delta_threshold": avg_delta_thr,
            "avg_cost": avg_cost,
            "avg_benign_intensity": avg_benign_intensity,
            "avg_attack_intensity": avg_attack_intensity,
            "avg_benign_reward": avg_benign_reward,
            "avg_attack_reward": avg_attack_reward,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "total_cost": total_cost,
            "intensity_distribution": intensity_distribution,
        },
        "closed_loop": {
            "response_threshold": response_threshold,
            "response_trigger_rate": response_trigger_rate,
            "silent_rate": silent_rate,
            "tp_trigger": int(tp_trigger),
            "fp_trigger": int(fp_trigger),
            "fn_missed": int(fn_missed),
            "tn_silent": int(tn_silent),
            "attack_mitigated": int(attack_mitigated),
            "attack_mitigation_rate": attack_mitigation_rate,
            "attack_response_given_detected": attack_response_given_detected,
            "attack_under_response_detected": int(attack_under_response_detected),
            "attack_under_response_detected_rate": attack_under_response_detected_rate,
            "missed_attack_rate": missed_attack_rate,
            "benign_over_response": int(benign_over_response),
            "benign_stability_rate": benign_stability_rate,
            "false_alarm_trigger_rate": false_alarm_trigger_rate,
            "closed_loop_success_rate": closed_loop_success_rate,
        },
        "meta": {
            "n_samples": int(n),
            "n_benign": n_benign,
            "n_attack": n_attack,
            "dqn_path": dqn_path,
            "td3_path": td3_path,
        },
    }

    if verbose:
        print(f"\n{'=' * 68}")
        print(f"Closed-Loop Evaluation Results (n={n})")
        print(f"{'=' * 68}")
        print("\n[Detection]")
        print(f"  Accuracy:           {det_acc:.4f}")
        print(f"  Precision (macro):  {det_precision:.4f}")
        print(f"  Recall (macro):     {det_recall:.4f}")
        print(f"  F1 (macro):         {det_f1:.4f}")
        if det_auc is not None:
            print(f"  AUC:                {det_auc:.4f}")
        print(f"  Attack recall:      {results['detection']['class_1_recall']:.4f}")

        print("\n[Response Under Detection Gate]")
        print(f"  Average Reward:     {avg_reward:.4f}")
        print(f"  Average Intensity:  {avg_intensity:.4f}")
        print(f"  Attack Intensity:   {avg_attack_intensity:.4f}")
        print(f"  Benign Intensity:   {avg_benign_intensity:.4f}")
        print(f"  FP Cost:            {fp_cost:.4f}")
        print(f"  FN Cost:            {fn_cost:.4f}")
        print(f"  Total Cost:         {total_cost:.4f}")

        print("\n[Closed Loop]")
        print(f"  Response Trigger:   {response_trigger_rate:.4f}")
        print(f"  Attack Mitigation:  {attack_mitigation_rate:.4f}")
        print(f"  Missed Attack Rate: {missed_attack_rate:.4f}")
        print(f"  Benign Stability:   {benign_stability_rate:.4f}")
        print(f"  Success Rate:       {closed_loop_success_rate:.4f}")
        print(f"  TP/FP/FN/TN:        {tp_trigger}/{fp_trigger}/{fn_missed}/{tn_silent}")
        print(f"{'=' * 68}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate closed-loop DQN + TD3 pipeline")
    parser.add_argument("--dqn", type=str, required=True, help="Path to DQN detection model (.zip)")
    parser.add_argument("--td3", type=str, required=True, help="Path to TD3 response model (.zip)")
    parser.add_argument("--data", type=str, required=True, help="Path to test data (.npz)")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples to evaluate (0=all)")
    parser.add_argument("--response_threshold", type=float, default=0.5, help="Intensity threshold for effective response")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X_num, X_kg, y = load_npz_with_kg(args.data)
    print(f"X_num: {X_num.shape}, X_kg: {X_kg.shape}, y: {y.shape}")

    results = eval_closed_loop(
        dqn_path=args.dqn,
        td3_path=args.td3,
        X_num=X_num,
        X_kg=X_kg,
        y=y,
        n_samples=args.n_samples,
        response_threshold=args.response_threshold,
        verbose=True,
    )

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

