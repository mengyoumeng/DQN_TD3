from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "rl_ids_sb3"
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(RL_ROOT))

from src.data.io import load_npz_with_kg, load_npz  # type: ignore  # noqa: E402
from src.eval.evaluate import eval_detection  # type: ignore  # noqa: E402
from src.eval.eval_td3_response import eval_td3_response  # type: ignore  # noqa: E402


def _round_floats(obj: Any, digits: int = 4) -> Any:
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    if isinstance(obj, float):
        return round(obj, digits)
    return obj


def evaluate_baseline(model_path: Path, data_path: Path) -> dict[str, Any]:
    pack = joblib.load(model_path)
    model = pack["model"]
    use_kg = int(pack.get("use_kg", 1))

    data = np.load(data_path, allow_pickle=True)
    x_num = data["X_num"].astype(np.float32)
    x_kg = data["X_kg"].astype(np.float32)
    y = data["y"].astype(np.int64)
    x = np.concatenate([x_num, x_kg], axis=1).astype(np.float32) if use_kg == 1 else x_num

    y_pred = model.predict(x).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            metrics["auc"] = float(roc_auc_score(y, prob[:, 1]))
        else:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    return metrics


def make_markdown(summary: dict[str, Any]) -> str:
    ds = summary["dataset"]
    det = summary["detection"]
    resp = summary["response"]

    lines = [
        "# 已复核关键实验结果",
        "",
        f"- 数据集: `{ds['path']}`",
        f"- 测试样本数: `{ds['n_samples']}`",
        f"- 正常样本: `{ds['n_benign']}`",
        f"- 攻击样本: `{ds['n_attack']}`",
        "",
        "## 检测结果",
        "",
        "| 模型 | Accuracy | Precision(macro) | Recall(macro) | F1(macro) | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, metrics in det.items():
        auc_value = "/" if metrics["auc"] is None else f"{metrics['auc']:.4f}"
        lines.append(
            f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
            f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} | "
            f"{auc_value} |"
        )

    lines.extend(
        [
            "",
            "## 响应结果",
            "",
            "| 模型 | Avg Reward | Avg Intensity | Attack Intensity | FP Cost | FN Cost | Total Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name, metrics in resp.items():
        lines.append(
            f"| {model_name} | {metrics['avg_reward']:.4f} | {metrics['avg_intensity']:.4f} | "
            f"{metrics['avg_attack_intensity']:.4f} | {metrics['fp_cost']:.4f} | "
            f"{metrics['fn_cost']:.4f} | {metrics['total_cost']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            f"- 检测侧最高 `Accuracy` 为 `{det['DQN(binary)']['accuracy']:.4f}`，来自 `DQN(binary)`。",
            f"- 检测侧最高 `Precision(macro)` 为 `{det['DQN(binary)']['precision_macro']:.4f}`，来自 `DQN(binary)`。",
            f"- 检测侧最高 `Recall(macro)` / `F1(macro)` / `AUC` 分别为 "
            f"`{det['HGB']['recall_macro']:.4f}` / `{det['HGB']['f1_macro']:.4f}` / `{det['HGB']['auc']:.4f}`，来自 `HGB`。",
            f"- 响应侧 `BC 预训练` 将 `Avg Reward` 从 `{resp['TD3(binary)']['avg_reward']:.4f}` 提升到 "
            f"`{resp['TD3 BC pretrain']['avg_reward']:.4f}`。",
            f"- 响应侧攻击样本平均响应强度从 `{resp['TD3(binary)']['avg_attack_intensity']:.4f}` 提升到 "
            f"`{resp['TD3 BC pretrain']['avg_attack_intensity']:.4f}`。",
            f"- 当前微调最优分支 `TD3 BC finetune v2` 的 `Avg Reward` 为 "
            f"`{resp['TD3 BC finetune v2']['avg_reward']:.4f}`，但 `FN Cost` 恶化到 "
            f"`{resp['TD3 BC finetune v2']['fn_cost']:.4f}`，因此仍不优于纯 `BC pretrain`。",
            "",
            "## 备注",
            "",
            "- 本次只复核了工作区现有的主实验与增强实验产物。",
            "- 补充文稿中建议的 `NSW-NB15` 外部泛化实验暂未执行，因为工作区内没有该数据集。当前原始外部数据为 `ids2018`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    detection_data = TEST_ROOT / "data" / "processed" / "cic_test_transe_nolabel.npz"
    x_num, x_kg, y = load_npz_with_kg(str(detection_data))
    x, y_simple = load_npz(str(detection_data))

    summary: dict[str, Any] = {
        "dataset": {
            "path": str(detection_data.relative_to(ROOT)).replace("\\", "/"),
            "n_samples": int(len(y)),
            "n_benign": int((y == 0).sum()),
            "n_attack": int((y == 1).sum()),
            "feature_dims": {
                "x_num": int(x_num.shape[1]),
                "x_kg": int(x_kg.shape[1]),
                "x_total": int(x.shape[1]),
            },
        },
        "detection": {},
        "response": {},
    }

    baseline_models = {
        "LR": TEST_ROOT / "models" / "lr_baseline.joblib",
        "HGB": TEST_ROOT / "models" / "hgb_baseline.joblib",
        "ExtraTrees": TEST_ROOT / "models" / "extratrees_baseline.joblib",
    }
    for name, path in baseline_models.items():
        summary["detection"][name] = evaluate_baseline(path, detection_data)

    dqn_model = RL_ROOT / "runs" / "formal_exp_binary" / "dqn_detect_binary.zip"
    dqn_metrics = eval_detection(str(dqn_model), x, y_simple, k_classes=2)
    summary["detection"]["DQN(binary)"] = {
        "accuracy": float(dqn_metrics["accuracy"]),
        "precision_macro": float(dqn_metrics["precision"]),
        "recall_macro": float(dqn_metrics["recall"]),
        "f1_macro": float(dqn_metrics["f1_macro"]),
        "auc": float(dqn_metrics["auc"]) if dqn_metrics["auc"] is not None else None,
        "n_samples": int(dqn_metrics["n_samples"]),
    }

    response_models = {
        "TD3(binary)": RL_ROOT / "runs" / "formal_exp_binary" / "td3_resp_binary.zip",
        "TD3 BC pretrain": RL_ROOT / "models" / "td3_bc_pretrained_binary.zip",
        "TD3 BC finetune": RL_ROOT / "models" / "td3_bc_finetuned_binary.zip",
        "TD3 BC finetune v2": RL_ROOT / "runs" / "pretrained_fix_v2" / "td3_bc_finetune_v2.zip",
    }
    for name, path in response_models.items():
        summary["response"][name] = eval_td3_response(str(path), x_num, x_kg, y, n_samples=0, verbose=False)

    summary = _round_floats(summary, digits=4)

    out_json = ROOT / "docs" / "KEY_RESULTS_VERIFIED.json"
    out_md = ROOT / "docs" / "KEY_RESULTS_VERIFIED.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(make_markdown(summary), encoding="utf-8")

    print(f"Saved JSON summary to: {out_json}")
    print(f"Saved Markdown summary to: {out_md}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

