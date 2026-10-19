from __future__ import annotations
import os, json, argparse
import numpy as np
import yaml

from src.data.io import load_npz, ensure_dir
from src.train.train_dqn import train_dqn
from src.eval.evaluate import eval_detection

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--out_dir", type=str, default="runs/formal_exp_binary")
    ap.add_argument("--config", type=str, default="src/configs/default.yaml")
    ap.add_argument("--train_file", type=str, default="train_binary.npz")
    ap.add_argument("--test_file", type=str, default="test_binary.npz")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    ensure_dir(args.out_dir)

    # 1) 加载二分类数据
    train_path = os.path.join(args.data_dir, args.train_file)
    test_path = os.path.join(args.data_dir, args.test_file)

    print(f"Loading train data: {train_path}")
    Xtr, ytr = load_npz(train_path)
    print(f"Loading test data: {test_path}")
    Xte, yte = load_npz(test_path)

    print(f"Train: X={Xtr.shape}, y={ytr.shape}, classes={np.unique(ytr)}")
    print(f"Test: X={Xte.shape}, y={yte.shape}, classes={np.unique(yte)}")

    k_classes = int(cfg.get("k_classes", 2))
    print(f"k_classes: {k_classes}")

    # 2) 训练 DQN
    dqn_path = os.path.join(args.out_dir, "dqn_detect_binary")
    train_dqn(
        Xtr, ytr, k_classes,
        out_path=dqn_path,
        seed=seed,
        total_steps=int(cfg["train"]["dqn_steps"]),
        **cfg["dqn"]
    )

    # 3) 评估 DQN
    det = eval_detection(dqn_path + ".zip", Xte, yte, k_classes)

    # 4) 保存结果
    metrics = {"detection": det}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== DONE ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

