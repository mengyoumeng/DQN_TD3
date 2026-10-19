"""Retrain and evaluate TD3 response model using the rebuilt data."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rl_ids_sb3"))
from src.eval.evaluate import eval_response
from src.train.train_td3 import train_td3

data_dir = ROOT / "rebuild_data/processed"
train = np.load(data_dir / "cic_train_transe_nolabel.npz")
test = np.load(data_dir / "cic_test_transe_nolabel.npz")
Xtr = np.column_stack((train["X_num"], train["X_kg"]))
Xte = np.column_stack((test["X_num"], test["X_kg"]))
ytr, yte = train["y"], test["y"]
embed_tr = np.zeros((len(ytr), 1), dtype="float32")
embed_te = np.zeros((len(yte), 1), dtype="float32")
out_dir = ROOT / "rl_ids_sb3/runs/rebuild_20260928"
out_dir.mkdir(parents=True, exist_ok=True)
model_path = out_dir / "td3_resp"
train_td3(
    Xtr, ytr, embed_tr, out_path=str(model_path), seed=0, total_steps=20000,
    learning_rate=1e-3, buffer_size=50000, learning_starts=1000,
    batch_size=256, gamma=0.99, tau=0.005, train_freq=1,
    gradient_steps=1, action_noise_sigma=0.1,
)
metrics = eval_response(str(model_path) + ".zip", Xte, yte, embed_te, n_samples=len(yte))
(out_dir / "td3_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(metrics)

