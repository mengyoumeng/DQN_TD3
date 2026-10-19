"""Retrain comparison classifiers on the rebuilt binary CIC data."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
train = np.load(ROOT / "processed/cic_train_transe_nolabel.npz")
test = np.load(ROOT / "processed/cic_test_transe_nolabel.npz")
Xtr = np.column_stack((train["X_num"], train["X_kg"]))
Xte = np.column_stack((test["X_num"], test["X_kg"]))
ytr, yte = train["y"], test["y"]

models = {
    "lr": LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced"),
    "hgb": HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, random_state=0),
    "extratrees": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=0, class_weight="balanced"),
}
results = {}
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    score = model.predict_proba(Xte)[:, 1]
    results[name] = {
        "accuracy": float(accuracy_score(yte, pred)),
        "precision_macro": float(precision_score(yte, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yte, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(yte, score)),
        "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(yte)),
    }
    print(name, results[name], flush=True)

out = ROOT.parent / "rl_ids_sb3/runs/rebuild_20260928/baseline_metrics.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved", out)

