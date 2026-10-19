"""
================================================================================
正式无泄漏版本：NPZ 数据生成脚本
================================================================================

【用途】
此脚本是正式实验使用的数据生成脚本。

【与 make_cic_transe_npz_split.py 的区别】
- 旧版：X_kg 包含 Label 嵌入，存在标签泄漏
- 本版：X_kg 仅包含 Port 和 Protocol 嵌入，无标签泄漏

【输入文件】
- data/raw/cic_small.csv
- data/kg/transe_embeddings_nolabel.npz（由 kg_train_transe_nolabel.py 生成）

【输出文件】
- data/processed/cic_train_transe_nolabel.npz
- data/processed/cic_test_transe_nolabel.npz

【相关文档】
详见项目根目录 DATA_VERSION.md 和 docs/ARCHITECTURE.md

================================================================================
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

csv_in = r"D:\\DOQTQ3\\rebuild_data\\raw\\cic_small.csv"
# 使用无泄漏版本的嵌入文件
emb_in = r"D:\\DOQTQ3\\rebuild_data\\kg\transe_embeddings_nolabel.npz"

out_train = r"D:\\DOQTQ3\\rebuild_data\\processed\cic_train_transe_nolabel.npz"
out_test  = r"D:\\DOQTQ3\\rebuild_data\\processed\cic_test_transe_nolabel.npz"

emb = np.load(emb_in, allow_pickle=True)
E = emb["entity_emb"].astype(np.float32)
id2ent = emb["id2ent"].tolist()
ent2id = {e:i for i,e in enumerate(id2ent)}

def get_ent_vec(key):
    i = ent2id.get(key, None)
    return None if i is None else E[i]

df = pd.read_csv(csv_in, low_memory=False)

df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"].notna()]
df = df[df["Label"].str.lower() != "label"]
df = df[df["Label"] != ""]

y_raw = df["Label"].astype(str)
classes, y = np.unique(y_raw, return_inverse=True)
y = y.astype(np.int64)

drop_cols = ["Label", "Timestamp", "Flow ID", "Src IP", "Dst IP", "Source IP", "Destination IP"]
X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
for c in X_df.columns:
    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_num_raw = X_df.to_numpy(dtype=np.float32)

dst_port = pd.to_numeric(df.get("Dst Port", 0), errors="coerce").fillna(-1).astype(int).to_numpy()
proto    = pd.to_numeric(df.get("Protocol", 0), errors="coerce").fillna(-1).astype(int).to_numpy()

X_kg = np.zeros((len(df), 32), dtype=np.float32)
for i, (p, pr) in enumerate(zip(dst_port, proto)):
    vecs = []
    if p >= 0:
        v = get_ent_vec(f"Port:{p}")
        if v is not None: vecs.append(v)
    if pr >= 0:
        v = get_ent_vec(f"Proto:{pr}")
        if v is not None: vecs.append(v)
    if vecs:
        X_kg[i] = np.mean(np.stack(vecs, axis=0), axis=0)

idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

Xn_tr, Xn_te = X_num_raw[train_idx], X_num_raw[test_idx]
Xk_tr, Xk_te = X_kg[train_idx],      X_kg[test_idx]
y_tr,  y_te  = y[train_idx],         y[test_idx]

scaler = StandardScaler()
Xn_tr = scaler.fit_transform(Xn_tr).astype(np.float32)
Xn_te = scaler.transform(Xn_te).astype(np.float32)

np.savez_compressed(out_train, X_num=Xn_tr, X_kg=Xk_tr, y=y_tr, classes=classes, feature_names=np.array(X_df.columns, dtype=object))
np.savez_compressed(out_test,  X_num=Xn_te, X_kg=Xk_te, y=y_te, classes=classes, feature_names=np.array(X_df.columns, dtype=object))

print("Saved:", out_train, "rows=", len(y_tr))
print("Saved:", out_test,  "rows=", len(y_te))
print("Classes:", classes)
print("X_num:", Xn_tr.shape, "X_kg:", Xk_tr.shape)


