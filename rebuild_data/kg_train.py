"""
================================================================================
正式无泄漏版本：TransE 训练脚本
================================================================================

【用途】
此脚本是正式实验使用的 TransE 嵌入训练脚本。

【与 kg_train_transe.py 的区别】
- 旧版：使用包含 Label 实体的三元组
- 本版：使用无标签三元组（triples_nolabel.tsv）

【输入文件】
- data/kg/triples_nolabel.tsv（由 kg_make_triples_nolabel.py 生成）

【输出文件】
- data/kg/transe_embeddings_nolabel.npz

【相关文档】
详见项目根目录 DATA_VERSION.md 和 docs/ARCHITECTURE.md

================================================================================
"""
import os
import numpy as np
import torch

triples_path = r"D:\\DOQTQ3\\rebuild_data\\kg\triples_nolabel.tsv"
out_path = r"D:\\DOQTQ3\\rebuild_data\\kg\transe_embeddings_nolabel.npz"

# hyperparams
dim = 32
epochs = 50  # 增加训练轮数
batch_size = 1024
lr = 1e-2
margin = 1.0
neg_ratio = 1

# load triples
triples = []
entities = {}
relations = {}

def get_id(d, key):
    if key not in d:
        d[key] = len(d)
    return d[key]

with open(triples_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        h, r, t = parts
        hid = get_id(entities, h)
        rid = get_id(relations, r)
        tid = get_id(entities, t)
        triples.append((hid, rid, tid))

n_ent = len(entities)
n_rel = len(relations)
triples = np.array(triples, dtype=np.int64)

if len(triples) == 0:
    print("【警告】未找到有效三元组，请先运行 kg_make_triples_nolabel.py")
    exit(1)

device = torch.device("cpu")
E = torch.nn.Embedding(n_ent, dim, device=device)
R = torch.nn.Embedding(n_rel, dim, device=device)

# init
torch.nn.init.uniform_(E.weight, -0.1, 0.1)
torch.nn.init.uniform_(R.weight, -0.1, 0.1)

opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=lr)

def score(h, r, t):
    # L1 distance
    return torch.sum(torch.abs(E(h) + R(r) - E(t)), dim=1)

all_entities = torch.arange(n_ent, device=device)

for ep in range(1, epochs + 1):
    idx = np.random.permutation(len(triples))
    total_loss = 0.0
    for s in range(0, len(triples), batch_size):
        batch = triples[idx[s:s+batch_size]]
        h = torch.tensor(batch[:,0], device=device)
        r = torch.tensor(batch[:,1], device=device)
        t = torch.tensor(batch[:,2], device=device)

        # negatives: corrupt head or tail
        corrupt_head = torch.rand(len(h), device=device) < 0.5
        neg_h = h.clone()
        neg_t = t.clone()

        rand_ent = all_entities[torch.randint(0, n_ent, (len(h),), device=device)]
        neg_h[corrupt_head] = rand_ent[corrupt_head]
        neg_t[~corrupt_head] = rand_ent[~corrupt_head]

        pos = score(h, r, t)
        neg = score(neg_h, r, neg_t)

        loss = torch.relu(margin + pos - neg).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # normalize entity embeddings
        with torch.no_grad():
            E.weight.div_(torch.clamp(torch.norm(E.weight, p=2, dim=1, keepdim=True), min=1.0))

        total_loss += loss.item() * len(h)

    avg_loss = total_loss / len(triples)
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep}/{epochs} loss={avg_loss:.4f}")

# save
id2ent = {i:e for e,i in entities.items()}
id2rel = {i:r for r,i in relations.items()}

np.savez_compressed(
    out_path,
    entity_emb=E.weight.detach().cpu().numpy().astype(np.float32),
    rel_emb=R.weight.detach().cpu().numpy().astype(np.float32),
    id2ent=np.array([id2ent[i] for i in range(n_ent)], dtype=object),
    id2rel=np.array([id2rel[i] for i in range(n_rel)], dtype=object),
)

print("=" * 60)
print("【正式无泄漏版本】TransE 嵌入训练完成")
print("=" * 60)
print(f"输出文件: {out_path}")
print(f"实体数量: {n_ent}")
print(f"关系数量: {n_rel}")
print(f"嵌入维度: {dim}")
print()
print("【重要】此嵌入不包含标签信息，可用于正式实验")


