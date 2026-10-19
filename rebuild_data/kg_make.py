"""
================================================================================
正式无泄漏版本：知识图谱三元组构建脚本
================================================================================

【用途】
此脚本是正式实验使用的知识图谱三元组构建脚本。

【与 kg_make_triples.py 的区别】
- 旧版（kg_make_triples.py）：使用 Label:* 作为实体头，存在间接标签泄漏
- 本版（kg_make_triples_nolabel.py）：仅使用非标签字段（Port、Protocol），无标签泄漏

【三元组设计】
- 实体类型：Port:{端口号}、Proto:{协议号}
- 关系类型：uses_port、uses_protocol、port_proto_assoc
- 不包含任何 Label:* 实体

【输出文件】
- data/kg/triples_nolabel.tsv
- data/kg/entities_nolabel.txt
- data/kg/relations_nolabel.txt

【相关文档】
详见项目根目录 DATA_VERSION.md 和 docs/ARCHITECTURE.md

================================================================================
"""
import os
import pandas as pd
import numpy as np

inp = r"D:\\DOQTQ3\\rebuild_data\\raw\\cic_small.csv"
out_dir = r"D:\\DOQTQ3\\rebuild_data\\kg"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(inp, low_memory=False)

# clean label (保留用于后续数据生成，但不用于KG构建)
df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"].notna()]
df = df[df["Label"].str.lower() != "label"]
df = df[df["Label"] != ""]

# keep needed cols (仅非标签字段)
need = ["Dst Port", "Protocol"]
for c in need:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# coerce numeric for port/proto
df["Dst Port"] = pd.to_numeric(df["Dst Port"], errors="coerce").fillna(-1).astype(int)
df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(int)

# 端口离散化：将端口分为几个类别，减少实体数量
# - well-known: 0-1023
# - registered: 1024-49151
# - dynamic: 49152-65535
# - invalid: -1
def discretize_port(p):
    if p < 0:
        return "invalid"
    elif p <= 1023:
        return f"wellknown_{p}"  # 保留知名端口的精确值
    elif p <= 49151:
        return "registered"
    else:
        return "dynamic"

# 构建三元组：仅使用 Port 和 Protocol
triples = set()
entities = set()
relations = ["uses_port", "uses_protocol", "port_proto_assoc"]

# 统计端口-协议共现关系
port_proto_pairs = {}

for port, proto in zip(df["Dst Port"], df["Protocol"]):
    if port >= 0:
        port_ent = f"Port:{port}"
        entities.add(port_ent)

    if proto >= 0:
        proto_ent = f"Proto:{proto}"
        entities.add(proto_ent)

    # 记录端口-协议共现
    if port >= 0 and proto >= 0:
        pair = (port, proto)
        port_proto_pairs[pair] = port_proto_pairs.get(pair, 0) + 1

# 构建端口-协议关联三元组（仅保留共现次数>=10的关联，减少噪声）
min_cooccurrence = 10
for (port, proto), count in port_proto_pairs.items():
    if count >= min_cooccurrence:
        h = f"Port:{port}"
        t = f"Proto:{proto}"
        triples.add((h, "port_proto_assoc", t))

# 添加协议类型之间的层次关系（可选）
# TCP=6, UDP=17, ICMP=1
proto_hierarchy = {
    6: "TCP",
    17: "UDP",
    1: "ICMP"
}
for proto, name in proto_hierarchy.items():
    proto_ent = f"Proto:{proto}"
    name_ent = f"ProtoName:{name}"
    entities.add(name_ent)
    triples.add((proto_ent, "has_name", name_ent))

# 添加 has_name 关系
if "has_name" not in relations:
    relations.append("has_name")

triples = sorted(triples)
entities = sorted(entities)

# write files
with open(os.path.join(out_dir, "triples_nolabel.tsv"), "w", encoding="utf-8") as f:
    for h, r, t in triples:
        f.write(f"{h}\t{r}\t{t}\n")

with open(os.path.join(out_dir, "entities_nolabel.txt"), "w", encoding="utf-8") as f:
    for e in entities:
        f.write(e + "\n")

with open(os.path.join(out_dir, "relations_nolabel.txt"), "w", encoding="utf-8") as f:
    for r in relations:
        f.write(r + "\n")

print("=" * 60)
print("【正式无泄漏版本】知识图谱三元组构建完成")
print("=" * 60)
print(f"输出目录: {out_dir}")
print(f"三元组数量: {len(triples)}")
print(f"实体数量: {len(entities)}")
print(f"关系类型: {relations}")
print()
print("输出文件:")
print(f"  - {os.path.join(out_dir, 'triples_nolabel.tsv')}")
print(f"  - {os.path.join(out_dir, 'entities_nolabel.txt')}")
print(f"  - {os.path.join(out_dir, 'relations_nolabel.txt')}")
print()
print("【重要】此KG不包含任何Label实体，可用于正式实验")


