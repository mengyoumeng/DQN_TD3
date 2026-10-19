from __future__ import annotations
import numpy as np
import pandas as pd

def build_demo_triples(y: np.ndarray):
    """
    粗浅示例：用标签 y 构造一个“攻击手段-漏洞-资产-路径”的小图谱。
    真实毕业设计里，你可以把 attack_type 映射到：
      - technique（攻击手段）
      - vuln（漏洞）
      - asset（目标资产）
      - path（传播路径）
    然后输出 triples: (head, relation, tail)
    """
    # 这里为了可跑通，简单硬编码几类
    mapping = {
        0: dict(tech="benign", vuln="none", asset="host", path="none"),
        1: dict(tech="dos", vuln="cve-demo-1", asset="web",  path="lateral-move"),
        2: dict(tech="scan",vuln="cve-demo-2", asset="db",   path="recon"),
        3: dict(tech="bot", vuln="cve-demo-3", asset="iot",  path="cmd-control"),
    }
    triples = []
    for cls in np.unique(y):
        m = mapping.get(int(cls), dict(tech=f"attack-{cls}", vuln="cve-unk", asset="host", path="unk"))
        atk = f"attack_type:{cls}"
        triples += [
            (atk, "uses", m["tech"]),
            (atk, "targets", m["asset"]),
            (atk, "exploits", m["vuln"]),
            (atk, "spreads_via", m["path"]),
        ]
    df = pd.DataFrame(triples, columns=["head","relation","tail"])
    return df

