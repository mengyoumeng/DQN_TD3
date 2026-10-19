from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def _fallback_random_entity_embeddings(entities, dim: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {e: rng.normal(0, 1.0, size=(dim,)).astype("float32") for e in entities}

def train_transe_with_pykeen(triples_df, dim: int = 16, epochs: int = 20, seed: int = 0) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    用 PyKEEN 训练 TransE（少量 epoch，毕业设计演示够用）。
    兼容：PyKEEN 某些版本要求 pipeline() 必须同时给 training + testing
    """
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except Exception as e:
        raise RuntimeError("没有安装/导入 pykeen。请 pip install pykeen 或把 use_pykeen=false") from e

    triples = triples_df[["head", "relation", "tail"]].astype(str).values
    tf = TriplesFactory.from_labeled_triples(triples)

    # ✅ 关键修复：PyKEEN pipeline 需要 training + testing（部分版本不允许只给 training）
    # 但我们的 demo 图谱 triples 很少，split 可能切出空 testing，做兜底。
    try:
        training_tf, testing_tf = tf.split(random_state=seed)
        # 如果 testing 为空（极少数情况下），就让 testing=training（只是为了跑通流程）
        if testing_tf.num_triples == 0:
            testing_tf = training_tf
    except Exception:
        # split 如果因版本/数据太小异常，也兜底
        training_tf = tf
        testing_tf = tf

    result = pipeline(
        training=training_tf,
        testing=testing_tf,
        model="TransE",
        model_kwargs=dict(embedding_dim=dim),
        training_kwargs=dict(num_epochs=epochs),
        random_seed=seed,
        device="cpu",
    )

    # PyKEEN 内部存储为 torch tensor，取出来转 numpy
    ent_rep = result.model.entity_representations[0]
    ent_tensor = ent_rep(indices=None).detach().cpu().numpy()

    entity_to_id = training_tf.entity_to_id
    inv = {label: ent_tensor[idx].astype("float32") for label, idx in entity_to_id.items()}
    return inv, entity_to_id

def get_attacktype_embedding(entity_embeddings: Dict[str, np.ndarray], cls: int, dim: int, seed: int = 0) -> np.ndarray:
    key = f"attack_type:{int(cls)}"
    if key in entity_embeddings:
        return entity_embeddings[key]
    return _fallback_random_entity_embeddings([key], dim, seed)[key]

