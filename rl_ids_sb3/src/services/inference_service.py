from __future__ import annotations
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import DQN, TD3

# 这里假设你把模型放在 runs/demo/ 下；也可以用环境变量覆盖
DQN_PATH = os.environ.get("DQN_MODEL", "runs/demo/dqn_detect.zip")
TD3_PATH = os.environ.get("TD3_MODEL", "runs/demo/td3_resp.zip")

app = FastAPI(title="RL-IDS Inference (SB3)")

class InferenceReq(BaseModel):
    x_num: list[float]
    attack_embed: list[float]  # 你可以把 TransE embedding 当作外部输入（或服务端再生成）

class InferenceResp(BaseModel):
    pred_class: int
    td3_action: list[float]

_dqn = None
_td3 = None

@app.on_event("startup")
def _load_models():
    global _dqn, _td3
    if os.path.exists(DQN_PATH):
        _dqn = DQN.load(DQN_PATH)
    if os.path.exists(TD3_PATH):
        _td3 = TD3.load(TD3_PATH)

@app.post("/predict", response_model=InferenceResp)
def predict(req: InferenceReq):
    x = np.asarray(req.x_num, dtype="float32")
    e = np.asarray(req.attack_embed, dtype="float32")
    # DQN：输入 state = [x_num; attack_embed]
    state = np.concatenate([x, e], axis=0).astype("float32")
    pred, _ = _dqn.predict(state, deterministic=True)

    # TD3：输入 obs = [x_num; attack_embed]
    a, _ = _td3.predict(state, deterministic=True)

    return InferenceResp(pred_class=int(pred), td3_action=[float(a[0]), float(a[1])])

