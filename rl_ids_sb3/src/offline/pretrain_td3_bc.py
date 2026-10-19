import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from src.offline.make_env import make_response_env_from_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True, help="offline buffer npz, e.g. data/offline/llm_buffer_A.npz")
    ap.add_argument("--npz", required=True, help="dataset npz used to build env spaces, e.g. data/processed/train.npz")
    ap.add_argument("--save", default="models/td3_bc_init.zip")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    data = np.load(args.buffer, allow_pickle=True)
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    act = torch.tensor(data["act"], dtype=torch.float32)

    ds = TensorDataset(obs, act)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    env = DummyVecEnv([lambda: make_response_env_from_npz(args.npz, embed_dim=args.embed_dim)])

    # 建一个 TD3 模型（只是为了拿到 policy/actor 结构）
    model = TD3("MlpPolicy", env, verbose=0, device=args.device)
    actor = model.policy.actor

    # 训练 actor 逼近 LLM 动作
    actor.train()
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    # SB3 会把网络放到 model.device 上
    device = model.device
    for ep in range(args.epochs):
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = actor(xb)
            loss = mse(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())

        print(f"epoch {ep+1}/{args.epochs} bc_mse={total/len(dl):.6f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model.save(args.save)
    print("Saved TD3 BC init:", args.save)

if __name__ == "__main__":
    main()
