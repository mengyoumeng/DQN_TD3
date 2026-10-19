from __future__ import annotations
import requests
import streamlit as st

st.title("RL-IDS Demo (SB3)")

st.markdown("输入一条样本的数值特征 + 攻击 embedding（可用随机向量演示），调用 FastAPI 推理。")

api = st.text_input("API 地址", "http://127.0.0.1:8000/predict")

x_str = st.text_area("x_num（用逗号分隔）", "0.1,0.2,0.3")
e_str = st.text_area("attack_embed（用逗号分隔）", "0.0,0.0,0.0,0.0")

def parse_vec(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

if st.button("Predict"):
    payload = {"x_num": parse_vec(x_str), "attack_embed": parse_vec(e_str)}
    r = requests.post(api, json=payload, timeout=10)
    st.json(r.json())

