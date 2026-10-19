# RL + 知识图谱的一体化网络流量检测（SB3 版，小组作业）

这个项目用 **Stable-Baselines3 (SB3)** 的 **DQN**（离散攻击分类）+ **TD3**（连续响应/阈值调整）做一个“粗浅可跑”的一体化流程：
- **多维特征**：数值特征（基础/统计/时序） + KG/TransE 嵌入
- **检测**：DQN 在离散动作空间做攻击类别判别
- **响应**：TD3 在连续动作空间做“响应强度/阈值调整”等连续决策
- **融合推理**：DQN 给出攻击类别 + TD3 给出连续响应参数
- **微服务/可视化**：FastAPI 推理服务 + Streamlit 简单 UI（可选）

> 设计目标：强化学习课程小组作业水平，以“能复现、能跑通、能解释”为主，不追求最强指标。

---

## 1. 一键跑通（最推荐）

### 1) 安装依赖（PowerShell）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) 准备数据
项目默认读取 `data/processed/` 下的 `.npz`：
- `train.npz`：包含 `X_num`（数值特征，float32），`y`（0..K-1 int）
- `test.npz`：同上

如果你已经有自己的 `cic_train_*.npz` / `cic_test_*.npz`，也没关系：按 README 下面的“数据适配”改一下键名即可。

把文件放到：
```
data/processed/train.npz
data/processed/test.npz
```

### 3) 一键训练 + 评估（DQN + TD3 + 融合推理）
```powershell
python -m src.train.train_all --data_dir data/processed --out_dir runs/demo
```

训练结束会产出：
- `runs/demo/dqn_detect.zip`
- `runs/demo/td3_resp.zip`
- `runs/demo/metrics.json`

---

## 2. 项目结构（小组作业说明用）

```
src/
  data/                # 特征工程（示例）
  kg/                  # 构建攻击知识图谱 + 训练 TransE + 导出 embedding
  envs/                # Gymnasium 环境（DetectionEnv / ResponseEnv）
  train/               # 训练脚本（SB3：DQN/TD3）
  eval/                # 评估 + 指标
  services/            # FastAPI：特征服务/推理服务（可选）
  ui/                  # Streamlit 简单展示（可选）
```

---

## 3. 数据适配（你已有 npz 的情况）

项目默认读取键：
- `X_num`: shape = (N, D_num)
- `y`: shape = (N,)

如果你的 npz 键叫 `X`, `X_kg`, `X_num` 等：
打开 `src/data/io.py`，在 `load_npz()` 里改映射即可。

---

## 4. 运行微服务（可选）

### 1) 启动推理服务
```powershell
uvicorn src.services.inference_service:app --host 0.0.0.0 --port 8000
```

### 2) 启动 Streamlit UI
```powershell
streamlit run src/ui/app.py
```

---

## 5. 小组作业汇报怎么讲（简短模板）

- **MDP 统一建模**：状态 s = [数值特征; KG embedding]；动作包含离散检测 a_d 与连续响应 a_c；奖励 = w_det * R_det + w_resp * R_resp（多目标加权）。
- **工程化**：SB3 负责训练；环境按 Gymnasium 接口实现；模型保存/加载；提供 FastAPI 推理接口与 Streamlit 可视化。


