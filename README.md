# DQN_TD3：基于强化学习的网络流量攻击检测与响应

> 强化学习小组作业：使用 DQN 完成离散攻击检测，使用 TD3 学习连续响应策略，并用无标签 TransE 嵌入增强流量特征。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.x-orange.svg)](https://stable-baselines3.readthedocs.io/)

本仓库记录从数据清洗、知识图谱特征构建、监督学习基线，到 DQN 检测、TD3 响应和闭环评估的完整小组作业代码。代码以“可复现、可解释、便于实验复核”为目标；训练输出、原始数据和个人文档不随仓库提交。

---

## 项目总体目标

本项目旨在设计并实现一个基于强化学习（Reinforcement Learning, RL）和知识图谱（Knowledge Graph, KG）的网络流量攻击检测系统，实现"检测-分析-响应"一体化的网络攻击检测框架。

### 核心技术架构

| 模块 | 技术方案 | 功能描述 |
|------|---------|---------|
| **检测** | DQN（离散动作空间） | 攻击类型判别/二分类检测 |
| **响应** | TD3（连续动作空间） | 响应强度/阈值调整等连续决策 |
| **特征增强** | 知识图谱 + TransE嵌入 | 多维特征融合（数值特征 + KG嵌入） |
| **工程化** | FastAPI + Streamlit | 推理服务与可视化展示 |

---

## 目录结构

```text
.
├── rl_ids_sb3/                 # 主线：DQN 检测 + TD3 响应 + 服务/评估
│   ├── src/data/               # NPZ 读取、特征生成
│   ├── src/kg/                 # 无标签 KG 与 TransE 嵌入
│   ├── src/envs/               # DetectionEnv、ResponseEnv
│   ├── src/train/              # DQN、TD3、联合流程训练入口
│   ├── src/offline/            # TD3-BC 离线预训练/微调实验
│   ├── src/eval/               # 指标、响应、闭环评估
│   ├── src/services/           # FastAPI 推理服务
│   ├── src/ui/                 # Streamlit 演示界面
│   └── requirements.txt
├── test/                       # CIC-IDS2018 数据准备与基线脚本
├── rebuild_data/               # 独立重建流程（KG、NPZ、基线、TD3）
├── stable_baseline3_DQNTD3/   # 早期单独 DQN/TD3 实验与诊断参考
├── docs/                       # 架构、实验、结果和限制说明
├── scripts/                    # 结果汇总和辅助脚本
└── DATA_VERSION.md             # 数据版本、泄漏边界和使用约定
```

### 目录职责

#### `rl_ids_sb3/` — 主实验目录

**职责**：最终"检测-分析-响应"一体化主线，对应项目设计中的 DQN 检测 + TD3 响应主链路。

**核心文件**：
- `src/train/train_all.py` — **主入口**：DQN+TD3一体化训练
- `src/envs/detection_env.py` — DQN检测环境
- `src/envs/response_env.py` — TD3响应环境
- `src/train/train_dqn.py` — DQN训练模块
- `src/train/train_td3.py` — TD3训练模块

**使用方式**：
```bash
cd rl_ids_sb3
python -m src.train.train_all --data_dir data/processed --out_dir runs/demo
```

#### `test/` — 数据准备与基线实验

**职责**：数据准备、知识图谱构建、TransE训练、基线模型实验。

**核心文件**：
- `run_prepare_kg_nolabel.cmd` — **正式KG准备主链路**（无泄漏）
- `run_baselines.cmd` — LR基线训练
- `run_hgb_baseline.cmd` — HGB基线训练
- `run_tree_baselines.cmd` — 决策树基线训练
- `kg_make_triples_nolabel.py` — 无标签版KG三元组构建
- `kg_train_transe_nolabel.py` — 无标签版TransE训练
- `make_cic_transe_npz_split_nolabel.py` — 无泄漏版NPZ生成

**使用方式**：
```bash
cd test
# 1. 准备无泄漏数据
run_prepare_kg_nolabel.cmd

# 2. 训练基线模型
run_baselines.cmd
```

#### `stable_baseline3_DQNTD3/` — 历史实验与诊断

**职责**：单独DQN实验、诊断脚本、对照参考。

**核心文件**：
- `src/train/train_dqn.py` — DQN训练
- `src/envs/traffic_env.py` — 检测环境
- `_*.py` — 诊断脚本（见 `DIAGNOSTICS.md`）

**注意**：此目录下的诊断脚本不纳入正式实验结论。

---

## 方法概览

1. 从 CIC-IDS2018 流量中清洗数值字段，移除标签、时间戳、IP 等不应直接进入模型的字段。
2. 仅使用端口和协议构建不含标签实体的知识图谱，并训练 32 维 TransE 嵌入。
3. 将数值特征和 KG 向量拼接为状态：DQN 在离散动作空间预测正常/攻击，TD3 在连续动作空间输出阈值变化和响应强度。
4. 在测试集上分别报告检测指标、响应代价和闭环触发/缓解指标。

> 旧版含标签实体的 `*_transe.npz` 仅用于诊断，正式结果必须使用 `*_nolabel.npz`。

## 快速开始

以下命令在 Windows PowerShell 中执行。Linux/macOS 可将激活命令替换为对应写法。

### 1. 安装依赖

```powershell
cd rl_ids_sb3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

### 2. 准备数据

将已清洗的 NPZ 文件放在 `rl_ids_sb3/data/processed/`，每个文件至少包含 `X_num`（二维浮点特征）和 `y`（从 0 开始的整数标签）；若包含 `X_kg`，TD3 入口会单独读取它。

已有 CIC-IDS2018 原始 CSV 时，可参考 `test/README.md` 和 `test/run_prepare_kg_nolabel.cmd` 生成无标签 KG 与 NPZ。`rebuild_data/` 保存了独立重建脚本，但其中部分脚本仍含本机路径，运行前请按当前机器修改输入输出路径。

### 3. 训练 DQN 检测器

```powershell
cd rl_ids_sb3
python -m src.train.train_all `
  --data_dir data/processed `
  --train_file train_binary.npz `
  --test_file test_binary.npz `
  --out_dir runs/dqn_binary
```

输出包括 `dqn_detect_binary.zip` 和 `metrics.json`。`train_all.py` 当前是 DQN 检测入口；它不会自动训练 TD3。

### 4. 训练与评估 TD3

```powershell
python -m src.train.train_td3_binary `
  --data_dir data/processed `
  --train_file train_binary.npz `
  --out_dir runs/td3_binary

python -m src.eval.eval_td3_response `
  --model runs/td3_binary/td3_resp_binary.zip `
  --data data/processed/test_binary.npz `
  --out runs/td3_binary/metrics.json
```

### 5. 闭环评估

```powershell
python -m src.eval.eval_closed_loop `
  --dqn runs/dqn_binary/dqn_detect_binary.zip `
  --td3 runs/td3_binary/td3_resp_binary.zip `
  --data data/processed/test_binary.npz `
  --out runs/closed_loop.json
```

### 6. 启动推理服务（可选）

```powershell
$env:DQN_MODEL = "runs/dqn_binary/dqn_detect_binary.zip"
$env:TD3_MODEL = "runs/td3_binary/td3_resp_binary.zip"
uvicorn src.services.inference_service:app --host 127.0.0.1 --port 8000
```

请求体为 `{"x_num": [...], "attack_embed": [...]}`。服务返回 `pred_class` 和两个连续 TD3 动作。需要图形界面时另开终端运行 `streamlit run src/ui/app.py`。

## 评估指标

- 检测：Accuracy、macro Precision、macro Recall、macro F1、ROC-AUC，以及攻击类召回率。
- 响应：平均奖励、平均强度、正常/攻击样本强度、FP/FN 成本和总成本。
- 闭环：响应触发率、攻击缓解率、漏检率、正常流量稳定率和闭环成功率。

## 已复核结果

现有正式复核结果见 [`docs/KEY_RESULTS_VERIFIED.md`](docs/KEY_RESULTS_VERIFIED.md)。结果只代表对应数据版本、随机种子、奖励函数和训练步数；更换数据预处理或环境参数后应重新训练并报告。

## 数据与实验边界

- 原始 CSV、NPZ、模型压缩包和 `runs/` 训练产物默认被忽略，避免仓库膨胀；请通过脚本自行生成。
- 无标签 KG 只排除了标签实体，不等同于严格的训练集独立预处理；项目报告应在方法和限制中说明这一点。
- `stable_baseline3_DQNTD3/` 中以下划线开头的脚本是历史诊断工具，不应作为正式结论的唯一依据。

## 相关文档

- [`DATA_VERSION.md`](DATA_VERSION.md)：数据版本与泄漏边界
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：系统架构
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)：实验设计
- [`docs/KEY_RESULTS_VERIFIED.md`](docs/KEY_RESULTS_VERIFIED.md)：复核指标
- [`rebuild_data/README.md`](rebuild_data/README.md)：独立重建说明
- [`rl_ids_sb3/README.md`](rl_ids_sb3/README.md)：主线模块说明

## 许可证

项目代码建议在公开仓库中使用 MIT License；数据集和第三方依赖分别遵循其原始许可证。当前仓库未重新分发 CIC-IDS2018 原始数据。


