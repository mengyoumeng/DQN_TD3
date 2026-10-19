# CIC-IDS2018 二分类复现指南

## 1. 项目架构

- `test/`：原始 CSV、无标签 KG/TransE、训练测试 NPZ 的原始实现。
- `rl_ids_sb3/src/envs/`：DQN 检测环境与 TD3 响应环境。
- `rl_ids_sb3/src/train/`：训练入口；当前 `train_all.py` 实际只训练 DQN。
- `rl_ids_sb3/src/eval/`：模型评估。
- `rebuild_data/`：本次独立重建的数据与可重复运行脚本，保留原项目产物。

## 2. 环境和数据

从仓库根目录 `D:\DOQTQ3` 打开 PowerShell。现有 Python 环境位于 `test/.venv` 和 `rl_ids_sb3/.venv`。如果缺失，可参考各目录 `requirements.txt` 创建环境。

输入 `rebuild_data/raw/cic_small.csv` 是从项目自身 Git 历史中的 `test/data/raw/cic_small.csv` 恢复的 199,989 行二分类子集。若缺失，用以下命令恢复并复制：

```powershell
New-Item -ItemType Directory -Force test\data\raw, rebuild_data\raw | Out-Null
Push-Location test
cmd /c "git show HEAD:data/raw/cic_small.csv > data\raw\cic_small.csv"
Pop-Location
Copy-Item test\data\raw\cic_small.csv rebuild_data\raw\cic_small.csv
```

## 3. 重新生成数据

在仓库根目录依次运行：

```powershell
.\test\.venv\Scripts\python.exe rebuild_data\kg_make.py
.\test\.venv\Scripts\python.exe rebuild_data\kg_train.py
.\test\.venv\Scripts\python.exe rebuild_data\make_npz.py
```

第一步构建只含端口和协议的 KG；第二步训练 32 维 TransE；第三步清理标签、将无法解析的数值和无穷值填为 0，按 80/20 分层划分，并只在训练集上拟合 `StandardScaler`。应产生 `rebuild_data/processed/cic_train_transe_nolabel.npz`（159,991 行）和 `cic_test_transe_nolabel.npz`（39,998 行）。特征维度分别是 78 和 32。

## 4. DQN 检测复现

```powershell
Push-Location rl_ids_sb3
.\.venv\Scripts\python.exe -m src.train.train_all `
  --data_dir D:\DOQTQ3\rebuild_data\processed `
  --train_file cic_train_transe_nolabel.npz `
  --test_file cic_test_transe_nolabel.npz `
  --out_dir runs\rebuild_20260928
Pop-Location
```

配置见 `rl_ids_sb3/src/configs/default.yaml`：随机种子 0、50,000 个 DQN 步数。`train_all.py` 名称虽含 all，当前只执行 DQN。结果写入 `rl_ids_sb3/runs/rebuild_20260928/metrics.json`，模型为 `dqn_detect_binary.zip`。

## 5. 基线和 TD3

```powershell
.\test\.venv\Scripts\python.exe rebuild_data\run_baselines.py
.\rl_ids_sb3\.venv\Scripts\python.exe rebuild_data\run_td3.py
```

基线包括 LR、HGB、ExtraTrees，指标写入 `baseline_metrics.json`。TD3 训练 20,000 步，结果写入 `td3_metrics.json`。TD3 使用 `[数值特征; KG 向量; 一维零占位]` 作为响应状态，这是本次独立复现时采用的明确输入设置。TD3 的平均奖励只能在相同环境、输入和奖励定义下比较，不能直接与旧的 `td3_resp_binary.zip` 指标视为同一实验。

## 6. 验收与局限

已完成运行的汇总见 `reproduction_summary.json`。在测试集上 DQN Accuracy 为 0.7254、Macro-F1 为 0.6343、AUC 为 0.6765；TD3 平均奖励为 0.6897，NaN 奖励为 0。

注意：TransE 脚本未设置随机种子，所以再次运行的 KG 向量和模型指标可能略有变化。无标签 KG 虽不含标签实体，但原脚本在划分训练/测试之前统计全体样本的端口与协议共现，因此这是无标签的转导式特征构建；若课程要求严格的训练集独立预处理，应先划分，再仅用训练集构图，并重新报告结果。原脚本把缺失或无穷数值填 0，不执行重复样本去重；报告中应按实际处理步骤描述“清洗”。

