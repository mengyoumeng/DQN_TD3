# 强化学习小组作业：三人分工与协作框架

## 1. 作业目标

本作业围绕网络流量攻击检测与响应流程展开。小组共同完成一条可运行的实验链路：

`原始流量 → 数据清洗与知识图谱特征 → DQN 离散检测 → TD3 连续响应 → 闭环评估`

三名成员分别负责一个可独立复现的实验包。每名成员提交一份独立报告，报告应能在不依赖其他成员叙述的情况下说明自己的问题、方法、代码、实验和结论。最终小组材料再将三份报告按系统流程合并。

## 2. 总体分工

| 成员 | 独立主题 | 主要问题 | 主要目录 | 核心输出 |
|---|---|---|---|---|
| 成员 1 | 数据处理与知识图谱特征 | 如何在避免标签实体的前提下构建可供强化学习使用的状态特征？ | `test/`、`rebuild_data/`、`rl_ids_sb3/src/kg/`、`rl_ids_sb3/src/data/` | 清洗规则、KG 三元组、32 维 TransE、训练/测试 NPZ、数据质量报告 |
| 成员 2 | DQN 离散攻击检测 | DQN 能否根据数值特征与 KG 特征区分正常流量和攻击流量？ | `rl_ids_sb3/src/envs/detection_env.py`、`src/train/train_dqn.py`、`src/eval/evaluate.py` | DQN 模型、检测指标、混淆矩阵、基线对比、误报漏报分析 |
| 成员 3 | TD3 连续响应与闭环评估 | TD3 能否在检测结果基础上学习响应强度与阈值调整，并控制响应代价？ | `rl_ids_sb3/src/envs/response_env.py`、`src/train/train_td3*.py`、`src/eval/eval_td3_response.py`、`src/eval/eval_closed_loop.py` | TD3 模型、响应指标、成本分析、DQN+TD3 闭环结果、服务演示 |

## 3. 共享实验协议

为保证三份报告可以比较，三名成员统一使用以下设置：

- 数据：同一版二分类数据，标签统一为 0=Benign、1=Attack。
- 特征：数值特征和不含标签实体的 KG/TransE 特征；不得把标签或标签派生字段放入输入。
- 划分：固定使用项目提供的训练集和测试集，不在测试集上调参。
- 随机种子：默认 `seed=0`。如需额外种子，至少记录 0、1、2 三次结果和均值/标准差。
- 记录：保存命令、配置文件、运行时间、样本数量、特征维度、模型路径和指标 JSON。
- 结果：所有表格注明数据文件、训练步数、随机种子和评估脚本。
- 结论：只解释实际运行得到的结果，不把预期指标写成实验结果。

## 4. 成员 1：数据与 KG 特征

### 研究任务

1. 说明原始 CSV 的字段、标签分布和数据清洗规则。
2. 移除标签、时间戳、IP 等不应直接进入模型的字段。
3. 仅使用端口和协议构建无标签 KG，训练 32 维 TransE。
4. 生成 `X_num`、`X_kg`、`y`，检查形状、有限值、类别和重复情况。
5. 设计一个特征对照：仅数值特征 vs 数值特征+KG 特征。

### 运行范围

- `test/kg_make_triples_nolabel.py`
- `test/kg_train_transe_nolabel.py`
- `test/make_cic_transe_npz_split_nolabel.py`
- `rebuild_data/kg_make.py`
- `rebuild_data/kg_train.py`
- `rebuild_data/make_npz.py`
- `rl_ids_sb3/src/data/io.py`
- `rl_ids_sb3/src/kg/`

### 必交结果

- 数据处理流程图。
- 清洗前后样本数、类别数、标签分布表。
- KG 实体/关系数量和示例三元组。
- 数值特征维度、KG 维度、拼接后维度。
- 数据质量检查表：NaN、Inf、空值、未知实体。
- 两种特征输入的对照结果或可复现实验命令。

## 5. 成员 2：DQN 检测

### 研究任务

1. 将检测任务建模为离散动作强化学习问题。
2. 说明状态、动作、奖励、终止条件和采样方式。
3. 训练 DQN 二分类检测器。
4. 与 LR、HGB、ExtraTrees 等基线比较。
5. 分析准确率、macro Precision、macro Recall、macro F1、AUC、攻击类召回率和混淆矩阵。
6. 重点分析误报和漏报，不只报告 Accuracy。

### 运行范围

- `rl_ids_sb3/src/envs/detection_env.py`
- `rl_ids_sb3/src/train/train_dqn.py`
- `rl_ids_sb3/src/train/train_all.py`
- `rl_ids_sb3/src/eval/evaluate.py`
- `rebuild_data/run_baselines.py`

### 推荐命令

```powershell
cd rl_ids_sb3
python -m src.train.train_all --data_dir data/processed --train_file train_binary.npz --test_file test_binary.npz --out_dir runs/member2_dqn
python -m src.eval.evaluate --model runs/member2_dqn/dqn_detect_binary.zip --data data/processed/test_binary.npz --out runs/member2_dqn/metrics.json
```

### 必交结果

- DQN 环境定义和训练流程图。
- DQN 超参数表。
- DQN 与三个监督学习基线的统一测试集对比表。
- 混淆矩阵和至少一张训练曲线或指标图。
- 误报样本与漏报样本的原因分析。
- 对状态特征、奖励和采样策略的局限性说明。

## 6. 成员 3：TD3 响应与闭环

### 研究任务

1. 将响应任务建模为连续动作强化学习问题。
2. 说明 TD3 状态、动作、奖励和响应成本。
3. 训练 TD3，输出阈值变化和响应强度。
4. 比较原始 TD3、TD3-BC 预训练和微调方案（若对应脚本可运行）。
5. 使用 DQN 输出作为检测门控，完成闭环评估。
6. 分析平均奖励、响应强度、FP/FN 成本、总成本、攻击缓解率和闭环成功率。

### 运行范围

- `rl_ids_sb3/src/envs/response_env.py`
- `rl_ids_sb3/src/train/train_td3.py`
- `rl_ids_sb3/src/train/train_td3_binary.py`
- `rl_ids_sb3/src/offline/`
- `rl_ids_sb3/src/eval/eval_td3_response.py`
- `rl_ids_sb3/src/eval/eval_closed_loop.py`

### 推荐命令

```powershell
cd rl_ids_sb3
python -m src.train.train_td3_binary --data_dir data/processed --train_file train_binary.npz --out_dir runs/member3_td3
python -m src.eval.eval_td3_response --model runs/member3_td3/td3_resp_binary.zip --data data/processed/test_binary.npz --out runs/member3_td3/metrics.json
python -m src.eval.eval_closed_loop --dqn runs/member2_dqn/dqn_detect_binary.zip --td3 runs/member3_td3/td3_resp_binary.zip --data data/processed/test_binary.npz --out runs/member3_td3/closed_loop.json
```

### 必交结果

- TD3 环境和奖励函数说明。
- TD3 超参数表及动作范围。
- 原始 TD3 与 TD3-BC 结果对比。
- 响应强度分布、成本分解和攻击/正常样本对比。
- DQN 单独检测与 DQN+TD3 闭环结果对比。
- 对漏检、过度响应和奖励函数错配的分析。

## 7. 三份独立报告的统一结构

每个人的报告单独保存为一个 Markdown 或文档文件，建议包含：

1. 作业题目与个人负责范围。
2. 个人研究问题和评价指标。
3. 数据、环境和实验设置。
4. 方法原理与关键代码位置。
5. 实验过程和可复现命令。
6. 结果表格、图表和误差分析。
7. 局限性与改进建议。
8. 个人完成内容、提交文件和运行记录。
9. 附录：配置、日志摘要和关键 JSON 结果。

每份报告至少包含一张流程图、一张核心结果表和一段失败或限制分析。三份报告共享数据说明，但方法、实验结果和结论必须分别书写。

## 8. 协作接口

成员 1 向成员 2、3 提供：

- 训练/测试 NPZ 路径。
- 特征键名和形状。
- 标签编码和类别分布。
- 数据版本说明和质量检查结果。

成员 2 向成员 3 提供：

- DQN 模型路径。
- 检测预测格式。
- 检测阈值或门控规则。
- 测试集检测指标和混淆矩阵。

成员 3 向全组提供：

- TD3 模型路径。
- 响应动作含义和范围。
- 响应及闭环 JSON。
- 成本与成功率分析。

## 9. 最终验收清单

- 三份报告都能独立说明本人的任务和结果。
- 三个人使用同一数据版本和测试集。
- 所有结果都有运行命令和配置来源。
- 代码路径、模型路径和结果文件可以互相对应。
- README 中的总流程与三份个人报告一致。
- 报告中的数字可以在 JSON、日志或图表中找到来源。
