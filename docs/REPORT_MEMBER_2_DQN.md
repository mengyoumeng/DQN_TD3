# 成员 2 独立完成报告：DQN 离散攻击检测

## 1. 报告定位

本报告负责将攻击检测建模为离散动作强化学习问题，完成 DQN 训练、测试和监督学习基线对比，并分析误报与漏报。

## 2. 需要回答的问题

- DQN 环境中的状态、动作、奖励和终止条件如何定义？
- DQN 是否能区分正常流量和攻击流量？
- DQN 与 LR、HGB、ExtraTrees 相比表现如何？
- 哪些指标最能反映攻击检测质量？
- 模型在哪些样本上容易误报或漏报？

## 3. 实验范围

代码范围：

- `rl_ids_sb3/src/envs/detection_env.py`
- `rl_ids_sb3/src/train/train_dqn.py`
- `rl_ids_sb3/src/train/train_all.py`
- `rl_ids_sb3/src/eval/evaluate.py`
- `rebuild_data/run_baselines.py`

输入由成员 1 提供，必须记录数据文件名、特征维度和标签编码。

## 4. 报告章节

### 4.1 MDP 建模

填写：

| 要素 | 本项目定义 |
|---|---|
| 状态 | 数值特征与 KG 特征拼接向量 |
| 动作 | 离散类别动作 |
| 奖励 | 根据预测类别与真实标签给出检测奖励 |
| 终止 | 单样本交互结束或环境规定的终止条件 |
| 采样 | 项目环境中的均衡或随机采样方式 |

结合代码说明每个定义如何实现。

### 4.2 DQN 配置

| 参数 | 值 |
|---|---:|
| 随机种子 | 待填写 |
| 总训练步数 | 待填写 |
| 学习率 | 待填写 |
| Batch size | 待填写 |
| Replay buffer | 待填写 |
| Gamma | 待填写 |
| Target update interval | 待填写 |
| Exploration 参数 | 待填写 |

### 4.3 统一测试结果

| 模型 | Accuracy | Precision macro | Recall macro | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| LR | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| HGB | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ExtraTrees | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| DQN | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

必须注明所有模型使用相同测试集。

### 4.4 混淆矩阵与错误分析

填写：

| 真实/预测 | Benign | Attack |
|---|---:|---:|
| Benign | 待填写 | 待填写 |
| Attack | 待填写 | 待填写 |

重点解释：

- 误报对正常流量的影响。
- 漏报对攻击检测的影响。
- Accuracy 与攻击类 Recall 不一致时应如何解释。
- 数据不平衡是否影响结论。

### 4.5 可复现命令

```powershell
cd rl_ids_sb3
python -m src.train.train_all --data_dir data/processed --train_file train_binary.npz --test_file test_binary.npz --out_dir runs/member2_dqn
```

报告中附上配置文件、模型路径、指标 JSON 和运行日志摘要。

## 5. 个人结论

结论应明确回答：DQN 的优势指标是什么、主要错误类型是什么、与监督学习基线相比是否值得采用、下一步需要改进奖励或采样策略的哪一部分。

## 6. 个人交付物

- 本报告。
- DQN 模型文件路径或模型生成说明。
- DQN 测试指标 JSON。
- 基线对比表。
- 混淆矩阵和错误分析图表。
- 运行命令与配置副本。
