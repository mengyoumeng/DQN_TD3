# 成员 1 独立完成报告：DQN 离散攻击检测

## 1. 报告定位

本报告围绕 DQN 离散动作强化学习展开。数据清洗和 KG 特征只作为状态输入准备，核心工作是设计检测环境、训练 DQN、分析探索与奖励，并完成检测结果解释。

## 2. 核心问题

- 如何把正常/攻击分类转换为离散动作 MDP？
- 状态特征、动作和奖励如何影响 DQN？
- DQN 能否提高攻击类 Recall 和 Macro-F1？
- 均衡采样和探索策略会怎样影响误报、漏报？

## 3. 负责代码

- `rl_ids_sb3/src/envs/detection_env.py`
- `rl_ids_sb3/src/train/train_dqn.py`
- `rl_ids_sb3/src/train/train_all.py`
- `rl_ids_sb3/src/eval/evaluate.py`

公共输入由统一数据流程生成，建议在报告中简要记录 `X_num`、`X_kg` 和 `y` 的形状。

## 4. 报告内容

### 4.1 MDP 定义

| 要素 | 填写内容 |
|---|---|
| 状态 | 数值特征 + KG 特征拼接向量 |
| 动作 | 0=Benign，1=Attack |
| 奖励 | 正确/错误分类对应奖励 |
| 采样 | 环境定义的随机或均衡采样 |
| 终止 | 单样本交互或环境规定的终止条件 |

结合源码解释每一项如何实现。

### 4.2 DQN 配置

| 参数 | 值 |
|---|---:|
| 随机种子 | 待填写 |
| 训练步数 | 待填写 |
| 学习率 | 待填写 |
| Replay buffer | 待填写 |
| Batch size | 待填写 |
| Gamma | 待填写 |
| Target update interval | 待填写 |
| Exploration fraction | 待填写 |
| Exploration final epsilon | 待填写 |

### 4.3 结果表

| 模型 | Accuracy | Precision macro | Recall macro | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| LR | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| HGB | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ExtraTrees | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| DQN | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 4.4 误报与漏报

给出混淆矩阵，并分别解释：

- 正常流量被判为攻击的原因；
- 攻击流量被判为正常的原因；
- Accuracy 与攻击类 Recall 不一致时的含义；
- 奖励函数是否鼓励了当前行为。

### 4.5 运行记录

```powershell
cd rl_ids_sb3
python -m src.train.train_all --data_dir data/processed --train_file train_binary.npz --test_file test_binary.npz --out_dir runs/member1_dqn
```

记录配置、模型路径、metrics JSON、运行时间和随机种子。

## 5. 个人结论

结论必须直接回答 DQN 的检测能力、最重要的错误类型、奖励与采样的影响，以及后续需要改进的 RL 部分。

## 6. 个人交付物

- DQN 训练和评估结果；
- 检测指标 JSON；
- 混淆矩阵和训练曲线；
- 误报/漏报分析；
- 本独立报告。
