# 成员 3 独立完成报告：TD3-BC 离线预训练与闭环控制

## 1. 报告定位

本报告围绕离线强化学习和 TD3-BC 展开。成员 3 负责从专家或 LLM buffer 中学习初始策略，再进行在线微调，并把检测策略与响应策略连接成完整闭环。

## 2. 核心问题

- 离线行为数据能否提供比随机初始化更好的 TD3 起点？
- BC 预训练、TD3-BC 和在线微调的差异是什么？
- 微调后奖励上升时，FN Cost 是否可能恶化？
- DQN 门控与 TD3 响应连接后，闭环成功率和攻击缓解率如何变化？

## 3. 负责代码

- `rl_ids_sb3/src/offline/gen_expert_buffer.py`
- `rl_ids_sb3/src/offline/gen_llm_buffer.py`
- `rl_ids_sb3/src/offline/pretrain_td3_bc.py`
- `rl_ids_sb3/src/offline/finetune_td3_binary.py`
- `rl_ids_sb3/src/offline/finetune_td3_fix_v1.py`
- `rl_ids_sb3/src/offline/finetune_td3_fix_v2.py`
- `rl_ids_sb3/src/policy/`
- `rl_ids_sb3/src/eval/eval_closed_loop.py`

## 4. 报告内容

### 4.1 离线数据与策略初始化

说明：

- buffer 来源；
- 状态、动作、奖励和标签的键；
- 专家动作如何生成；
- buffer 样本数量和类别分布；
- BC 损失或 TD3-BC 目标；
- 预训练模型如何接入 TD3 在线训练。

### 4.2 实验对照

至少比较以下方案：

| 方案 | 初始化 | 是否在线微调 | Avg Reward | FN Cost | Total Cost |
|---|---|---|---:|---:|---:|
| 原始 TD3 | 随机 | 是 | 待填写 | 待填写 | 待填写 |
| TD3-BC pretrain | 离线 BC/TD3-BC | 否或短微调 | 待填写 | 待填写 | 待填写 |
| TD3-BC finetune | 离线预训练 | 是 | 待填写 | 待填写 | 待填写 |
| TD3-BC fix v1/v2 | 离线预训练 | 是 | 待填写 | 待填写 | 待填写 |

解释不同方案的训练目标和评估目标是否一致。

### 4.3 闭环评估

```powershell
cd rl_ids_sb3
python -m src.eval.eval_closed_loop --dqn runs/member1_dqn/dqn_detect_binary.zip --td3 runs/member3_td3_bc/td3_bc.zip --data data/processed/test_binary.npz --out runs/member3_td3_bc/closed_loop.json
```

填写：

| 指标 | DQN 检测 | DQN + 原始 TD3 | DQN + TD3-BC |
|---|---:|---:|---:|
| Detection Accuracy | 待填写 | 待填写 | 待填写 |
| Attack Recall | 待填写 | 待填写 | 待填写 |
| Response Trigger Rate | 不适用 | 待填写 | 待填写 |
| Attack Mitigation Rate | 不适用 | 待填写 | 待填写 |
| Missed Attack Rate | 待填写 | 待填写 | 待填写 |
| Total Cost | 不适用 | 待填写 | 待填写 |
| Closed-loop Success Rate | 不适用 | 待填写 | 待填写 |

### 4.4 稳定性和失败分析

至少分析：

- buffer 动作质量不足；
- 预训练策略过度模仿单一行为；
- 在线微调造成策略退化；
- 平均奖励与 FN Cost 方向不一致；
- 检测漏报导致响应无法触发；
- 正常样本被错误触发响应。

## 5. 个人结论

结论必须明确说明 TD3-BC 是否改善初始化、哪种方案最稳定、闭环是否优于单独检测，以及奖励函数和离线数据还需要怎样改进。

## 6. 个人交付物

- buffer 生成或检查记录；
- TD3-BC 预训练模型；
- 微调模型；
- 原始 TD3/TD3-BC/微调对照表；
- 闭环评估 JSON；
- 成本和成功率图表；
- 本独立报告。
