# 成员 3 独立完成报告：TD3 连续响应与闭环评估

## 1. 报告定位

本报告负责将攻击响应建模为连续动作强化学习问题，并把 DQN 检测输出接入 TD3，评估完整的检测与响应闭环。

## 2. 需要回答的问题

- TD3 的状态、动作和奖励如何定义？
- 阈值变化与响应强度分别代表什么？
- 响应强度增加是否同时带来更高的响应成本？
- TD3-BC 预训练和在线微调是否稳定？
- DQN+TD3 闭环能否降低漏报或提高攻击缓解率？

## 3. 实验范围

代码范围：

- `rl_ids_sb3/src/envs/response_env.py`
- `rl_ids_sb3/src/train/train_td3.py`
- `rl_ids_sb3/src/train/train_td3_binary.py`
- `rl_ids_sb3/src/offline/`
- `rl_ids_sb3/src/eval/eval_td3_response.py`
- `rl_ids_sb3/src/eval/eval_closed_loop.py`

成员 2 提供 DQN 模型路径和检测输出格式。

## 4. 报告章节

### 4.1 响应 MDP

| 要素 | 本项目定义 |
|---|---|
| 状态 | 数值特征、KG 特征及响应环境输入 |
| 动作 | `[delta_threshold, response_intensity]` |
| 动作范围 | 通常为 [-1, 1] |
| 正常样本目标 | 控制过度响应和响应成本 |
| 攻击样本目标 | 提高响应强度并降低漏报代价 |
| 终止 | 单样本响应完成或环境规定的终止条件 |

说明环境中的成本项、正常样本奖励和攻击样本奖励如何共同构成总奖励。

### 4.2 TD3 配置

| 参数 | 原始 TD3 | TD3-BC/微调 |
|---|---:|---:|
| 总训练步数 | 待填写 | 待填写 |
| 学习率 | 待填写 | 待填写 |
| Batch size | 待填写 | 待填写 |
| Replay buffer | 待填写 | 待填写 |
| Action noise | 待填写 | 待填写 |
| Attack reward weight | 待填写 | 待填写 |
| Cost coefficient | 待填写 | 待填写 |

### 4.3 响应结果

| 模型 | Avg Reward | Avg Intensity | Attack Intensity | FP Cost | FN Cost | Total Cost |
|---|---:|---:|---:|---:|---:|---:|
| TD3 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| TD3-BC pretrain | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| TD3-BC finetune | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

解释奖励较高但 FN Cost 仍较高的情况，避免只依据单一指标判断模型优劣。

### 4.4 闭环评估

```powershell
cd rl_ids_sb3
python -m src.eval.eval_closed_loop --dqn runs/member2_dqn/dqn_detect_binary.zip --td3 runs/member3_td3/td3_resp_binary.zip --data data/processed/test_binary.npz --out runs/member3_td3/closed_loop.json
```

填写：

| 指标 | DQN 单独 | DQN+TD3 闭环 |
|---|---:|---:|
| Detection Accuracy | 待填写 | 待填写 |
| Attack Recall | 待填写 | 待填写 |
| Response Trigger Rate | 不适用 | 待填写 |
| Attack Mitigation Rate | 不适用 | 待填写 |
| Missed Attack Rate | 待填写 | 待填写 |
| Total Cost | 不适用 | 待填写 |
| Closed-loop Success Rate | 不适用 | 待填写 |

### 4.5 错误与稳定性分析

至少讨论：

- 检测漏报导致响应未触发的情况。
- 正常样本被触发响应的情况。
- 响应强度过低或过高的情况。
- 奖励函数与最终评估成本不一致的情况。
- 不同随机种子或训练步数下的变化。

## 5. 个人结论

结论应明确回答：TD3 学到的动作是否有实际区分度、响应成本是否可接受、闭环是否改善关键指标、TD3-BC 是否值得保留，以及下一步应优先调整奖励函数还是训练流程。

## 6. 个人交付物

- 本报告。
- TD3 模型路径或生成说明。
- 响应评估 JSON。
- 闭环评估 JSON。
- 成本分解表和响应强度图。
- 运行命令、配置副本和日志摘要。
