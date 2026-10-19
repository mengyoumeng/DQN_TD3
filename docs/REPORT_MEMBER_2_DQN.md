# 成员 2 独立完成报告：TD3 连续响应控制

## 1. 报告定位

本报告围绕 TD3 连续动作强化学习展开。数据和检测结果只作为环境输入，核心工作是设计响应环境、训练原始 TD3、分析动作行为和成本权衡。

## 2. 核心问题

- 如何把阈值调整和响应强度建模为连续动作？
- TD3 是否学到正常样本和攻击样本之间的响应差异？
- 响应奖励、动作噪声和成本系数如何影响策略？
- 平均奖励、FN Cost 和攻击缓解率是否一致？

## 3. 负责代码

- `rl_ids_sb3/src/envs/response_env.py`
- `rl_ids_sb3/src/train/train_td3.py`
- `rl_ids_sb3/src/train/train_td3_binary.py`
- `rl_ids_sb3/src/eval/eval_td3_response.py`

## 4. 报告内容

### 4.1 连续 MDP

| 要素 | 填写内容 |
|---|---|
| 状态 | 数值特征、KG 特征及环境状态 |
| 动作 | `[delta_threshold, response_intensity]` |
| 动作范围 | 通常为 [-1, 1] |
| 正常奖励 | 降低过度响应和响应成本 |
| 攻击奖励 | 提高有效响应强度 |
| 终止 | 单样本响应或环境终止 |

结合 `ResponseEnv.step()` 解释任务奖励、成本项和总奖励。

### 4.2 TD3 配置

| 参数 | 值 |
|---|---:|
| 随机种子 | 待填写 |
| 训练步数 | 待填写 |
| 学习率 | 待填写 |
| Replay buffer | 待填写 |
| Batch size | 待填写 |
| Gamma | 待填写 |
| Tau | 待填写 |
| Action noise sigma | 待填写 |
| Attack reward weight | 待填写 |
| Cost coefficient | 待填写 |

### 4.3 响应结果

| 模型 | Avg Reward | Avg Intensity | Attack Intensity | FP Cost | FN Cost | Total Cost |
|---|---:|---:|---:|---:|---:|---:|
| TD3 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

补充正常样本与攻击样本的强度分布图。

### 4.4 运行记录

```powershell
cd rl_ids_sb3
python -m src.train.train_td3_binary --data_dir data/processed --train_file train_binary.npz --out_dir runs/member2_td3
python -m src.eval.eval_td3_response --model runs/member2_td3/td3_resp_binary.zip --data data/processed/test_binary.npz --out runs/member2_td3/metrics.json
```

记录模型路径、配置、动作范围、运行时间和 JSON 指标。

## 5. 个人结论

结论必须说明 TD3 是否形成类别相关的响应行为，成本系数和攻击奖励权重如何影响策略，以及原始 TD3 适合作为什么基准。

## 6. 个人交付物

- 原始 TD3 模型；
- 响应评估 JSON；
- 响应强度和成本图；
- 超参数表；
- 本独立报告。
