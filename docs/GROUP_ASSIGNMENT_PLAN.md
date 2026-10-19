# 强化学习小组作业：三人 RL 主线分工框架

## 1. 分工原则

本作业由三名成员分别负责三个高度独立的强化学习实验包。每个人都有自己的环境、训练入口、算法问题、评价指标和独立报告；三人的结果最后通过统一测试集和闭环流程连接起来。

整体链路为：

`公共数据状态 → 成员 1：DQN 检测 → 成员 2：TD3 连续响应 → 成员 3：TD3-BC 离线预训练/微调 + 闭环评估`

数据清洗、KG/TransE 特征只作为公共输入准备，控制在每个人报告的“实验设置”章节中，不作为任何一个人的主要工作量。

## 2. 三人职责总览

| 成员 | RL 主题 | 核心研究问题 | 主要算法 | 主要代码范围 | 独立核心产物 |
|---|---|---|---|---|---|
| 成员 1 | DQN 离散检测 | 离散动作强化学习能否完成攻击类别识别？ | DQN | `src/envs/detection_env.py`、`src/train/train_dqn.py`、`src/eval/evaluate.py` | DQN 模型、检测指标、奖励曲线、误报/漏报分析 |
| 成员 2 | TD3 连续响应 | 连续动作强化学习能否学习响应强度和阈值调整？ | TD3 | `src/envs/response_env.py`、`src/train/train_td3.py`、`src/train/train_td3_binary.py`、`src/eval/eval_td3_response.py` | TD3 模型、响应强度分布、奖励和成本分析 |
| 成员 3 | TD3-BC 与闭环控制 | 离线行为克隆预训练能否改善 TD3 初始化，并在闭环中降低漏报或响应成本？ | TD3-BC、离线 RL、在线微调 | `src/offline/`、`src/policy/`、`src/eval/eval_closed_loop.py` | 专家/LLM buffer、预训练模型、微调模型、闭环评估 |

## 3. 统一实验条件

三人必须共享以下条件：

- 使用相同的训练集、测试集和标签编码：0=Benign，1=Attack。
- 使用同一套数值特征与不含标签实体的 KG/TransE 特征。
- 默认随机种子为 0；如果比较稳定性，统一使用 0、1、2。
- 测试集只用于最终评估，不能用来选择超参数。
- 每次实验记录配置、训练步数、模型路径、日志和 JSON 指标。
- 报告中的数字必须能在结果 JSON、日志或图表中找到来源。
- 三人可以复用公共数据准备脚本，但每个人必须独立完成自己的 RL 训练和分析。

## 4. 成员 1：DQN 离散检测

### 研究任务

1. 将攻击检测定义为离散动作 MDP。
2. 分析状态向量、动作类别、奖励函数、采样模式和终止条件。
3. 训练 DQN 二分类检测器。
4. 研究训练步数、探索率、奖励设置或均衡采样对检测效果的影响。
5. 与 LR、HGB、ExtraTrees 做辅助对照，但报告重点放在 DQN。
6. 分析攻击类 Recall、漏报和误报。

### 代码与命令

- 环境：`rl_ids_sb3/src/envs/detection_env.py`
- 训练：`rl_ids_sb3/src/train/train_dqn.py`
- 入口：`rl_ids_sb3/src/train/train_all.py`
- 评估：`rl_ids_sb3/src/eval/evaluate.py`

```powershell
cd rl_ids_sb3
python -m src.train.train_all --data_dir data/processed --train_file train_binary.npz --test_file test_binary.npz --out_dir runs/member1_dqn
```

### 必须回答

- DQN 的奖励是否与攻击检测目标一致？
- Accuracy 较高时，攻击类 Recall 是否也足够高？
- 均衡采样是否改变了误报/漏报比例？
- DQN 相对监督学习基线的主要优势和不足是什么？

## 5. 成员 2：TD3 连续响应

### 研究任务

1. 将响应策略定义为连续动作 MDP。
2. 解释 `delta_threshold` 和 `response_intensity` 的含义。
3. 训练原始 TD3 响应模型。
4. 研究 `attack_reward_weight`、`cost_coefficient`、动作噪声和训练步数的影响。
5. 分析正常样本过度响应、攻击样本响应不足和奖励成本之间的关系。
6. 完成原始 TD3 的独立报告，不依赖成员 3 的离线预训练结果。

### 代码与命令

- 环境：`rl_ids_sb3/src/envs/response_env.py`
- 训练：`rl_ids_sb3/src/train/train_td3.py`、`train_td3_binary.py`
- 评估：`rl_ids_sb3/src/eval/eval_td3_response.py`

```powershell
cd rl_ids_sb3
python -m src.train.train_td3_binary --data_dir data/processed --train_file train_binary.npz --out_dir runs/member2_td3
python -m src.eval.eval_td3_response --model runs/member2_td3/td3_resp_binary.zip --data data/processed/test_binary.npz --out runs/member2_td3/metrics.json
```

### 必须回答

- TD3 是否学到不同类别的响应强度？
- 平均奖励提高是否一定代表 FN Cost 降低？
- 响应成本和攻击缓解率之间如何权衡？
- 哪个奖励参数最影响策略行为？

## 6. 成员 3：TD3-BC 离线预训练、微调与闭环

### 研究任务

1. 说明离线 RL、行为克隆和 TD3-BC 的关系。
2. 生成或检查专家 buffer、LLM buffer 和混合 buffer。
3. 完成 TD3-BC 预训练，得到可用于在线 TD3 的初始策略。
4. 比较随机初始化 TD3、BC 预训练和在线微调。
5. 分析微调后奖励、FN Cost 和总成本是否同步变化。
6. 使用成员 1 的 DQN 和成员 2/3 的 TD3 模型完成闭环评估。
7. 解释检测门控、响应触发、攻击缓解和漏报之间的关系。

### 代码与命令

- buffer：`rl_ids_sb3/src/offline/gen_expert_buffer.py`、`gen_llm_buffer.py`
- 预训练：`rl_ids_sb3/src/offline/pretrain_td3_bc.py`
- 微调：`finetune_td3_binary.py`、`finetune_td3_fix_v1.py`、`finetune_td3_fix_v2.py`
- 策略：`rl_ids_sb3/src/policy/`
- 闭环：`rl_ids_sb3/src/eval/eval_closed_loop.py`

```powershell
cd rl_ids_sb3
python -m src.offline.pretrain_td3_bc --data_dir data/processed --out_dir runs/member3_td3_bc
python -m src.eval.eval_closed_loop --dqn runs/member1_dqn/dqn_detect_binary.zip --td3 runs/member3_td3_bc/td3_bc.zip --data data/processed/test_binary.npz --out runs/member3_td3_bc/closed_loop.json
```

实际参数以脚本的 `--help` 和配置文件为准。若某个 buffer 不存在，应在报告中记录生成方式和样本数量。

### 必须回答

- BC 预训练是否让 TD3 更快或更稳定？
- 离线数据的动作质量如何影响策略？
- 在线微调是否出现奖励上升但 FN Cost 恶化？
- 闭环指标是否比单独检测或单独响应更能反映系统效果？

## 7. 三份独立报告的写作边界

### 成员 1 报告

题目建议：**基于 DQN 的网络流量离散攻击检测实验**

报告重点：

- DQN 环境和奖励；
- 探索与采样；
- 训练曲线；
- 检测指标；
- 误报/漏报；
- DQN 与基线对照。

### 成员 2 报告

题目建议：**基于 TD3 的网络流量连续响应控制实验**

报告重点：

- 连续动作环境；
- 响应奖励和成本；
- TD3 超参数；
- 响应强度分布；
- 正常/攻击样本策略差异；
- 成本权衡。

### 成员 3 报告

题目建议：**TD3-BC 离线预训练及检测响应闭环实验**

报告重点：

- buffer 和离线数据；
- BC/TD3-BC 目标；
- 预训练、微调和随机初始化对比；
- DQN 门控与 TD3 响应；
- 闭环成功率、攻击缓解率、漏报率和总成本。

三份报告都应包含：摘要、问题定义、环境设计、算法原理、实验设置、结果、失败分析、局限性和个人完成内容。

## 8. 交接接口

成员 1 提供：

- DQN 模型路径；
- 预测标签格式；
- 检测指标和混淆矩阵；
- 测试集版本。

成员 2 提供：

- 原始 TD3 模型路径；
- 动作维度和范围；
- 响应指标与成本分解；
- 可供成员 3 对比的基线策略。

成员 3 提供：

- buffer 统计；
- TD3-BC 预训练模型和微调模型；
- 闭环 JSON；
- 原始 TD3、BC 预训练和微调的统一比较表。

## 9. 最终检查

- 三人的主要代码都直接涉及 RL 环境、策略、训练或评估。
- 每个人都有独立模型或独立 RL 实验结果。
- 三份报告使用同一测试集，但研究问题互不重复。
- 成员 3 的离线预训练结果能与成员 2 的原始 TD3 对齐比较。
- 最终闭环只作为成员 3 的综合实验，不改变成员 1、2 的独立结论。
