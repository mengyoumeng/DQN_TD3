# 诊断脚本说明

> 本文档列出 `stable_baseline3_DQNTD3/` 目录下所有诊断脚本的用途，以及相关注意事项。

---

## 一、诊断脚本清单

以下脚本文件名以 `_` 开头，属于诊断/分析脚本，**不纳入正式实验结论**：

### 1.1 泄漏检测与分析

| 脚本 | 用途 | 使用数据 |
|------|------|---------|
| `_check_auc.py` | AUC检查，验证模型性能 | ⚠️ 泄漏数据 |
| `_check_leaky_feature.py` | 泄漏特征检测 | ⚠️ 泄漏数据 |
| `_leak_localize.py` | 泄漏来源定位 | ⚠️ 泄漏数据 |
| `_find_leaky_dims.py` | 泄漏维度定位 | ⚠️ 泄漏数据 |
| `_check_overlap.py` | 数据重叠检查 | ⚠️ 泄漏数据 |

### 1.2 模型评估与诊断

| 脚本 | 用途 | 使用数据 |
|------|------|---------|
| `_eval_fixed.py` | 模型评估（使用泄漏数据） | ⚠️ 泄漏数据 |
| `_eval_dqn_with_threshold.py` | DQN阈值评估 | ⚠️ 泄漏数据 |
| `_score_auc.py` | AUC评分 | ⚠️ 泄漏数据 |
| `_sweep_dqn_threshold.py` | 阈值搜索 | ⚠️ 泄漏数据 |

### 1.3 对照实验

| 脚本 | 用途 | 使用数据 |
|------|------|---------|
| `_shuffle_train_eval_test.py` | 打乱标签对照实验 | ⚠️ 泄漏数据 |
| `_xkg_train_eval_test.py` | X_kg泄漏验证 | ⚠️ 泄漏数据 |
| `_xnum_train_eval_test.py` | X_num对比实验 | ⚠️ 泄漏数据 |
| `_train_gbdt_shuffle_y.py` | GBDT打乱标签对照 | ⚠️ 泄漏数据 |

### 1.4 数据处理

| 脚本 | 用途 | 使用数据 |
|------|------|---------|
| `_make_clean_test.py` | 数据清洗 | ⚠️ 泄漏数据 |
| `_make_consistent_scaled_npz.py` | 数据缩放 | ⚠️ 泄漏数据 |
| `_scale_npz_pair.py` | 数据缩放 | ⚠️ 泄漏数据 |

### 1.5 其他诊断

| 脚本 | 用途 | 使用数据 |
|------|------|---------|
| `_check_npz.py` | NPZ文件检查 | - |
| `_check_hgb_flip.py` | HGB模型检查 | - |
| `_check_shift.py` | 分布偏移检查 | - |
| `_check_xkg_linear_sep.py` | X_kg线性分离检查 | - |
| `_check_xkg_norm.py` | X_kg范数检查 | - |
| `_hgb_baseline.py` | HGB基线训练 | - |
| `_scan_csv_headers.py` | CSV头扫描 | - |
| `_triage_by_threshold.py` | 阈值分流 | - |
| `_triage_topk.py` | TopK分流 | - |

---

## 二、来源不明文件

### baseline_gbdt.pkl

**问题**：
- 此模型文件来源不明
- 代码库中未找到创建该模型的训练脚本
- 无法确认训练数据是否使用了标签泄漏版本

**处置**：
- 已归档到 `experiments/_unknown_source/baseline_gbdt.pkl`
- 不纳入正式实验结论
- 仅用于诊断分析
- 正式基线模型请使用 `test/run_baselines.cmd` 等脚本重新训练

---

## 三、归档文件说明

### 3.1 已归档的历史产物

| 原位置 | 归档位置 | 说明 |
|-------|---------|------|
| `experiments/baseline_gbdt.pkl` | `experiments/_unknown_source/` | 来源不明模型 |
| `experiments/benign_sc/` | `experiments/_archived_scenarios/` | 历史场景产物 |
| `data/processed/*_transe.npz` | `data/_archive/leaky_data/` | 泄漏版数据文件 |
| `data/processed/cic_features*.npz` | `data/_archive/legacy_data/` | Legacy数据文件 |
| `data/processed/cic_test.npz` | `data/_archive/legacy_data/` | Legacy数据文件 |
| `data/processed/cic_train.npz` | `data/_archive/legacy_data/` | Legacy数据文件 |

### 3.2 保留的正式数据

| 文件 | 说明 |
|------|------|
| `data/processed/*_transe_nolabel.npz` | 正式无泄漏数据 |
| `data/processed/toy_features.npz` | 测试用玩具数据 |
| `data/processed/ids2018_sb3/` | IDS2018数据 |

---

## 三、使用限制

### 3.1 禁止用于正式实验

以上诊断脚本和 `baseline_gbdt.pkl` 模型：

- ❌ 不得用于正式作业实验结果
- ❌ 不得用于正式模型对比
- ❌ 不得用于性能评估报告

### 3.2 允许用途

- ✅ 问题诊断与调试
- ✅ 数据验证
- ✅ 泄漏分析
- ✅ 算法原理验证

---

## 四、正式实验入口

如需正式实验，请使用以下入口：

| 用途 | 入口脚本 | 数据版本 |
|------|---------|---------|
| 数据准备 | `test/run_prepare_kg_nolabel.cmd` | 无泄漏 |
| LR基线 | `test/run_baselines.cmd` | 无泄漏 |
| HGB基线 | `test/run_hgb_baseline.cmd` | 无泄漏 |
| 决策树基线 | `test/run_tree_baselines.cmd` | 无泄漏 |
| DQN+TD3训练 | `rl_ids_sb3/src/train/train_all.py` | 无泄漏 |

---

## 五、相关文档

- `DATA_VERSION.md` — 数据版本与标签泄漏说明
- `docs/ARCHITECTURE.md` — 系统架构详细说明

---

*最后更新：2026-04-14*

