# 超参数搜索空间缩减记录

本文档记录 Tree 与 TNNS 分类 baseline 在 2026-07-21 进行的搜索空间缩减。缩减目标是在保留主要模型容量和学习率差异的同时，降低新增消融表的实验耗时。

最终重复实验次数、最优超参数选择标准、模型权重保存、日志记录和 early stopping 设置保持不变。

## 1. Tree 原搜索空间

```text
XGBoost:
  n_estimators: {300, 500, 1000}
  max_depth: {4, 6, 8}
  learning_rate: {0.01, 0.03, 0.05}
  subsample: {0.8, 0.9}
  colsample_bytree: {0.8, 0.9}
  combinations: 108

CatBoost:
  iterations: {300, 500, 1000}
  depth: {4, 6, 8}
  learning_rate: {0.01, 0.03, 0.05}
  subsample: {0.8, 0.9}
  rsm: {0.8, 0.9}
  combinations: 108

LightGBM:
  num_boost_round: {300, 500, 1000}
  num_leaves: {31, 63, 127}
  learning_rate: {0.01, 0.03, 0.05}
  feature_fraction: {0.8, 0.9}
  bagging_fraction: {0.8, 0.9}
  combinations: 108
```

Tree 原搜索预算为 `108 x 3 = 324` 组。

## 2. Tree 缩减后搜索空间

```text
XGBoost:
  n_estimators: {300, 1000}
  max_depth: {6, 8}
  learning_rate: {0.01, 0.05}
  subsample: {0.9}
  colsample_bytree: {0.8}
  combinations: 8

CatBoost:
  iterations: {300, 1000}
  depth: {6, 8}
  learning_rate: {0.01, 0.05}
  subsample: {0.9}
  rsm: {0.8}
  combinations: 8

LightGBM:
  num_boost_round: {300, 1000}
  num_leaves: {63, 127}
  learning_rate: {0.01, 0.05}
  feature_fraction: {0.8}
  bagging_fraction: {0.9}
  combinations: 8
```

Tree 新搜索预算为 `8 x 3 = 24` 组，相比原配置减少约 `92.6%`。三种模型的组合数量相同，early stopping 仍为 50 个 validation rounds。

## 3. TNNS 原搜索空间

以下空间由 FTTransformer、TabTransformer、ExcelFormer、SAINT 和 TromptNet 共享：

```text
hidden_dim: {32, 64, 128}
layers: {2, 3, 4}
learning_rate: {1e-3, 5e-4, 1e-4}
weight_decay: {1e-4, 5e-4, 1e-3}
batch_size: {256}
combinations per model: 81
```

TNNS 原搜索预算为 `81 x 5 = 405` 组。

## 4. TNNS 缩减后搜索空间

```text
hidden_dim: {64, 128}
layers: {2, 3, 4}
learning_rate: {1e-3, 5e-4, 1e-4}
weight_decay: {5e-4}
batch_size: {256}
combinations per model: 18
```

TNNS 新搜索预算为 `18 x 5 = 90` 组，相比原配置减少约 `77.8%`。Grid search 最大 epoch 仍为 500，early stopping patience 仍为 10。

## 5. 总体变化

```text
Tree + TNNS 原搜索预算: 324 + 405 = 729
Tree + TNNS 新搜索预算:  24 +  90 = 114
总体减少: 约 84.4%
```

缩减后，Tree 仍使用 validation accuracy 作为主要选择指标，并使用 validation F1 作为并列情况下的次级指标；TNNS 仍按 validation accuracy 选择最优配置。最优配置会重新训练并保存权重，随后执行 10 次独立重复实验。
