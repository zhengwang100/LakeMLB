# Dataset.py 修改说明

## 修改概述

本文档记录了对 `transtab/dataset.py` 的改进，这些修改使得该模块能够更灵活地处理本地CSV文件和预定义的数据划分。

## 主要改进

### 1. **灵活的文件名和编码支持**
- **改进**: 添加了 `filename` 和 `encoding` 参数
- **目的**: 
  - 允许用户指定任意CSV文件名（不再硬编码为 `data_processed.csv`）
  - 支持不同编码格式的CSV文件（如 `utf-8`, `gbk`, `latin1` 等）
- **函数签名变化**:
  ```python
  # 之前
  def load_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123)
  
  # 之后
  def load_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123, 
                filename=None, encoding: Optional[str] = None)
  ```

### 2. **灵活的目标列配置**
- **改进**: 通过 `dataset_config["target_col"]` 指定目标列名
- **目的**: 不再硬编码目标列为 `target_label`，支持任意列名
- **使用示例**:
  ```python
  dataset_config = {
      "target_col": "label",  # 可以是任意列名
      "num": [...],
      "cat": [...],
      "bin": [...]
  }
  ```

### 3. **无标签场景支持（自监督学习）**
- **改进**: 当 `target_col` 未指定或为 `None` 时，支持无标签数据加载
- **目的**: 支持对比学习等自监督学习场景
- **特性**:
  - `y` 会被设置为 `None`
  - `X` 包含整个数据框
  - 数据划分时正确处理 `y=None` 的情况
  - 打印输出中不显示 `pos_rate`

### 4. **预定义数据划分支持（mask_path）**
- **改进**: 添加了从 `.pt` 文件加载预定义训练/验证/测试划分的功能
- **目的**: 
  - 确保实验的可重复性
  - 支持自定义的数据划分策略
  - 与其他模型保持一致的数据划分
- **使用示例**:
  ```python
  dataset_config = {
      "mask_path": "path/to/mask.pt",  # torch.save({'train_mask': ..., 'val_mask': ..., 'test_mask': ...})
      "target_col": "label",
      # ... other configs
  }
  ```
- **mask.pt 格式**:
  ```python
  {
      'train_mask': torch.Tensor([True, False, ...]),  # 布尔掩码
      'val_mask': torch.Tensor([False, True, ...]),
      'test_mask': torch.Tensor([False, False, ...])
  }
  ```

### 5. **更健壮的缺失值处理**
- **改进**: 改进了分类列的 mode 填充逻辑
- **目的**: 处理某些列可能没有 mode 的边缘情况
- **代码变化**:
  ```python
  # 之前
  for col in cat_cols:
      X[col].fillna(X[col].mode()[0], inplace=True)
  
  # 之后
  for col in cat_cols:
      mode_series = X[col].mode()
      if not mode_series.empty:
          X[col].fillna(mode_series[0], inplace=True)
      else:
          X[col].fillna("Unknown", inplace=True)  # 使用默认值
  ```

### 6. **改进的目录检查**
- **改进**: 使用 `os.path.isdir()` 替代 `os.path.exists()`
- **目的**: 更准确地判断是否为本地目录
- **好处**: 避免与文件名混淆

## 使用示例

### 示例 1: 基本使用（有监督学习）
```python
dataset_config = {
    "target_col": "label",
    "num": ["age", "income"],
    "cat": ["city", "occupation"],
    "bin": ["gender"]
}

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
    dataname="./data/my_dataset",
    dataset_config=dataset_config,
    filename="my_data.csv",
    encoding="utf-8"
)
```

### 示例 2: 使用预定义数据划分
```python
dataset_config = {
    "target_col": "target",
    "mask_path": "./data/my_dataset/mask.pt",
    "num": ["feature1", "feature2"],
    "cat": ["category1"]
}

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
    dataname="./data/my_dataset",
    dataset_config=dataset_config,
    filename="dataset.csv"
)
```

### 示例 3: 无标签场景（自监督学习）
```python
dataset_config = {
    # 不指定 target_col
    "num": ["f1", "f2", "f3"],
    "cat": ["c1", "c2"]
}

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
    dataname="./data/unsupervised_dataset",
    dataset_config=dataset_config,
    filename="data.csv"
)
# 此时 allset = (X, None), trainset = (train_X, None), 等等
```

## 向后兼容性

这些修改保持了向后兼容性：
- OpenML数据集加载仍然正常工作
- 不使用新参数时，行为与原版本相同
- 原有的 `data_split_idx` 功能仍然可用

## 代码质量改进

- 移除了未使用的 `pdb` 导入
- 修复了代码风格问题（单行多语句等）
- 改进了代码可读性和一致性
- 添加了更详细的文档字符串

## 总结

这些修改使得 `transtab` 的数据加载模块更加灵活和强大，同时保持了简洁性和易用性。主要优势包括：

✅ 支持任意CSV文件名和编码  
✅ 灵活的目标列配置  
✅ 支持自监督学习场景  
✅ 支持预定义数据划分  
✅ 更健壮的错误处理  
✅ 保持向后兼容性  



