# TransTab Dataset Configuration 简化指南

## 概述

从 TransTab 改进版本开始，我们提供了简化的配置生成工具，让您无需手动编写冗长的 `dataset_config` 字典。

## 🎯 问题场景

**之前的做法**（繁琐）：
```python
dataset_config = {
    './data/MyDataset': {
        'bin': [],
        'cat': ['feature1', 'feature2', 'feature3', ...],  # 需要手动列举所有分类特征
        'num': ['feature10', 'feature11', ...],            # 需要手动列举所有数值特征
        "cols": ['feature1', 'feature2', ...],             # 需要再次列举所有特征
        "binary_indicator": ["1", "yes", "true", "positive", "t", "y"],
        "mask_path": "./data/MyDataset/mask.pt",
        "target_col": "label",
    }
}
```

**现在的做法**（简洁）：
```python
import transtab

# 定义列类型
col_types = {
    'feature1': 'categorical',
    'feature2': 'categorical',
    'feature10': 'numerical',
    # ...
}

# 一行代码生成配置
config = transtab.create_dataset_config(
    col_types,
    target_col='label',
    mask_path='./data/MyDataset/mask.pt'
)
```

## 🚀 核心功能

### 1. `create_dataset_config()` - 单数据集配置

为单个数据集创建配置。

**函数签名**:
```python
transtab.create_dataset_config(
    col_types_dict,          # 列类型字典
    target_col,              # 目标列名
    mask_path=None,          # mask文件路径（可选）
    binary_indicator=None,   # 二值指示符（可选）
    lowercase=True           # 是否转小写（默认True）
)
```

**参数说明**:
- `col_types_dict`: 字典，键为列名，值为类型
  - 类型可以是字符串: `'categorical'`, `'numerical'`, `'binary'`
  - 也可以是 `rllm.types.ColType` 对象
- `target_col`: 目标列名（不会被包含在特征列表中）
- `mask_path`: 预定义数据划分的路径
- `binary_indicator`: 二值特征的正类指示符列表
- `lowercase`: 是否将所有列名转为小写（TransTab 需要小写）

**示例 1: 基本使用**
```python
import transtab

col_types = {
    "Age": "numerical",
    "Gender": "binary",
    "City": "categorical",
    "Income": "numerical",
    "Label": "categorical"
}

config = transtab.create_dataset_config(
    col_types,
    target_col="Label",
    mask_path="./data/mask.pt"
)

# 生成的 config:
# {
#     'bin': ['gender'],
#     'cat': ['city'],
#     'num': ['age', 'income'],
#     'cols': ['gender', 'age', 'income', 'city'],
#     'binary_indicator': ["1", "yes", "true", "positive", "t", "y"],
#     'target_col': 'label',
#     'mask_path': './data/mask.pt'
# }
```

**示例 2: 与 rllm.types.ColType 一起使用**
```python
from rllm.types import ColType
import transtab

col_types = {
    "Age": ColType.NUMERICAL,
    "City": ColType.CATEGORICAL,
    "Label": ColType.CATEGORICAL
}

config = transtab.create_dataset_config(
    col_types,
    target_col="Label"
)
```

### 2. `create_multi_dataset_config()` - 多数据集配置

为多个数据集批量创建配置。

**函数签名**:
```python
transtab.create_multi_dataset_config(
    datasets_info,    # 数据集信息字典
    lowercase=True    # 是否转小写
)
```

**参数说明**:
- `datasets_info`: 嵌套字典，外层键为数据集路径，值为包含以下键的字典：
  - `'col_types'`: 列类型字典
  - `'target_col'`: 目标列名
  - `'mask_path'`: mask文件路径（可选）
  - `'binary_indicator'`: 二值指示符（可选）

**示例: 多数据集配置**
```python
import transtab

datasets_info = {
    './data/dataset1': {
        'col_types': {
            'Age': 'numerical',
            'City': 'categorical',
            'Label': 'categorical'
        },
        'target_col': 'Label',
        'mask_path': './data/dataset1/mask.pt'
    },
    './data/dataset2': {
        'col_types': {
            'Income': 'numerical',
            'Country': 'categorical',
            'Target': 'categorical'
        },
        'target_col': 'Target',
        'mask_path': './data/dataset2/mask.pt'
    }
}

config = transtab.create_multi_dataset_config(datasets_info)

# 生成的 config:
# {
#     './data/dataset1': { 'bin': [], 'cat': ['city'], 'num': ['age'], ... },
#     './data/dataset2': { 'bin': [], 'cat': ['country'], 'num': ['income'], ... }
# }
```

## 📝 完整工作流示例

### 场景：MSTraffic 数据集迁移学习

```python
import transtab
import numpy as np
from sklearn.metrics import accuracy_score

# ==================== 步骤 1: 定义列类型 ====================

maryland_col_types = {
    "Report Number": "categorical",
    "Distance": "numerical",
    "Latitude": "numerical",
    "Longitude": "numerical",
    "Weather": "categorical",
    "Collision Type": "categorical",
    # ... 更多特征
}

seattle_col_types = {
    "OBJECTID": "numerical",
    "REPORTNO": "categorical",
    "WEATHER": "categorical",
    "COLLISIONTYPE": "categorical",
    # ... 更多特征
}

# ==================== 步骤 2: 生成配置 ====================

dataset_config = transtab.create_multi_dataset_config({
    './data/MSTraffic/T1': {
        'col_types': maryland_col_types,
        'target_col': 'Collision Type',
        'mask_path': './data/MSTraffic/T1/mask.pt',
    },
    './data/MSTraffic/T2': {
        'col_types': seattle_col_types,
        'target_col': 'COLLISIONTYPE',
        'mask_path': './data/MSTraffic/T2/mask.pt',
    }
})

# ==================== 步骤 3: 加载数据 ====================

# 加载 Seattle 数据（预训练）
allset1, trainset1, valset1, testset1, cat_cols1, num_cols1, bin_cols1 = \
    transtab.load_data(
        ['./data/MSTraffic/T2'],
        dataset_config=dataset_config,
        filename='Seattle.csv'
    )

# ==================== 步骤 4: 构建和训练模型 ====================

model = transtab.build_classifier(
    categorical_columns=cat_cols1,
    numerical_columns=num_cols1,
    binary_columns=bin_cols1,
    num_class=9,
    num_layer=4
)

transtab.train(model, trainset1, valset1, num_epoch=10)

# ==================== 步骤 5: 迁移学习 ====================

# 加载 Maryland 数据（目标任务）
allset2, trainset2, valset2, testset2, cat_cols2, num_cols2, bin_cols2 = \
    transtab.load_data(
        ['./data/MSTraffic/T1'],
        dataset_config=dataset_config,
        filename='Maryland.csv'
    )

# 更新模型以适应新数据集
model.update({
    'cat': cat_cols2,
    'num': num_cols2,
    'bin': bin_cols2,
    'num_class': 9
})

# 在目标任务上微调
transtab.train(model, trainset2, valset2, num_epoch=25)

# ==================== 步骤 6: 评估 ====================

x_test, y_test = testset2[0]
ypred = transtab.predict(model, x_test, y_test)
accuracy = accuracy_score(y_test, np.argmax(ypred, axis=1))
print(f'Accuracy: {accuracy:.4f}')
```

## 🔄 与现有代码集成

### 从 rllm 数据集定义中提取

如果你已经在 `rllm` 框架中定义了数据集（如 `mstraffic_datasets.py`），可以轻松复用：

```python
# 在 mstraffic_datasets.py 中
from rllm.types import ColType

maryland_col_types = {
    "Report Number": ColType.CATEGORICAL,
    "Distance": ColType.NUMERICAL,
    "Collision Type": ColType.CATEGORICAL,
    # ...
}

# 在 transtab_clf.py 中
import transtab
from datasets.mstraffic_datasets import maryland_col_types, seattle_col_types

# 直接使用 ColType 对象
config = transtab.create_multi_dataset_config({
    './data/MSTraffic/T1': {
        'col_types': maryland_col_types,  # 直接使用 rllm 的定义！
        'target_col': 'Collision Type',
        'mask_path': './data/MSTraffic/T1/mask.pt',
    }
})
```

## ⚙️ 高级特性

### 自定义二值指示符

```python
config = transtab.create_dataset_config(
    col_types,
    target_col="Label",
    binary_indicator=["yes", "no", "1", "0", "true", "false"]
)
```

### 禁用小写转换

```python
config = transtab.create_dataset_config(
    col_types,
    target_col="Label",
    lowercase=False  # 保持原始大小写
)
```

### 无 mask 的情况

```python
# 不提供 mask_path，将使用随机划分
config = transtab.create_dataset_config(
    col_types,
    target_col="Label"
    # 不指定 mask_path
)
```

## 📊 类型识别规则

函数会自动识别列类型：

| 输入类型字符串 | 识别为 | 说明 |
|--------------|--------|------|
| `'numerical'`, `'num'`, `ColType.NUMERICAL` | 数值列 | 大小写不敏感 |
| `'categorical'`, `'cat'`, `ColType.CATEGORICAL` | 分类列 | 大小写不敏感 |
| `'binary'`, `'bin'` | 二值列 | 大小写不敏感 |
| 其他 | 分类列 | 默认当作分类 |

## ✅ 最佳实践

1. **保持列类型定义与数据集类定义同步**
   - 如果使用 rllm，在同一个地方维护 `col_types`
   - 避免重复定义

2. **使用小写列名**
   - TransTab 内部需要小写列名
   - 使用 `lowercase=True`（默认）自动转换

3. **复用配置**
   - 将配置保存为独立文件（如 `config.py`）
   - 在多个实验脚本中导入使用

4. **版本控制**
   - 将列类型定义纳入版本控制
   - 确保实验可重现

## 🆚 对比

| 特性 | 旧方法 | 新方法 |
|-----|--------|--------|
| 代码行数 | ~50行 | ~10行 |
| 列名重复定义 | 3次（bin/cat/num, cols, 手动列举） | 1次 |
| 与 rllm 集成 | 需要手动转换 | 自动识别 ColType |
| 小写转换 | 手动处理 | 自动处理 |
| 维护成本 | 高（多处同步） | 低（单一来源） |

## 🔗 相关文档

- [DATASET_MODIFICATIONS.md](./DATASET_MODIFICATIONS.md) - dataset.py 修改详情
- [TransTab 官方文档](https://github.com/RyanWangZf/transtab)

## 💡 提示

- 如果遇到列名不匹配的问题，检查 CSV 文件中的实际列名
- mask.pt 文件应该包含 `train_mask`, `val_mask`, `test_mask` 三个布尔张量
- 所有列名会自动转为小写（除非设置 `lowercase=False`）

