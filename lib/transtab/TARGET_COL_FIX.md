# Target Column 大小写处理修复

## 🐛 问题描述

### 错误信息
```
KeyError: 'collisiontype'
```

### 错误位置
```python
File "transtab/dataset.py", line 331, in load_single_data
    y = df[target_col]
KeyError: 'collisiontype'
```

### 问题原因

在 `create_dataset_config` 函数中，`target_col` 被错误地转换为小写：

```python
config = {
    "bin": bin_cols,
    "cat": cat_cols,
    "num": num_cols,
    "cols": all_cols,
    "binary_indicator": binary_indicator,
    "target_col": target_col.lower() if lowercase else target_col,  # ❌ 错误
}
```

**问题流程**：
1. 用户指定 `target_col='COLLISIONTYPE'`（原始大小写）
2. `create_dataset_config` 将其转为小写：`'collisiontype'`
3. 在 `load_single_data` 中读取 CSV，列名保持原始大小写：`'COLLISIONTYPE'`
4. 尝试用小写的 `'collisiontype'` 去访问 DataFrame
5. 失败！因为 DataFrame 中的列名是 `'COLLISIONTYPE'`

## ✅ 修复方案

### 修改内容

**修改前：**
```python
config = {
    "bin": bin_cols,
    "cat": cat_cols,
    "num": num_cols,
    "cols": all_cols,
    "binary_indicator": binary_indicator,
    "target_col": target_col.lower() if lowercase else target_col,  # ❌ 错误
}
```

**修改后：**
```python
config = {
    "bin": bin_cols,
    "cat": cat_cols,
    "num": num_cols,
    "cols": all_cols,
    "binary_indicator": binary_indicator,
    "target_col": target_col,  # ✅ 保持原始大小写
}
```

## 🔍 正确的处理逻辑

### 在 `create_dataset_config` 中

```python
for col_name, col_type in col_types_dict.items():
    # 特征列名转为小写（如果 lowercase=True）
    col_name_processed = col_name.lower() if lowercase else col_name
    
    # 跳过目标列（不加入特征列表）
    if col_name == target_col:
        continue
    
    # 根据类型添加到相应列表
    if 'categorical' in col_type_str:
        cat_cols.append(col_name_processed)  # 小写
    elif 'numerical' in col_type_str:
        num_cols.append(col_name_processed)  # 小写
    # ...

# 构建配置
config = {
    "bin": bin_cols,      # 特征列，小写
    "cat": cat_cols,      # 特征列，小写
    "num": num_cols,      # 特征列，小写
    "cols": all_cols,     # 特征列，小写（不包含目标列）
    "target_col": target_col,  # 保持原始大小写！
}
```

### 在 `load_single_data` 中

```python
# 1. 读取 CSV（列名保持原始大小写）
df = pd.read_csv(filepath, index_col=None)

# 2. 获取目标列（原始大小写）
target_col = dataset_config.get("target_col", None)

# 3. 从原始 DataFrame 中提取目标列
y = df[target_col]  # ✓ 能找到，因为 target_col 是原始大小写
X = df.drop([target_col], axis=1)

# 4. 将特征列名转为小写
all_cols = [col.lower() for col in X.columns.tolist()]
X.columns = all_cols

# 5. 使用配置中的小写列名列表
if dataset_config is not None:
    if 'cat' in dataset_config:
        cat_cols = dataset_config['cat']  # 已经是小写
    if 'num' in dataset_config:
        num_cols = dataset_config['num']  # 已经是小写
```

## 📊 配置示例

### Seattle 数据集

**CSV 文件中的列名**：
```
OBJECTID, REPORTNO, COLLISIONTYPE, WEATHER, ...
```

**生成的配置**：
```python
seattle_config = {
    'bin': [],
    'cat': ['objectid', 'reportno', 'weather', ...],  # 小写，不包含 collisiontype
    'num': ['objectid', ...],                         # 小写
    'cols': ['objectid', 'reportno', 'weather', ...], # 小写，不包含 collisiontype
    'binary_indicator': ['1', 'yes', 'true', ...],
    'target_col': 'COLLISIONTYPE'  # ✓ 保持原始大小写
}
```

### Maryland 数据集

**CSV 文件中的列名**：
```
Report Number, Distance, Collision Type, Weather, ...
```

**生成的配置**：
```python
maryland_config = {
    'bin': [],
    'cat': ['report number', 'distance unit', ...],  # 小写，不包含 collision type
    'num': ['distance', 'latitude', 'longitude'],    # 小写
    'cols': ['distance', 'latitude', ..., 'weather'], # 小写，不包含 collision type
    'binary_indicator': ['1', 'yes', 'true', ...],
    'target_col': 'Collision Type'  # ✓ 保持原始大小写
}
```

## 🎯 设计原则

### 为什么目标列要保持原始大小写？

1. **DataFrame 访问需要**：从原始 CSV 读取后，需要用原始列名提取目标列
2. **标签编码之前**：目标列在标签编码前就被提取，此时列名还是原始大小写
3. **不参与特征处理**：目标列不是特征，不需要遵循特征列的小写规则

### 为什么特征列要转为小写？

1. **TransTab 要求**：TransTab 模型要求特征列名为小写
2. **统一命名**：避免大小写不一致导致的匹配问题
3. **配置复用**：小写后的特征列名在不同数据集间更容易匹配

## 📋 检查清单

修复后，确保以下行为正确：

- [ ] `target_col` 在配置中保持原始大小写
- [ ] 能够从原始 DataFrame 中成功提取目标列
- [ ] `cat`/`num`/`bin` 列表中不包含目标列
- [ ] `cols` 列表中不包含目标列
- [ ] 特征列名都是小写
- [ ] 特征列名不包含目标列

## 🧪 测试验证

```python
import transtab

# 测试 Seattle 数据集
seattle_col_types = {
    "OBJECTID": "numerical",
    "COLLISIONTYPE": "categorical",  # 目标列，原始大小写
    "WEATHER": "categorical",
}

config = transtab.create_dataset_config(
    col_types_dict=seattle_col_types,
    target_col='COLLISIONTYPE',  # 原始大小写
)

print("目标列:", config['target_col'])  # 应该是 'COLLISIONTYPE'
print("分类特征:", config['cat'])       # 应该是 ['weather']，不包含 collisiontype
print("所有特征:", config['cols'])      # 应该是 ['objectid', 'weather']

# 验证
assert config['target_col'] == 'COLLISIONTYPE'  # 保持原始大小写
assert 'collisiontype' not in config['cat']     # 不包含目标列
assert 'collisiontype' not in config['cols']    # 不包含目标列
```

## 🔗 相关文件

- `transtab/dataset.py` - 第 145 行（已修复）
- `union/mstraffic/baseline/transtab_clf_simplified.py` - 使用配置的脚本

## 📝 总结

**核心规则**：
- ✅ `target_col`: 保持原始大小写（用于从原始 DataFrame 提取）
- ✅ `cat`/`num`/`bin`/`cols`: 特征列小写，不包含目标列
- ✅ 目标列在提取后被删除，不参与特征处理

**修复后的行为**：
```
1. 读取 CSV（原始列名）
2. 用原始大小写的 target_col 提取目标列 ✓
3. 删除目标列，得到特征 DataFrame
4. 将特征列名转为小写
5. 使用小写的特征列名进行后续处理
```

---

**修复完成！** ✅ 现在可以正确处理不同大小写的目标列了。

