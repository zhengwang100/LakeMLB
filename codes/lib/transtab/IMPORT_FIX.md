# TransTab 循环导入修复

## 🐛 问题描述

### 错误信息
```
ImportError: cannot import name 'constants' from partially initialized module 'transtab' 
(most likely due to a circular import)
```

### 错误原因
在 `transtab/transtab.py` 文件中使用了绝对导入：
```python
from transtab import constants
from transtab.modeling_transtab import TransTabClassifier
# ... 等等
```

当 Python 导入 `transtab` 包时：
1. 首先执行 `transtab/__init__.py`
2. `__init__.py` 中有 `from .transtab import *`
3. 这会执行 `transtab/transtab.py`
4. `transtab.py` 中又尝试 `from transtab import constants`
5. 但此时 `transtab` 模块还没有完全初始化
6. 导致循环导入错误

## ✅ 修复方案

### 修改内容
将 `transtab/transtab.py` 中的所有绝对导入改为相对导入：

**修改前：**
```python
from transtab import constants
from transtab.modeling_transtab import TransTabClassifier, ...
from transtab.dataset import load_data, ...
from transtab.evaluator import predict, evaluate
from transtab.trainer import Trainer
from transtab.trainer_utils import TransTabCollatorForCL, random_seed
```

**修改后：**
```python
from . import constants
from .modeling_transtab import TransTabClassifier, ...
from .dataset import load_data, ...
from .evaluator import predict, evaluate
from .trainer import Trainer
from .trainer_utils import TransTabCollatorForCL, random_seed
```

## 📖 相对导入说明

### 为什么使用相对导入？

在包内部的模块之间导入时，应该使用相对导入：

1. **避免循环导入** - 相对导入不会触发包的重新初始化
2. **更清晰的意图** - 明确表示导入的是同一个包内的模块
3. **更好的可移植性** - 如果包名改变，相对导入不需要修改

### 相对导入语法

```python
# 导入同级模块
from . import module_name

# 导入同级模块的内容
from .module_name import something

# 导入子模块
from .subpackage import module_name

# 导入上级目录的模块
from .. import module_name
```

## 🔍 验证修复

运行以下命令验证导入正常：

```python
import sys
sys.path.insert(0, './transtab')
import transtab

print(transtab.__version__)  # 应该显示: 0.0.6
print(transtab.load_data)    # 应该显示函数对象
```

或者运行测试脚本：

```bash
cd union/mstraffic/baseline
python transtab_clf_simplified.py
```

应该不再出现 `ImportError` 错误。

## 📋 检查清单

修复循环导入问题时，检查以下项目：

- [x] 将 `transtab.py` 中的绝对导入改为相对导入
- [x] 确保 `__init__.py` 使用相对导入（已经正确）
- [x] 测试导入是否正常工作

## 🎯 最佳实践

### 包内模块导入的推荐做法

**在包内部的模块（如 `transtab/transtab.py`）中：**
```python
# ✅ 推荐：使用相对导入
from . import constants
from .dataset import load_data

# ❌ 避免：使用绝对导入
from transtab import constants
from transtab.dataset import load_data
```

**在包的 `__init__.py` 中：**
```python
# ✅ 推荐：使用相对导入
from .transtab import *
from .dataset import load_data

# ❌ 避免：使用绝对导入
from transtab.transtab import *
```

**在包外部的脚本中：**
```python
# ✅ 正确：使用绝对导入
import transtab
from transtab import load_data
```

## 🔗 相关文件

- `transtab/__init__.py` - 包初始化文件
- `transtab/transtab.py` - 主模块（已修复）
- `transtab/dataset.py` - 数据加载模块
- `union/mstraffic/baseline/transtab_clf_simplified.py` - 测试脚本

## 📝 注意事项

1. **不要混用绝对和相对导入** - 在同一个包内保持一致
2. **包名冲突** - 避免文件名与包名相同（如 `transtab/transtab.py`）
3. **测试导入** - 每次修改后测试导入是否正常

---

**修复完成！** ✅ 现在可以正常使用 transtab 库了。

