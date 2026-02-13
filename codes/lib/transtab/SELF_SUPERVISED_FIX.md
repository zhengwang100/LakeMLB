# Self-Supervised Learning Support (target_col=None)

## 🔧 Fix Summary

Fixed `create_dataset_config()` to properly support `target_col=None` for self-supervised/contrastive learning.

## 🐛 The Problem

When using `target_col=None` for self-supervised learning (e.g., contrastive learning), the function was not correctly handling the `None` value.

### Previous Behavior (Incorrect)
```python
# In create_dataset_config()
for col_name, col_type in col_types_dict.items():
    if col_name == target_col:  # This check fails when target_col=None
        continue
    # ... add to feature lists
```

**Issue:** When `target_col=None`, the condition `col_name == target_col` is always `False`, so all columns are processed. This is actually **correct** for self-supervised learning! 

However, the config was always adding `"target_col": None`, which could cause confusion.

## ✅ The Fix

### 1. Explicit None Check
```python
# Skip target column only if target_col is not None
if target_col is not None and col_name == target_col:
    continue
```

**Benefit:** Makes the logic explicit and clear.

### 2. Conditional target_col in Config
```python
# Only add target_col if it's not None
if target_col is not None:
    config["target_col"] = target_col
```

**Benefit:** Config dictionary doesn't include `target_col` key when it's `None`, making it clearer that this is for unsupervised learning.

### 3. Updated Documentation
```python
target_col : str or None
    The name of the target/label column. 
    Set to None for unsupervised/self-supervised learning (all columns become features).
```

## 📝 Usage Examples

### Supervised Learning (with target)
```python
col_types = {
    "Feature1": ColType.NUMERICAL,
    "Feature2": ColType.CATEGORICAL,
    "Label": ColType.CATEGORICAL,
}

config = transtab.create_dataset_config(
    col_types_dict=col_types,
    target_col='Label',  # Label is excluded from features
    mask_path='./data/mask.pt',
)

# Result:
# config['cols'] = ['feature1', 'feature2']  # Label NOT included
# config['target_col'] = 'Label'
```

### Self-Supervised Learning (no target)
```python
col_types = {
    "Feature1": ColType.NUMERICAL,
    "Feature2": ColType.CATEGORICAL,
    "Feature3": ColType.CATEGORICAL,  # Could be label, but used as feature
}

config = transtab.create_dataset_config(
    col_types_dict=col_types,
    target_col=None,  # No target column
    mask_path='./data/mask.pt',
)

# Result:
# config['cols'] = ['feature1', 'feature2', 'feature3']  # ALL included
# config does NOT have 'target_col' key
```

### MSTraffic Contrastive Learning Example
```python
# For Seattle dataset in contrastive learning
seattle_col_types = {
    "OBJECTID": ColType.NUMERICAL,
    "COLLISIONTYPE": ColType.CATEGORICAL,  # Include as feature, not target
    "WEATHER": ColType.CATEGORICAL,
    # ... other columns
}

seattle_config = transtab.create_dataset_config(
    col_types_dict=seattle_col_types,
    target_col=None,  # All columns become features for contrastive learning
    mask_path=os.path.join(DATA_DIR, 'seattle_mask.pt'),
)

# COLLISIONTYPE is now in seattle_config['cat'] and seattle_config['cols']
# It will be used as a feature for contrastive learning
```

## 🔍 How It Works Internally

### In `create_dataset_config()`
1. If `target_col=None`:
   - All columns are processed and added to feature lists
   - No column is skipped
   - `target_col` key is NOT added to config

2. If `target_col='SomeColumn'`:
   - 'SomeColumn' is skipped when processing features
   - Other columns are added to feature lists
   - `config['target_col'] = 'SomeColumn'` is added

### In `load_single_data()`
```python
target_col = dataset_config.get("target_col", None)
if target_col is not None:
    y = allset[target_col].values  # Extract labels
else:
    y = None  # No labels for self-supervised learning

# Later in data splitting:
if y is not None:
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
else:
    y_train = y_val = y_test = None  # No labels
```

## 📊 Comparison Table

| Aspect | Supervised (`target_col='Label'`) | Self-Supervised (`target_col=None`) |
|--------|-----------------------------------|-------------------------------------|
| **Feature Columns** | All except target | All columns (including former target) |
| **config['cols']** | Excludes target | Includes all columns |
| **config['target_col']** | Present (e.g., `'Label'`) | Not present |
| **y values** | Extracted from target column | `None` |
| **Use Case** | Classification, Regression | Contrastive Learning, Clustering |

## 🎯 Updated Files

1. **`transtab/dataset.py`**
   - ✅ Fixed `create_dataset_config()` logic
   - ✅ Updated documentation
   - ✅ Added self-supervised example

2. **`union/mstraffic/baseline/transtab_cl_simplified.py`**
   - ✅ Updated comments for Seattle config
   - ✅ Uses `target_col=None` correctly

## 🚀 Testing

### Test Supervised Learning
```python
config = transtab.create_dataset_config(
    {"A": "num", "B": "cat", "C": "cat"},
    target_col="C"
)
assert "a" in config['cols']
assert "b" in config['cols']
assert "c" not in config['cols']  # Target excluded
assert config['target_col'] == "C"
```

### Test Self-Supervised Learning
```python
config = transtab.create_dataset_config(
    {"A": "num", "B": "cat", "C": "cat"},
    target_col=None
)
assert "a" in config['cols']
assert "b" in config['cols']
assert "c" in config['cols']  # All included
assert 'target_col' not in config  # Key not present
```

## ✨ Benefits

1. **Clearer Intent**: Code explicitly shows when target column handling is needed
2. **Correct Behavior**: All columns included for self-supervised learning
3. **Better Config**: Config dictionary structure reflects the learning mode
4. **Documentation**: Clear examples for both use cases

## 🎉 Summary

**Before Fix:**
- Implicit behavior with `target_col=None`
- Config always had `target_col` key (even when `None`)
- Unclear intent

**After Fix:**
- ✅ Explicit `target_col is not None` check
- ✅ Config only has `target_col` key when needed
- ✅ Clear documentation and examples
- ✅ Works correctly for both supervised and self-supervised learning

**You can now use `target_col=None` for contrastive learning!** 🚀

---

**Related Files:**
- `transtab/dataset.py` - Core fix
- `union/mstraffic/baseline/transtab_cl_simplified.py` - Usage example
- `CONTRASTIVE_VS_SUPERVISED.md` - Comparison guide

