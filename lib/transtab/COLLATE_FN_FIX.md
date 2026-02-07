# Contrastive Learning Collate Function Fix

## 🐛 The Problem

When using `target_col=None` for self-supervised contrastive learning, the training failed with:

```
ValueError: All objects passed were None
```

**Location:** `transtab/trainer_utils.py`, line 120 in `TransTabCollatorForCL.__call__()`

## 🔍 Root Cause

The collate function tried to concatenate `None` values when `y=None`:

```python
def __call__(self, data):
    df_x = pd.concat([row[0] for row in data])
    df_y = pd.concat([row[1] for row in data])  # ← Fails when all y are None
    # ...
```

When:
- `target_col=None` in config
- Data is loaded with `y=None` for all samples
- Collate function receives a batch: `[(x1, None), (x2, None), ...]`
- `pd.concat([None, None, ...])` → **ValueError**

## ✅ The Fix

Added explicit handling for when all `y` values are `None`:

```python
def __call__(self, data):
    df_x = pd.concat([row[0] for row in data])
    
    # Handle y values: if all are None (unsupervised), keep as None
    y_list = [row[1] for row in data]
    if all(y is None for y in y_list):
        df_y = None  # ← Return None instead of trying to concat
    else:
        df_y = pd.concat(y_list)
    
    # ... rest of the function
    return res, df_y
```

## 📊 How It Works

### Self-Supervised Learning (y=None)
```
1. Config: target_col=None
   ↓
2. load_data() returns: y_train = None, y_val = None
   ↓
3. Dataset yields: (x_batch, None)
   ↓
4. Collate function receives: [(x1, None), (x2, None), ...]
   ↓
5. Collate function returns: (res, None)  ← Fixed!
   ↓
6. Model forward: model(data[0], data[1])  # data[1] = None
   ↓
7. TransTabForCL handles: if y is None → self_supervised_contrastive_loss()
```

### Supervised Contrastive Learning (y not None)
```
1. Config: target_col='Label'
   ↓
2. load_data() returns: y_train = [...], y_val = [...]
   ↓
3. Dataset yields: (x_batch, y_batch)
   ↓
4. Collate function receives: [(x1, y1), (x2, y2), ...]
   ↓
5. Collate function returns: (res, df_y)  # df_y = pd.concat([y1, y2, ...])
   ↓
6. Model forward: model(data[0], data[1])  # data[1] = df_y
   ↓
7. TransTabForCL handles: if y is not None → supervised_contrastive_loss()
```

## 🎯 Model Support

The `TransTabForCL.forward()` method already supports both cases:

```python
def forward(self, x, y=None):
    # ... encoding logic ...
    
    if y is not None and self.supervised:
        # Supervised contrastive learning
        y = torch.tensor(y.values, device=feat_x_multiview.device)
        loss = self.supervised_contrastive_loss(feat_x_multiview, y)
    else:
        # Self-supervised contrastive learning
        loss = self.self_supervised_contrastive_loss(feat_x_multiview)
    
    return None, loss
```

**Key:** The model checks `if y is not None` before using `y.values`, so passing `y=None` is safe!

## 🔧 Complete Flow for Self-Supervised Learning

### 1. Create Config
```python
seattle_config = transtab.create_dataset_config(
    col_types_dict=seattle_col_types,
    target_col=None,  # All columns become features
    mask_path=os.path.join(DATA_DIR, 'seattle_mask.pt'),
)
```

### 2. Load Data
```python
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(
    [DATA_DIR], 
    dataset_config={DATA_DIR: seattle_config}, 
    filename='seattle.csv'
)
# trainset = [(X_train, None)]  # y is None
```

### 3. Build Contrastive Learner
```python
model_pretrain, collate_fn = transtab.build_contrastive_learner(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    supervised=False,  # Self-supervised
    num_partition=4,
    overlap_ratio=0.5,
)
```

### 4. Train
```python
transtab.train(
    model_pretrain, 
    trainset,  # [(X, None)]
    valset,    # [(X_val, None)]
    collate_fn=collate_fn,  # Fixed collate_fn handles None correctly
    num_epoch=2,
    output_dir='./ckpt_cl/pretrained'
)
# ✅ Works! No ValueError
```

## 📝 Testing

### Test Case 1: Self-Supervised (y=None)
```python
data = [
    (pd.DataFrame({'a': [1, 2]}), None),
    (pd.DataFrame({'a': [3, 4]}), None),
]

collate_fn = TransTabCollatorForCL(...)
result, df_y = collate_fn(data)

assert df_y is None  # ✓ Pass
```

### Test Case 2: Supervised (y not None)
```python
data = [
    (pd.DataFrame({'a': [1, 2]}), pd.Series([0, 1])),
    (pd.DataFrame({'a': [3, 4]}), pd.Series([1, 0])),
]

collate_fn = TransTabCollatorForCL(...)
result, df_y = collate_fn(data)

assert df_y is not None  # ✓ Pass
assert len(df_y) == 4     # ✓ Pass
```

## ✨ Summary

**The Fix:**
- Check if all `y` values are `None` before calling `pd.concat()`
- Return `None` directly when all labels are `None`
- Preserve normal behavior when labels exist

**Benefits:**
- ✅ Self-supervised contrastive learning now works
- ✅ Supervised contrastive learning still works
- ✅ Clean and explicit handling
- ✅ No breaking changes

**Files Modified:**
1. `transtab/trainer_utils.py` - Fixed `TransTabCollatorForCL.__call__()`
2. `transtab/dataset.py` - Fixed `create_dataset_config()` for `target_col=None`

**Now you can run self-supervised contrastive learning!** 🚀

---

**Related:**
- `SELF_SUPERVISED_FIX.md` - Config generation fix
- `SELF_SUPERVISED_GUIDE.md` - Usage guide
- `union/mstraffic/baseline/transtab_cl_simplified.py` - Working example

