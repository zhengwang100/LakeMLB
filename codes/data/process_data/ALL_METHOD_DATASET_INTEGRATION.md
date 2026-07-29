# LakeMLB 全方法数据集接入清单

本文档用于在新增数据集或新增表后，将其接入 Tree、TNNS、TransTab、TabPFN、TabICL 和 CARTE 的完整实验流程。

## 1. 准备原始数据

将 LakeMLB 原始 CSV 和固定 mask 放入：

```text
codes/data/table_<dataset>/raw/
```

监督分类表需要确认：

```text
1. 标签列存在且没有意外缺失值。
2. CSV 行数与 mask 长度一致。
3. train_mask、val_mask、test_mask 互不重叠。
4. 三个 mask 完整覆盖预期样本，test_mask 非空。
5. 删除标签泄露列和不参与建模的列。
```

## 2. 定义 RLLM 数据集

修改：

```text
codes/lib/rllm/datasets/lakemlb/<dataset>.py
```

需要同步：

```text
raw_filenames
processed_filenames
process()
data_list
__len__()
```

在 `process()` 中为监督表构造：

```python
TableData(
    df=df,
    col_types=col_types,
    target_col="label",
    train_mask=masks["train_mask"],
    val_mask=masks["val_mask"],
    test_mask=masks["test_mask"],
)
```

`table_idx` 就是 `processed_filenames` 和 `data_list` 中从0开始的顺序，两处顺序必须完全一致。

如果是全新 Dataset 类，还要更新：

```text
codes/lib/rllm/datasets/__init__.py
```

## 3. 注册通用 Baseline

新增数据集需要加入 `_DATASET_REGISTRY`；新增已有数据集中的表需要补充 `_TABLE_TAGS` 或对应 table-name 字典：

```text
codes/baseline/tree_models.py
codes/baseline/tnns_test.py
codes/baseline/tabpfnv2_extend.py
codes/baseline/tabicl_v2.py
codes/baseline/transtab_lakemlb_utils.py
```

同步 shell 输出目录 tag：

```text
codes/scripts/run_tree_models.sh
codes/scripts/run_nn_grid_search.sh
codes/scripts/run_tabpfn.sh
codes/scripts/run_tabicl.sh
```

TransTab 的 shell 默认可使用通用 `<dataset>_table<idx>` 目录名，可读表名在 `transtab_lakemlb_utils.py` 中维护。

## 4. 各方法的数据要求

```text
Tree      : 需要 target_col 和固定 train/val/test mask。
TNNS      : 需要 target_col 和固定 train/val/test mask。
TabPFN    : 需要 target_col 和固定 mask；fit 使用 train+val，test 独立评估。
TabICL    : 需要 target_col 和固定 mask；fit 使用 train+val，test 独立评估。
TransTab  : 有固定 mask 时复用；无 mask 时 prepare_table 会按 seed 生成 80/10/10。
CARTE     : single/joint 的任务表需要 mask；joint 的辅助表不需要 mask。
```

无标签辅助表不能直接用于 Tree、TNNS、TabPFN、TabICL 或 CARTE single 的监督分类测试。

## 5. 接入 CARTE

CARTE 不使用 RLLM 的 `--dataset/--table_idx`，而是独立的数据名。

复制 CSV：

```text
codes/lib/carte_ai/data/data_raw/<carte_data_name>.csv
```

任务表 mask 命名为：

```text
codes/lib/carte_ai/data/data_raw/mask_<mask_basename>.pt
```

在以下文件增加或复用 `data_name` 分支：

```text
codes/lib/carte_ai/scripts/preprocess_raw.py
```

每个分支至少定义：

```python
target_name = "label"       # 无标签辅助表为 None
entity_name = "entity_col"
task = "classification"
repeated = False
```

并完成缺失值处理、数值/类别类型转换、泄露列删除和标签编码。

执行预处理：

```bash
conda activate lake
cd /path/to/LakeMLB

PYTHONPATH="$PWD/codes/lib${PYTHONPATH:+:$PYTHONPATH}" \
python codes/lib/carte_ai/scripts/preprocess_raw.py \
  -dt <carte_data_name_1> <carte_data_name_2>
```

成功后应生成：

```text
codes/lib/carte_ai/data/data_singletable/<carte_data_name>/raw.parquet
codes/lib/carte_ai/data/data_singletable/<carte_data_name>/config_data.json
```

CARTE single 示例：

```bash
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name <carte_data_name> \
  --mask_basename <mask_basename> \
  --gpu 1 --num_runs 10 --max_jobs 1
```

CARTE joint 示例：

```bash
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name <target_name> \
  --source_data_name <source_name> \
  --mask_basename <target_mask_basename> \
  --gpu 1 --num_runs 10 --max_jobs 1
```

## 6. 验证

先强制重新处理 RLLM 数据集，检查每张表的行数、列、标签和 mask：

```python
dataset = DatasetClass("/path/to/LakeMLB/codes/data", force_reload=True)
for idx, table in enumerate(dataset.data_list):
    print(idx, table.df.shape, table.target_col)
```

然后执行：

```bash
python -m py_compile \
  codes/lib/rllm/datasets/lakemlb/<dataset>.py \
  codes/baseline/tree_models.py \
  codes/baseline/tnns_test.py \
  codes/baseline/tabpfnv2_extend.py \
  codes/baseline/tabicl_v2.py \
  codes/baseline/transtab_lakemlb_utils.py \
  codes/lib/carte_ai/scripts/preprocess_raw.py

bash -n \
  codes/scripts/run_tree_models.sh \
  codes/scripts/run_nn_grid_search.sh \
  codes/scripts/run_transtab.sh \
  codes/scripts/run_tabpfn.sh \
  codes/scripts/run_tabicl.sh \
  codes/scripts/run_carte.sh
```

最后对每类方法至少进行一次数据加载或单 run 冒烟测试，并确认 results、logs 和 artifacts 使用正确的数据 tag。

## 7. 文档同步

正式实验表需要同步更新：

```text
codes/scripts/EXPERIMENT_WORKFLOW.md
```

记录 table_idx、CARTE data name、mask、运行命令和特殊列处理规则。
