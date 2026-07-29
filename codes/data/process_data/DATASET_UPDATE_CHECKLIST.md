# 新增数据集或新增表维护清单

本文档用于提醒：当新增 LakeMLB 数据集，或在已有数据集中新增一张表时，需要同步修改哪些位置，避免实验脚本无法通过 `--dataset` / `--table_idx` 正确调用，或结果目录命名不一致。

## 1. 必须修改的数据集定义

数据集定义文件：

```text
codes/lib/rllm/datasets/lakemlb/<dataset_name>.py
```

需要检查：

```text
raw_filenames
processed_filenames
process()
__len__()
__getitem__()
data_list
```

新增表时必须做的事：

```text
1. 将新原始文件加入 raw_filenames，如果新表复用已有 raw 文件则不一定新增。
2. 将新 processed 文件加入 processed_filenames。
3. 在 process() 中构造新 TableData，并 save 到对应 processed_paths[index]。
4. 在 __init__ 的 data_list 中追加 TableData.load(self.processed_paths[index])。
5. 更新 __len__() 返回值。
6. 确认 target_col 是否正确。
7. 如果该表用于监督分类，确认 train_mask / val_mask / test_mask 已设置。
8. 确认 col_types 不包含不应作为特征的列。
```

如果是全新数据集，还需要在导出文件中注册：

```text
codes/lib/rllm/datasets/__init__.py
```

需要添加：

```python
from .lakemlb.<dataset_name> import <DatasetClass>
```

并在 `__all__` 中加入对应类名。

## 2. Baseline Python 中的数据集注册

目前多类实验脚本各自维护了 `_DATASET_REGISTRY` 和 `_TABLE_TAGS` / `TABLE_TAGS`。

新增全新数据集时，需要把 Dataset class import 进去，并加入 registry。

新增已有数据集中的表时，通常只需要更新 tag 映射。

需要检查的文件：

```text
codes/baseline/tree_models.py
codes/baseline/tnns_test.py
codes/baseline/tabpfnv2_extend.py
codes/baseline/tabicl_v2.py
codes/baseline/transtab_lakemlb_utils.py
```

需要检查的变量：

```text
_DATASET_REGISTRY / DATASET_REGISTRY
_TABLE_TAGS / TABLE_TAGS
```

示例：

```python
_TABLE_TAGS = {
    "agbooks": {
        0: "agbooks_amazon",
        2: "agbooks_amazon_enriched",
        4: "agbooks_amazon_no_features",
    },
}
```

说明：

```text
tag 映射不影响数据能否加载；如果缺失，脚本通常会退回到 <dataset>_table<table_idx>。
但 tag 会影响 results/logs/artifacts 目录名，所以建议显式补全。
```

## 3. Shell 入口脚本中的 DATA_TAG 映射

这些脚本负责组织多次实验、日志目录和结果目录。新增表后建议同步补充 `DATA_TAG` 的 case 映射。

需要检查：

```text
codes/scripts/run_tree_models.sh
codes/scripts/run_nn_grid_search.sh
codes/scripts/run_tabpfn.sh
codes/scripts/run_tabicl.sh
```

搜索关键词：

```bash
DATA_TAG
case "${DATASET}:${TABLE_IDX}" in
```

示例：

```bash
agbooks:4) DATA_TAG="agbooks_amazon_no_features" ;;
```

TransTab 当前 shell 脚本：

```text
codes/scripts/run_transtab.sh
```

默认使用：

```text
<dataset>_table<table_idx>
```

真正的 tag 映射主要在：

```text
codes/baseline/transtab_lakemlb_utils.py
```

CARTE 当前不走 LakeMLB 的 `--dataset --table_idx` 逻辑，而是使用 CARTE 自己的数据目录和 data name：

```text
codes/scripts/run_carte.sh
codes/lib/carte_ai/data/data_singletable/
codes/lib/carte_ai/data/data_raw/
```

## 4. 文档同步

如果新增的数据集/表会作为正式实验使用，建议同步更新：

```text
codes/scripts/EXPERIMENT_WORKFLOW.md
```

至少更新：

```text
1. 常用数据集指定方式。
2. 新增 table_idx 的含义。
3. 如果是特殊消融表，说明删除/保留了哪些列。
```

## 5. 处理后验证

修改后建议运行：

```bash
conda activate lake
```

语法检查：

```bash
python -m py_compile \
  codes/lib/rllm/datasets/lakemlb/<dataset_name>.py \
  codes/baseline/tree_models.py \
  codes/baseline/tnns_test.py \
  codes/baseline/tabpfnv2_extend.py \
  codes/baseline/tabicl_v2.py \
  codes/baseline/transtab_lakemlb_utils.py
```

shell 检查：

```bash
bash -n \
  codes/scripts/run_tree_models.sh \
  codes/scripts/run_nn_grid_search.sh \
  codes/scripts/run_tabpfn.sh \
  codes/scripts/run_tabicl.sh \
  codes/scripts/run_transtab.sh
```

强制重新处理并检查表：

```bash
python - <<'PY'
import sys
sys.path.insert(0, "/path/to/LakeMLB/codes/lib")
from rllm.datasets import AGBooksDataset

ds = AGBooksDataset(cached_dir="/path/to/LakeMLB/codes/data", force_reload=True)
print("len:", len(ds), "data_list:", len(ds.data_list))
for i, data in enumerate(ds.data_list):
    print(i, len(data.df), len(data.df.columns), data.target_col)
    print(list(data.df.columns))
    if hasattr(data, "train_mask"):
        print(int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()))
PY
```

将 `AGBooksDataset` 替换为实际新增的数据集类。

## 6. 给 Codex 的复用提示词

后续新增数据集或新增表时，可以直接把下面这段发给 Codex：

```text
我新增/修改了 LakeMLB 数据集，请按照 `codes/data/process_data/DATASET_UPDATE_CHECKLIST.md` 的清单帮我同步代码。

具体信息：
1. 数据集名：<dataset_name>
2. Dataset 类名：<DatasetClass>
3. raw 路径：<raw_dir>
4. 新增/修改的表：
   - table_idx: <idx>
   - raw 文件: <file.csv>
   - processed 文件名: <name>_data.pt
   - target_col: <label_col 或 None>
   - mask 文件: <mask.pt 或无>
   - 需要删除/保留的特殊列：<说明>
   - 建议 tag: <dataset_table_tag>

请你完成：
1. 修改 rllm 数据集定义文件。
2. 如果是全新数据集，更新 codes/lib/rllm/datasets/__init__.py。
3. 同步 tree_models.py、tnns_test.py、tabpfnv2_extend.py、tabicl_v2.py、transtab_lakemlb_utils.py 中的 dataset registry 和 table tag。
4. 同步 run_tree_models.sh、run_nn_grid_search.sh、run_tabpfn.sh、run_tabicl.sh 中的 DATA_TAG 映射。
5. 必要时更新 codes/scripts/EXPERIMENT_WORKFLOW.md。
6. 激活 conda 环境 lake，运行 py_compile、bash -n，并 force_reload 验证新增表可以加载，检查列、target、mask 和 table_idx。
```

## 7. 当前已知 agbooks 表索引

```text
agbooks[0] = agbooks_amazon
agbooks[1] = agbooks_goodreads
agbooks[2] = agbooks_amazon_enriched
agbooks[3] = agbooks_da
agbooks[4] = agbooks_amazon_no_features
```
