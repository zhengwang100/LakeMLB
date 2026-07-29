# 实验脚本工作流说明

本文档总结当前分类任务的四大类实验脚本：决策树类、TNNS 深度表格模型类、迁移学习类、基础模型类。所有命令默认从项目根目录执行。

## 通用约定

默认数据集与表：

```bash
--dataset mstraffic --table_idx 0
```

也就是 `MSTrafficDataset.data_list[0]`，对应 `mstraffic_maryland` 任务表。

常用数据集指定方式：

```bash
# mstraffic 第一张表
--dataset mstraffic --table_idx 0

# nctaxi 第一张表，即 newyork_taxi
--dataset nctaxi --table_idx 0

# agbooks 第三张表，即 amazon_enriched
--dataset agbooks --table_idx 2

# nnstocks 指定表
--dataset nnstocks --table_idx 3
```

NNStocks 当前表索引：

```text
0  nnlist
1  nnwiki（无标签辅助表，不能用于普通监督分类）
2  nnstocks_da
3  nnstocks_fa
4  stocks_wiki_llm_1nn
5  t1_enriched_rank2
6  t1_enriched_rank4
7  t1_enriched_rank8
8  stocks_wiki_tfidf_1nn
9  t1_enriched_random
```

索引 `4-9` 均使用任务表标签 `sector` 和 `mask_nnlist.pt`，可直接用于 Tree、TNNS、TransTab single、TabPFN 和 TabICL。CARTE 需要单独预处理和定义。

DSMusic 书名/歌曲名近邻消融表索引：

```text
4  dsmusic_1nn
5  dsmusic_2nn
6  dsmusic_4nn
7  dsmusic_8nn
8  dsmusic_random
```

这些表以 `discogs.title` 匹配 `spotify.track_name`，使用任务表标签 `genres` 和 `mask_discogs.pt`。

AGBooks 书名近邻消融表索引：

```text
2   amazon_enriched（原 FA 对照表）
6   agbooks_1nn
7   agbooks_2nn
8   agbooks_4nn
9   agbooks_8nn
10  agbooks_random
```

索引 `6-10` 以 `amazon.title` 匹配 `goodreads.title`，使用任务表标签 `categories` 和 `amazon_mask.pt`。上述 DSMusic 与 AGBooks 消融表均可直接用于 Tree、TNNS、TransTab single、TabPFN、TabICL 和 CARTE single。

例如，将 `<idx>` 替换为上表索引：

```bash
bash codes/scripts/run_tree_models.sh --dataset dsmusic --table_idx <idx> --device 1 --num_runs 10
bash codes/scripts/run_nn_grid_search.sh --dataset dsmusic --table_idx <idx> --gpu 1 --num_runs 10
bash codes/scripts/run_transtab.sh --mode single --dataset dsmusic --table_idx <idx> --gpu 1 --num_runs 10
bash codes/scripts/run_tabpfn.sh --dataset dsmusic --table_idx <idx> --gpu 1 --num_runs 10
bash codes/scripts/run_tabicl.sh --dataset dsmusic --table_idx <idx> --gpu 1 --num_runs 10
```

AGBooks 使用相同命令格式，将 `--dataset dsmusic` 改为 `--dataset agbooks`。

CARTE 使用独立的 data name，不使用 `--dataset/--table_idx`：

```text
dsmusic_1nn     mask_dsmusic_1nn.pt
dsmusic_2nn     mask_dsmusic_2nn.pt
dsmusic_4nn     mask_dsmusic_4nn.pt
dsmusic_8nn     mask_dsmusic_8nn.pt
dsmusic_random  mask_dsmusic_random.pt

agbooks_1nn     mask_agbooks_1nn.pt
agbooks_2nn     mask_agbooks_2nn.pt
agbooks_4nn     mask_agbooks_4nn.pt
agbooks_8nn     mask_agbooks_8nn.pt
agbooks_random  mask_agbooks_random.pt
```

CARTE 单表运行示例：

```bash
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name dsmusic_1nn \
  --mask_basename dsmusic_1nn \
  --gpu 1 --num_runs 10 --max_jobs 1

bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name agbooks_1nn \
  --mask_basename agbooks_1nn \
  --gpu 1 --num_runs 10 --max_jobs 1
```

其余近邻和随机表只需同步替换 `--data_name` 与 `--mask_basename` 的后缀。

注意：被测试的表必须包含 `train_mask`、`val_mask`、`test_mask`。辅助表如果没有 mask，不能直接作为监督测试表。

GPU 约定：

```bash
--gpu 1
```

表示脚本设置 `CUDA_VISIBLE_DEVICES=1`，Python 进程内部使用 `cuda:0`。

随机种子约定：

```text
重复实验的每个 run 使用系统熵生成随机 seed，不再使用 seed+run_id 的等差序列。
实际使用的 seed 会写入每个 run 的 JSON、日志和文件名，便于之后复查。
```

后台运行约定：

长实验建议使用 `nohup`，避免 SSH 或终端连接中断导致任务停止。推荐使用 `bash -lc` 包住 conda 激活、进入项目目录和实际实验命令。

通用模板：

```bash
mkdir -p codes/results/nohup

nohup bash -lc '
conda activate lake
cd /path/to/LakeMLB

bash codes/scripts/<script_name>.sh \
  <args>
' > codes/results/nohup/<job_name>_$(date +"%Y%m%d_%H%M%S").out 2>&1 &
```

示例，后台跑 TNNS：

```bash
mkdir -p codes/results/nohup

nohup bash -lc '
conda activate lake
cd /path/to/LakeMLB

bash codes/scripts/run_nn_grid_search.sh \
  --dataset nctaxi \
  --table_idx 2 \
  --gpu 1 \
  --num_runs 10 \
  --num_tasks 2 \
  --models saint
' > codes/results/nohup/tnns_nctaxi_table2_saint_$(date +"%Y%m%d_%H%M%S").out 2>&1 &
```

示例，后台跑 TransTab：

```bash
mkdir -p codes/results/nohup

nohup bash -lc '
conda activate lake
cd /path/to/LakeMLB

bash codes/scripts/run_transtab.sh \
  --mode single \
  --dataset agbooks \
  --table_idx 0 \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
' > codes/results/nohup/transtab_agbooks_amazon_$(date +"%Y%m%d_%H%M%S").out 2>&1 &
```

查看后台任务：

```bash
jobs -l
ps -fp <PID>
nvidia-smi
```

查看 nohup 外层日志：

```bash
tail -f codes/results/nohup/<job_name>_<timestamp>.out
```

查看实验脚本自身日志：

```bash
tail -f codes/results/logs/<method>/<dataset_tag>/<log_file>.log
```

说明：

```text
1. `nohup ... &` 会让任务在当前 SSH 断开后继续运行。
2. `> ... 2>&1` 会把外层终端输出保存到指定 .out 文件。
3. 当前实验脚本本身也会写 results/logs 下的正式日志；nohup 的 .out 主要用于查看外层启动过程和报错。
4. 如果当前 shell 已经 `conda activate lake`，也可以直接 `nohup bash codes/scripts/xxx.sh ... > log.out 2>&1 &`。
   但更推荐上面的 `bash -lc` 模板，因为它对重新登录、复制命令、环境切换更稳。
```

## 1. 决策树类

入口脚本：

```bash
codes/scripts/run_tree_models.sh
```

覆盖方法：

```text
XGBoost
CatBoost
LightGBM
```

功能流程：

```text
对指定数据集/表：
  1. 依次串行运行 XGBoost、CatBoost、LightGBM
  2. 每个模型内部先进行 grid search
  3. 用 validation accuracy 选择最优超参
  4. 如果 validation accuracy 相同，用 validation macro-F1 打平
  5. 用最优超参重新训练并保存 best grid model
  6. 用最优超参重复运行 num_runs 次
  7. 保存每次 run、均值/方差、耗时、日志
```

默认运行：

```bash
bash codes/scripts/run_tree_models.sh --device 1 --num_runs 10
```

指定数据集：

```bash
bash codes/scripts/run_tree_models.sh \
  --dataset nctaxi \
  --table_idx 0 \
  --device 1 \
  --num_runs 10
```

只跑部分模型：

```bash
bash codes/scripts/run_tree_models.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --models xgboost,lightgbm \
  --num_runs 10
```

控制 tree learner 内部线程数：

```bash
bash codes/scripts/run_tree_models.sh \
  --num_threads 16
```

`--num_threads 0` 表示使用库默认线程策略。

超参搜索空间：

```text
XGBoost:
  n_estimators: {300, 1000}
  max_depth: {6, 8}
  learning_rate: {0.01, 0.05}
  subsample: {0.9}
  colsample_bytree: {0.8}

CatBoost:
  iterations: {300, 1000}
  depth: {6, 8}
  learning_rate: {0.01, 0.05}
  subsample: {0.9}
  rsm: {0.8}

LightGBM:
  num_boost_round: {300, 1000}
  num_leaves: {63, 127}
  learning_rate: {0.01, 0.05}
  feature_fraction: {0.8}
  bagging_fraction: {0.9}
```

early stopping:

```text
patience = 50 validation rounds
```

搜索预算：

```text
XGBoost: 8 组
CatBoost: 8 组
LightGBM: 8 组
```

三种模型保持相同的 8 组搜索预算。CatBoost 的 `subsample` 会配合 `bootstrap_type=Bernoulli` 使用，`rsm` 对应列采样比例。

输出目录：

```text
codes/results/tree_models/<dataset_tag>/<model>/
  grid_search_<timestamp>.json
  final_<num_runs>runs_<timestamp>.json

codes/results/artifacts/tree_models/<dataset_tag>/<model>/grid/
  best_grid_seed<seed>.<json|cbm|txt>
  best_grid_metadata.json

codes/results/logs/tree_models/<dataset_tag>/
  run_tree_models_<timestamp>.log
  <model>_<dataset_tag>_<timestamp>.log
```

并行策略：

```text
三个模型之间：串行
每个模型内部 grid search：串行
每个 tree learner 内部：多线程，可由 --num_threads 控制
```

## 2. TNNS 深度表格模型类

入口脚本：

```bash
codes/scripts/run_nn_grid_search.sh
```

覆盖方法：

```text
FTTransformer
TabTransformer
ExcelFormer
SAINT
TromptNet
```

功能流程：

```text
对指定数据集/表：
  1. 对每个 TNNS 模型进行 grid search
  2. grid search 可按 num_tasks 切分并行
  3. 按 validation accuracy 选择最优超参
  4. 用最优超参重新训练并保存 grid checkpoint
  5. 用最优超参重复运行 num_runs 次
  6. 保存 final checkpoint、每次 run、均值/方差、耗时、日志
```

默认运行：

```bash
bash codes/scripts/run_nn_grid_search.sh
```

指定数据集：

```bash
bash codes/scripts/run_nn_grid_search.sh \
  --dataset nctaxi \
  --table_idx 0 \
  --gpu 1 \
  --num_runs 10
```

只跑部分模型：

```bash
bash codes/scripts/run_nn_grid_search.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --models fttransformer,saint \
  --gpu 1
```

控制 grid search 并行 shard 数：

```bash
bash codes/scripts/run_nn_grid_search.sh \
  --num_tasks 2
```

超参搜索空间：

```text
hidden_dim: {64, 128}
layers: {2, 3, 4}
learning_rate: {1e-3, 5e-4, 1e-4}
weight_decay: {5e-4}
batch_size: 256
max_epochs: 500
early_stopping_patience: 10
combinations per model: 18
```

输出目录：

```text
codes/results/grid_search/tnns/<dataset_tag>/<model>/<timestamp>/
  <model>_grid_task_000.json
  <model>_grid_task_001.json
  <model>_grid_merged.json

codes/results/tnns/<dataset_tag>/<model>/
  final_<num_runs>runs_<timestamp>.json

codes/results/artifacts/tnns/<dataset_tag>/<model>/
  grid/best_seed<seed>.pt
  grid/best_model_metadata.json
  final/best_seed<seed>.pt
  final/best_model_metadata.json

codes/results/logs/tnns/<dataset_tag>/
  run_tnns_grid_<timestamp>.log
  <model>_task_<task_id>_<timestamp>.log
  <model>_merge_<timestamp>.log
```

并行策略：

```text
grid search：支持 --num_tasks 并行切分
final num_runs 重复实验：当前串行，避免同一 GPU 上 OOM
```

建议：

```text
单 GPU 先用 --num_tasks 2
显存充足再尝试更大值
```

## 3. 迁移学习类

当前迁移学习类包含：

```text
TransTab single-table
TransTab labeled-auxiliary transfer
TransTab unlabeled-auxiliary contrastive transfer
CARTE multi-table transfer
```

### 3.1 TransTab

入口脚本：

```bash
codes/scripts/run_transtab.sh
```

该脚本是 TransTab 的统一入口，通过 `--mode` 选择三种场景：

```text
--mode single      只做单表测试，调用 transtab_single.py
--mode transfer    辅助表有标签，调用 transtab_transfer.py
--mode contrastive 辅助表没有标签，调用 transtab_transfer_cl.py
```

表指定方式：

```text
--dataset / --table_idx 指定任务表
--aux_dataset / --aux_table_idx 指定辅助表，仅 transfer 和 contrastive 模式需要
```

默认运行：

```bash
bash codes/scripts/run_transtab.sh --gpu 1 --num_runs 10
```

默认等价于：

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic \
  --table_idx 0 \
  --aux_dataset mstraffic \
  --aux_table_idx 1 \
  --gpu 1 \
  --num_runs 10
```

单表测试：

```bash
bash codes/scripts/run_transtab.sh \
  --mode single \
  --dataset nnstocks \
  --table_idx 4 \
  --gpu 1 \
  --num_runs 10
```

辅助表有标签的迁移测试：

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic \
  --table_idx 0 \
  --aux_dataset mstraffic \
  --aux_table_idx 1 \
  --gpu 1 \
  --num_runs 10
```

MSTraffic 的 Seattle 辅助表样本量消融索引：

```text
--aux_table_idx 4  Seattle 25%（2700 行，预训练 train=2160）
--aux_table_idx 5  Seattle 50%（5400 行，预训练 train=4320）
--aux_table_idx 6  Seattle 75%（8100 行，预训练 train=6480）
--aux_table_idx 1  Seattle 100%（10800 行，预训练 train=8640）
```

例如运行 25% 辅助表消融：

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic \
  --table_idx 0 \
  --aux_dataset mstraffic \
  --aux_table_idx 4 \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

50% 和 75% 实验只需分别将 `--aux_table_idx` 改为 `5` 和 `6`。为保证消融可比性，四档实验应使用相同的 epoch、重复次数和并行设置。

NCTaxi 的 Chicago 有标签辅助表样本量消融同样使用索引 `4/5/6`：

```text
--aux_table_idx 4  Chicago 25%（25000 行，每类 500 行）
--aux_table_idx 5  Chicago 50%（50000 行，每类 1000 行）
--aux_table_idx 6  Chicago 75%（75000 行，每类 1500 行）
--aux_table_idx 1  Chicago 100%（100000 行，每类 2000 行）
```

四张 Chicago 表均不绑定固定 mask，由 TransTab 根据每次运行的随机种子生成 `80%/10%/10%` 划分。例如运行 25% 辅助表：

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset nctaxi \
  --table_idx 0 \
  --aux_dataset nctaxi \
  --aux_table_idx 4 \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

辅助表无标签的 contrastive transfer 测试：

```bash
bash codes/scripts/run_transtab.sh \
  --mode contrastive \
  --dataset lhstocks \
  --table_idx 0 \
  --aux_dataset lhstocks \
  --aux_table_idx 1 \
  --gpu 1 \
  --num_runs 10
```

控制重复实验并行数：

```bash
bash codes/scripts/run_transtab.sh \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 2
```

控制训练 epoch：

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic \
  --table_idx 0 \
  --aux_dataset mstraffic \
  --aux_table_idx 1 \
  --pretrain_epochs 100 \
  --finetune_epochs 100
```

功能流程：

```text
single 每个 run：
  1. 在指定任务表上训练
  2. 在测试集评估
  3. 保存单 run JSON、checkpoint、日志

transfer 每个 run：
  1. 在有标签辅助表上 pretrain
  2. 在任务表上 fine-tune
  3. 在测试集评估
  4. 保存单 run JSON、checkpoint、pretrained checkpoint、日志

contrastive 每个 run：
  1. 在无标签辅助表上 contrastive pretrain
  2. 在任务表上 fine-tune
  3. 在测试集评估
  4. 保存单 run JSON、checkpoint、pretrained checkpoint、日志

所有 runs 完成后：
  1. 汇总 available metrics，如 accuracy/AUC/precision/recall/F1
  2. 汇总 runtime
  3. 保存 summary JSON
```

输出目录：

```text
codes/results/transfer/transtab/<mode>/<dataset_tag>/
  run_<id>_seed<seed>_<timestamp>.json
  summary_<num_runs>runs_<timestamp>.json

codes/results/logs/transfer/transtab/<mode>/<dataset_tag>/
  run_transtab_<mode>_<timestamp>.log
  run_<id>_seed<seed>_<timestamp>.log

codes/results/artifacts/transfer/transtab/<mode>/<dataset_tag>/
  run_<id>_seed<seed>/
    checkpoint/
    pretrained/
```

### 3.2 CARTE

入口脚本：

```bash
codes/scripts/run_carte.sh
```

默认运行：

```bash
bash codes/scripts/run_carte.sh --gpu 1 --num_runs 10
```

默认等价于多表模式：

```bash
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name maryland \
  --source_data_name seattle \
  --mask_basename maryland \
  --gpu 1 \
  --num_runs 10
```

CARTE 使用自己的预处理数据名，数据目录为：

```text
codes/lib/carte_ai/data/data_singletable/<data_name>/
codes/lib/carte_ai/data/data_raw/mask_<mask_basename>.pt
```

单表测试：

```bash
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name stocks_wiki_llm_1nn \
  --mask_basename nnlist \
  --gpu 1 \
  --num_runs 10
```

NNStocks 新增表在 CARTE 中的数据名如下，均使用 `mask_nnlist.pt`：

```text
stocks_wiki_llm_1nn
t1_enriched_rank2
t1_enriched_rank4
t1_enriched_rank8
stocks_wiki_tfidf_1nn
t1_enriched_random
```

例如测试 rank-2 表：

```bash
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name t1_enriched_rank2 \
  --mask_basename nnlist \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

其他五张表只需替换 `--data_name`。六张表均为1078行，mask 划分为 train/val/test=`748/99/231`。

多表测试：

```bash
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name maryland \
  --source_data_name seattle \
  --mask_basename maryland \
  --gpu 1 \
  --num_runs 10
```

NCTaxi 的 Chicago 辅助表样本量消融使用以下 CARTE 数据名：

```text
nctaxi_chicago_taxi_25pct  Chicago 25%（25000 行，每类 500 行）
nctaxi_chicago_taxi_50pct  Chicago 50%（50000 行，每类 1000 行）
nctaxi_chicago_taxi_75pct  Chicago 75%（75000 行，每类 1500 行）
nctaxi_chicago_taxi        Chicago 100%（100000 行，每类 2000 行）
```

辅助表不需要 mask；只有任务表 `nctaxi_newyork_taxi` 使用 `mask_nctaxi_newyork_taxi.pt`。例如运行 25% 辅助表：

```bash
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name nctaxi_newyork_taxi \
  --source_data_name nctaxi_chicago_taxi_25pct \
  --mask_basename nctaxi_newyork_taxi \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

50%、75% 和 100% 实验只需替换 `--source_data_name`，其余参数保持一致。

控制重复实验并行数：

```bash
bash codes/scripts/run_carte.sh \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 2
```

控制 CARTE ensemble 数：

```bash
bash codes/scripts/run_carte.sh \
  --num_model 5
```

功能流程：

```text
single 每个 run：
  1. 加载 data_singletable/<data_name>
  2. 使用 mask_<mask_basename>.pt 固定 train/val/test
  3. 构建 CARTE 图表示
  4. 训练 CARTEClassifier
  5. 在测试集评估
  6. 保存单 run JSON、日志

joint 每个 run：
  1. 加载目标表 Maryland
  2. 加载辅助表 Seattle
  3. 构建 multi-table CARTE 图表示
  4. 训练 CARTEMultitableClassifier
  5. 在测试集评估
  6. 保存单 run JSON、日志
  7. 尝试 pickle 保存 estimator

所有 runs 完成后：
  1. 汇总 available metrics，如 test accuracy、train accuracy、overfitting
  2. 汇总 runtime
  3. 保存 summary JSON
```

输出目录：

```text
codes/results/transfer/carte/<mode>/<dataset_tag>/
  run_<id>_seed<seed>_<timestamp>.json
  summary_<num_runs>runs_<timestamp>.json

codes/results/logs/transfer/carte/<mode>/<dataset_tag>/
  run_carte_<mode>_<timestamp>.log
  run_<id>_seed<seed>_<timestamp>.log

codes/results/artifacts/transfer/carte/<mode>/<dataset_tag>/
  run_<id>_seed<seed>/
    carte_joint.pkl     # joint mode
```

并行策略：

```text
TransTab 和 CARTE 的重复 runs 支持 --max_jobs
默认 --max_jobs 1
建议先串行，显存足够再尝试 --max_jobs 2
```

## 4. 基础模型类

当前基础模型类包含：

```text
TabPFN v3
TabICLv2
```

这类方法不进行超参搜索。脚本记录基础模型包版本、模型版本或 checkpoint 路径、每次 run 的 seed、accuracy、macro-F1 和耗时。

### 4.1 TabPFN

入口脚本：

```bash
codes/scripts/run_tabpfn.sh
```

底层 Python 文件：

```bash
codes/baseline/tabpfnv2_extend.py
```

当前调用方式：

```text
tabpfn==8.0.7
TabPFNClassifier.create_default_for_version(ModelVersion.V3)
默认 checkpoint: `/path/to/tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt`
```

默认运行：

```bash
bash codes/scripts/run_tabpfn.sh --gpu 1 --num_runs 10
```

指定数据集：

```bash
bash codes/scripts/run_tabpfn.sh \
  --dataset nctaxi \
  --table_idx 0 \
  --gpu 1 \
  --num_runs 10
```

控制重复实验并行数：

```bash
bash codes/scripts/run_tabpfn.sh --max_jobs 2
```

控制 TabPFN 参数：

```bash
bash codes/scripts/run_tabpfn.sh \
  --n_estimators 8 \
  --model_path /path/to/tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt
```

如果 checkpoint 已存在，`--model_path` 可以省略，TabPFN 会使用默认缓存路径。首次下载 TabPFN v3 权重需要 Prior Labs API key，并且账号需要接受 `tabpfn_3` license。

输出目录：

```text
codes/results/foundation/tabpfn/<dataset_tag>/
  run_<id>_seed<seed>_<timestamp>.json
  summary_<num_runs>runs_<timestamp>.json

codes/results/logs/foundation/tabpfn/<dataset_tag>/
  run_tabpfn_<timestamp>.log
  run_<id>_seed<seed>_<timestamp>.log
```

### 4.2 TabICL

入口脚本：

```bash
codes/scripts/run_tabicl.sh
```

底层 Python 文件：

```bash
codes/baseline/tabicl_v2.py
```

当前调用方式：

```text
tabicl==2.0.2
TabICLClassifier
默认 checkpoint: tabicl-classifier-v2-20260212.ckpt
默认自动下载到 Hugging Face cache
```

默认运行：

```bash
bash codes/scripts/run_tabicl.sh --gpu 1 --num_runs 10
```

指定数据集：

```bash
bash codes/scripts/run_tabicl.sh \
  --dataset agbooks \
  --table_idx 2 \
  --gpu 1 \
  --num_runs 10
```

控制重复实验并行数：

```bash
bash codes/scripts/run_tabicl.sh --max_jobs 2
```

控制 TabICL 参数：

```bash
bash codes/scripts/run_tabicl.sh \
  --n_estimators 4 \
  --batch_size 8
```

控制 checkpoint：

```bash
bash codes/scripts/run_tabicl.sh \
  --checkpoint tabicl-classifier-v2-20260212.ckpt

bash codes/scripts/run_tabicl.sh \
  --model_path /path/to/tabicl-classifier-v2-20260212.ckpt \
  --no_auto_download
```

输出目录：

```text
codes/results/foundation/tabicl/<dataset_tag>/
  run_<id>_seed<seed>_<timestamp>.json
  summary_<num_runs>runs_<timestamp>.json

codes/results/logs/foundation/tabicl/<dataset_tag>/
  run_tabicl_<timestamp>.log
  run_<id>_seed<seed>_<timestamp>.log
```

并行策略：

```text
TabPFN 和 TabICL 的重复 runs 支持 --max_jobs
默认 --max_jobs 1
同一 GPU 上并行过多可能 OOM，建议先尝试 --max_jobs 2
```

## 推荐执行顺序

如果目标是完整跑 `mstraffic[0]`：

```bash
# 1. 决策树类
bash codes/scripts/run_tree_models.sh --dataset mstraffic --table_idx 0 --device 1 --num_runs 10

# 2. TNNS 深度表格模型类
bash codes/scripts/run_nn_grid_search.sh --dataset mstraffic --table_idx 0 --gpu 1 --num_runs 10 --num_tasks 2

# 3. 迁移学习类
bash codes/scripts/run_transtab.sh --mode transfer --dataset mstraffic --table_idx 0 --aux_dataset mstraffic --aux_table_idx 1 --gpu 1 --num_runs 10 --max_jobs 1
bash codes/scripts/run_carte.sh --gpu 1 --num_runs 10 --max_jobs 1

# 4. 基础模型类
bash codes/scripts/run_tabpfn.sh --dataset mstraffic --table_idx 0 --gpu 1 --num_runs 10 --max_jobs 1
bash codes/scripts/run_tabicl.sh --dataset mstraffic --table_idx 0 --gpu 1 --num_runs 10 --max_jobs 1
```

## 注意事项

1. `--max_jobs` 是同一张 GPU 上并发重复实验，显存不足时会 OOM。
2. TNNS 的 `--num_tasks` 是并行切分 grid search，不是 final runs 的并行数。
3. 决策树默认外层串行，内部 tree learner 多线程。
4. 基础模型类不保存新训练权重，只记录基础模型版本或 checkpoint path。
5. CARTE 依赖 FastText 模型：

```text
codes/lib/FastText/cc.en.300.bin
```

6. TabPFN v3 默认 checkpoint 缓存位置：

```text
/path/to/tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt
```

7. TabICLv2 默认 checkpoint 名称：

```text
tabicl-classifier-v2-20260212.ckpt
```
