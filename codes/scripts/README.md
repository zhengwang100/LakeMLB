# Scripts

Shell scripts for running baseline experiments. All Python source files are in `../baseline/`.

| Script | Description |
|---|---|
| `example.sh` | Quick-start examples showing how to run each method with minimal arguments. |
| `run_tree_models.sh` | Grid search + repeated evaluation for XGBoost, CatBoost, and LightGBM. |
| `run_nn_grid_search.sh` | Parallel grid search (memory-efficient) for tabular neural networks (FTTransformer, TabTransformer, ExcelFormer, SAINT, TromptNet). |
| `run_transtab.sh` | Repeated runs of TransTab with concurrency control. |
| `run_carte.sh` | Run CARTE single-table and multi-table experiments. Supports mode selection via argument (`single`, `multi`, or both). |
| `run_fundation_models.sh` | Repeated runs of foundation models (TabPFN v2, TabICL) with accuracy statistics. |

Results and logs are saved under `../results/`.

## Note

CARTE methods (`carte_single.py`, `carte_joint.py`) require the FastText model `cc.en.300.bin`.
Download it and place it at `../lib/FastText/cc.en.300.bin` before running.

Download: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz

TabICL methods (`tabicl_clf.py`) require the model checkpoint `tabicl-classifier-v1.1-0506.ckpt`.
Due to its large file size (103 MB), it is not included in this repository.
Please obtain the checkpoint and place it at `../lib/huggingface/hub/models--jingang--TabICL-clf/snapshots/main/tabicl-classifier-v1.1-0506.ckpt` before running.

Detailed grid search ranges and hyperparameter settings are described in the paper appendix.

