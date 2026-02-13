# LakeMLB: Data Lake Machine Learning Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2602.10441-b31b1b.svg)](https://arxiv.org/abs/2602.10441)

**Official implementation of the LakeMLB benchmark**

> **LakeMLB: Data Lake Machine Learning Benchmark**  
> Feiyu Pan, Tianbin Zhang, Aoqian Zhang, Yu Sun, Zheng Wang, Lixing Chen, Li Pan, Jianhua Li  
> *arXiv preprint arXiv:2602.10441, 2026*  
> [[Paper](https://arxiv.org/abs/2602.10441)]

---

## About

**LakeMLB** is a standardized benchmark for evaluating machine learning methods on multi-table scenarios in data lake environments. It addresses the critical challenge of leveraging weakly-associated heterogeneous tables to improve downstream ML performance.

### Key Features

- **Real-world datasets**: Six curated datasets covering finance, government, and e-commerce domains
- **Two core scenarios**: Join-based and union-based multi-table integration
- **Standardized evaluation**: Fixed train/validation/test splits for reproducibility
- **Comprehensive baselines**: 12+ methods including tree models, neural networks, transfer learning, and foundation models
- **Augmentation support**: Mapping files for Feature Augmentation (FA) and guidelines for Data Augmentation (DA)

---

## Repository Structure

```
LakeMLB/
├── benckmark/              # Benchmark datasets
│   ├── join_based/         # dsmusic, lhstocks, nnstocks
│   └── union_based/        # gacars, mstraffic, ncbuilding
├── codes/                  # Implementation code
│   ├── baseline/           # Model implementations
│   ├── scripts/             # Experiment scripts
│   └── lib/                 # Modified third-party libraries
└── benchmark_details.png   # Dataset statistics figure
```

---

## Datasets

### Statistics

![Dataset Statistics](benchmark_details.png)

All datasets are located in `benckmark/` with two categories:

- **Join-based** (`join_based/`): dsmusic, lhstocks, nnstocks
- **Union-based** (`union_based/`): gacars, mstraffic, ncbuilding

Each dataset includes:
- Source CSV files for each table
- Pre-computed split masks (`mask.pt`) for train/validation/test splits
- **Mapping file** (`mapping.csv`) for Feature Augmentation (FA) strategy
- Detailed documentation (`README.md`)

### Augmentation Strategies

To reproduce the augmentation experiments in the paper:

**Feature Augmentation (FA)**: Each dataset includes a `mapping.csv` file that provides row index correspondences between two tables for feature joining. The mapping contains:
- `T1_index`, `T2_index`: Row indices for table matching
- `cosine_similarity`, `cosine_distance`: Similarity metrics for the matched pairs

Users can use this mapping to perform feature augmentation by joining tables based on the provided correspondences.

**Data Augmentation (DA)**: Implemented via vertical concatenation (union) of tables. See the paper for detailed implementation specifications.

---

## Baselines

We provide implementations of 12+ methods across four categories:

| Category | Methods | Entry Point |
|----------|---------|-------------|
| **Tree-based** | XGBoost, CatBoost, LightGBM | `codes/baseline/tree_models.py` |
| **Neural networks** | FT-Transformer, TabTransformer, ExcelFormer, SAINT, TromptNet | `codes/baseline/tnns_test.py` |
| **Transfer learning** | TransTab, CARTE | `codes/baseline/transtab_*.py`, `codes/baseline/carte_*.py` |
| **Foundation models** | TabPFN v2, TabICL | `codes/baseline/tabpfnv2.py`, `codes/baseline/tabicl_clf.py` |

---

## Quick Start

Run all baseline methods with default hyperparameters:

```bash
bash codes/scripts/example.sh
```

For systematic evaluation:

```bash
# Tree models (CPU)
bash codes/scripts/run_tree_models.sh

# Neural networks (GPU)
bash codes/scripts/run_nn_grid_search.sh

# Transfer learning (GPU)
bash codes/scripts/run_transtab.sh
bash codes/scripts/run_carte.sh

# Foundation models (GPU)
bash codes/scripts/run_fundation_models.sh
```

---

## Requirements

### Dependencies

Modified third-party libraries are bundled in `codes/lib/` with unified data loading and standardized preprocessing:

- **rllm**: Our team's open-source library for tabular learning ([rllm-team/rllm](https://github.com/rllm-team/rllm))
  - To reproduce experiments, clone the repository and place it in `codes/lib/rllm/`
  - ```bash
    git clone https://github.com/rllm-team/rllm.git codes/lib/rllm
    ```
- **transtab**, **carte_ai**: Modified versions for benchmark compatibility

### External Resources

**CARTE** requires FastText embeddings:
- Download [`cc.en.300.bin.gz`](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)
- Extract and place at `codes/lib/FastText/cc.en.300.bin`

**TabICL** requires model checkpoint (103 MB, not included):
- Obtain `tabicl-classifier-v1.1-0506.ckpt`
- Place at `codes/lib/huggingface/hub/models--jingang--TabICL-clf/snapshots/main/tabicl-classifier-v1.1-0506.ckpt`

---

## Citation

If you find LakeMLB useful in your research, please cite:

```bibtex
@article{pan2026lakemlb,
  title={LakeMLB: Data Lake Machine Learning Benchmark},
  author={Pan, Feiyu and Zhang, Tianbin and Zhang, Aoqian and Sun, Yu and Wang, Zheng and Chen, Lixing and Pan, Li and Li, Jianhua},
  journal={arXiv preprint arXiv:2602.10441},
  year={2026}
}
```

---

## License

This project is provided for academic and research purposes. Refer to individual dataset README files for data provenance and licensing details.
