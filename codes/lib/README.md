# lib

Among the 13 baselines evaluated in this study, fair comparison required local
modifications to the `rllm` library used by the TNN methods and to the
third-party code used by the transfer-based TransTab and CARTE methods. Due to
the anonymous artifact size limit, this directory includes only the code files
that we modified; the remaining upstream code can be obtained from public
resources.

Modified source code of third-party libraries (`transtab`, `carte_ai`, etc.).

To ensure fair and reproducible experiments, we patched these libraries to use **unified data loading**, **fixed train/test splits**, and **standardized data preprocessing**. The modified code is kept here instead of modifying the installed packages directly.

## Anonymous artifact packaging

The anonymous submission has a strict size limit. Therefore, this directory
retains only the locally modified portions of third-party libraries that are
needed by the experiment code. Large datasets, downloaded model caches, and
publicly available pretrained weights are intentionally excluded. They can be
restored from the public sources listed below.

In particular, `carte_ai/data/` is not included in the anonymous artifact. It
previously contained local raw and preprocessed dataset copies
(`data_raw/`, `data_singletable/`, and auxiliary files under `etc/`). These
files are experiment data rather than modified third-party source code.

## Excluded public model assets

### FastText

The following file is excluded:

- `FastText/cc.en.300.bin` (approximately 6.8 GiB)
- SHA-256:
  `14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca`

Download the official compressed English Common Crawl vectors from:

- Documentation: <https://fasttext.cc/docs/en/crawl-vectors>
- Direct download:
  <https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz>

After extraction, place the file at
`codes/lib/FastText/cc.en.300.bin`.

### TabICLv2 checkpoint

The local Hugging Face cache is excluded from the artifact. The TabICL
experiments use `tabicl==2.0.2` and its default TabICLv2 classifier checkpoint:

- Repository: [`jingang/TabICL`](https://huggingface.co/jingang/TabICL)
- Checkpoint: `tabicl-classifier-v2-20260212.ckpt` (110,368,038 bytes)
- SHA-256:
  `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
- Direct download:
  <https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt?download=true>

By default, `run_tabicl.sh` downloads this checkpoint to the standard
Hugging Face cache. A manually downloaded checkpoint can instead be supplied
with `--model_path /path/to/tabicl-classifier-v2-20260212.ckpt`; combine this
with `--no_auto_download` to require local-only execution.


### Other local Hugging Face model copies

The removed `models/` directory contained these public model snapshots:

- [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased),
  previously stored at `models/bert-base-uncased/`. Its
  `model.safetensors` SHA-256 was
  `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.
- [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  previously stored at `models/all-MiniLM-L6-v2/`. Its
  `model.safetensors` SHA-256 was
  `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

These snapshots can be restored with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir="codes/lib/models/bert-base-uncased",
)
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="codes/lib/models/all-MiniLM-L6-v2",
)
```
