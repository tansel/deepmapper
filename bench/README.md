# Paper analysis scripts

These are the scripts that reproduce the figures and quantitative claims in the
DeepMapper T-cell paper. Each one consumes a public dataset, runs the DeepMapper
pipeline (or a control), and writes a JSON or CSV result. `docs/REPRODUCIBILITY.md`
maps every figure and result to the script that makes it.

The scripts are not part of the importable `pydeepmapper` package. They are research
drivers: install the package first, fetch the data, then run a script from the repo
root.

## Setup

```bash
pip install -e ".[all]"          # the package + io/backbones/denovo extras
bash scripts/download_data.sh    # fetch the public 10x / SMART-seq datasets
```

See `docs/data-sources.md` for the dataset accessions and where each lands on disk.

## Run

```bash
# from the repo root, so the relative paths resolve
python bench/ribosomal_validation.py        # Fig 1
python bench/hvg_ribosomal_rank.py          # Fig 2
python bench/review_controls.py             # depth / cell-cycle / effectorness controls
python bench/backbone_headtohead.py         # linear vs cnn
python bench/mouse_dm_run.py --help         # mouse cross-species run (phases + gates)
```

GPU steps (DeepMapper training and attribution) want torch with Apple MPS or CUDA.
The sklearn-only controls run on CPU in seconds. A typical invocation:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python bench/review_controls.py
```

Scripts not in this folder were left out on purpose: they are either internal
tooling or probes that did not make the paper.
