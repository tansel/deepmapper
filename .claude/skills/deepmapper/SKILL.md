---
name: deepmapper
description: Run a DeepMapper analysis with Claude. Use this when the user wants to analyse a single-cell RNA-seq or other high-dimensional omics matrix WITHOUT dimension reduction, find the genes that separate a cell state (including distributed "gene chords" that only matter in combination), attribute a classifier back to named features, compare against a linear baseline, or reproduce a figure from the DeepMapper paper. Triggers on "analyse this with DeepMapper", "find gene chords", "which genes separate these states", "no-filtering / keep every feature", "attribute back to genes", or "reproduce the DeepMapper figure".
---

# Using DeepMapper

DeepMapper keeps every feature (no HVG selection, no PCA), lays the feature vector
out as a small pseudo-image, reads it with a configurable backbone over several
seeded arrangements, and attributes the result back to named features. It surfaces
gene chords: sets of genes that separate a state together while each one alone looks
weak.

Follow this workflow. Read the real signatures in `docs/USER_MANUAL.md` or the API
reference; do not invent parameters.

## 1. Confirm the environment

```bash
python -c "import pydeepmapper; print('ok')" || pip install "deepmapper[all]"
```

The pure core imports with the standard library only. Loading 10x / h5ad data needs
the `[io]` extra; the image backbones need `[backbones]`.

## 2. Get the matrix and labels

Ask the user for the data if it is not already in hand. Do not filter or run PCA;
DeepMapper wants the full matrix.

```python
from pydeepmapper.io import load_h5ad, load_10x_populations

ds = load_h5ad("data.h5ad", label_key="cell_type")          # one AnnData file
# or merge FACS-sorted 10x matrices:
ds = load_10x_populations([("data/naive/", "naive"),
                           ("data/memory/", "memory")], n_per_class=2000, seed=0)
X, y, genes = ds.X, ds.y, ds.var_names
```

If the user hands you a plain matrix, use it directly: `X` is `(n_cells, n_genes)`,
`y` is integer labels, `genes` is the feature names.

## 3. Run a fast linear check first

The linear backbone is exact, deterministic, and quick. It tells you immediately
whether the signal is there before spending time on the CNN.

```python
from pydeepmapper import linear_baseline
clf, scaler, ranking = linear_baseline.fit(X, y, feature_names=genes)
print(linear_baseline.top_genes(ranking, 20))
```

## 4. Run DeepMapper and read the chord

```python
from pydeepmapper.config import DeepMapperConfig, BackboneSpec
from pydeepmapper.runner import run

cfg = DeepMapperConfig(n_passes=3, backbone=BackboneSpec(kind="cnn_small"))
findings = run(X, y, cfg, feature_names=genes)

for name, freq, importance in findings.ranking(20):
    print(name, round(freq, 3), round(importance, 4))
```

Start with `n_passes=3` for a quick look. For a publishable ranking raise it (10 to
30) until the top-k membership stops changing between two halves of the runs.

## 5. Interpret honestly

- A feature is trustworthy only when it has BOTH a high selection frequency (picked
  in most passes) and a high median importance.
- The CNN ranking should roughly agree with the linear ranking. If it does not, the
  model is probably undertrained; check `findings.accuracies` and `stop_epochs`.
- Report the ribosomal / interferon / lncRNA finding as a MODULE (a set of
  interchangeable genes), not a fixed gene list. The moderate gene-level overlap
  across seeds is the gene-chord point, not noise.
- Do not over-read one run. Re-run with more passes or seeds before making a claim.

## 6. Held-out evaluation (when you need a class call)

```python
from pydeepmapper.evaluate import evaluate
res = evaluate(X, y, cfg, class_names=ds.label_names)
print(res.accuracy, res.macro_f1)
```

## Reproducing a paper figure

`docs/REPRODUCIBILITY.md` maps every figure to its script. Install, fetch the public
data, then run the script from the repo root:

```bash
pip install -e ".[all]"
bash scripts/download_data.sh
python bench/ribosomal_validation.py     # Fig 1, and so on
```

## Pitfalls to avoid

- Do not select highly variable genes or run PCA first. That throws away the
  distributed signal DeepMapper exists to recover.
- Do not use `resnet18` on sparse single-cell data. Its ImageNet stem downsamples
  hard and destroys the single-pixel gene signal. Use `cnn_small` (the default).
- On Apple GPUs set `PYTORCH_ENABLE_MPS_FALLBACK=1` and keep attribution bounded with
  `attribution_samples` and `ig_internal_batch` on large gene sets.
- Keep runs reproducible by fixing `config.seed`; the arrangements are seeded from it.
