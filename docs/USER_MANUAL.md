# DeepMapper user manual

DeepMapper analyses high-dimensional data (single-cell RNA-seq and other omics)
without throwing features away. It keeps every feature, lays the feature vector out
as a small pseudo-image, reads it with a configurable backbone over several feature
arrangements, and attributes the result back to named features. The features that
matter only in combination show up as gene chords: sets of genes that separate a
state together while each one alone looks weak.

This manual covers installation, the core workflow, configuration, data loading,
evaluation, attribution, and how to reproduce the paper.

## Contents

1. [Install](#install)
2. [The core idea](#the-core-idea)
3. [Quickstart](#quickstart)
4. [Configuration](#configuration)
5. [Backbones](#backbones)
6. [Reading the results](#reading-the-results)
7. [The linear baseline](#the-linear-baseline)
8. [Loading data](#loading-data)
9. [Held-out evaluation](#held-out-evaluation)
10. [Attribution](#attribution)
11. [Early stopping](#early-stopping)
12. [De-novo transcript recovery (optional)](#de-novo-transcript-recovery-optional)
13. [Package layout](#package-layout)
14. [Reproducing the paper](#reproducing-the-paper)
15. [Practical notes](#practical-notes)
16. [Using DeepMapper with Claude](#using-deepmapper-with-claude)

## Install

```bash
pip install pydeepmapper                 # core engine: run(X, y) + linear/mlp/cnn + attribution
pip install "pydeepmapper[io]"           # + 10x / h5ad loaders (scanpy, anndata)
pip install "pydeepmapper[backbones]"    # + ResNet / ViT / timm backbones
pip install "pydeepmapper[denovo]"       # + de-novo ingestion (biopython; external tools below)
pip install "pydeepmapper[all]"          # everything
```

From a clone, install editable with the dev extras and run the tests:

```bash
pip install -e ".[dev]"
python -m pytest
```

The pure core (the `transform`, `accumulate`, `earlystop` and de-novo namespacing
logic) imports with only the standard library, so those tests run without numpy or
torch. Python 3.9 or newer.

## The core idea

A standard scRNA-seq pipeline filters cells, selects a few thousand highly variable
genes, compresses them with PCA, and clusters. That is built to find the largest
sources of variation, so it discards low-amplitude, distributed signal. DeepMapper
does the opposite:

1. **Keep every feature.** No HVG selection, no PCA. The input is the full matrix.
2. **Arrange and read.** Each feature maps to one pixel of a small square image. A
   seeded permutation gives one arrangement; the backbone is trained on it.
3. **Repeat N times.** Different arrangements average out the layout, so the result
   does not depend on which gene landed next to which.
4. **Attribute back.** Per-pixel attribution maps back to per-feature importance
   through the inverse permutation, a clean one-pixel-per-feature gather.
5. **Accumulate.** Per-pass importances become a ranking plus stability statistics
   (selection frequency, median importance, Borda mean rank).

## Quickstart

```python
import numpy as np
from pydeepmapper.config import DeepMapperConfig, BackboneSpec
from pydeepmapper.runner import run

X = ...            # (n_cells, n_genes) expression matrix, UNFILTERED (no HVG, no PCA)
y = ...            # integer class labels, shape (n_cells,)
genes = [...]      # feature names, len == X.shape[1]

cfg = DeepMapperConfig(n_passes=3, backbone=BackboneSpec(kind="cnn_small"))
findings = run(X, y, cfg, feature_names=genes)

for name, freq, importance in findings.ranking(20):
    print(name, round(freq, 3), round(importance, 4))
```

`run` returns a `Findings` object. `findings.ranking(20)` gives the top 20 features
as `(name, selection_frequency, median_importance)` tuples.

## Configuration

Everything is a `DeepMapperConfig` (a dataclass). The fields you reach for most:

| Field | Default | Meaning |
|---|---|---|
| `n_passes` | 30 | number of arrangements to accumulate over |
| `top_k` | 20 | top-k cut for the selection-frequency stability statistic |
| `epochs` | 20 | max epochs per pass; early stopping ends a pass sooner at a plateau |
| `batch_size` | 64 | training batch size |
| `lr` | 1e-3 | learning rate |
| `test_size` | 0.25 | held-out fraction for the reported accuracy |
| `min_accuracy` | 0.0 | per-pass quality gate; a pass below this is not accumulated |
| `index_features` | False | True fixes the identity arrangement (no shuffle) |
| `seed` | 0 | base seed; pass t uses `seed + t`, so runs are reproducible |
| `backbone` | `BackboneSpec()` | which model sits behind the pseudo-image |

Early-stopping knobs (see [Early stopping](#early-stopping)): `early_stop` (default
True), `patience` (20), `min_epochs` (40), `min_delta` (1e-3), `val_size` (0.15).

Attribution knobs: `attribution` (default `"integrated_gradients"`),
`attribution_samples` (64, caps the test samples fed to attribution to bound memory),
`ig_steps` (32), `ig_internal_batch` (16).

## Backbones

Set `backbone=BackboneSpec(kind=...)`. Valid kinds:

| kind | What it is |
|---|---|
| `cnn_small` | tiny 3-conv net. Default. Fast and strong on small pseudo-images |
| `cnn` | a larger convolutional net |
| `mlp` | plain multilayer perceptron |
| `linear` | logistic regression on the full feature set, deterministic, single pass |
| `resnet18` | torchvision ResNet-18 (legacy parity baseline) |
| `vit_cct` | Compact Convolutional Transformer (the ViT pick for pseudo-images) |
| `timm:<name>` | any timm model, e.g. `timm:mobilevit_xxs`, `timm:convnext_nano` |
| `conv_vae` | convolutional autoencoder, adds a latent space for clustering / anomaly |

`BackboneSpec` also takes `img_size` (0 derives it from the feature count),
`in_chans` (1), `pretrained` (False; ImageNet weights do not transfer to
pseudo-images), and `extra` (a dict passed to the builder).

The image backbones need the `[backbones]` extra. The default `cnn_small` needs only
torch.

## Reading the results

`run` returns a `Findings` dataclass:

- `n_features`, `n_passes_run`, `n_accepted`: run bookkeeping.
- `accuracies`: per-pass held-out accuracy.
- `stop_epochs`: the epoch each pass stopped at (with early stopping on).
- `median_importance`, `mean_rank`, `selection_frequency`: per-feature statistics,
  each a list of length `n_features`.
- `feature_names`: the names you passed, or None.
- `ranking(top=20)`: the top features by selection frequency (then median
  importance), as `(name_or_index, selection_frequency, median_importance)` tuples.

A feature is worth trusting when it has both a high selection frequency (it is picked
in most passes) and a high median importance.

## The linear baseline

For a fast, exact, deterministic check with no imagification and no passes:

```python
from pydeepmapper import linear_baseline

clf, scaler, ranking = linear_baseline.fit(X, y, feature_names=genes)
top = linear_baseline.top_genes(ranking, 20)
```

`fit(X, y, feature_names=None, C=1.0, max_iter=5000)` returns the fitted classifier,
the scaler, and a ranking of `(name, importance)` sorted descending, where importance
is the largest standardised coefficient across classes. In the paper the linear
backbone closely matches the CNN, which is the point: the signal is real, not an
artefact of a heavy model.

## Loading data

The `[io]` extra adds loaders that return a `Dataset` (`X`, `y`, `var_names`,
`label_names`).

```python
from pydeepmapper.io import load_h5ad, load_10x_populations

# one AnnData file with a label column in .obs
ds = load_h5ad("pbmc.h5ad", label_key="cell_type", normalize=True, target_sum=1e4)

# several FACS-sorted 10x matrices merged into one labelled set
ds = load_10x_populations(
    [("data/cd4_naive/", "naive"), ("data/cd4_memory/", "memory")],
    n_per_class=2000, seed=0,
)

findings = run(ds.X, ds.y, cfg, feature_names=ds.var_names)
```

`load_10x_populations` concatenates the matrices on their shared genes (inner join).
`normalize=True` applies normalize-total plus log1p.

Helpers for the experiments in the paper:

- `highly_variable_subset(ds, n_top_genes=2000)`: HVG-filter a Dataset and return the
  kept gene indices, to measure what dimension reduction throws away.
- `intersect_genes(a, b)`: restrict two datasets to shared genes (the common-genes
  interferon analysis).
- `exclusive_genes(a, b)`: the genes unique to each dataset.
- `export_for_seurat(ds, outdir)`: write `counts.csv` + `labels.csv` for an external
  Seurat comparison.

## Held-out evaluation

When you want the final class call and a confusion matrix rather than a feature
ranking, use `evaluate`, which ensembles the N arrangements into one prediction:

```python
from pydeepmapper.evaluate import evaluate

res = evaluate(X, y, cfg, class_names=ds.label_names, test_size=0.25, seed=0)
print(res.accuracy, res.macro_f1)
print(res.report)            # sklearn classification report as a dict
```

`EvalResult` carries `y_true`, `y_pred`, `y_proba` (ensembled softmax), `classes`,
`accuracy`, `macro_f1`, `per_pass_acc`, and `report`.

## Attribution

`run` attributes for you. To attribute a model directly, call
`attribution.feature_importances(model, X_images, targets, method, perm, n_features,
baseline=None, ig_steps=32, ig_internal_batch=16)`. It returns a per-feature
importance vector. The method names: `integrated_gradients` (default), `saliency`,
`occlusion` for CNNs (via Captum), and `attention_rollout`, `chefer` for ViTs. torch
and captum are imported lazily; a missing dependency raises `AttributionUnavailable`.

Integrated gradients can be memory-hungry. Bound it with `attribution_samples`,
`ig_steps`, and `ig_internal_batch` in the config.

## Early stopping

DeepMapper trains each pass to a plateau rather than for a fixed number of epochs.
`EarlyStopper` monitors the held-out loss carved from the training split (size
`val_size`), and stops when the loss has not improved by `min_delta` for `patience`
epochs, but never before `min_epochs` (the floor keeps a pass out of the
undertrained zone). The best weights are restored, so the reported accuracy and the
attribution come from the held-out optimum, not the last epoch. `epochs` is the
maximum cap. Turn it off with `early_stop=False`.

## De-novo transcript recovery (optional)

The `[denovo]` extra builds the feature matrix from raw reads, recovering transcripts
the reference annotation never lists and folding them in, so DeepMapper can use
features a reference-only pipeline discards.

```python
from pydeepmapper import hybrid_assembly

table = hybrid_assembly.run_hybrid_assembly(
    reads_by_sample, hisat2_index, reference_fasta, work_dir="work/",
)
```

The pipeline aligns reads, captures the unaligned fraction, assembles it de novo with
Trinity, tags the novel transcripts `DENOVO_*`, builds a hybrid Salmon index,
quantifies, and assembles a transcripts-by-samples table. The pure helpers
(`is_denovo`, `tag_denovo`, `merge_reference_and_denovo`, `build_sc_table`,
`denovo_fraction`) need no external tools. The orchestration needs biopython plus
hisat2, Trinity, and salmon on PATH:

```bash
conda install -c bioconda hisat2 trinity salmon
```

A missing tool raises `BioToolUnavailable`.

## Package layout

| Module | What it holds |
|---|---|
| `transform` | feature-to-pixel mapping; seeded permutations; back-projection. Pure |
| `accumulate` | per-pass aggregation: selection frequency, mean rank, median importance, stability bound. Pure |
| `config` | `DeepMapperConfig`, `BackboneSpec`, `AugmentConfig`. Pure |
| `runner` | the iterate-N-accumulate loop; returns `Findings` |
| `backbones` | the swappable model registry behind one `build()` |
| `attribution` | per-pixel attribution to per-feature importance |
| `evaluate` | held-out ensembled prediction and metrics |
| `linear_baseline` | the deterministic linear backbone |
| `io` | 10x / h5ad loaders and dataset helpers |
| `earlystop` | plateau early-stopping logic |
| `hybrid_assembly` | the optional de-novo ingestion path |
| `augment`, `plots`, `unsupervised` | light augmentation, figures, unsupervised helpers |

"Pure" modules import with the standard library only; the heavy dependencies load
lazily, so importing the package never forces torch.

## Reproducing the paper

The `bench/` folder holds the scripts that produce the figures and quantitative
claims, each one driven by a public dataset. `docs/REPRODUCIBILITY.md` maps every
figure and result to its script and lists the dataset accessions. `bench/README.md`
has the run instructions. In short:

```bash
pip install -e ".[all]"
bash scripts/download_data.sh
python bench/ribosomal_validation.py     # Fig 1, and so on
```

## Practical notes

- **Determinism.** The arrangements are seeded from `config.seed`, so a re-run with
  the same seed reproduces the ranking. The `linear` backbone is exactly
  deterministic.
- **GPU.** Training and attribution want torch with Apple MPS or CUDA. CPU works but
  is slow. On MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Memory.** On large gene sets, keep attribution bounded with `attribution_samples`
  and `ig_internal_batch`. The ResNet ImageNet stem downsamples hard and can destroy
  single-pixel gene signal, so `cnn_small` is the safe default for sparse data.
- **How many passes.** More passes give a more stable ranking. Grow `n_passes` until
  the top-k membership stops changing between two halves of the runs.

## Using DeepMapper with Claude

The repository ships a [Claude Code](https://claude.com/claude-code) skill at
`.claude/skills/deepmapper/SKILL.md`. Open the repo in Claude Code and ask in plain
language, for example "analyse this h5ad with DeepMapper and tell me which genes
separate the states" or "reproduce Figure 1". Claude loads the skill and walks the
workflow: it checks the install, loads your data with `pydeepmapper.io`, runs the fast
deterministic linear baseline first, then runs `pydeepmapper.runner.run`, and reads
`findings.ranking(...)`.

The skill follows the same honest-analysis rules as this manual: keep every feature,
run the linear baseline before the CNN, report a finding as a module rather than a
fixed gene list, default to `cnn_small`, and do not over-read one run. You can invoke
it explicitly with `/deepmapper`, or edit the Markdown to add your own datasets and
defaults. See [Using DeepMapper with Claude](claude-skill.md) for the full page.

## Citation

If you use DeepMapper, please cite the article in `CITATION.cff`:

> Ersavas T., Smith M.A., Mattick J.S. (2024). Novel applications of Convolutional
> Neural Networks in the age of Transformers. Scientific Reports 14.
> https://doi.org/10.1038/s41598-024-60709-z
