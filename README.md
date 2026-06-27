# DeepMapper

**No-filtering, attribute-back analysis of high-dimensional omics.** DeepMapper keeps
every feature (no highly-variable-gene selection, no dimension reduction), reads the
full matrix with a configurable backbone over several feature arrangements, and
attributes each result back to named features. That surfaces distributed *gene chords*,
sets of genes that separate a cell state only together, which standard pipelines
discard. An optional de-novo step recovers transcripts absent from the reference
annotation.

## Install

```bash
pip install deepmapper                 # core engine: run(X, y) + linear/mlp/cnn + attribution
pip install "deepmapper[io]"           # + 10x / h5ad loaders (scanpy, anndata)
pip install "deepmapper[backbones]"    # + ResNet / ViT / timm backbones
pip install "deepmapper[denovo]"       # + de-novo ingestion (biopython; external tools below)
pip install "deepmapper[all]"          # everything
```

From a clone:

```bash
git clone https://github.com/tansel/deepmapper.git
cd deepmapper
pip install -e ".[dev]"
python -m pytest
```

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
    print(name, round(freq, 3), round(importance, 4))   # the gene chord, ranked
```

Swap the backbone with `BackboneSpec(kind=...)`: `linear`, `mlp`, `cnn_small`
(default), `cnn`, `resnet18`, `vit_cct`, `timm:<name>`, or `conv_vae`. For a fast,
exact linear check:

```python
from pydeepmapper import linear_baseline
clf, scaler, ranking = linear_baseline.fit(X, y, feature_names=genes)
```

## How it works

1. Keep every feature. No HVG, no PCA.
2. Lay the feature vector out as a small pseudo-image, one pixel per feature, under a
   seeded permutation.
3. Train the backbone, attribute per pixel, project back to per-feature importance.
4. Repeat over N arrangements and accumulate into a ranking plus stability statistics.

## Documentation

- **[User manual](docs/USER_MANUAL.md)** is the full guide: configuration, data
  loading, evaluation, attribution, early stopping, de-novo recovery, and the package
  layout.
- **[Reproducibility map](docs/REPRODUCIBILITY.md)** maps every paper figure to its
  script and dataset.
- **[Data sources](docs/data-sources.md)** lists the dataset accessions.
- **[bench/](bench/)** holds the paper analysis scripts.

## Reproducing the paper

```bash
pip install -e ".[all]"
bash scripts/download_data.sh
python bench/ribosomal_validation.py     # Fig 1, and so on (see docs/REPRODUCIBILITY.md)
```

## Citation

If you use DeepMapper, please cite the peer-reviewed article (see
[`CITATION.cff`](CITATION.cff)):

> Ersavas T., Smith M.A., Mattick J.S. (2024). *Novel applications of Convolutional
> Neural Networks in the age of Transformers.* **Scientific Reports** 14.
> https://doi.org/10.1038/s41598-024-60709-z

## License

Licensed under the **Apache License, Version 2.0**, see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). Apache-2.0 is permissive (commercial use, modification, and
redistribution allowed) and adds an explicit patent grant covering the method.
