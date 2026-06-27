# DeepMapper

**No-filtering, attribute-back analysis of high-dimensional omics.**

DeepMapper keeps every feature (no highly-variable-gene selection, no dimension
reduction), reads the full matrix with a configurable backbone over several feature
arrangements, and attributes each result back to named features. That surfaces
distributed *gene chords*, sets of genes that separate a cell state only together,
which standard pipelines discard. An optional de-novo step recovers transcripts
absent from the reference annotation.

## Install

```bash
pip install pydeepmapper            # core engine
pip install "pydeepmapper[all]"     # + io, backbones, de-novo
```

## Quickstart

```python
from pydeepmapper.config import DeepMapperConfig, BackboneSpec
from pydeepmapper.runner import run

cfg = DeepMapperConfig(n_passes=3, backbone=BackboneSpec(kind="cnn_small"))
findings = run(X, y, cfg, feature_names=genes)   # X unfiltered, y integer labels
for name, freq, importance in findings.ranking(20):
    print(name, round(freq, 3), round(importance, 4))
```

## Where to go next

- **[User manual](USER_MANUAL.md)** is the full guide: configuration, data loading,
  evaluation, attribution, early stopping, de-novo recovery, package layout.
- **[API reference](api.md)** is generated from the docstrings.
- **[Reproducing the paper](REPRODUCIBILITY.md)** maps every figure to its script.
- **[Using DeepMapper with Claude](claude-skill.md)** drives an analysis through the
  bundled Claude Code skill.

## Citation

> Ersavas T., Smith M.A., Mattick J.S. (2024). Novel applications of Convolutional
> Neural Networks in the age of Transformers. Scientific Reports 14.
> https://doi.org/10.1038/s41598-024-60709-z
