"""Data intake for real scRNA-seq benchmarks (AnnData / .h5ad).

Turns an AnnData into the plain ``(X, y, var_names, label_names)`` the DeepMapper
runner consumes, and provides the preprocessing parity helpers the paper cases need:
log-normalization, highly-variable-gene selection (to *measure what filtering costs*),
and the gene-**intersection** of two datasets (the stim-vs-unstim "common genes only"
case, C5). scanpy/anndata are imported lazily, importing this module is cheap.

See docs/paper-cases.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Dataset:
    """A benchmark-ready matrix: cells × genes, integer labels, names."""
    X: "object"                     # numpy ndarray (n_cells, n_genes), float32
    y: "object"                     # numpy ndarray (n_cells,), int64
    var_names: List[str]            # gene names (len n_genes)
    label_names: List[str]          # class index -> label string
    obs_label_key: str = ""

    @property
    def n_cells(self): return self.X.shape[0]
    @property
    def n_genes(self): return self.X.shape[1]


def _densify(X):
    import numpy as np
    return np.asarray(X.todense(), dtype=np.float32) if hasattr(X, "todense") \
        else np.asarray(X, dtype=np.float32)


def load_h5ad(path: str, label_key: str, normalize: bool = True,
              target_sum: float = 1e4) -> Dataset:
    """Load an AnnData ``.h5ad`` → :class:`Dataset`. ``label_key`` is the ``obs`` column
    holding the ground-truth class. ``normalize`` applies ``normalize_total`` + ``log1p``
    (skip if the matrix is already normalized)."""
    import numpy as np
    import anndata as ad

    a = ad.read_h5ad(path)
    if normalize:
        a = _lognorm(a, target_sum)
    X = _densify(a.X)
    labels = a.obs[label_key].astype("category")
    y = labels.cat.codes.to_numpy().astype(np.int64)
    return Dataset(X=X, y=y, var_names=list(map(str, a.var_names)),
                   label_names=list(map(str, labels.cat.categories)),
                   obs_label_key=label_key)


def _lognorm(a, target_sum=1e4):
    import scanpy as sc
    a = a.copy()
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    return a


def _find_mtx_dir(path: str) -> str:
    """A CellRanger mtx dir holds matrix.mtx(.gz) + genes/features.tsv + barcodes.tsv.
    The Zheng tarballs nest it under filtered_gene_bc_matrices/<genome>/, descend to it."""
    import os
    for root, _dirs, files in os.walk(path):
        names = set(files)
        if any(n.startswith("matrix.mtx") for n in names) and \
           any(n.startswith(("genes.tsv", "features.tsv")) for n in names):
            return root
    raise FileNotFoundError(f"no CellRanger mtx (matrix.mtx + genes/features.tsv) under {path}")


def load_10x_populations(specs, normalize: bool = True, target_sum: float = 1e4,
                         n_per_class: Optional[int] = None, seed: int = 0) -> Dataset:
    """Load several FACS-sorted 10x populations and merge into one labeled Dataset.

    ``specs``: list of ``(mtx_dir, label)``, e.g. the Zheng CD4 subsets. Cells from each
    are tagged with ``label`` (the ground-truth FACS class); matrices are concatenated on
    shared genes (these share the same 10x reference, so genes align). Optionally subsample
    ``n_per_class`` cells per population (cap big runs). NO gene filtering, that's the point.
    """
    import numpy as np
    import anndata as ad
    import scanpy as sc

    parts = []
    for mtx_dir, label in specs:
        a = sc.read_10x_mtx(_find_mtx_dir(mtx_dir), var_names="gene_symbols")
        a.var_names_make_unique()
        if n_per_class and a.n_obs > n_per_class:
            rng = np.random.default_rng(seed)
            a = a[rng.choice(a.n_obs, n_per_class, replace=False)].copy()
        a.obs["label"] = label
        parts.append(a)
    merged = ad.concat(parts, join="inner", label="batch")    # inner = shared genes
    if normalize:
        merged = _lognorm(merged, target_sum)
    X = _densify(merged.X)
    labels = merged.obs["label"].astype("category")
    return Dataset(X=X, y=labels.cat.codes.to_numpy().astype(np.int64),
                   var_names=list(map(str, merged.var_names)),
                   label_names=list(map(str, labels.cat.categories)), obs_label_key="label")


def highly_variable_subset(ds: Dataset, n_top_genes: int = 2000) -> Tuple[Dataset, List[int]]:
    """Return the HVG-filtered dataset + the kept gene indices, used to *measure what
    dimension reduction discards* (run DeepMapper on full vs HVG, see if attribution genes
    survive selection)."""
    import numpy as np
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(np.asarray(ds.X))
    a.var_names = ds.var_names
    sc.pp.highly_variable_genes(a, n_top_genes=n_top_genes)
    keep = list(np.where(a.var["highly_variable"].to_numpy())[0])
    sub = Dataset(X=ds.X[:, keep], y=ds.y,
                  var_names=[ds.var_names[i] for i in keep],
                  label_names=ds.label_names, obs_label_key=ds.obs_label_key)
    return sub, keep


def intersect_genes(a: Dataset, b: Dataset) -> Tuple[Dataset, Dataset, List[str]]:
    """Restrict two datasets to their shared genes (C5 "common genes only"). Returns the
    two restricted datasets + the shared gene list, both column-aligned to that order."""
    shared = [g for g in a.var_names if g in set(b.var_names)]
    ai = {g: i for i, g in enumerate(a.var_names)}
    bi = {g: i for i, g in enumerate(b.var_names)}
    aidx = [ai[g] for g in shared]
    bidx = [bi[g] for g in shared]
    A = Dataset(X=a.X[:, aidx], y=a.y, var_names=shared,
                label_names=a.label_names, obs_label_key=a.obs_label_key)
    B = Dataset(X=b.X[:, bidx], y=b.y, var_names=shared,
                label_names=b.label_names, obs_label_key=b.obs_label_key)
    return A, B, shared


def exclusive_genes(a: Dataset, b: Dataset) -> Tuple[List[str], List[str]]:
    """Genes unique to each dataset (the "trivially separating" features in C5)."""
    sa, sb = set(a.var_names), set(b.var_names)
    return [g for g in a.var_names if g not in sb], [g for g in b.var_names if g not in sa]


def export_for_seurat(ds: Dataset, outdir: str) -> str:
    """Write ``counts.csv`` (cells × genes) + ``labels.csv`` (cell,label) for
    ``bench/seurat_pipeline.R``. Returns ``outdir``. Cell ids are ``cell0..cellN``."""
    import os
    import numpy as np
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    cells = [f"cell{i}" for i in range(ds.n_cells)]
    pd.DataFrame(np.asarray(ds.X), index=cells, columns=ds.var_names).to_csv(
        os.path.join(outdir, "counts.csv"))
    pd.DataFrame({"cell": cells,
                  "label": [ds.label_names[i] for i in ds.y]}).to_csv(
        os.path.join(outdir, "labels.csv"), index=False)
    return outdir
