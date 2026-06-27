"""Where do the memory-defining ribosomal genes fall in HVG selection? Computes each gene's
highly-variable-gene rank (normalised dispersion) on the sorted CD4 data and records where the
ribosomal-protein genes land relative to the top-2000 cutoff. Feeds Fig2 (the honest mechanism:
HVG discards the genes that carry memory).

Run: conda run -n deepmapper python bench/hvg_ribosomal_rank.py
Out: results/hvg_ribosomal.json
"""
from __future__ import annotations
import os, sys, json, re, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc
from anndata import AnnData
from pydeepmapper.io import load_10x_populations

N_TOP = 2000
OUT = "results"; os.makedirs(OUT, exist_ok=True)

specs = [("data/cd4/cd4_t_helper", "Helper"), ("data/cd4/regulatory_t", "Treg"),
         ("data/cd4/naive_t", "Naive"), ("data/cd4/memory_t", "Memory")]
ds = load_10x_populations(specs, normalize=True, n_per_class=1000, seed=0)
a = AnnData(X=np.asarray(ds.X, np.float32))
a.var_names = [str(g) for g in ds.var_names]; a.var_names_make_unique()
sc.pp.highly_variable_genes(a, n_top_genes=N_TOP)

genes = np.array(a.var_names)
disp = a.var["dispersions_norm"].to_numpy()
order = np.argsort(-disp)                              # rank 1 = most variable
rank = np.empty(len(genes), int); rank[order] = np.arange(1, len(genes) + 1)
hvg = a.var["highly_variable"].to_numpy()
is_rp = np.array([bool(re.match(r"^(RPL|RPS|MRPL|MRPS|RPLP|FAU)", g, re.I)) for g in genes])  # paper's 186-gene def

ribo_ranks = sorted(int(r) for r in rank[is_rp])
res = {
    "n_genes": int(len(genes)), "n_top_genes": N_TOP,
    "n_ribosomal": int(is_rp.sum()),
    "ribosomal_retained_by_hvg": int((is_rp & hvg).sum()),
    "ribosomal_dropped_by_hvg": int((is_rp & ~hvg).sum()),
    "ribosomal_dropped_frac": round(float((is_rp & ~hvg).sum() / is_rp.sum()), 3),
    "ribosomal_hvg_ranks": ribo_ranks,
}
json.dump(res, open(f"{OUT}/hvg_ribosomal.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "ribosomal_hvg_ranks"}, indent=1))
print("median ribosomal HVG rank:", int(np.median(ribo_ranks)), "(cutoff", N_TOP, ")")
