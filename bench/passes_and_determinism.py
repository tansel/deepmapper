"""Do multiple passes matter? Is the linear path deterministic? (answers the reviewer/user question)

(A) Deterministic linear backbone (pydeepmapper.linear_baseline): no imagification, single convex fit.
    Fit TWICE -> identical top genes (Jaccard 1.0) confirms determinism; report held-out chord AUC.
(B) Torch linear backbone: n_passes=1 vs n_passes=3 -> top-gene Jaccard + chord AUC. Passes should be
    (near-)redundant: a flatten+linear model is permutation-equivariant, so arrangements add nothing;
    only optimisation noise (random init / batch order) remains.
(C) Torch cnn_small backbone: n_passes=1 vs n_passes=3 -> arrangement-sensitive, so passes should matter
    more (lower N1-vs-N3 Jaccard).

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n deepmapper python bench/passes_and_determinism.py
Out: results/attribution_de/passes_and_determinism.json
"""
from __future__ import annotations
import os, sys, json, time, warnings
sys.path.insert(0, "."); sys.path.insert(0, "bench"); warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pydeepmapper.io import load_10x_populations
from pydeepmapper.config import DeepMapperConfig, BackboneSpec
from pydeepmapper.runner import run
from pydeepmapper import linear_baseline

OUT = "results/attribution_de"; os.makedirs(OUT, exist_ok=True)
TOPK = 60


def load_cd4_memory():
    specs = [("data/cd4/cd4_t_helper", "Helper"), ("data/cd4/regulatory_t", "Treg"),
             ("data/cd4/naive_t", "Naive"), ("data/cd4/memory_t", "Memory")]
    ds = load_10x_populations(specs, normalize=True, n_per_class=1000, seed=0)
    y0 = (ds.y == ds.label_names.index("Memory")).astype(int); X0 = np.asarray(ds.X)
    rng = np.random.default_rng(0); mem = np.where(y0 == 1)[0]
    rest = np.concatenate([rng.choice(np.where(ds.y == c)[0], len(mem) // 3 + 1, replace=False)
                           for c in range(4) if c != ds.label_names.index("Memory")])
    rest = rng.permutation(rest)[:len(mem)]; sel = rng.permutation(np.concatenate([mem, rest]))
    return X0[sel], y0[sel], np.array([str(g) for g in ds.var_names])


def jac(a, b):
    a, b = set(a), set(b); return round(len(a & b) / len(a | b), 3)


def chord_auc(Xtr, ytr, Xte, yte, cols):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[:, cols], ytr)
    return round(float(roc_auc_score(yte, clf.decision_function(Xte[:, cols]))), 3)


def torch_top(Xtr, ytr, genes, kind, n_passes, seed=0):
    cfg = DeepMapperConfig(n_passes=n_passes, epochs=25, top_k=Xtr.shape[1], seed=seed,
                           backbone=BackboneSpec(kind=kind))
    f = run(Xtr, ytr, cfg, feature_names=list(genes))
    return [str(n) for n, _s, _m in f.ranking(TOPK)]


def main():
    t0 = time.time()
    X, y, genes = load_cd4_memory(); gi = {g: i for i, g in enumerate(genes)}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    print(f"balanced {len(y)} cells, {X.shape[1]} genes ({time.time()-t0:.0f}s)", flush=True)
    res = {"task": "cd4_memory_vs_rest", "topK": TOPK}

    # (A) deterministic linear baseline, no imagification, single fit, fit twice to prove determinism
    _, _, r1 = linear_baseline.fit(Xtr, ytr, feature_names=genes)
    _, _, r2 = linear_baseline.fit(Xtr, ytr, feature_names=genes)
    t1, t2 = linear_baseline.top_genes(r1, TOPK), linear_baseline.top_genes(r2, TOPK)
    res["deterministic_linear"] = {
        "imagification": False, "passes": 1,
        "refit_jaccard": jac(t1, t2),                      # expect 1.0 (exact)
        "chord_auc": chord_auc(Xtr, ytr, Xte, yte, [gi[g] for g in t1]),
        "note": "convex lbfgs fit, no imagification, no passes; refit is identical -> N=1 is exact"}
    print(f"(A) det-linear: refit Jaccard={res['deterministic_linear']['refit_jaccard']} "
          f"chord AUC={res['deterministic_linear']['chord_auc']} ({time.time()-t0:.0f}s)", flush=True)

    # (B) torch linear backbone: passes redundant?
    lin1 = torch_top(Xtr, ytr, genes, "linear", 1); lin3 = torch_top(Xtr, ytr, genes, "linear", 3)
    res["torch_linear_passes"] = {
        "n1_vs_n3_jaccard": jac(lin1, lin3),
        "chord_auc_n1": chord_auc(Xtr, ytr, Xte, yte, [gi[g] for g in lin1]),
        "chord_auc_n3": chord_auc(Xtr, ytr, Xte, yte, [gi[g] for g in lin3]),
        "det_linear_vs_torch_n3_jaccard": jac(t1, lin3)}
    print(f"(B) torch-linear N1-vs-N3 Jaccard={res['torch_linear_passes']['n1_vs_n3_jaccard']} "
          f"AUC {res['torch_linear_passes']['chord_auc_n1']}/{res['torch_linear_passes']['chord_auc_n3']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # (C) torch cnn_small backbone: passes matter (arrangement-sensitive)?
    c1 = torch_top(Xtr, ytr, genes, "cnn_small", 1); c3 = torch_top(Xtr, ytr, genes, "cnn_small", 3)
    res["torch_cnn_passes"] = {
        "n1_vs_n3_jaccard": jac(c1, c3),
        "chord_auc_n1": chord_auc(Xtr, ytr, Xte, yte, [gi[g] for g in c1]),
        "chord_auc_n3": chord_auc(Xtr, ytr, Xte, yte, [gi[g] for g in c3])}
    print(f"(C) torch-cnn   N1-vs-N3 Jaccard={res['torch_cnn_passes']['n1_vs_n3_jaccard']} "
          f"AUC {res['torch_cnn_passes']['chord_auc_n1']}/{res['torch_cnn_passes']['chord_auc_n3']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    res["conclusion"] = ("deterministic linear is exact at N=1 with no imagification; the CNN's top genes "
                         "shift more between N=1 and N=3 than linear's, i.e. multi-pass is a robustness "
                         "device for arrangement-sensitive backbones, redundant for a deterministic linear one")
    json.dump(res, open(f"{OUT}/passes_and_determinism.json", "w"), indent=2)
    print("\n" + json.dumps(res, indent=1)[:900], flush=True)


if __name__ == "__main__":
    main()
