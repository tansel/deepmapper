"""Statistical validation + the 'ribosomal easy catch' for the DeepMapper biology paper.

(1) VALIDATION: cross-validated separability with a permutation null (CI + p-value) for each of the
    paper's classification tasks, so the headline accuracies are backed by statistics, not a single
    split. Fast linear classifier (Ridge) on the FULL unfiltered gene set (no HVG, no filtering).
(2) EASY CATCH: re-run each task using ONLY ribosomal-protein genes (RPL*/RPS*/MRP*/RPLP*/FAU). If
    ribosomal genes alone separate cells -- including non-CD4 lineages (PBMC broad) -- then a
    coordinated ribosomal programme is a general axis of cell identity, not a CD4-memory quirk.

    python bench/ribosomal_validation.py
"""
from __future__ import annotations
import json, os, re, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "bench")

CD4 = [("data/cd4/cd4_t_helper", "Helper"), ("data/cd4/regulatory_t", "Treg"),
       ("data/cd4/naive_t", "Naive"), ("data/cd4/memory_t", "Memory")]
CD8 = [("data/pbmc/cytotoxic_t", "Cytotoxic"), ("data/pbmc/naive_cytotoxic", "NaiveCytotoxic")]
BROAD = [("data/cd4/cd4_t_helper", "CD4_Helper"), ("data/pbmc/cytotoxic_t", "CD8_Cytotoxic"),
         ("data/pbmc/b_cells", "B"), ("data/pbmc/cd14_monocytes", "Monocyte"),
         ("data/pbmc/cd56_nk", "NK")]
RIBO = re.compile(r"^(RPL|RPS|MRPL|MRPS|RPLP|FAU)", re.I)   # ribosomal-protein genes by symbol
PER = 400


def cv_and_null(X, y, n_perm=100, seed=0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    clf = lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    real = cross_val_score(clf(), X, y, cv=cv).mean()
    rng = np.random.default_rng(seed)
    null = np.array([cross_val_score(clf(), X, rng.permutation(y),
                                     cv=StratifiedKFold(5, shuffle=True, random_state=1000 + p)).mean()
                     for p in range(n_perm)])
    # 95% CI on the real accuracy via repeated CV
    reps = [cross_val_score(clf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=r)).mean()
            for r in range(10)]
    return {"acc": float(np.mean(reps)), "sd": float(np.std(reps)),
            "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "null_mean": float(null.mean()), "p_value": float((1 + (null >= real).sum()) / (1 + n_perm))}


def main():
    from pydeepmapper.io import load_10x_populations, Dataset
    t0 = time.time()
    tasks = {"cd4_4way": CD4, "cd8_2way": CD8, "pbmc_broad": BROAD}
    out = {}
    for name, spec in tasks.items():
        ds = load_10x_populations(spec, normalize=True, n_per_class=PER, seed=0)
        X = np.asarray(ds.X, np.float32); y = np.asarray(ds.y, np.int64)
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        Xn = np.log1p(X / lib * 1e6)
        ribo = np.array([bool(RIBO.match(str(g))) for g in ds.var_names])
        full = cv_and_null(Xn, y)
        rib = cv_and_null(Xn[:, ribo], y)
        out[name] = {"classes": ds.label_names, "n": int(len(y)), "n_genes": int(X.shape[1]),
                     "n_ribo": int(ribo.sum()), "full": full, "ribosomal_only": rib}
        print(f"[{name}] classes={ds.label_names}", flush=True)
        print(f"   FULL  ({X.shape[1]} genes): acc={full['acc']:.3f} CI{full['ci95'][0]:.2f}-{full['ci95'][1]:.2f} "
              f"null={full['null_mean']:.3f} p={full['p_value']:.4f}", flush=True)
        print(f"   RIBO-ONLY ({ribo.sum()} genes): acc={rib['acc']:.3f} CI{rib['ci95'][0]:.2f}-{rib['ci95'][1]:.2f} "
              f"null={rib['null_mean']:.3f} p={rib['p_value']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        # CD4 memory-vs-rest binary (the paper headline) from the same load
        if name == "cd4_4way":
            mi = ds.label_names.index("Memory")
            yb = (ds.y == mi).astype(np.int64)
            fb = cv_and_null(Xn, yb); rb = cv_and_null(Xn[:, ribo], yb)
            out["cd4_memory_vs_rest"] = {"n": int(len(yb)), "n_ribo": int(ribo.sum()),
                                         "full": fb, "ribosomal_only": rb}
            print(f"[cd4_memory_vs_rest] FULL acc={fb['acc']:.3f} p={fb['p_value']:.4f} | "
                  f"RIBO-ONLY acc={rb['acc']:.3f} p={rb['p_value']:.4f}", flush=True)
    json.dump(out, open("results/ribosomal_validation.json", "w"), indent=2)
    print(f"\nDONE ({time.time()-t0:.0f}s) -> results/ribosomal_validation.json", flush=True)


if __name__ == "__main__":
    main()
