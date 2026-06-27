"""Backbone head-to-head (answers the recurring reviewer ask: does the CNN matter?).

Same no-filtering + multi-pass + held-out attribution pipeline, only the backbone changes:
  linear, logistic regression on the full feature set (architecture removed)
  mlp, dense non-linear, NO convolution
  cnn_small, the paper's convolutional backbone
For each backbone x seed: attribution + top-K chord-gene selection on TRAIN; held-out chord AUC on TEST
(logistic on those genes); ribosomal fraction of the chord. Then cross-backbone Jaccard of the top-K
attributed genes. If linear matches cnn AND they recover the same genes, the architecture is incidental
and the contribution is the no-filtering + attribution pipeline.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n deepmapper python bench/backbone_headtohead.py
Out: results/attribution_de/backbone_headtohead.json
"""
from __future__ import annotations
import os, sys, json, time, re, warnings
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

OUT = "results/attribution_de"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]; TOPK = 60; BACKBONES = ["linear", "mlp", "cnn_small"]
RIBO = re.compile(r"^(RPL|RPS|MRPL|MRPS|RPLP|FAU)", re.I)


def load_cd4_memory():
    specs = [("data/cd4/cd4_t_helper", "Helper"), ("data/cd4/regulatory_t", "Treg"),
             ("data/cd4/naive_t", "Naive"), ("data/cd4/memory_t", "Memory")]
    ds = load_10x_populations(specs, normalize=True, n_per_class=1000, seed=0)
    y0 = (ds.y == ds.label_names.index("Memory")).astype(int); X0 = np.asarray(ds.X)
    rng = np.random.default_rng(0); mem = np.where(y0 == 1)[0]
    rest = np.concatenate([rng.choice(np.where(ds.y == c)[0], len(mem) // 3 + 1, replace=False)
                           for c in range(4) if c != ds.label_names.index("Memory")])
    rest = rng.permutation(rest)[:len(mem)]
    sel = rng.permutation(np.concatenate([mem, rest]))
    return X0[sel], y0[sel], np.array([str(g) for g in ds.var_names])


def main():
    t0 = time.time()
    X, y, genes = load_cd4_memory()
    gi = {g: i for i, g in enumerate(genes)}
    print(f"balanced {len(y)} cells, {X.shape[1]} genes ({time.time()-t0:.0f}s)", flush=True)

    per = {b: {"chord": [], "ribo": [], "topsets": []} for b in BACKBONES}
    for b in BACKBONES:
        for s in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=s)
            cfg = DeepMapperConfig(n_passes=3, epochs=25, top_k=Xtr.shape[1], seed=s,
                                   backbone=BackboneSpec(kind=b))
            f = run(Xtr, ytr, cfg, feature_names=list(genes))
            top = [str(n) for n, _sc, _mi in f.ranking(TOPK)]
            cols = [gi[g] for g in top]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[:, cols], ytr)
            auc = roc_auc_score(yte, clf.decision_function(Xte[:, cols]))
            per[b]["chord"].append(auc); per[b]["topsets"].append(set(top))
            per[b]["ribo"].append(float(np.mean([bool(RIBO.match(g)) for g in top])))
            print(f"  {b:10} seed {s}: held-out chord AUC={auc:.3f}  ribo-frac={per[b]['ribo'][-1]:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    def jac(a, b):
        vals = [len(sa & sb) / len(sa | sb) for sa, sb in zip(per[a]["topsets"], per[b]["topsets"])]
        return round(float(np.mean(vals)), 3)

    res = {"task": "cd4_memory_vs_rest", "topK": TOPK, "seeds": SEEDS,
           "per_backbone": {b: {"held_out_chord_auc_mean": round(float(np.mean(per[b]["chord"])), 3),
                                "held_out_chord_auc_sd": round(float(np.std(per[b]["chord"])), 3),
                                "ribosomal_fraction_mean": round(float(np.mean(per[b]["ribo"])), 3)}
                            for b in BACKBONES},
           "topK_gene_jaccard": {"linear_vs_cnn_small": jac("linear", "cnn_small"),
                                 "mlp_vs_cnn_small": jac("mlp", "cnn_small"),
                                 "linear_vs_mlp": jac("linear", "mlp")},
           "note": "if linear chord AUC ~ cnn AND linear-vs-cnn gene Jaccard is high, the architecture is "
                   "incidental; the recovered chord is a property of no-filtering + attribution, not the CNN"}
    json.dump(res, open(f"{OUT}/backbone_headtohead.json", "w"), indent=2)
    print("\n" + json.dumps(res["per_backbone"], indent=1), flush=True)
    print("Jaccard:", res["topK_gene_jaccard"], f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
