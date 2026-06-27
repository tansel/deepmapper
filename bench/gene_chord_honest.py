"""Selection-bias-free ('honest') gene-chord AUC, answering the reviewer's double-dipping objection.

The original number selected the top-K attributed genes on the FULL data, then cross-validated a logistic
on those genes -> feature selection outside the CV loop -> optimistic. Here, for each seed we split
train/test, run DeepMapper attribution + pick the chord genes + the best single gene ON THE TRAIN SPLIT
ONLY, fit the linear chord on train, and report AUC on the held-out TEST. No information leaks from test
into gene selection.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n deepmapper \
     python bench/gene_chord_honest.py
Out: results/attribution_de/gene_chord_honest.json
"""
from __future__ import annotations
import os, sys, json, time, warnings
sys.path.insert(0, "."); sys.path.insert(0, "bench")
warnings.filterwarnings("ignore")
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
SEEDS = [0, 1, 2]
TOPK = 60


def single_auc(x, y):
    a = roc_auc_score(y, x); return max(a, 1 - a)


def main():
    t0 = time.time()
    specs = [("data/cd4/cd4_t_helper", "Helper"), ("data/cd4/regulatory_t", "Treg"),
             ("data/cd4/naive_t", "Naive"), ("data/cd4/memory_t", "Memory")]
    ds = load_10x_populations(specs, normalize=True, n_per_class=1000, seed=0)
    y0 = (ds.y == ds.label_names.index("Memory")).astype(int); X0 = np.asarray(ds.X)
    rng = np.random.default_rng(0)
    mem = np.where(y0 == 1)[0]
    rest = np.concatenate([rng.choice(np.where(ds.y == c)[0], len(mem) // 3 + 1, replace=False)
                           for c in range(4) if c != ds.label_names.index("Memory")])
    rest = rng.permutation(rest)[:len(mem)]
    sel = rng.permutation(np.concatenate([mem, rest])); X, y = X0[sel], y0[sel]
    genes = np.array([str(g) for g in ds.var_names])
    print(f"balanced {len(y)} cells ({time.time()-t0:.0f}s)", flush=True)

    chord_test, single_test = [], []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=s)
        cfg = DeepMapperConfig(n_passes=3, epochs=25, top_k=Xtr.shape[1], seed=s, backbone=BackboneSpec(kind="cnn"))
        f = run(Xtr, ytr, cfg, feature_names=list(genes))           # attribution on TRAIN only
        top = [str(n) for n, _sc, _mi in f.ranking(TOPK)]
        gi = {g: i for i, g in enumerate(genes)}
        cols = [gi[g] for g in top]
        # chord: fit linear model on train genes, evaluate on held-out test
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xtr[:, cols], ytr)
        c_auc = roc_auc_score(yte, clf.decision_function(Xte[:, cols]))
        # best single gene: pick by TRAIN single-AUC, evaluate on test
        tr_aucs = [single_auc(Xtr[:, j], ytr) for j in cols]
        bestcol = cols[int(np.argmax(tr_aucs))]
        s_auc = single_auc(Xte[:, bestcol], yte)
        chord_test.append(c_auc); single_test.append(s_auc)
        print(f"  seed {s}: held-out chord AUC={c_auc:.3f}  best-single AUC={s_auc:.3f} "
              f"(gene {genes[bestcol]}) ({time.time()-t0:.0f}s)", flush=True)

    ct = np.array(chord_test); st = np.array(single_test)
    res = {"task": "cd4_memory_vs_rest", "topK": TOPK, "seeds": SEEDS, "protocol":
           "attribution + gene selection on TRAIN split; AUC on held-out TEST (no selection leak)",
           "held_out_chord_auc": {"mean": round(float(ct.mean()), 3), "sd": round(float(ct.std()), 3),
                                  "values": [round(float(v), 3) for v in ct]},
           "held_out_best_single_auc": {"mean": round(float(st.mean()), 3), "sd": round(float(st.std()), 3),
                                        "values": [round(float(v), 3) for v in st]}}
    json.dump(res, open(f"{OUT}/gene_chord_honest.json", "w"), indent=2)
    print(f"\nHONEST: chord {res['held_out_chord_auc']['mean']}+/-{res['held_out_chord_auc']['sd']} "
          f"vs best single {res['held_out_best_single_auc']['mean']} (held-out, no leak) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
