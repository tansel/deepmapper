"""Major-5 benchmark: does the LEARNED exhaustion chord add value over a standard multi-marker exhaustion
SCORE (and over the best single marker)? TIL-vs-blood CD8, patient/sample held out, both cohorts.

Three readouts on the same 8 canonical exhaustion genes:
  best_single, best single-gene AUC (the single-marker baseline, e.g. PDCD1)
  mean_score, unweighted mean of per-gene z-scored expression (the "standard exhaustion score" clinicians
                 use: average of co-expressed checkpoint markers); no training, so no held-out needed
  learned_chord, logistic regression on the 8 genes, patient-held-out GroupKFold (learns the weights)
If learned ~ mean_score, the honest message is that the value is DE-NOVO RECOVERY of the relevant gene set
by the no-filtering + attribution pipeline, not a performance gain over a curated score.

Run: conda run -n deepmapper python bench/independent_validation/exhaustion_vs_score.py
Out: results/attribution_de/exhaustion_vs_score.json
"""
from __future__ import annotations
import os, re, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "..", "results", "attribution_de")
NAME = re.compile(r"^([NPT])T([CHRY])(\d+)")
EXH = ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "CTLA4", "TOX", "ENTPD1", "LAYN"]
COHORTS = [("NSCLC", "GSE99254_NSCLC.TCell.TPM.txt.gz"), ("HCC", "GSE98638_HCC.TCell.S5063.TPM.txt.gz")]


def single_auc(x, y):
    a = roc_auc_score(y, x); return max(a, 1 - a)


def main():
    res = {}
    for name, fn in COHORTS:
        df = pd.read_csv(os.path.join(D, fn), sep="\t")
        sym = df["symbol"].astype(str).to_numpy(); cells = list(df.columns[2:])
        keep, y, grp = [], [], []
        for i, c in enumerate(cells):
            m = NAME.match(c)
            if m and m.group(2) == "C" and m.group(1) in ("T", "P") and "-" in c:
                keep.append(i); y.append(1 if m.group(1) == "T" else 0); grp.append(c.split("-", 1)[1])
        keep = np.array(keep); y = np.array(y); grp = np.array(grp)
        M = df.iloc[:, 2:].to_numpy(np.float32).T[keep]
        lib = M.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(M / lib * 1e6)
        idx = {g.upper(): i for i, g in enumerate(sym)}
        cols = [idx[g] for g in EXH if g in idx]
        Xe = X[:, cols]

        singles = {EXH[k]: round(single_auc(Xe[:, k], y), 3) for k in range(len(cols))}
        best_single = max(singles.values()); best_gene = max(singles, key=singles.get)
        # standard exhaustion score = mean of per-gene z-scored expression (unsupervised, no weights)
        Z = StandardScaler().fit_transform(Xe)
        score = Z.mean(1)
        mean_score_auc = single_auc(score, y)
        # learned chord = logistic, patient/sample held out
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        dec = cross_val_predict(pipe, Xe, y, groups=grp, cv=GroupKFold(min(5, len(np.unique(grp)))),
                                method="decision_function")
        learned_auc = roc_auc_score(y, dec)
        res[name] = {
            "n_groups": int(len(np.unique(grp))),
            "best_single_marker": best_gene, "best_single_auc": round(best_single, 3),
            "mean_score_auc": round(float(mean_score_auc), 3),
            "learned_chord_auc_heldout": round(float(learned_auc), 3),
            "single_marker_aucs": singles}
        print(f"{name}: single({best_gene})={best_single:.3f}  mean-score={mean_score_auc:.3f}  "
              f"learned-chord={learned_auc:.3f}", flush=True)
    res["_note"] = ("learned chord vs a standard unweighted multi-marker exhaustion score vs the best single "
                    "marker, TIL-vs-blood CD8. If learned ~ mean-score, the contribution is de-novo recovery "
                    "of the gene set by the pipeline, not a gain over a curated score.")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "exhaustion_vs_score.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
