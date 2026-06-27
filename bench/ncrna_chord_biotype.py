"""All-non-coding chord on GSE99254 using a GENCODE biotype-defined lncRNA set (Reviewer B/R4.2), instead
of the symbol regex. Confirms whether the antisense/chord result holds under a proper annotation.

Run: conda run -n deepmapper python bench/ncrna_chord_biotype.py
Out: results/attribution_de/ncrna_chord_biotype.json
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IV = f"{ROOT}/bench/independent_validation"
OUT = f"{ROOT}/results/attribution_de"
TYPE = {"C": "CD8", "H": "CD4_helper", "R": "Treg", "Y": "CD8_other"}


def chord_auc(X, y, seed=0):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    s = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="decision_function")
    return float(roc_auc_score(y, s))


def main():
    lnc_symbols = set(json.load(open(f"{IV}/annot/gencode_lncRNA_symbols.json")))
    df = pd.read_csv(f"{IV}/data/GSE99254_NSCLC.TCell.TPM.txt.gz", sep="\t")
    sym = df["symbol"].astype(str).to_numpy()
    cells = df.columns[2:]
    labels = np.array([TYPE.get(str(c)[2:3], None) for c in cells])
    keep = labels != None
    M = df.iloc[:, 2:].to_numpy(np.float32).T[keep]
    y = labels[keep]
    lib = M.sum(1, keepdims=True); lib[lib == 0] = 1
    X = np.log1p(M / lib * 1e6)
    det = (M > 0).mean(0)
    is_lnc = np.array([g in lnc_symbols for g in sym]) & (det >= 0.05)   # GENCODE biotype, detected>=5%
    Xl, gl = X[:, is_lnc], sym[is_lnc]
    print(f"cells={len(y)}  GENCODE-lncRNAs(detected>=5%)={is_lnc.sum()} (vs ~209 by regex)", flush=True)

    res = {"dataset": "GSE99254", "annotation": "GENCODE v44 lncRNA biotype",
           "n_lncRNA_features": int(is_lnc.sum()), "states": {}}
    for s in ["Treg", "CD8", "CD4_helper", "CD8_other"]:
        yb = (y == s).astype(int)
        if yb.sum() < 20:
            continue
        aucs = np.maximum.reduce([np.array([roc_auc_score(yb, Xl[:, j]) for j in range(Xl.shape[1])]),
                                  1 - np.array([roc_auc_score(yb, Xl[:, j]) for j in range(Xl.shape[1])])])
        bj = int(aucs.argmax())
        chord = chord_auc(Xl, yb)
        res["states"][s] = {"n_pos": int(yb.sum()), "best_single_lncRNA": gl[bj],
                            "best_single_auc": round(float(aucs[bj]), 3),
                            "ncRNA_chord_auc": round(chord, 3), "lift": round(chord - float(aucs[bj]), 3)}
        r = res["states"][s]
        print(f"  {s:<11} best single {r['best_single_lncRNA']} AUC={r['best_single_auc']} | "
              f"all-ncRNA chord AUC={r['ncRNA_chord_auc']} (+{r['lift']})", flush=True)
    json.dump(res, open(f"{OUT}/ncrna_chord_biotype.json", "w"), indent=2)
    print("saved ->", f"{OUT}/ncrna_chord_biotype.json", flush=True)


if __name__ == "__main__":
    main()
