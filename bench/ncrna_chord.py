"""All-ncRNA gene chord on GSE99254 (full-length SMART-seq2 T cells).

Tests the claim that a gene chord can be entirely non-coding: for each T-cell state, no single lncRNA is
a strong marker, yet the LINEAR COMBINATION of lncRNAs (an all-ncRNA chord) separates that state well
above the best single lncRNA. lncRNA-only feature set; nothing protein-coding.

Run: conda run -n deepmapper python bench/ncrna_chord.py
Out: results/attribution_de/ncrna_chord.json
"""
import os, re, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IV = f"{ROOT}/bench/independent_validation"
OUT = f"{ROOT}/results/attribution_de"; os.makedirs(OUT, exist_ok=True)
LNC = re.compile(r"^(LINC\d|LOC\d|SNHG\d|MIR\d+HG|.*-AS\d+|A[CLP]\d{6})$|^("
                 r"MALAT1|NEAT1|XIST|TSIX|KCNQ1OT1|MEG3|HOTAIR|GAS5|NORAD|FTX|TUG1|HOTTIP|H19|"
                 r"FLICR|MIR155HG|MIR4435-2HG|ZFAS1|DLEU2|PVT1|SNHG1|SNHG5|CRNDE)$", re.I)
TYPE = {"C": "CD8", "H": "CD4_helper", "R": "Treg", "Y": "CD8_other"}


def chord_auc(X, y, seed=0):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    s = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="decision_function")
    return float(roc_auc_score(y, s))


def main():
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
    is_lnc = np.array([bool(LNC.match(g)) for g in sym]) & (det >= 0.05)
    Xl, gl = X[:, is_lnc], sym[is_lnc]
    print(f"cells={len(y)}  lncRNAs(detected>=5%)={is_lnc.sum()}  states={sorted(set(y))}", flush=True)

    res = {"dataset": "GSE99254", "n_lncRNA_features": int(is_lnc.sum()), "states": {}}
    for s in ["Treg", "CD8", "CD4_helper", "CD8_other"]:
        yb = (y == s).astype(int)
        if yb.sum() < 20:
            continue
        aucs = np.array([roc_auc_score(yb, Xl[:, j]) for j in range(Xl.shape[1])])
        aucs = np.maximum(aucs, 1 - aucs)
        best_j = int(aucs.argmax())
        chord = chord_auc(Xl, yb)
        res["states"][s] = {
            "n_pos": int(yb.sum()), "best_single_lncRNA": gl[best_j],
            "best_single_auc": round(float(aucs[best_j]), 3),
            "median_single_auc": round(float(np.median(aucs)), 3),
            "ncRNA_chord_auc": round(chord, 3),
            "lift": round(chord - float(aucs[best_j]), 3),
        }
        r = res["states"][s]
        print(f"  {s:<11} best single lncRNA {r['best_single_lncRNA']} AUC={r['best_single_auc']}  "
              f"| all-ncRNA chord AUC={r['ncRNA_chord_auc']}  (+{r['lift']})", flush=True)
    json.dump(res, open(f"{OUT}/ncrna_chord.json", "w"), indent=2)
    print("saved ->", f"{OUT}/ncrna_chord.json", flush=True)


if __name__ == "__main__":
    main()
