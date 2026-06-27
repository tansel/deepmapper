"""Disease-relevant reanalysis (round-3 Reviewer I / Tier 4.1): do the signatures separate a clinically
pivotal T-cell state? We test tumour-infiltrating (TIL) versus blood CD8 T cells in the cohorts already in
the paper, evaluated with PATIENT held out, and ask (a) are they separable, (b) are canonical exhaustion
genes coordinately up in TIL, and (c) does a multi-gene exhaustion 'chord' beat the best single marker.

Cell naming (Guo NSCLC, Zheng HCC): [tissue N/P/T][T][subtype C/H/R/Y][cell#]-[patient/sample]. TIL CD8 =
TTC*, blood CD8 = PTC*; the patient/sample id is the suffix after the dash. CRC (GSE108989) uses a
different scheme and is skipped here.

Run: conda run -n deepmapper python bench/independent_validation/exhaustion_til_vs_blood.py
Out: results/attribution_de/exhaustion_til_vs_blood.json
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
NAME = re.compile(r"^([NPT])T([CHRY])(\d+)")              # tissue, subtype, patient
EXH = ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "CTLA4", "TOX", "ENTPD1", "LAYN"]
COHORTS = [("NSCLC", "GSE99254_NSCLC.TCell.TPM.txt.gz"), ("HCC", "GSE98638_HCC.TCell.S5063.TPM.txt.gz")]


def patient_cv_auc(X, y, groups, seed=0):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    n_splits = min(5, len(np.unique(groups)))
    s = cross_val_predict(pipe, X, y, groups=groups, cv=GroupKFold(n_splits),
                          method="decision_function")
    return float(roc_auc_score(y, s))


def main():
    res = {}
    for name, fn in COHORTS:
        df = pd.read_csv(os.path.join(D, fn), sep="\t")
        sym = df["symbol"].astype(str).to_numpy()
        cells = list(df.columns[2:])
        parsed = [NAME.match(c) for c in cells]
        # CD8 cytotoxic cells (subtype C), tumour (T) vs blood (P)
        keep, y, grp = [], [], []
        for i, (c, m) in enumerate(zip(cells, parsed)):
            if m and m.group(2) == "C" and m.group(1) in ("T", "P") and "-" in c:
                keep.append(i); y.append(1 if m.group(1) == "T" else 0)
                grp.append(c.split("-", 1)[1])               # patient/sample id = suffix after the dash
        keep = np.array(keep); y = np.array(y); grp = np.array(grp)
        M = df.iloc[:, 2:].to_numpy(np.float32).T[keep]
        lib = M.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(M / lib * 1e6)
        idx = {g.upper(): i for i, g in enumerate(sym)}
        exh_idx = [idx[g] for g in EXH if g in idx]

        full_auc = patient_cv_auc(X, y, grp)
        chord_auc = patient_cv_auc(X[:, exh_idx], y, grp)
        singles = {}
        for g in EXH:
            if g in idx:
                col = X[:, idx[g]]
                a = roc_auc_score(y, col); singles[g] = round(max(a, 1 - a), 3)
        # exhaustion direction: mean(TIL) - mean(blood) in log-CPM
        direction = {g: round(float(X[y == 1, idx[g]].mean() - X[y == 0, idx[g]].mean()), 3)
                     for g in EXH if g in idx}
        res[name] = {
            "n_til_cd8": int((y == 1).sum()), "n_blood_cd8": int((y == 0).sum()),
            "n_patients": int(len(np.unique(grp))),
            "til_vs_blood_full_auc_patient_cv": round(full_auc, 3),
            "exhaustion_chord_auc_patient_cv": round(chord_auc, 3),
            "best_single_exhaustion_marker_auc": round(max(singles.values()), 3),
            "best_single_marker": max(singles, key=singles.get),
            "single_marker_aucs": singles,
            "til_minus_blood_logCPM": direction,
            "n_exhaustion_up_in_til": int(sum(v > 0 for v in direction.values())),
        }
        print(f"{name}: TIL={res[name]['n_til_cd8']} blood={res[name]['n_blood_cd8']} "
              f"patients={res[name]['n_patients']} | full AUC {full_auc:.3f} | "
              f"exhaustion-chord AUC {chord_auc:.3f} vs best single "
              f"{res[name]['best_single_exhaustion_marker_auc']:.3f} "
              f"({res[name]['best_single_marker']}) | {res[name]['n_exhaustion_up_in_til']}/{len(direction)} "
              f"exhaustion genes up in TIL", flush=True)
    res["_note"] = ("TIL vs blood CD8 T cells, patient-held-out CV, in two cohorts already in the paper. "
                    "A multi-gene exhaustion chord separating held-out patients, with the canonical "
                    "exhaustion genes coordinately up in TIL, is the disease-relevant gene-chord result.")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "exhaustion_til_vs_blood.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
