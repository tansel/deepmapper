"""Follow-up: are the state-discriminating lncRNAs enriched for natural antisense transcripts (NATs)
of canonical T-cell genes, and do those antisense lncRNAs co-vary with their SENSE partner?
If yes, the 'novel catch' is a distributed antisense-lncRNA programme tracking T-cell identity."""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TPM = os.path.join(HERE, "data", "GSE99254_NSCLC.TCell.TPM.txt.gz")
AS = re.compile(r"^(.+?)-AS\d+$")            # antisense: SENSE-AS1 etc.
TYPE = {"C": "CD8", "H": "CD4_helper", "R": "Treg", "Y": "CD8_other"}
TCELL = {"CD27","ENTPD1","ZEB2","LEF1","PRKCQ","IL21R","TRG","CTLA4","TIGIT","CCR7","TCF7","BCL2",
         "IL2RA","FOXP3","GZMB","PDCD1","LAG3","TOX","IKZF2","CD28","SELL","KLRG1","TBX21"}


def main():
    df = pd.read_csv(TPM, sep="\t")
    sym = df["symbol"].astype(str).to_numpy(); sym_u = np.char.upper(sym.astype(str))
    cells = df.columns[2:]
    labels = np.array([TYPE.get(str(c)[2:3], None) for c in cells])
    keep = np.array([l is not None for l in labels]); y = labels[keep]
    M = df.iloc[:, 2:].to_numpy(np.float32).T[keep]
    lib = M.sum(1, keepdims=True); lib[lib == 0] = 1
    X = np.log1p(M / lib * 1e6)
    idx = {g: i for i, g in enumerate(sym_u)}

    hunt = pd.read_csv(os.path.join(HERE, "lncrna_hunt_all.csv"))
    top = hunt.drop_duplicates("lncRNA").sort_values("effect", ascending=False).head(40)
    # antisense enrichment among top discriminators
    def sense_of(g):
        m = AS.match(str(g)); return m.group(1).upper() if m else None
    top = top.copy(); top["sense"] = top["lncRNA"].map(sense_of)
    n_as = top["sense"].notna().sum()
    n_as_tcell = top["sense"].isin(TCELL).sum()
    print(f"top 40 discriminating lncRNAs: {n_as} are antisense (-AS); "
          f"{n_as_tcell} are antisense to a canonical T-cell gene", flush=True)

    print("\n=== sense-antisense coordination (Spearman across cells) for antisense hits ===")
    rows = []
    for _, r in top[top["sense"].notna()].iterrows():
        s = r["sense"]
        if s in idx and r["lncRNA"].upper() in idx:
            a = X[:, idx[r["lncRNA"].upper()]]; b = X[:, idx[s]]
            rho, p = spearmanr(a, b)
            rows.append((r["lncRNA"], s, r["marks"], round(r["auc"], 3), round(float(rho), 3), p))
    cc = pd.DataFrame(rows, columns=["antisense_lncRNA", "sense_gene", "marks", "auc", "rho", "p"])
    cc = cc.sort_values("rho", ascending=False)
    print(cc.to_string(index=False))
    pos = (cc["rho"] > 0).sum(); strong = (cc["rho"].abs() > 0.2).sum()
    print(f"\n{len(cc)} sense-antisense pairs present; {pos} positively correlated; "
          f"{strong} with |rho|>0.2 -> coordinated antisense module", flush=True)
    cc.to_csv(os.path.join(HERE, "lncrna_antisense_coordination.csv"), index=False)


if __name__ == "__main__":
    main()
