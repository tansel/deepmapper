"""Donor/batch control for the interferon signature (round-3 Reviewer G #2).

Is the stim-vs-ctrl ISG separation the designed IFN-beta response, or could it be an ex-vivo handling /
donor-batch artefact? We compute a per-cell ISG score (mean log-CPM over canonical ISGs) on the Kang
GSE96583 batch-2 lanes, align each cell to its donor (`ind`) via the published metadata, restrict to CD4
T cells, and report: (a) the stim-vs-ctrl ISG-score AUC WITHIN each of the 8 donors, and (b) each donor's
control-baseline mean ISG. If the separation holds in every donor and the ctrl baseline is uniformly low,
the signature is the stimulation, not donor/handling.

Run: conda run -n deepmapper python bench/c5_kang/c5_kang_donor_isg.py
Out: results/attribution_de/kang_donor_isg.json
"""
from __future__ import annotations
import gzip, io, json, os, tarfile
import numpy as np, pandas as pd
from scipy.io import mmread
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "..", "results", "attribution_de")
RAW = os.path.join(D, "GSE96583_RAW.tar")
ISG = ["ISG15","IFI6","IFIT1","IFIT3","MX1","MX2","OAS1","OAS2","RSAD2","ISG20","IRF7","STAT1",
       "IFI44","IFI44L","LY6E","IFITM3","TNFSF10","CXCL10","GBP1","APOBEC3A"]


def isg_score_lane(member_mtx, member_bc, isg_idx):
    """Return per-cell ISG score (mean log-CPM over ISG rows) and barcodes, slicing the sparse matrix."""
    with tarfile.open(RAW) as t:
        M = mmread(io.BytesIO(gzip.decompress(t.extractfile(member_mtx).read()))).tocsr()  # genes x cells
        bc = gzip.decompress(t.extractfile(member_bc).read()).decode().split()
    lib = np.asarray(M.sum(0)).ravel(); lib[lib == 0] = 1                # library size per cell
    sub = M[isg_idx, :].toarray()                                       # ISG genes x cells (small)
    logcpm = np.log1p(sub / lib * 1e6)
    return logcpm.mean(0), bc                                          # mean over ISG genes -> per-cell score


def main():
    genes = pd.read_csv(os.path.join(D, "GSE96583_batch2.genes.tsv.gz"), sep="\t", header=None)
    sym = genes.iloc[:, 1].astype(str).to_numpy()
    isg_idx = [i for i, g in enumerate(sym) if g in set(ISG)]
    meta = pd.read_csv(os.path.join(D, "GSE96583_batch2.total.tsne.df.tsv.gz"), sep="\t")

    frames = []
    for lane_mtx, lane_bc, cond in [("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz", "ctrl"),
                                    ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz", "stim")]:
        score, bc = isg_score_lane(lane_mtx, lane_bc, isg_idx)
        m = meta[meta["stim"] == cond]                                  # align within condition (unique barcodes)
        ind = pd.Series(m["ind"].values, index=m.index.astype(str))
        cell = pd.Series(m["cell"].values, index=m.index.astype(str))
        df = pd.DataFrame({"bc": [b for b in bc], "isg": score, "stim": cond})
        df["ind"] = df["bc"].map(ind); df["cell"] = df["bc"].map(cell)
        frames.append(df.dropna(subset=["ind"]))
    dat = pd.concat(frames, ignore_index=True)
    cd4 = dat[dat["cell"] == "CD4 T cells"].copy()

    res = {"n_cells_aligned": int(len(dat)), "n_cd4": int(len(cd4)),
           "isg_genes_used": int(len(isg_idx)),
           "overall_stim_vs_ctrl_auc_cd4": round(float(
               roc_auc_score((cd4["stim"] == "stim").astype(int), cd4["isg"])), 3),
           "per_donor": {}}
    for ind, g in cd4.groupby("ind"):
        ys = (g["stim"] == "stim").astype(int)
        auc = float(roc_auc_score(ys, g["isg"])) if ys.nunique() == 2 else None
        res["per_donor"][str(int(ind))] = {
            "n": int(len(g)),
            "stim_vs_ctrl_isg_auc": round(auc, 3) if auc is not None else None,
            "ctrl_baseline_mean_isg": round(float(g[g["stim"] == "ctrl"]["isg"].mean()), 3),
            "stim_mean_isg": round(float(g[g["stim"] == "stim"]["isg"].mean()), 3)}
    aucs = [d["stim_vs_ctrl_isg_auc"] for d in res["per_donor"].values() if d["stim_vs_ctrl_isg_auc"]]
    bases = [d["ctrl_baseline_mean_isg"] for d in res["per_donor"].values()]
    res["per_donor_auc_min"] = round(min(aucs), 3); res["per_donor_auc_max"] = round(max(aucs), 3)
    res["ctrl_baseline_spread"] = round(max(bases) - min(bases), 3)
    res["conclusion"] = ("if per-donor AUC is high in every donor and the ctrl baseline spread is small, "
                         "the ISG signature is the IFN-beta stimulation, not a donor/handling artefact")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "kang_donor_isg.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
