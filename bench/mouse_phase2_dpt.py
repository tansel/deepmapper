"""Phase 2b: proper effectorness control for the mouse ribosomal signal, using
scanpy diffusion pseudotime (DPT) instead of the marker-based proxy from the
linear preview. Residualises the ribosomal-only naive-vs-TEM separation against
sequencing depth, cell cycle, and the DPT effectorness axis, and reports whether
the ribosomal signal survives. Matrix-level, CPU only (no torch). Read-only.

  python bench/mouse_phase2_dpt.py --per-class 2000
"""
import argparse, json, os, re, sys, random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

D = "data/SCP490/"

def is_ribo(g):
    if g.startswith("Rps6k"): return False
    return bool(re.match(r"^Rp[ls]\d", g)) or g in {"Rplp0","Rplp1","Rplp2","Rpsa","Fau"}

S_GENES = "Mcm5 Pcna Tyms Mcm2 Mcm4 Rrm1 Mcm6 Cdca7 Uhrf1 Hells Nasp Gmnn Rrm2 Cdc6 Cdc45 Rad51 Ccne2".split()
G2M_GENES = "Hmgb2 Cdk1 Nusap1 Ube2c Birc5 Tpx2 Top2a Mki67 Ccnb2 Aurkb Bub1 Kif11 Cdc20 Aurka Anln".split()
NAIVE_MK = "Sell Ccr7 Lef1 Tcf7".split()

def auc(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    s = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                        X, y, cv=cv, scoring="roc_auc")
    return round(float(s.mean()), 3)

def residualise(X, covs, n):
    C = np.column_stack([np.ones(n)] + covs)
    return X - C @ (np.linalg.pinv(C) @ X)

def main():
    import scanpy as sc, anndata as ad, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--out", default="results/mouse_cd4_naive_vs_tem/phase2_dpt.json")
    a = ap.parse_args()
    random.seed(0)

    lab = {}
    with open(D+"Metadata.txt", encoding="latin-1") as fh:
        fh.readline(); fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t"); bc, sub = f[0], f[7]
            if sub == "TEM": lab[bc] = 1
            elif sub.startswith("Na") and "Isg" not in sub: lab[bc] = 0
    naive = [b for b,v in lab.items() if v==0]; tem = [b for b,v in lab.items() if v==1]
    k = min(a.per_class, len(tem))
    pick = random.sample(naive, k) + random.sample(tem, k)
    y = np.array([lab[b] for b in pick], dtype=np.int64)
    print(f"loading {2*k} cells ...", flush=True)
    df = pd.read_csv(D+"RawData1.csv", usecols=["GENE"]+pick, index_col="GENE")[pick]
    genes = df.index.tolist()
    raw = df.values.T.astype(np.float32); del df          # cells x genes raw counts

    ad_obj = ad.AnnData(raw.copy()); ad_obj.var_names = genes
    sc.pp.normalize_total(ad_obj, target_sum=1e4); sc.pp.log1p(ad_obj)
    Xn = ad_obj.X.copy()                                   # normalised, for the AUC + scores
    # DPT effectorness axis
    sc.pp.scale(ad_obj, max_value=10)
    sc.tl.pca(ad_obj, n_comps=30)
    sc.pp.neighbors(ad_obj, n_neighbors=15, n_pcs=30)
    sc.tl.diffmap(ad_obj)
    gi = {g:i for i,g in enumerate(genes)}
    naive_score = Xn[:, [gi[g] for g in NAIVE_MK if g in gi]].mean(1)
    root = int(np.where(y==0)[0][np.argmax(naive_score[y==0])])   # most-naive cell
    ad_obj.uns["iroot"] = root
    sc.tl.dpt(ad_obj)
    dpt = np.asarray(ad_obj.obs["dpt_pseudotime"].values, float)
    dpt = np.nan_to_num(dpt, nan=np.nanmedian(dpt))
    print(f"DPT range {dpt.min():.3f}..{dpt.max():.3f}; root cell {root}", flush=True)

    # covariates + ribosomal block
    numis = raw.sum(1)
    ln_nu = np.log1p(numis)
    def score(gs):
        cols = [gi[g] for g in gs if g in gi]
        return Xn[:, cols].mean(1) if cols else np.zeros(len(y))
    S, G2M = score(S_GENES), score(G2M_GENES)
    ribo_idx = [i for i,g in enumerate(genes) if is_ribo(g)]
    Xr = Xn[:, ribo_idx]
    n = len(y)

    res = {"ribosomal_only": auc(Xr, y),
           "resid_depth": auc(residualise(Xr, [ln_nu], n), y),
           "resid_cellcycle": auc(residualise(Xr, [S, G2M], n), y),
           "resid_DPT_effectorness": auc(residualise(Xr, [dpt], n), y),
           "resid_joint_depth+cycle+DPT": auc(residualise(Xr, [ln_nu, S, G2M, dpt], n), y),
           "n_ribo_genes": len(ribo_idx), "per_class": int(k),
           "note": "DPT replaces the marker-based effectorness proxy; marker-proxy preview gave eff 0.874, joint 0.865"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k_:v for k_,v in res.items() if k_!="note"}, indent=2))
    print("wrote", a.out)

if __name__ == "__main__":
    main()
