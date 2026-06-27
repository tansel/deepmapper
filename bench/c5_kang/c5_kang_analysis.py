"""C5: stimulated vs unstimulated, SHARED-GENES-ONLY, on Kang 2018 (GSE96583 batch 2).
GSM2560248 (2.1) = control, GSM2560249 (2.2) = IFN-beta stimulated.

The paper's claim: removing genes exclusive to one condition (the trivial split), the conditions are
still separable by a coordinated shift across SHARED genes. We test that, and report which pathway the
attribution surfaces. NB: Kang is IFN-beta stimulation, so the coordinated shift is INTERFERON-driven
(ISGs), not IL2 -- an honest correction to the draft's "IL2-led" wording (which needs a TCR/IL2 set).
"""
from __future__ import annotations
import gzip, io, json, os, tarfile
import numpy as np
import pandas as pd
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
RAW = os.path.join(D, "GSE96583_RAW.tar")
ISG = {"ISG15","IFI6","IFIT1","IFIT3","MX1","MX2","OAS1","OAS2","RSAD2","ISG20","IRF7","STAT1",
       "IFI44","IFI44L","LY6E","IFITM3","TNFSF10","CXCL10","GBP1","APOBEC3A"}
IL2 = {"IL2RA","IL2RB","IL2RG","JAK3","STAT5A","STAT5B","MYC","IL2"}


def load_lane(member_mtx, member_bc):
    with tarfile.open(RAW) as t:
        M = mmread(io.BytesIO(gzip.decompress(t.extractfile(member_mtx).read())))   # genes x cells
        bc = gzip.decompress(t.extractfile(member_bc).read()).decode().split()
    return np.asarray(M.todense(), dtype=np.float32).T, bc                          # cells x genes


def cv_and_null(X, y, n_perm=100, seed=0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    clf = lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
    real = cross_val_score(clf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
    rng = np.random.default_rng(seed)
    null = np.array([cross_val_score(clf(), X, rng.permutation(y),
                                     cv=StratifiedKFold(5, shuffle=True, random_state=1000 + p)).mean()
                     for p in range(n_perm)])
    return {"acc": float(real), "null_mean": float(null.mean()),
            "p_value": float((1 + (null >= real).sum()) / (1 + n_perm))}


def main():
    genes = pd.read_csv(os.path.join(D, "GSE96583_batch2.genes.tsv.gz"), sep="\t", header=None)
    sym = genes.iloc[:, 1].astype(str).to_numpy()
    Xc, _ = load_lane("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz")
    Xs, _ = load_lane("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz")
    print(f"ctrl {Xc.shape}  stim {Xs.shape}  genes {len(sym)}", flush=True)
    # subsample for speed, balanced
    rng = np.random.default_rng(0); n = 2000
    Xc = Xc[rng.choice(len(Xc), min(n, len(Xc)), replace=False)]
    Xs = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    X = np.vstack([Xc, Xs]); y = np.r_[np.zeros(len(Xc)), np.ones(len(Xs))].astype(int)
    lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    Xn = np.log1p(X / lib * 1e6).astype(np.float32)

    det_c = (Xc > 0).mean(0); det_s = (Xs > 0).mean(0)
    shared = (det_c >= 0.10) & (det_s >= 0.10)                 # expressed in BOTH conditions
    exclusive = ((det_c >= 0.10) ^ (det_s >= 0.10))
    res = {"n": int(len(y)), "n_genes": int(len(sym)), "n_shared": int(shared.sum()),
           "n_exclusive": int(exclusive.sum())}
    print(f"shared genes={shared.sum()}  exclusive={exclusive.sum()}", flush=True)
    res["all_genes"] = cv_and_null(Xn, y)
    res["shared_only"] = cv_and_null(Xn[:, shared], y)
    print(f"  all-genes   : acc={res['all_genes']['acc']:.3f} p={res['all_genes']['p_value']:.4f}", flush=True)
    print(f"  SHARED-only : acc={res['shared_only']['acc']:.3f} p={res['shared_only']['p_value']:.4f}", flush=True)

    # which shared genes drive it (effect size = mean stim - mean ctrl, on shared)
    eff = Xn[y == 1][:, shared].mean(0) - Xn[y == 0][:, shared].mean(0)
    sg = sym[shared]; order = np.argsort(np.abs(eff))[::-1]
    top = [(sg[i], float(eff[i])) for i in order[:30]]
    res["top_shared_genes"] = top
    n_isg = sum(g in ISG for g, _ in top); n_il2 = sum(g in IL2 for g, _ in top)
    res["top30_ISG"] = n_isg; res["top30_IL2"] = n_il2
    print("  top shared-gene markers:", [g for g, _ in top[:15]], flush=True)
    print(f"  of top 30 shared markers: {n_isg} interferon-stimulated genes, {n_il2} IL2-pathway", flush=True)
    json.dump(res, open(os.path.join(HERE, "c5_kang_metrics.json"), "w"), indent=2)
    print("DONE -> c5_kang_metrics.json", flush=True)


if __name__ == "__main__":
    main()
