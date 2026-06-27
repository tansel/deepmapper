"""Controls answering reviewer points, all CPU/sklearn:
 (1) ribosomal gene count reconciled to the paper's 186-gene definition; HVG drop recomputed on it.
 (2) supervised-after-PCA: does the standard DR (HVG+PCA) preserve memory for a SUPERVISED classifier?
     (separates 'genes removed by HVG' from 'signal lost', for 2.5).
 (3) depth/size confound: does library size separate memory? does ribosomal-only separation survive
     residualising against library size? (Reviewer 2's #1 point.)

Run: conda run -n deepmapper python bench/review_controls.py
Out: results/review_controls.json
"""
from __future__ import annotations
import os, sys, json, re, warnings
sys.path.insert(0, "."); sys.path.insert(0, "bench")
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc
from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from pydeepmapper.io import load_10x_populations

OUT = "results"; RIBO = re.compile(r"^(RPL|RPS|MRPL|MRPS|RPLP|FAU)", re.I)

# Tirosh/Regev cell-cycle marker lists (classic hg19 symbols); missing symbols are dropped on intersection.
S_GENES = ("MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP HELLS RFC2 "
           "RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 "
           "TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8").split()
G2M_GENES = ("HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 "
             "FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 "
             "HN1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 "
             "ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA").split()


def cv_auc(X, y, seed=0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, X, y, scoring="roc_auc",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean())


def residualise(X, confs):
    """Regress each column of X on [1, *confs] and return the residuals (same approach as the depth control)."""
    L = np.c_[np.ones(X.shape[0]), *[np.asarray(c, float) for c in confs]]
    beta, *_ = np.linalg.lstsq(L, X, rcond=None)
    return X - L @ beta


def main():
    specs = [("data/cd4/cd4_t_helper", "Helper"), ("data/cd4/regulatory_t", "Treg"),
             ("data/cd4/naive_t", "Naive"), ("data/cd4/memory_t", "Memory")]
    raw = load_10x_populations(specs, normalize=False, n_per_class=1000, seed=0)   # raw counts
    Xr = np.asarray(raw.X, np.float32); genes = np.array([str(g) for g in raw.var_names])
    y = (raw.y == raw.label_names.index("Memory")).astype(int)
    lib = Xr.sum(1)                                              # library size per cell
    Xn = np.log1p(Xr / lib[:, None] * 1e4)                       # CP10K + log1p (as in the paper)
    ribo = np.array([bool(RIBO.match(g)) for g in genes])
    res = {"n_genes": int(len(genes)), "n_ribosomal_186def": int(ribo.sum())}

    # (1) HVG drop on the 186-gene definition
    a = AnnData(X=Xn.copy()); a.var_names = list(genes); a.var_names_make_unique()
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    hvg = a.var["highly_variable"].to_numpy()
    res["hvg"] = {"n_ribosomal": int(ribo.sum()), "retained": int((ribo & hvg).sum()),
                  "dropped": int((ribo & ~hvg).sum()),
                  "dropped_frac": round(float((ribo & ~hvg).sum() / ribo.sum()), 3)}

    # (2) supervised-after-PCA vs full
    a2 = AnnData(X=Xn.copy()); a2.var_names = list(genes); a2.var_names_make_unique()
    sc.pp.highly_variable_genes(a2, n_top_genes=2000); a2 = a2[:, a2.var.highly_variable].copy()
    sc.pp.scale(a2, max_value=10); a2.X = np.nan_to_num(a2.X); sc.tl.pca(a2, n_comps=50)
    res["supervised"] = {"full_genes_auc": round(cv_auc(Xn, y), 3),
                         "hvg_pca50_auc": round(cv_auc(a2.obsm["X_pca"], y), 3),
                         "note": "if HVG+PCA AUC stays high, DR keeps the signal for SUPERVISED tasks; "
                                 "the loss is the unsupervised clustering + HVG gene removal, not DR per se"}

    # (3) depth/size confound
    libsize_auc = roc_auc_score(y, lib); libsize_auc = max(libsize_auc, 1 - libsize_auc)
    Xrib = Xn[:, ribo]
    base = cv_auc(Xrib, y)
    # residualise each ribosomal gene against log library size, re-test
    L = np.c_[np.ones_like(lib), np.log1p(lib)]
    beta, *_ = np.linalg.lstsq(L, Xrib, rcond=None)
    Xrib_resid = Xrib - L @ beta
    resid = cv_auc(Xrib_resid, y)
    res["depth_control"] = {"library_size_auc": round(float(libsize_auc), 3),
                            "ribosomal_only_auc": round(base, 3),
                            "ribosomal_only_auc_after_residualising_libsize": round(resid, 3),
                            "note": "if separation survives residualising on library size, it is not a "
                                    "pure depth artefact; library_size_auc shows how much depth alone separates"}

    # (4) cell-cycle / proliferation confound (round-3 Reviewer F #3)
    gset = set(genes)
    acc = AnnData(X=Xn.copy()); acc.var_names = list(genes); acc.var_names_make_unique()
    sc.tl.score_genes_cell_cycle(acc, s_genes=[g for g in S_GENES if g in gset],
                                 g2m_genes=[g for g in G2M_GENES if g in gset])
    s_sc = acc.obs["S_score"].to_numpy(); g2m_sc = acc.obs["G2M_score"].to_numpy()
    cc_auc = roc_auc_score(y, g2m_sc); cc_auc = max(cc_auc, 1 - cc_auc)
    Xrib_cc = residualise(Xrib, [s_sc, g2m_sc])
    res["cellcycle_control"] = {
        "s_genes_used": int(sum(g in gset for g in S_GENES)),
        "g2m_genes_used": int(sum(g in gset for g in G2M_GENES)),
        "g2m_score_auc": round(float(cc_auc), 3),
        "ribosomal_only_auc": round(base, 3),
        "ribosomal_only_auc_after_residualising_cellcycle": round(cv_auc(Xrib_cc, y), 3),
        "note": "depth residualisation does not cover proliferation; if the ribosomal separation survives "
                "partialling the S/G2M scores it is not a pure cell-cycle readout"}

    # (5) effectorness continuum (round-3 Reviewer G #1): diffusion pseudotime as a graded naive->effector axis
    ae = AnnData(X=Xn.copy()); ae.var_names = list(genes); ae.var_names_make_unique()
    sc.pp.highly_variable_genes(ae, n_top_genes=2000); ae = ae[:, ae.var.highly_variable].copy()
    sc.pp.scale(ae, max_value=10); ae.X = np.nan_to_num(ae.X)
    sc.tl.pca(ae, n_comps=50); sc.pp.neighbors(ae, n_neighbors=15, use_rep="X_pca"); sc.tl.diffmap(ae)
    naive_idx = np.where(raw.y == raw.label_names.index("Naive"))[0]
    dm = ae.obsm["X_diffmap"][:, 1:11]; mem_centroid = dm[y == 1].mean(0)
    d_to_mem = np.linalg.norm(dm - mem_centroid, axis=1)        # root = naive cell furthest from memory
    ae.uns["iroot"] = int(naive_idx[np.argmax(d_to_mem[naive_idx])]); sc.tl.dpt(ae)
    dpt = ae.obs["dpt_pseudotime"].to_numpy(); dpt = np.nan_to_num(dpt, nan=float(np.nanmedian(dpt)))
    dpt_auc = roc_auc_score(y, dpt); dpt_auc = max(dpt_auc, 1 - dpt_auc)
    Xrib_eff = residualise(Xrib, [dpt])
    Xrib_all = residualise(Xrib, [np.log1p(lib), s_sc, g2m_sc, dpt])
    res["effectorness_control"] = {
        "dpt_pseudotime_auc": round(float(dpt_auc), 3),
        "ribosomal_only_auc": round(base, 3),
        "ribosomal_only_auc_after_residualising_effectorness": round(cv_auc(Xrib_eff, y), 3),
        "ribosomal_only_auc_after_residualising_depth_cellcycle_effectorness": round(cv_auc(Xrib_all, y), 3),
        "note": "diffusion-pseudotime proxy for the naive->effector continuum; if the ribosomal separation "
                "collapses after partialling it, the memory signal is largely continuum position, if it "
                "survives it is memory identity beyond effectorness"}

    json.dump(res, open(f"{OUT}/review_controls.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
