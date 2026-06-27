"""Full DeepMapper CNN + integrated-gradients run on mouse naive vs TEM CD4
(Elyahu SCP490). Shows whether unbiased attribution over the full unfiltered
gene set recovers the ribosomal module de novo, the cross-species method claim.

Phase 0 smoke:  python bench/mouse_dm_run.py --per-class 800 --passes 1 --epochs 60
Phase 1 full:   python bench/mouse_dm_run.py --per-class 1500 --passes 3 --epochs 250

Pre-normalises (the runner does not). Memory-safe: balanced subsample first,
then read only those columns. Read-only on the data.
"""
import argparse, json, re, os, sys, random
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydeepmapper.config import DeepMapperConfig, BackboneSpec
from pydeepmapper.runner import run

D = "data/SCP490/"

def is_ribo(g):
    if g.startswith("Rps6k"): return False
    return bool(re.match(r"^Rp[ls]\d", g)) or g in {"Rplp0","Rplp1","Rplp2","Rpsa","Fau"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=800)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone", default="cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mouse_cd4_naive_vs_tem/smoke.json")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    # labels: barcode -> 0 naive / 1 TEM
    lab = {}
    with open(D+"Metadata.txt", encoding="latin-1") as fh:
        fh.readline(); fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            bc, sub = f[0], f[7]
            if sub == "TEM": lab[bc] = 1
            elif sub.startswith("Na") and "Isg" not in sub: lab[bc] = 0
    naive = [b for b,v in lab.items() if v==0]; tem = [b for b,v in lab.items() if v==1]
    k = min(a.per_class, len(tem))
    pick = random.sample(naive, k) + random.sample(tem, k)
    random.shuffle(pick)
    y = np.array([lab[b] for b in pick], dtype=np.int64)
    print(f"subset: naive={ (y==0).sum() } TEM={ (y==1).sum() } (balanced {k}/class)")

    # read only the picked columns (+ GENE), C parser
    print("reading matrix columns ...", flush=True)
    df = pd.read_csv(D+"RawData1.csv", usecols=["GENE"]+pick, index_col="GENE")
    df = df[pick]                                   # enforce pick order -> aligns with y
    genes = df.index.tolist()
    X = df.values.T.astype(np.float32)              # cells x genes
    del df
    # normalise: CPM 1e4 + log1p
    tot = X.sum(1, keepdims=True); tot[tot==0] = 1
    X = np.log1p(X / tot * 1e4).astype(np.float32)
    print(f"X {X.shape} (cells x genes), {len(genes)} genes", flush=True)

    cfg = DeepMapperConfig(
        n_passes=a.passes, epochs=a.epochs, batch_size=a.batch_size,
        backbone=BackboneSpec(kind=a.backbone),
        attribution="integrated_gradients", ig_steps=32, ig_internal_batch=16,
        attribution_samples=64, lr=1e-3, test_size=0.25, seed=a.seed)
    print(f"running: {a.backbone} passes={a.passes} epochs={a.epochs} batch={a.batch_size}", flush=True)
    F = run(X, y, cfg, feature_names=genes)

    print(f"\nper-pass accuracy: {[round(x,3) for x in F.accuracies]}  accepted={F.n_accepted}")
    print(f"stop epochs (plateau): {F.stop_epochs}  (cap was {a.epochs})")
    if F.n_accepted == 0:
        print("NO accepted passes"); return
    order = sorted(range(F.n_features), key=lambda i:-F.median_importance[i])
    top40 = [genes[i] for i in order[:40]]
    rp = [g for g in top40 if is_ribo(g)]
    print(f"top-40 attributed genes, ribosomal fraction: {len(rp)}/40 = {len(rp)/40:.0%}")
    print("top-15:", ", ".join(genes[i] for i in order[:15]))
    print("RP in top-40:", ", ".join(rp))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"accuracies":F.accuracies, "n_accepted":F.n_accepted,
               "ribo_fraction_top40": len(rp)/40, "top40": top40,
               "rp_top40": rp, "per_class": int(k),
               "config": {"passes":a.passes,"epochs":a.epochs,"backbone":a.backbone,
                          "batch_size":a.batch_size}}, open(a.out,"w"), indent=2)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
