"""Co-quantification control for the antisense-lncRNA finding (the feasible substitute for a literal
strand-aware re-quant, which is impossible: SMART-seq2 is unstranded and the raw reads are ~7.5 TB).

The worry: a sense-antisense correlation could be an artefact of reads from overlapping loci being
counted for both genes. Test: does the correlation track genomic OVERLAP? If the saturating pairs
(rho~0.97) overlap heavily and the moderate reproducible pairs (rho~0.6) overlap little/none, the
moderate concordances are biological. Coordinates from mygene (Ensembl/NCBI)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def coords(symbols):
    import mygene
    mg = mygene.MyGeneInfo()
    out = mg.querymany(list(symbols), scopes="symbol", fields="genomic_pos,genomic_pos_hg19",
                       species="human", verbose=False)
    pos = {}
    for r in out:
        q = r.get("query")
        gp = r.get("genomic_pos") or r.get("genomic_pos_hg19")
        if isinstance(gp, list):
            gp = gp[0]
        if gp and "chr" in gp:
            pos[q] = (str(gp["chr"]), int(gp["start"]), int(gp["end"]), int(gp.get("strand", 0)))
    return pos


def overlap_bp(a, b):
    if a is None or b is None or a[0] != b[0]:
        return 0, 0.0
    lo = max(a[1], b[1]); hi = min(a[2], b[2]); ov = max(0, hi - lo)
    span = min(a[2] - a[1], b[2] - b[1]) or 1
    return ov, ov / span


def main():
    try:
        import mygene  # noqa
    except ImportError:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mygene"])
    cc = pd.read_csv(os.path.join(HERE, "lncrna_antisense_coordination.csv"))
    genes = set(cc["antisense_lncRNA"]) | set(cc["sense_gene"])
    pos = coords(genes)
    rows = []
    for _, r in cc.iterrows():
        a = pos.get(r["antisense_lncRNA"]); b = pos.get(r["sense_gene"])
        ov, frac = overlap_bp(a, b)
        rows.append((r["antisense_lncRNA"], r["sense_gene"], r["rho"], ov, round(frac, 2),
                     "" if a and b else "no-coords"))
    R = pd.DataFrame(rows, columns=["antisense", "sense", "rho", "overlap_bp", "overlap_frac", "note"])
    R = R.sort_values("rho", ascending=False)
    R.to_csv(os.path.join(HERE, "antisense_overlap_control.csv"), index=False)
    ok = R[R.note == ""]
    print(R.to_string(index=False), flush=True)
    if len(ok) > 3:
        rho_ov = np.corrcoef(ok["rho"], ok["overlap_frac"])[0, 1]
        print(f"\nSpearman-ish corr(rho, overlap_frac) over {len(ok)} pairs with coords: {rho_ov:.3f}", flush=True)
    KEY = ["ZEB2-AS1", "LEF1-AS1", "ENTPD1-AS1", "IL21R-AS1", "ADORA2A-AS1", "CD27-AS1"]
    print("\n=== key pairs: overlap vs correlation ===")
    print(R[R.antisense.isin(KEY)][["antisense", "sense", "rho", "overlap_frac", "note"]].to_string(index=False))
    print("\nReading: moderate-rho key pairs with LOW overlap_frac -> correlation is biological, not "
          "read co-quantification.", flush=True)


if __name__ == "__main__":
    main()
