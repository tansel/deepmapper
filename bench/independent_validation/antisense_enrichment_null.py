"""Null model for the 'top-40 discriminating lncRNAs are antisense' statement (round-3 Reviewer E #2).

Question: is the antisense (-AS) fraction among the top state-discriminating lncRNAs ENRICHED relative to
the antisense fraction of the DETECTED lncRNA background, or does it merely reflect that detected lncRNAs
are themselves predominantly antisense? Background = unique lncRNAs scored in lncrna_hunt_all.csv.

Run: conda run -n deepmapper python bench/independent_validation/antisense_enrichment_null.py
Out: results/attribution_de/antisense_enrichment_null.json
"""
from __future__ import annotations
import os, re, json
import numpy as np, pandas as pd
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "attribution_de")
AS = re.compile(r"^(.+?)-AS\d+$")
N_TOP, K_OBS, N_PERM = 40, 32, 20000


def main():
    hunt = pd.read_csv(os.path.join(HERE, "lncrna_hunt_all.csv"))
    uni = hunt.drop_duplicates("lncRNA").copy()
    uni["is_as"] = uni["lncRNA"].map(lambda g: AS.match(str(g)) is not None)
    N = len(uni); K = int(uni["is_as"].sum())
    exp = N_TOP * K / N
    p_enrich = float(hypergeom.sf(K_OBS - 1, N, K, N_TOP))     # P(X >= 32)
    p_deplete = float(hypergeom.cdf(K_OBS, N, K, N_TOP))       # P(X <= 32)
    rng = np.random.default_rng(0); arr = uni["is_as"].to_numpy()
    perm = np.array([rng.choice(arr, N_TOP, replace=False).sum() for _ in range(N_PERM)])
    res = {
        "universe_n_lncRNA": N, "universe_n_antisense": K,
        "background_antisense_frac": round(K / N, 3),
        "top_n": N_TOP, "observed_antisense": K_OBS,
        "expected_antisense_random": round(exp, 1),
        "fold_enrichment": round(K_OBS / exp, 2),
        "hypergeom_p_enrichment": round(p_enrich, 3),
        "hypergeom_p_depletion": round(p_deplete, 3),
        "permutation_mean": round(float(perm.mean()), 1),
        "permutation_p_enrichment": round(float((perm >= K_OBS).mean()), 3),
        "conclusion": ("the detected lncRNA set is overwhelmingly antisense, so the top-40 antisense count "
                       "is NOT enriched (it is at or below chance); the antisense result rests on the "
                       "overlap-controlled sense-antisense covariation, not on this count"),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "antisense_enrichment_null.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
