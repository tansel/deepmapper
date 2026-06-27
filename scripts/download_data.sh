#!/usr/bin/env bash
# DeepMapper verification datasets — downloader for VERIFIED public scRNA-seq sources.
# All URLs returned HTTP 200 on 2026-06-20. See docs/data-sources.md for provenance & ground-truth notes.
#
# Usage:
#   ./download_data.sh [target_dir]        # default: ./data
#   GROUP=cd4 ./download_data.sh           # only one group: cd4 | cd8 | other_sorted | kang | fresh_frozen | benchmark | all (default)
set -euo pipefail

DEST="${1:-./data}"
GROUP="${GROUP:-all}"
mkdir -p "$DEST"

CF="https://cf.10xgenomics.com/samples/cell"

get() {  # get <url> [outfile]
  local url="$1"; local out="${2:-$DEST/$(basename "$url")}"
  if [[ -s "$out" ]]; then echo "skip (exists): $out"; return; fi
  echo ">> $url"
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
}

tenx() {  # tenx <slug>   -> sorted/PBMC filtered matrix tar.gz
  local slug="$1"
  get "$CF/$slug/${slug}_filtered_gene_bc_matrices.tar.gz" "$DEST/${slug}_filtered_gene_bc_matrices.tar.gz"
}

want() { [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; }

# --- Case 1: CD4 subsets (FACS ground truth) ---
if want cd4; then
  tenx cd4_t_helper
  tenx regulatory_t
  tenx naive_t
  tenx memory_t
fi

# --- Case 2: CD8 subsets (FACS ground truth) ---
if want cd8; then
  tenx cytotoxic_t
  tenx naive_cytotoxic
fi

# --- Other sorted PBMC populations (extra multi-class GT) ---
if want other_sorted; then
  tenx b_cells
  tenx cd14_monocytes
  tenx cd56_nk
  tenx cd34
  # Pre-merged sorted populations + labels (fast path, R .rds):
  get "$CF/pbmc68k_rds/all_pure_select_11types.rds" "$DEST/all_pure_select_11types.rds"
  get "$CF/pbmc68k_rds/pbmc68k_data.rds"            "$DEST/pbmc68k_data.rds"
  # Large (1.5 GB) — uncomment if needed:
  # get "$CF/pbmc68k_rds/all_pure_pbmc_data.rds"     "$DEST/all_pure_pbmc_data.rds"
fi

# --- Case 3: Kang 2018 IFN-beta stim vs ctrl (GSE96583) ---
if want kang; then
  get "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_RAW.tar" "$DEST/GSE96583_RAW.tar"
  get "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz" "$DEST/GSE96583_batch2.total.tsne.df.tsv.gz"
  get "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.genes.tsv.gz" "$DEST/GSE96583_batch2.genes.tsv.gz"
fi

# --- Case 4: Fresh vs Frozen PBMC (Donor A) ---
if want fresh_frozen; then
  tenx fresh_68k_pbmc_donor_a
  tenx frozen_pbmc_donor_a
fi

# --- Case 5: General benchmarks ---
if want benchmark; then
  tenx pbmc3k
  tenx pbmc4k
  tenx pbmc8k
  # PBMC68k & sorted labels come via other_sorted (all_pure_select_11types.rds, pbmc68k_data.rds)
  echo "NOTE: scanpy/pertpy loaders (sc.datasets.pbmc3k_processed, sc.datasets.pbmc68k_reduced, pertpy.dt.kang_2018) fetch on first call from Python; no wget needed."
fi

echo "Done. Files in: $DEST"
