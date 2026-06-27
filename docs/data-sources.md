# DeepMapper, Public scRNA-seq Data Sources for Verification / Benchmarking

Curated, **link-verified** (HTTP 200, see "Verified" column) public single-cell RNA-seq
datasets that back each DeepMapper case study. Every URL below was
checked with an HTTP `HEAD`/`GET` on 2026-06-20 unless explicitly marked **UNVERIFIED**.

Ground-truth (GT) legend:
- **FACS** = cells were antibody/bead-sorted before sequencing → label is experimentally enforced (strongest GT).
- **Design** = condition label is the experimental design (stim/ctrl, fresh/frozen) → reliable GT for *that axis*, not for cell type.
- **Author** = manual/marker-based annotation by original authors (good, but not orthogonal GT).

---

## TL;DR ranking for verifying the CD4 subtype finding (case 1)

| Rank | Dataset | Why | GT |
|------|---------|-----|----|
| 1 | **10x Zheng 2017 FACS-sorted CD4 subsets** (helper / Treg / naive / memory) | This is the DeepMapper "CD4 subset data by 10X"; four populations are antibody-sorted, so the class label is physical, not inferred | **FACS** |
| 2 | 10x Zheng 2017 FACS-sorted CD8 subsets (cytotoxic / naive cytotoxic) | Same source, covers case 2 | **FACS** |
| 3 | `all_pure_select_11types.rds` (Zheng pre-packaged 11 sorted sub-pops) | One 687 KB file = all sorted pops + labels, instant load | **FACS** |
| 4 | Kang 2018 GSE96583 (IFN-β stim vs ctrl) | Case 3 ground-truth condition axis | **Design** |
| 5 | 10x Fresh-68k vs Frozen PBMC (Donor A) | Case 4 fresh/stored axis | **Design** |
| 6 | scanpy `pbmc68k_reduced` / `pbmc3k` | Quick smoke-test / general benchmark | Author (68k) / none (3k) |

---

## Case 1 & 2, CD4+ and CD8+ T-cell subtypes (Zheng et al. 2017, FACS-sorted)

**Citation:** Zheng GXY et al. "Massively parallel digital transcriptional profiling of single cells." *Nature Communications* 8:14049 (2017).
**Ground truth:** **FACS / bead-enriched sort = physical label.** Each population was immuno-sorted *before* droplet capture, so the cell-type label is the sort gate, not a downstream cluster call. This is the strongest GT available for cell-type classification and is exactly the "CD4+ subset data by 10X" the DeepMapper analyses use.
**Format:** 10x CellRanger `filtered_gene_bc_matrices` (`matrix.mtx` + `genes.tsv` + `barcodes.tsv`) inside a `.tar.gz`. Genes ≈ 32,738 (hg19 reference) before filtering.

Direct download URL pattern (all **verified 200 OK**, 2026-06-20):
`https://cf.10xgenomics.com/samples/cell/<slug>/<slug>_filtered_gene_bc_matrices.tar.gz`

### CD4 subsets (case 1 to 4 classes)

| Population (sort gate) | slug | ~Cells | tar.gz size | Verified |
|---|---|---|---|---|
| CD4+ Helper T | `cd4_t_helper` | ~11,213 | 21.0 MB | ✅ 200 |
| CD4+/CD25+ **Regulatory T (Treg)** | `regulatory_t` | ~10,263 | 19.3 MB | ✅ 200 |
| CD4+/CD45RA+/CD25− **Naïve T** | `naive_t` | ~10,479 | 17.6 MB | ✅ 200 |
| CD4+/CD45RO+ **Memory T** | `memory_t` | ~10,224 | 20.0 MB | ✅ 200 |

### CD8 subsets (case 2 to 2 classes)

| Population (sort gate) | slug | ~Cells | tar.gz size | Verified |
|---|---|---|---|---|
| CD8+ **Cytotoxic T** | `cytotoxic_t` | ~10,209 | 20.0 MB | ✅ 200 |
| CD8+/CD45RA+ **Naïve Cytotoxic T** | `naive_cytotoxic` | ~11,953 | 20.9 MB | ✅ 200 |

> Cell counts are the published Zheng-2017 sorted-population sizes (post-CellRanger filtering); treat as approximate (±a few hundred depending on filtering). The *byte sizes* and HTTP status were directly verified.

**Other sorted PBMC populations from the same release** (useful for multi-class GT / negatives), all **verified 200**:

| Population | slug | tar.gz |
|---|---|---|
| CD19+ B cells | `b_cells` | 18.0 MB |
| CD14+ Monocytes | `cd14_monocytes` | 4.2 MB |
| CD56+ NK cells | `cd56_nk` | 20.0 MB |
| CD34+ cells | `cd34` | 38.0 MB |

**How to get it (one-liner per population):**
```bash
wget https://cf.10xgenomics.com/samples/cell/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz
# swap slug: regulatory_t | naive_t | memory_t | cytotoxic_t | naive_cytotoxic
```

### Pre-packaged mirror (fastest path to labelled CD4/CD8 subsets)

The official 10x analysis repo ships the sorted populations already merged **with their sort labels**:

| File | Contents | Size | URL | Verified |
|---|---|---|---|---|
| `all_pure_select_11types.rds` | 11 sorted sub-populations + meta/labels | 687 KB (703,532 B) | `https://cf.10xgenomics.com/samples/cell/pbmc68k_rds/all_pure_select_11types.rds` | ✅ 200 |
| `all_pure_pbmc_data.rds` | Full expression of the 10 bead-enriched samples | 1.5 GB | `https://cf.10xgenomics.com/samples/cell/pbmc68k_rds/all_pure_pbmc_data.rds` | ⚠️ pattern confirmed live; size not byte-checked |
| `pbmc68k_data.rds` | ~68k fresh PBMC expression (the PBMC68k set) | ~77 MB (80,452,142 B) | `https://cf.10xgenomics.com/samples/cell/pbmc68k_rds/pbmc68k_data.rds` | ✅ 200 |

Repo / README (download instructions): `https://github.com/10XGenomics/single-cell-3prime-paper` (`pbmc68k_analysis/README.md`), ✅ reachable.

```r
# R, fastest route to labelled CD4/CD8 sorted subsets:
download.file("https://cf.10xgenomics.com/samples/cell/pbmc68k_rds/all_pure_select_11types.rds",
              "all_pure_select_11types.rds")
x <- readRDS("all_pure_select_11types.rds")   # contains the 11 sorted sub-populations + labels
```

---

## Case 3, Stimulated vs unstimulated T cells

Reference numbers: **stimulated 4,400 cells / 16,394 genes; unstimulated 5,092 cells / 16,612 genes.**

### Primary candidate, Kang et al. 2018 (IFN-β stim vs ctrl PBMC), RECOMMENDED

**Citation:** Kang HM et al. "Multiplexed droplet single-cell RNA-sequencing using natural genetic variation." *Nature Biotechnology* 36:89 to 94 (2018).
**Accession:** **GSE96583** (SRA SRP102802). ~15,000 PBMCs, two conditions: control vs IFN-β-stimulated (6 h).
**Ground truth:** **Design** (stim/ctrl is the experimental condition). Author cell-type annotations also provided in the `*.tsne.df.tsv` metadata.

| File | Size | Direct URL | Verified |
|---|---|---|---|
| `GSE96583_RAW.tar` (per-sample 10x mtx) | 72.7 MB (76,195,840 B) | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_RAW.tar` | ✅ 200 |
| same via GEO portal | 72.7 MB | `https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file` | ✅ 200 |
| `GSE96583_batch2.total.tsne.df.tsv.gz` (cell labels + condition) | 738.6 KB | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz` | ⚠️ listed on GEO; not byte-checked |
| `GSE96583_batch2.genes.tsv.gz` | 270.6 KB | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.genes.tsv.gz` | ⚠️ listed on GEO; not byte-checked |

GEO landing page: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583`, ✅ verified, confirms Kang 2018.

**Pre-packaged loaders (same data, easiest):**
```python
# pertpy (Python), bundles the Kang IFN-β PBMC set with stim/ctrl + cell_type labels:
import pertpy as pt
adata = pt.dt.kang_2018()        # AnnData with .obs['condition'] (ctrl/stimulated) + .obs['cell_type']
```
```r
# Seurat / SeuratData (R):
# install.packages("SeuratData"); SeuratData::InstallData("ifnb")
library(ifnb.SeuratData); data("ifnb")   # ifnb$stim = "CTRL"/"STIM"
```
> ⚠️ `pertpy.dt.kang_2018()` and `SeuratData::ifnb` loader names are widely documented but were **not executed/verified** in this environment, confirm the exact function name against your installed package version. The underlying GEO data IS verified.

### Closer match to the ~4,400 / ~5,092 counts (anti-CD3/CD28 T-cell activation)

The Kang set is IFN-β-stimulated PBMCs (~15k cells), not a perfect match to the reference ~4,400/~5,092 split. If DeepMapper used a **T-cell-activation** (anti-CD3/CD28) stim-vs-rest set with those exact counts, the most likely public source is a resting-vs-stimulated **purified T cell** 10x study (e.g. the "stimulated/frozen human PBMC" deep-sequencing benchmark in *Scientific Data* 2023, `s41597-023-02348-z`). **UNVERIFIED, accession not pinned down.** Recommend the author confirm the exact sub-cohort; the 4,400/5,092 numbers strongly suggest a single 10x lane each, which is consistent with one activation experiment rather than the multiplexed Kang design.

---

## Case 4, Fresh vs 24h-old / stored cells (viability / storage)

### Recommended, 10x Fresh-68k vs Frozen PBMC (Donor A)

Same Zheng-2017 1.1.0 release; a matched **fresh** and **frozen** aliquot of one donor, the canonical fresh-vs-stored benchmark.
**Ground truth:** **Design** (fresh vs frozen is the experimental axis). Format: 10x mtx `.tar.gz`.

| Sample | slug | tar.gz | Verified |
|---|---|---|---|
| Fresh 68k PBMC (Donor A) | `fresh_68k_pbmc_donor_a` | 124.4 MB (124,442,812 B) | ✅ 200 |
| Frozen PBMC (Donor A) | `frozen_pbmc_donor_a` | 7.5 MB (7,452,623 B) | ✅ 200 |

```bash
wget https://cf.10xgenomics.com/samples/cell/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
wget https://cf.10xgenomics.com/samples/cell/frozen_pbmc_donor_a/frozen_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
```

### Time-delay (fresh vs N-hour-old) literature sources

For a **fresh vs 24h-delayed-processing** axis specifically (as opposed to frozen), the published benchmarks are:
- Genome Biology 2017 "Single-cell transcriptome conservation in cryopreserved cells and tissues", `https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1171-9` (✅ page reachable; GEO accession in paper, **not pinned here, UNVERIFIED accession**).
- "Effects of Cryopreservation and Thawing on Single-Cell Transcriptomes of Human T Cells" (PMC7458795), relevant to T cells specifically; **accession UNVERIFIED**.

> If "24h-old" is literal (delayed processing, not freezing), confirm the exact GEO accession from the DeepMapper methods; the 10x fresh/frozen pair above is the closest single-download GT.

---

## Case 5, General benchmarking sets with ground truth

| Dataset | Loader / URL | Cells × Genes | GT | Verified |
|---|---|---|---|---|
| **PBMC3k** (10x healthy donor) | `import scanpy as sc; sc.datasets.pbmc3k()`, or `https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz` (7.6 MB) | 2,700 × 32,738 | None built-in (annotate via tutorial); `sc.datasets.pbmc3k_processed()` adds louvain labels | ✅ 200 (tar) |
| **PBMC68k** (Zheng 2017) | `sc.datasets.pbmc68k_reduced()` (subset 700×765 w/ labels) or `pbmc68k_data.rds` above | reduced 700 × 765 (full ~68k) | **Author** label key `bulk_labels` (reduced set) | ✅ (rds 200) |
| **Kang 2018 ifnb** | `pertpy.dt.kang_2018()` / `SeuratData ifnb` / GSE96583 | ~15,000 × ~35k | Design (stim/ctrl) + Author cell_type | ✅ GEO 200; loader UNVERIFIED |
| **Tabula Muris** | `https://figshare.com/projects/Tabula_Muris_Transcriptomic_characterization_of_20_organs_and_tissues/27733` | ~100k mouse | Author (FACS plate + droplet) | ⚠️ UNVERIFIED (figshare project id) |
| **Tabula Sapiens** | `https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219` / CELLxGENE | ~500k human | Author | ⚠️ UNVERIFIED |
| **CellTypist Immune_All** (training/annotation models) | `https://www.celltypist.org/models` |, | Author-curated | ⚠️ UNVERIFIED |
| **scIB integration benchmark sets** | `https://github.com/theislab/scib` (Immune_ALL_human etc.) | varies | Author | ⚠️ UNVERIFIED |

```python
import scanpy as sc
adata3k  = sc.datasets.pbmc3k()              # 2700 x 32738, raw
adata3kp = sc.datasets.pbmc3k_processed()    # adds 'louvain' cell-type labels
adata68k = sc.datasets.pbmc68k_reduced()     # 700 x 765.obs['bulk_labels'] = GT
```
> `scanpy.datasets` loader names/dimensions verified against scanpy docs; not executed locally.

---

## What is verified vs not

- **Fully verified (HTTP 200 + byte size):** all 6 Zheng CD4/CD8 sorted tar.gz; b_cells / cd14_monocytes / cd56_nk / cd34; pbmc3k/4k/8k; fresh & frozen Donor A; `pbmc68k_data.rds`; `all_pure_select_11types.rds`; GSE96583_RAW.tar (both mirrors).
- **Reachable but byte size not re-checked:** `all_pure_pbmc_data.rds` (1.5 GB), GSE96583 per-batch `.tsv.gz` metadata files, GenomeBiology cryo paper page.
- **UNVERIFIED (names from docs/literature, not executed/pinned here):** `pertpy.dt.kang_2018()`, `SeuratData::ifnb`, Tabula Muris/Sapiens figshare IDs, CellTypist model URLs, scIB sets, the exact 4,400/5,092 T-cell-activation accession, and the fresh-vs-24h delayed-processing GEO accession.

See `scripts/download_data.sh` for an executable downloader covering all verified URLs.
