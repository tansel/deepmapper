# Reproducibility map

This file maps every figure and quantitative claim in the DeepMapper T-cell paper to
the script that produces it and the public dataset it consumes. No new data was
generated; everything runs on public accessions.

Install the package and fetch the data first (see the repo `README.md` and
`docs/data-sources.md`), then run each script from the repo root.

## Public datasets

| Accession | Platform | Use |
|---|---|---|
| SRP073767 (Zheng et al. 2017, 10x sorted PBMC) | 10x 3' | sorted CD4+/CD8+ subsets; ribosomal, chord, backbone analyses |
| GSE96583 (Kang et al. 2018) | 10x 3' | IFN-beta-stimulated vs control; interferon signature + per-donor control |
| GSE99254 (Guo et al. 2018, NSCLC) | SMART-seq2 | antisense lncRNA; all-ncRNA chord; exhaustion chord |
| GSE98638 (Zheng et al. 2017, HCC) | SMART-seq2 | antisense replication; exhaustion replication |
| GSE108989 (Zhang et al. 2018, CRC) | SMART-seq2 | antisense replication |
| Elyahu et al. 2019 (Single Cell Portal SCP490) | 10x 3' | mouse CD4 naive vs effector-memory; cross-species ribosomal validation |

## Figure / result to script

| Item | Script |
|---|---|
| Fig 1 (state separation + ribosomal-only) | `bench/ribosomal_validation.py` |
| Fig 2 (HVG discards ribosomal genes) | `bench/hvg_ribosomal_rank.py` |
| Fig 3 (interferon shared genes) | `bench/c5_kang/c5_kang_analysis.py` |
| Fig 4 (antisense overlap control + cross-cohort) | `bench/independent_validation/lncrna_antisense.py`, `antisense_overlap_control.py` |
| Fig 5 (gene chord, held-out) | `bench/gene_chord_honest.py` |
| Fig 6 (all-non-coding chord) | `bench/ncrna_chord.py`, `ncrna_chord_biotype.py` |
| Fig 7 (exhaustion chord + score benchmark) | `bench/independent_validation/exhaustion_til_vs_blood.py`, `exhaustion_vs_score.py` |
| Sec 2.2 confound controls (depth/cycle/effectorness) | `bench/review_controls.py` |
| Sec 2.3 per-donor interferon control | `bench/c5_kang/c5_kang_donor_isg.py` |
| Sec 2.4 antisense enrichment null | `bench/independent_validation/antisense_enrichment_null.py` |
| Sec 3 backbone head-to-head (linear approx cnn) | `bench/backbone_headtohead.py` |
| Sec 3 deterministic linear / passes | `bench/passes_and_determinism.py`, `pydeepmapper/linear_baseline.py` |
| Cross-species mouse (de-novo ribosomal recovery) | `bench/mouse_dm_run.py` |
| Cross-species confound control (scanpy DPT) | `bench/mouse_phase2_dpt.py` |

Each script writes its result to a JSON or CSV file under `results/` (gitignored).
Re-running a script regenerates its output from the public data.

## Environment

- Python 3.9 or newer. Install with `pip install -e ".[all]"`.
- GPU-heavy steps (DeepMapper training and attribution) need torch with Apple MPS or
  CUDA. CPU works but is slow. The sklearn-only controls run on CPU in seconds.
- Typical invocation: `PYTORCH_ENABLE_MPS_FALLBACK=1 python bench/<script>.py`.

## Citation

See `CITATION.cff`. If a release DOI is minted (for example via the Zenodo to GitHub
integration), cite that archive in the Code Availability statement.
