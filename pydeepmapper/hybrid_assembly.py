"""De-novo-hybrid assembly, recover NON-DOCUMENTED transcripts and fold them into the
expression "sc table" so DeepMapper sees features a reference-only pipeline discards.

Pipeline (validated on tissue-resident vs circulating memory T-cell bulk RNA, where the
top discriminators turned out to be novel DENOVO transcripts):

    reads --align(reference)--> UNALIGNED reads  (the non-documented fraction)
         --pool--> Trinity de-novo assembly --> novel transcripts, tagged ``DENOVO_*``
    hybrid_ref = reference_cdna + DENOVO  ->  Salmon index + quant  ->  counts matrix
    build_sc_table(...) -> transcripts × samples, each transcript flagged is_denovo

This is an optional DeepMapper *data-ingestion* path: instead of starting from a given matrix,
build the matrix (incl. non-documented features) from raw reads. The PURE core below
(namespacing + table assembly) is property-tested; the heavy tool orchestration (HISAT2/Trinity/Salmon/biopython) is gated behind
dependency checks and requirements-bio.txt so importing this module stays cheap.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Mapping, Sequence

DENOVO_PREFIX = "DENOVO_"


class BioToolUnavailable(RuntimeError):
    """A required external bioinformatics tool (hisat2/Trinity/salmon) is not on PATH."""


# =============================================================================
#  PURE CORE, deterministic, no I/O, no external tools  (contracted + tested)
# =============================================================================
def is_denovo(transcript_id: str) -> bool:
    """True iff ``transcript_id`` is a de-novo (non-documented) transcript."""
    return str(transcript_id).startswith(DENOVO_PREFIX)


def tag_denovo(transcript_ids: Sequence[str]) -> List[str]:
    """Namespace de-novo transcript ids with ``DENOVO_`` (idempotent, never double-tags),
    so they cannot collide with reference ids in the hybrid reference."""
    return [t if is_denovo(t) else DENOVO_PREFIX + t for t in transcript_ids]


def merge_reference_and_denovo(reference_ids: Sequence[str],
                               denovo_ids: Sequence[str]) -> List[str]:
    """Combined transcript id list for the hybrid reference: reference ids untouched,
    de-novo ids tagged. Asserts (via the contract) that the namespaces don't collide."""
    return list(reference_ids) + tag_denovo(denovo_ids)


def build_sc_table(per_sample_counts: Mapping[str, Mapping[str, float]]):
    """Assemble the expression "sc table" (transcripts × samples) from per-sample
    transcript->count maps (e.g. one Salmon ``quant.sf`` per sample).

    Returns a pandas DataFrame indexed by transcript, one column per sample (column order =
    insertion order of ``per_sample_counts``), missing transcripts filled 0. An extra boolean
    column is NOT added here (keep the matrix numeric); use :func:`is_denovo` on the index.
    """
    import pandas as pd
    cols = {s: pd.Series(counts, dtype="float64") for s, counts in per_sample_counts.items()}
    mat = pd.DataFrame(cols).fillna(0.0)
    mat = mat[list(per_sample_counts.keys())]          # preserve sample order
    return mat


def denovo_fraction(transcript_ids: Sequence[str]) -> float:
    """Fraction of ids that are de-novo, a quick diagnostic of how much non-documented
    signal the hybrid step recovered."""
    ids = list(transcript_ids)
    return sum(is_denovo(t) for t in ids) / len(ids) if ids else 0.0


# =============================================================================
#  ORCHESTRATION, side-effecting; needs HISAT2 / Trinity / Salmon / biopython
# =============================================================================
def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise BioToolUnavailable(
            f"'{tool}' not found on PATH. The de-novo path needs the Python extras AND the external tools:\n"
            f"  pip install 'deepmapper[denovo]'                    # biopython, pandas\n"
            f"  conda install -c bioconda hisat2 trinity salmon     # the assemblers (not pip-installable)")
    return path


def _require_biopython():
    """Clear message if the de-novo Python extra is missing."""
    try:
        import Bio  # noqa: F401
    except ImportError as e:
        raise BioToolUnavailable(
            "biopython is required for the de-novo path; install with: pip install 'deepmapper[denovo]'") from e


def align_capture_unaligned(reads_fastq: str, hisat2_index: str, out_dir: str,
                            strandness: str = "R") -> str:
    """Align ``reads_fastq`` to the reference with HISAT2 and capture the UNALIGNED reads, the non-documented fraction. Returns the path to the unaligned FASTA."""
    _require("hisat2")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(reads_fastq))[0]
    un_fastq = os.path.join(out_dir, f"unaligned_{base}.fastq")
    sam = os.path.join(out_dir, f"aligned_{base}.sam")
    subprocess.run(["hisat2", "-x", hisat2_index, "-U", reads_fastq, "--rna-strandness",
                    strandness, "-S", sam, "--un", un_fastq], check=True)
    un_fasta = os.path.join(out_dir, f"unaligned_{base}.fasta")
    _fastq_to_fasta(un_fastq, un_fasta)
    return un_fasta


def denovo_assemble(pooled_unaligned_fasta: str, out_dir: str, max_memory: str = "50G",
                    cpu: int = 16) -> str:
    """Trinity de-novo assembly of the pooled unaligned reads. Returns the path to the
    DENOVO-tagged FASTA (ready to concatenate into the hybrid reference)."""
    _require("Trinity")
    subprocess.run(["Trinity", "--seqType", "fa", "--single", pooled_unaligned_fasta,
                    "--max_memory", max_memory, "--CPU", str(cpu), "--output", out_dir],
                   check=True)
    assembled = out_dir + ".Trinity.fasta" if not out_dir.endswith(".Trinity.fasta") else out_dir
    coded = os.path.join(os.path.dirname(out_dir) or ".", "denovo_coded.fasta")
    _prefix_fasta_ids(assembled, coded, DENOVO_PREFIX)
    return coded


def build_hybrid_reference(reference_fasta: str, denovo_fasta: str, out_fasta: str,
                           salmon_index_dir: str, kmer: int = 15) -> str:
    """Concatenate reference + DENOVO into the hybrid reference and build a Salmon index.
    Returns the index directory."""
    _require("salmon")
    with open(out_fasta, "wb") as out:
        for src in (reference_fasta, denovo_fasta):
            with open(src, "rb") as f:
                shutil.copyfileobj(f, out)
    subprocess.run(["salmon", "index", "-t", out_fasta, "-i", salmon_index_dir,
                    "-k", str(kmer)], check=True)
    return salmon_index_dir


def quantify(reads_by_sample: Mapping[str, str], salmon_index_dir: str, out_dir: str,
             threads: int = 16) -> Dict[str, str]:
    """Salmon quant each sample against the hybrid index. Returns {sample: quant.sf path}."""
    _require("salmon")
    out = {}
    for sample, fastq in reads_by_sample.items():
        qd = os.path.join(out_dir, f"quant_{sample}")
        subprocess.run(["salmon", "quant", "-i", salmon_index_dir, "-l", "A", "-r", fastq,
                        "-p", str(threads), "--minAssignedFrags", "1", "-o", qd], check=True)
        out[sample] = os.path.join(qd, "quant.sf")
    return out


def read_quant(quant_sf: str) -> Dict[str, float]:
    """Read a Salmon ``quant.sf`` into {transcript: NumReads}."""
    import pandas as pd
    df = pd.read_csv(quant_sf, sep="\t", usecols=["Name", "NumReads"])
    return dict(zip(df["Name"], df["NumReads"]))


def run_hybrid_assembly(reads_by_sample: Mapping[str, str], hisat2_index: str,
                        reference_fasta: str, work_dir: str) -> "object":
    """End-to-end orchestration → the expression sc table (transcripts × samples), with the
    recovered DENOVO transcripts included. Each step is idempotent enough to resume."""
    os.makedirs(work_dir, exist_ok=True)
    un_dir = os.path.join(work_dir, "denovo")
    unaligned = [align_capture_unaligned(fq, hisat2_index, un_dir)
                 for fq in reads_by_sample.values()]
    pooled = os.path.join(un_dir, "pooled_unaligned.fasta")
    with open(pooled, "wb") as out:
        for u in unaligned:
            with open(u, "rb") as f:
                shutil.copyfileobj(f, out)
    denovo = denovo_assemble(pooled, os.path.join(work_dir, "trinity_denovo"))
    index = build_hybrid_reference(reference_fasta, denovo,
                                   os.path.join(work_dir, "combined.fasta"),
                                   os.path.join(work_dir, "combined_index"))
    quants = quantify(reads_by_sample, index, un_dir)
    table = build_sc_table({s: read_quant(q) for s, q in quants.items()})
    return table


# -------------------------------- small fasta/fastq helpers ------------------
def _fastq_to_fasta(fastq_path: str, fasta_path: str) -> None:
    _require_biopython()
    from Bio import SeqIO
    with open(fasta_path, "w") as out:
        for rec in SeqIO.parse(fastq_path, "fastq"):
            out.write(f">{rec.id}\n{rec.seq}\n")


def _prefix_fasta_ids(in_fasta: str, out_fasta: str, prefix: str) -> None:
    with open(in_fasta) as fin, open(out_fasta, "w") as fout:
        for line in fin:
            fout.write(">" + prefix + line[1:] if line.startswith(">") else line)
