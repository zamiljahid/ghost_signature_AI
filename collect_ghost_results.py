
import os
import sys
import argparse
import tempfile
from pathlib import Path
import pandas as pd
from Bio import SeqIO

# Ensure main pipeline components are visible if running from alternative directories
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from main import GhostSignatureReport
    from ghost_config import (
        INDEPENDENT_TEST_NATURAL, INDEPENDENT_TEST_VECTOR, INDEPENDENT_TEST_GHOST
    )
except ImportError:
    print("[ERROR] Could not find 'main.py' or 'GhostSignatureReport'. "
          "Make sure this script resides in your project root directory.")
    sys.exit(1)


def _clean_fasta(src: Path) -> Path:
    """
    Writes a localized clean temporary copy of the target dataset, skipping any
    blank lines or leading malformed whitespace blocks before the initial '>' header
    to guarantee clean parsing by BioPython.
    """
    lines = src.read_bytes().decode("utf-8", errors="replace").splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith(">")), 0)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, encoding="utf-8"
    )
    tmp.write("".join(lines[start:]))
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _combine_independent_test_sets() -> Path:
    """
    Combines all three independent test sets into a single temporary FASTA file.
    Embeds GHOST_SRC=<LABEL> in each header so _parse_metadata() can assign
    ground-truth labels even when headers are plain GenBank accessions.
    Returns path to the combined FASTA.
    """
    combined_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, encoding="utf-8"
    )

    test_sets = [
        (INDEPENDENT_TEST_NATURAL, "NATURAL"),
        (INDEPENDENT_TEST_VECTOR,  "VECTOR"),
        (INDEPENDENT_TEST_GHOST,   "GHOST"),
    ]

    total_seqs = 0
    for test_file, label in test_sets:
        if os.path.exists(test_file):
            for rec in SeqIO.parse(test_file, "fasta"):
                combined_tmp.write(f">{rec.id} GHOST_SRC={label}\n{str(rec.seq)}\n")
                total_seqs += 1

    combined_tmp.flush()
    combined_tmp.close()

    print(f"[INFO] Combined {total_seqs} sequences from all independent test sets")
    return Path(combined_tmp.name)


def _parse_metadata(seq_id: str, description: str) -> tuple[str, str]:
    """
    Extracts true labels and classification difficulty criteria dynamically
    by checking for targeted sequence attributes anywhere inside the fasta records.
    """
    import re
    combined_header = f"{seq_id} {description}".upper()

    # 1. Determine True Label
    # Priority: explicit GHOST_SRC tag injected when combining independent test sets
    src_match = re.search(r"GHOST_SRC=(NATURAL|VECTOR|GHOST)", combined_header)
    if src_match:
        label = src_match.group(1)
    elif "NATURAL" in combined_header:
        label = "NATURAL"
    elif "VECTOR" in combined_header:
        label = "VECTOR"
    elif "GHOST" in combined_header:
        label = "GHOST"
    else:
        # Fallback tracking if strict structural configurations are hidden
        parts = seq_id.split("_")
        if len(parts) >= 3 and parts[2].upper() in {"NATURAL", "VECTOR", "GHOST"}:
            label = parts[2].upper()
        else:
            label = "UNKNOWN"

    # 2. Determine Difficulty Slice
    difficulty = "UNKNOWN"
    for opt in ["EASY", "MEDIUM", "HARD", "CHALLENGING"]:
        if opt in combined_header:
            difficulty = opt
            break

    if difficulty == "UNKNOWN":
        parts = seq_id.split("_")
        if len(parts) >= 4:
            difficulty = parts[3].upper()

    return label, difficulty


def collect(fasta_path: Path, output_csv: Path) -> None:
    if not fasta_path.exists():
        print(f"[ERROR] Input dataset target file not found at: {fasta_path}")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize FASTA input stream to protect the strict standard parse loop
    clean_fasta = _clean_fasta(fasta_path)

    try:
        # ── Build structural metadata map from FASTA headers ─────────────────
        metadata_map: dict[str, tuple[str, str]] = {}
        for rec in SeqIO.parse(clean_fasta, "fasta"):
            metadata_map[rec.id] = _parse_metadata(rec.id, rec.description)

        total = len(metadata_map)
        print(f"\n[INFO] Identified {total} matching reference sequences.")

        # ── Run Analysis Suite ────────────────────────────────────────────────
        print(f"[INFO] Initializing Multi-Engine analysis over dataset: {fasta_path.name}")

        # Capture absolute path reference before passing to bypass context adjustments
        abs_clean_fasta = clean_fasta.resolve()

        # Context guard against internal relative execution pathways
        original_cwd = os.getcwd()
        os.chdir(Path(__file__).resolve().parent)

        try:
            report = GhostSignatureReport(query_fasta=abs_clean_fasta)
            results, _ = report.analyze()
        finally:
            os.chdir(original_cwd) # Always revert working directory cleanly

        # ── Collect results into rows ─────────────────────────────────────────
        rows = []
        for r in results:
            seq_id = r["id"]
            true_lbl, difficulty = metadata_map.get(seq_id, ("UNKNOWN", "UNKNOWN"))

            # Support variations in hit tables structures (DataFrames vs Lists)
            if hasattr(r["hits"], "empty"):
                blast_hits = len(r["hits"]) if not r["hits"].empty else 0
            else:
                blast_hits = len(r["hits"]) if r["hits"] else 0

            # ── Multi-engine fused suspicion score S_fused ────────────────────
            # OOD has ZERO voting power: its coefficient is 0.0 (ablation showed the
            # OOD score, AUROC 0.248, inverts rankings and dragged the fused system
            # to 0.564). The remaining engine weights are renormalised to sum to 1 so
            # S_fused stays in [0, 1]. ood_score is still emitted as a passive column.
            P_kmer  = r.get("ai_risk",  0.0) / 100.0
            P_blast = min(blast_hits / 10.0, 1.0)   # blast_hits computed above
            P_motif = 0.0   # placeholder
            P_codon = 0.0   # placeholder
            # ood weight dropped to 0.0; {0.45,0.20,0.05,0.05} renormalised over /0.75.
            weights = {"kmer": 0.60, "ood": 0.0, "blast": 0.2667, "motif": 0.0667, "codon": 0.0667}
            S_fused = round(
                weights["kmer"]  * P_kmer  +
                weights["blast"] * P_blast +
                weights["motif"] * P_motif +
                weights["codon"] * P_codon,
                4)

            rows.append({
                "id":         seq_id,
                "true_label": true_lbl,
                "difficulty": difficulty,
                "ai_risk":    round(r.get("ai_risk", 0.0), 2),
                "ood_score":  round(r.get("ood_score", 0.0), 2),
                "blast_hits": blast_hits,
                "S_fused":    S_fused,
                "verdict":    r.get("verdict", "UNKNOWN"),
                "length_bp":  r.get("length", 0),
                "gc_pct":     round(r.get("gc", 0.0), 2),
            })

        if not rows:
            print("[WARN] Analysis loop finalized with an empty metrics table.")
            return

        df = pd.DataFrame(rows, columns=[
            "id", "true_label", "difficulty",
            "ai_risk", "ood_score", "blast_hits", "S_fused",
            "verdict", "length_bp", "gc_pct",
        ])

        df.to_csv(output_csv, index=False)

        print(f"\n{'='*85}")
        print(f"BENCHMARK / EVALUATION PROFILE DATA ({len(df)} Sequences Captured)")
        print(f"{'='*85}")
        print(df[["id", "true_label", "difficulty",
                  "ai_risk", "ood_score", "blast_hits", "verdict"]].to_string(index=False))
        print(f"{'='*85}")
        print(f"\n[DONE] Diagnostic pipeline results saved → {output_csv}")

    finally:
        clean_fasta.unlink(missing_ok=True)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir / "outputs" / "benchmark" / "ghost_results_independent.csv"

    parser = argparse.ArgumentParser(description="Automated Benchmarking Suite for Ghost Signature Framework (Independent Test Sets).")
    parser.add_argument("--output", type=str, default=str(default_output), help="Output path for evaluation summary CSV")

    args = parser.parse_args()

    # Automatically combine all independent test sets
    print("[INFO] Using independent test sets (Natural, Vector, Ghost)")
    combined_fasta = _combine_independent_test_sets()
    try:
        collect(combined_fasta, Path(args.output))
    finally:
        combined_fasta.unlink(missing_ok=True)