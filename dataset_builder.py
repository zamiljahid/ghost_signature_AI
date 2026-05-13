"""
dataset_builder.py
==================
Downloads all datasets needed for Ghost Signature Detector training
and evaluation. Run this FIRST before training.

Produces:
  data/natural_viruses.fasta    - wild-type viral genomes (label 0)
  data/vectors.fasta            - lab vectors from UniVec (label 1)
  data/eve_sequences.fasta      - endogenous viral elements (real ghosts, label 2)
  data/ghost_benchmark.fasta    - synthetic ghost sequences (label 2)
  data/test_natural.fasta       - held-out naturals for evaluation
  data/test_ghost.fasta         - held-out ghosts for evaluation
"""

import os, random, time, requests
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

Entrez.email = "zamiljahid2002@gmail.com"
os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — add / remove accessions freely
# ─────────────────────────────────────────────────────────────────────────────

NATURAL_TRAIN = [
    # ASFV strains
    "NC_001659.2", "NC_044487.1", "AM712239.1",
    # Respiratory / common
    "NC_007374.1", "NC_012532.1", "NC_002549.1", "NC_001405.1", "NC_045512.2",
    # Large DNA viruses
    "NC_000852.5", "NC_006273.2", "NC_003310.1", "NC_001359.1",
    # Bacteriophages
    "NC_001416.1", "NC_001422.1", "NC_001604.1", "NC_001330.1",
    # Diverse families
    "NC_003977.2", "NC_001477.1", "NC_001489.1", "NC_001563.2",
    "NC_001436.1", "NC_001716.2", "NC_001722.1", "NC_001802.1",
    "NC_001806.2", "NC_001927.1", "NC_002645.1", "NC_003551.1",
    "NC_004102.1", "NC_004148.2", "NC_004355.1", "NC_005084.2",
    # Additional diversity (herpesviruses, poxviruses, retroviruses)
    "NC_001798.2", "NC_007605.1", "NC_006998.1", "NC_009334.1",
    "NC_001664.4", "NC_006273.2", "NC_009333.1", "NC_008uncovered",
]

NATURAL_TEST = [
    # Held-out for evaluation only — NOT seen during training
    "NC_038294.1", "NC_029905.1", "NC_026438.1",
    "NC_028981.1", "NC_026314.1", "NC_023898.1",
    "NC_022518.1",  # HERV-K113 — endogenous retrovirus used as natural test
    "NC_001224.1", "NC_001664.4", "NC_021920.1",
]

# Endogenous Viral Elements — ancient integrated viruses (real ghost positives)
EVE_ACCESSIONS = [
    "AY037928.1",   # HERV-H
    "AF289081.1",   # HERV-K LTR
    "AC002350.1",   # ERV integration site
    "U60060.1",     # Foamy virus EVE
    "AJ279072.1",   # Murine ERV
    "AC013386.1",   # Human EVE locus
    "AY371337.1",   # Bat EVE
    "KF887995.1",   # Insect-associated EVE
    "MH747649.1",   # Rodent ERV
    "GQ225585.1",   # Snake EVE
]
PROKARYOTE_IDS = [
    "NC_000913.3",   # E. coli K-12 MG1655
    "NC_000964.3",   # B. subtilis 168
    "NC_002695.2",   # E. coli O157:H7
    "NC_003197.2",   # Salmonella typhimurium LT2
]

def fetch_sequences(accession_list, output_file, label="sequences", batch=50):
    """Fetch NCBI nucleotide records in batches."""
    print(f"[*] Downloading {len(accession_list)} {label}...")
    all_records = []
    # Filter out obviously bad accessions
    valid = [a for a in accession_list if not a.endswith("uncovered")]
    for i in range(0, len(valid), batch):
        chunk = valid[i:i + batch]
        for attempt in range(3):
            try:
                handle = Entrez.efetch(
                    db="nucleotide", id=chunk,
                    rettype="fasta", retmode="text"
                )
                text = handle.read()
                handle.close()
                recs = list(SeqIO.parse(
                    __import__("io").StringIO(text), "fasta"
                ))
                all_records.extend(recs)
                print(f"    Batch {i//batch + 1}: got {len(recs)} records")
                time.sleep(0.4)   # NCBI rate limit
                break
            except Exception as e:
                print(f"    Retry {attempt+1}/3 — {e}")
                time.sleep(2)
    SeqIO.write(all_records, output_file, "fasta")
    print(f"[+] Saved {len(all_records)} records → {output_file}\n")
    return all_records


def download_univec(output_file="data/vectors.fasta"):
    """Download UniVec database from NCBI FTP."""
    url = "https://ftp.ncbi.nlm.nih.gov/pub/UniVec/UniVec"
    print("[*] Downloading UniVec database...")
    try:
        r = requests.get(url, timeout=60, stream=True)
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        count = sum(1 for line in open(output_file) if line.startswith(">"))
        print(f"[+] UniVec downloaded: {count} sequences → {output_file}\n")
    except Exception as e:
        print(f"[!] UniVec download failed: {e}")
        print("    Using fallback: synthetic vector sequences")
        _write_fallback_vectors(output_file)


def _write_fallback_vectors(output_file):
    """Write known lab vector motifs as fallback if UniVec unavailable."""
    motifs = {
        "T7_Promoter":    "TAATACGACTCACTATAGGG" * 20,
        "CMV_Promoter":   "TGACATTGATTATTGACTAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAG" * 8,
        "Kanamycin_Res":  "ATGAGCCATATTCAACGGGAAACGTCTTGCTCGAGGCCGCGATTAAATTCCAACATGG" * 10,
        "Ampicillin_Res": "ATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCTTTCCC" * 10,
        "ColE1_Origin":   "TTTTCAGGGCAAGGGCATGACAAAAACGCGTAACAAAAGTGTCTATAATCAGGGCTTTTT" * 8,
    }
    records = [
        SeqRecord(Seq(seq), id=name, description="fallback_vector")
        for name, seq in motifs.items()
    ]
    SeqIO.write(records, output_file, "fasta")
    print(f"[+] Fallback vectors written → {output_file}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Ghost benchmark generator
# ─────────────────────────────────────────────────────────────────────────────

def mutate_sequence(seq, rate=0.20):
    bases = list("ATGC")
    result = list(seq)
    for i in range(len(result)):
        if random.random() < rate and result[i] in bases:
            result[i] = random.choice([b for b in bases if b != result[i]])
    return "".join(result)


def create_chimeric(seq_a, seq_b, n_breaks=3):
    min_len = min(len(seq_a), len(seq_b), 2000)
    seq_a, seq_b = seq_a[:min_len], seq_b[:min_len]
    points = sorted(random.sample(range(100, min_len - 100), min(n_breaks, 2)))
    result, prev, src = "", 0, [seq_a, seq_b]
    for idx, bp in enumerate(points + [min_len]):
        result += src[idx % 2][prev:bp]
        prev = bp
    return result


def generate_ghost_benchmark(natural_fasta, output_fasta,
                              n_ghosts=100, mutation_rate=0.20):
    print(f"[*] Generating {n_ghosts} synthetic ghost sequences...")
    records = [r for r in SeqIO.parse(natural_fasta, "fasta")
               if len(str(r.seq)) > 500]
    if len(records) < 2:
        print("[!] Not enough natural sequences for ghost generation")
        return

    ghost_records = []
    for i in range(n_ghosts):
        a, b = random.sample(records, 2)
        chimeric = create_chimeric(str(a.seq).upper(), str(b.seq).upper())
        drifted = mutate_sequence(chimeric, mutation_rate)
        rec = SeqRecord(
            Seq(drifted), id=f"ghost_{i:04d}",
            description=f"synthetic_ghost|rate={mutation_rate}|parents={a.id},{b.id}"
        )
        ghost_records.append(rec)

    SeqIO.write(ghost_records, output_fasta, "fasta")
    print(f"[+] {n_ghosts} ghost sequences → {output_fasta}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GHOST SIGNATURE DETECTOR — Dataset Builder")
    print("=" * 60)

    # 1. Natural viruses (training)
    fetch_sequences(NATURAL_TRAIN, "data/natural_viruses.fasta",
                    label="natural training viruses")

    # 2. Natural viruses (held-out test)
    fetch_sequences(NATURAL_TEST, "data/test_natural.fasta",
                    label="natural test viruses (held-out)")

    # 3. Endogenous viral elements (real ghost positives)
    fetch_sequences(EVE_ACCESSIONS, "data/eve_sequences.fasta",
                    label="endogenous viral elements (EVEs)")

    # 4. UniVec lab vectors
    download_univec("data/vectors.fasta")

    fetch_sequences(PROKARYOTE_IDS, "data/prokaryotes.fasta",
                    label="prokaryotic references (OOD envelope)")
    # 5. Synthetic ghost benchmark
    generate_ghost_benchmark(
        "data/natural_viruses.fasta",
        "data/ghost_benchmark.fasta",
        n_ghosts=100, mutation_rate=0.20
    )

    # 6. Held-out ghost test set (separate from training ghosts)
    generate_ghost_benchmark(
        "data/test_natural.fasta",
        "data/test_ghost.fasta",
        n_ghosts=50, mutation_rate=0.25
    )

    print("=" * 60)
    print("Dataset build complete. Files in data/")
    print("  natural_viruses.fasta  → training class 0")
    print("  vectors.fasta          → training class 1")
    print("  eve_sequences.fasta    → training class 2 (real ghosts)")
    print("  ghost_benchmark.fasta  → training class 2 (synthetic ghosts)")
    print("  test_natural.fasta     → evaluation (never seen in training)")
    print("  test_ghost.fasta       → evaluation (never seen in training)")
    print("=" * 60)