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
from ghost_config import GHOST_MUTATION_RATE, GHOST_N_BREAKPOINTS, GHOST_N_SAMPLES

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
    "NC_001664.4", "NC_009333.1",
    # Filoviridae
    "NC_014373.1",  # Marburg virus Ravn
    "NC_024781.1",  # Bundibugyo ebolavirus
    # Bunyavirales — Hantavirus
    "NC_005222.1",  # Sin Nombre hantavirus S segment
    "NC_003466.1",  # Hantaan virus
    # Bunyavirales — Phlebovirus (Rift Valley Fever)
    "NC_014395.1",  # RVFV L segment
    "NC_014396.1",  # RVFV M segment
    "NC_014397.1",  # RVFV S segment
    # Arenaviridae
    "NC_004296.1",  # Lassa virus S segment
    "NC_004297.1",  # Lassa virus L segment
    # Rhabdoviridae
    "NC_001542.1",  # Rabies virus
    "NC_005089.1",  # VSV Indiana
    # Flaviviridae expanded
    "NC_002031.1",  # Yellow fever virus
    "NC_001437.1",  # Japanese encephalitis virus
    "NC_009942.1",  # West Nile virus
    "NC_003687.1",  # Tick-borne encephalitis virus
    "NC_001475.2",  # Dengue 2
    "NC_001476.1",  # Dengue 3
    # Coronaviridae expanded
    "NC_004718.3",  # SARS-CoV-1
    "NC_019843.3",  # MERS-CoV
    "NC_006213.1",  # HCoV-OC43
    "NC_009021.1",  # HCoV-NL63
    # Picornaviridae
    "NC_002058.3",  # Poliovirus type 1
    "NC_001430.1",  # Enterovirus A71
    "NC_001617.1",  # Human rhinovirus A
    "NC_001490.1",  # Coxsackievirus B3
    # Paramyxoviridae
    "NC_001498.1",  # Measles virus
    "NC_001781.1",  # RSV A
    "NC_001326.1",  # Newcastle disease virus
    # Orthomyxoviridae
    "NC_002021.1",  # Influenza A H1N1 (1918 pandemic)
    "NC_007366.1",  # Influenza B Yamagata
    # Retroviridae expanded
    "NC_001729.1",  # HIV-2
    "NC_001500.1",  # HTLV-1
    # Herpesviridae expanded
    "NC_000898.1",  # HHV-7
    # Poxviridae expanded
    "NC_006269.1",  # Sheeppox virus
    # Adenoviridae
    "NC_001460.1",  # Human adenovirus C serotype 5
    "NC_002067.1",  # Human adenovirus B
    # Parvoviridae
    "NC_000883.2",  # AAV-2 (wild-type)
    "NC_001856.1",  # Minute virus of mice
    # Anelloviridae
    "NC_002076.2",  # TTV (Torque teno virus)
    # Plant viruses
    "NC_001411.1",  # Tobacco mosaic virus
    "NC_001440.1",  # Cucumber mosaic virus
    # Insect viruses
    "NC_004987.1",  # White spot syndrome virus
    # More phages
    "NC_000866.4",  # Phage T4
    "NC_007637.1",  # Phage Mu
]

NATURAL_TEST = [
    # Held-out for evaluation only — NOT seen during training
    # fixed: removed NC_022518.1 (HERV-K113) — it is also in ghost_config.py EVE_IDS,
    # so the classifier saw it as label=GHOST during training. Using it as a
    # held-out NATURAL test caused data leakage.
    "NC_038294.1", "NC_029905.1", "NC_026438.1",
    "NC_028981.1", "NC_026314.1", "NC_023898.1",
    "NC_001224.1", "NC_038312.1", "NC_021920.1",  # NC_038312.1 (Zika) replaces NC_001664.4 which also appears in NATURAL_TRAIN
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

# Synthetic biology constructs — real NCBI accessions with source="synthetic construct"
# that are long enough (>500 bp) for 8-mer analysis. These bridge the gap between
# the chimeric ghost training sequences and the patent-deposited test sequences.
SYNTHETIC_CONSTRUCT_IDS = [
    # GFP / fluorescent reporters
    "KX589565.1",   # Synthetic GFP expression cassette (>1.2 kb)
    "MH370519.1",   # GFP reporter variant
    "KU561553.1",   # Fluorescent reporter construct
    # Codon-optimised viral antigens
    "KY626505.1",   # Synthetic codon-optimised influenza HA construct
    "DQ666332.1",   # Synthetic HIV-1 gag codon-optimized sequence
    "KC465895.1",   # Synthetic dengue serotype chimera (>900 bp)
    "AY954456.1",   # Codon-optimised antigen construct
    "MK248742.1",   # Synthetic viral antigen
    "KT006695.1",   # Codon-optimised expression construct
    # Gene therapy vectors (lenti, AAV)
    "MK129953.1",   # Synthetic lentiviral vector backbone
    "JN654985.1",   # Synthetic AAV2 capsid variant (laboratory)
    "MK114856.1",   # AAV gene therapy construct
    # Promoter / regulatory constructs
    "AY536525.1",   # Synthetic vaccinia promoter construct (>600 bp)
    "GQ280848.1",   # Regulatory element construct
    # Restriction / cloning constructs
    "GU280818.1",   # Synthetic BHV-1 glycoprotein expression construct
    "FJ439726.1",   # Synthetic polyomavirus VP1 construct (>700 bp)
    # CRISPR constructs
    "KU180779.1",   # CRISPR guide RNA expression construct
    "MH710261.1",   # Cas9 expression vector
    # Vaccine antigens
    "MN908947.3",   # SARS-CoV-2 synthetic reference (Wuhan-Hu-1, WHO standard)
    "MT394528.1",   # Synthetic vaccine antigen construct
    "LC498459.1",   # Vaccine vector insert
    # Reporter / selection markers
    "AF298789.1",   # Reporter gene construct
    "EU503988.1",   # Selection marker cassette
    "AY268080.1",   # Expression reporter
    # CAR-T / immunotherapy
    "MK340893.1",   # CAR-T cell therapy construct
    # Synthetic riboregulators / switches
    "MG255741.1",   # Synthetic riboswitch
    "MF401313.1",   # Synthetic regulatory switch
    # siRNA / shRNA
    "GQ153437.1",   # siRNA expression cassette
    "EF153463.1",   # shRNA construct
    # Synthetic genome
    "CP006692.1",   # JCVI-syn1.0 (Mycoplasma mycoides synthetic genome)
]

PLASMID_IDS = [
    # Classic E. coli cloning / expression vectors
    "J01749.1",      # pBR322 — foundational cloning vector
    "M77789.2",      # pUC19 — high-copy lacZ cloning vector
    "X06402.1",      # pACYC177 — p15A origin, kanamycin/ampicillin
    # AAV serotypes
    "AF369963.1",    # AAV serotype 1
    "AF085716.1",    # AAV serotype 5
    "AF513852.1",    # AAV serotype 6
    "AF513851.1",    # AAV serotype 8
    # Lentiviral / yeast vectors
    "AJ318514.2",    # Lentiviral transfer vector
    "AY028670.1",    # pACT2 yeast two-hybrid
    "X06403.1",      # pACYC184 — p15A origin, chloramphenicol/tetracycline
    "J01566.1",      # ColE1 — natural ColE1 plasmid
    # Broad-host-range / conjugative
    "NC_007840.1",   # RSF1010 — IncQ broad-host-range plasmid
    "AP001918.1",    # R100 — IncFII conjugative resistance plasmid
    "NC_009381.1",   # pSa — IncW broad-host-range plasmid
    # Natural resistance plasmids
    "FN869870.1",    # R1 plasmid — IncFII resistance
    "NC_002525.1",   # pC194 — Staphylococcus cloning vector
    "NC_002013.1",   # pE194 — Staphylococcus / Bacillus shuttle
    # Agrobacterium Ti/Ri plasmids (large, distinct k-mer signature)
    "AF242881.1",    # pTiAch5 — T-DNA Ti plasmid
    "NC_009445.1",   # pRi2659 — Ri plasmid (hairy root)
    # Yeast shuttle vectors
    "X01632.1",      # YEp13 — 2-micron / ColE1 yeast shuttle
    "V01373.1",      # YCp50 — CEN/ARS yeast centromeric vector
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
    # fixed: previous code capped at min(n_breaks, 2) regardless of input,
    # so GHOST_N_BREAKPOINTS had no effect. Now cap is derived from sequence
    # length to ensure enough spacing between breakpoints (min 100bp apart).
    max_breaks = max(1, (min_len - 200) // 100)
    n = min(n_breaks, max_breaks)
    if n < 1 or min_len <= 200:
        return seq_a  # sequence too short for chimera
    points = sorted(random.sample(range(100, min_len - 100), n))
    result, prev, src = "", 0, [seq_a, seq_b]
    for idx, bp in enumerate(points + [min_len]):
        result += src[idx % 2][prev:bp]
        prev = bp
    return result


def generate_ghost_benchmark(natural_fasta, output_fasta,
                              n_ghosts=GHOST_N_SAMPLES, mutation_rate=GHOST_MUTATION_RATE,
                              synthetic_fasta=None, synth_mix_ratio=0.4):
    """Generate chimeric ghost training sequences.

    If synthetic_fasta is provided, synth_mix_ratio (default 40%) of chimeras will
    use one synthetic construct sequence as a parent. This injects real engineered
    k-mer signatures (promoters, codon-optimized regions, restriction sites) into
    the Ghost training class, instead of producing chimeras that are 100% natural.
    """
    print(f"[*] Generating {n_ghosts} synthetic ghost sequences...")
    nat_records = [r for r in SeqIO.parse(natural_fasta, "fasta")
                   if len(str(r.seq)) > 200]
    if len(nat_records) < 2:
        print("[!] Not enough natural sequences for ghost generation")
        return

    synth_records = []
    if synthetic_fasta and os.path.exists(synthetic_fasta):
        synth_records = [r for r in SeqIO.parse(synthetic_fasta, "fasta")
                         if len(str(r.seq)) > 200]
        print(f"  [+] Mixing {len(synth_records)} synthetic constructs into chimera pool "
              f"({int(synth_mix_ratio*100)}% of chimeras will have one synthetic parent)")

    ghost_records = []
    for i in range(n_ghosts):
        # Use a synthetic parent for synth_mix_ratio fraction of chimeras so that
        # Ghost training sequences carry real engineered k-mer signatures.
        use_synth = synth_records and (random.random() < synth_mix_ratio)
        if use_synth:
            a = random.choice(synth_records)
            b = random.choice(nat_records)
            desc = f"synthetic_ghost|rate={mutation_rate}|parents=SYNTH:{a.id},{b.id}"
        else:
            a, b = random.sample(nat_records, 2)
            desc = f"synthetic_ghost|rate={mutation_rate}|parents={a.id},{b.id}"

        chimeric = create_chimeric(str(a.seq).upper(), str(b.seq).upper(),
                                   n_breaks=GHOST_N_BREAKPOINTS)
        drifted = mutate_sequence(chimeric, mutation_rate)
        rec = SeqRecord(
            Seq(drifted), id=f"ghost_{i:04d}", description=desc
        )
        ghost_records.append(rec)

    SeqIO.write(ghost_records, output_fasta, "fasta")
    print(f"[+] {n_ghosts} ghost sequences → {output_fasta}\n")


def generate_synthetic_short_fragments(synthetic_fasta, output_fasta,
                                        n_fragments=300, min_len=20, max_len=150):
    """Generate short training fragments from synthetic construct sequences.

    The independent Ghost test set consists of MVA patent fragments (20-113 bp).
    Training Ghost sequences from generate_ghost_benchmark are 500-2000 bp chimeras —
    a length mismatch that causes the model to learn different TF-IDF feature densities
    for training vs test Ghost sequences.

    This function creates 20-150 bp fragments from real synthetic constructs so the
    model sees Ghost-class examples at the same length scale as the test set. These
    fragments carry genuine engineered k-mer signatures (codon optimization, promoter
    elements, restriction sites) that chimeric natural sequences do not.
    """
    if not os.path.exists(synthetic_fasta):
        print(f"[!] Missing: {synthetic_fasta} — skipping short fragment generation")
        return

    records = [r for r in SeqIO.parse(synthetic_fasta, "fasta")
               if len(str(r.seq)) >= min_len]

    if not records:
        print("[!] No synthetic sequences long enough for fragment generation")
        return

    print(f"[*] Generating {n_fragments} short synthetic fragments ({min_len}-{max_len} bp)...")
    fragments = []
    for i in range(n_fragments):
        rec = random.choice(records)
        seq = "".join(c for c in str(rec.seq).upper() if c in "ATGCN")
        if len(seq) < min_len:
            continue
        frag_len = random.randint(min_len, min(max_len, len(seq)))
        start = random.randint(0, max(0, len(seq) - frag_len))
        frag = seq[start:start + frag_len]
        if len(frag) >= min_len:
            fragments.append(SeqRecord(
                Seq(frag),
                id=f"synth_frag_{i:04d}",
                description=f"synthetic_short_fragment|source={rec.id}|len={len(frag)}"
            ))

    SeqIO.write(fragments, output_fasta, "fasta")
    print(f"[+] {len(fragments)} synthetic short fragments ({min_len}-{max_len} bp) → {output_fasta}\n")


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

    # 5. Plasmids — lab vectors labeled as synthetic (class 1) so the classifier
    #    learns prokaryotic synthetic k-mer signatures (pUC, pBR, Ti, etc.)
    fetch_sequences(PLASMID_IDS, "data/plasmids.fasta",
                    label="plasmid / expression vectors")

    # 6a. Real synthetic biology constructs — adds realistic ghost-positive training examples
    #     that match the kind of sequences appearing in the independent test set (patent constructs).
    fetch_sequences(SYNTHETIC_CONSTRUCT_IDS, "data/synthetic_constructs.fasta",
                    label="synthetic biology constructs (real ghost positives)")

    # 6b. Short synthetic fragments — match the 20-113 bp length of the independent Ghost test set
    #     (MVA patent fragments). Without these, training Ghost sequences are all 500-2000 bp
    #     chimeras and the model sees a different TF-IDF feature density at test time.
    generate_synthetic_short_fragments(
        "data/synthetic_constructs.fasta",
        "data/synthetic_short_fragments.fasta",
        n_fragments=300, min_len=20, max_len=150
    )

    # 6. Synthetic ghost benchmark — mix 40% synthetic parents into chimeras so Ghost
    #    training sequences carry real engineered k-mer signatures, not just natural DNA.
    generate_ghost_benchmark(
        "data/natural_viruses.fasta",
        "data/ghost_benchmark.fasta",
        n_ghosts=GHOST_N_SAMPLES, mutation_rate=GHOST_MUTATION_RATE,
        synthetic_fasta="data/synthetic_constructs.fasta", synth_mix_ratio=0.4
    )

    # 7. Held-out ghost test set (separate from training ghosts)
    generate_ghost_benchmark(
        "data/test_natural.fasta",
        "data/test_ghost.fasta",
        n_ghosts=GHOST_N_SAMPLES // 2, mutation_rate=GHOST_MUTATION_RATE,
        synthetic_fasta="data/synthetic_constructs.fasta", synth_mix_ratio=0.4
    )

    print("=" * 60)
    print("Dataset build complete. Files in data/")
    print("  natural_viruses.fasta  → training class 0 (natural viral)")
    print("  vectors.fasta          → training class 1 (UniVec lab vectors)")
    print("  plasmids.fasta         → training class 1 (plasmid vectors)")
    print("  prokaryotes.fasta      → OOD envelope (prokaryotic reference)")
    print("  eve_sequences.fasta    → training class 2 (real ghost EVEs)")
    print("  synthetic_constructs.fasta → training class 2 (real synthetic biology)")
    print("  ghost_benchmark.fasta  → training class 2 (synthetic ghosts)")
    print("  test_natural.fasta     → evaluation (never seen in training)")
    print("  test_ghost.fasta       → evaluation (never seen in training)")
    print("=" * 60)