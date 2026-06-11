"""
generate_plasmids.py
====================
Downloads plasmid sequences from NCBI and saves to data/plasmids.fasta.
Uses Entrez search (more reliable than hardcoded accessions).
Falls back to synthetic lab-vector motifs if NCBI is unavailable.
"""

import os, time, io
import requests
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

Entrez.email = "zamiljahid2002@gmail.com"
os.makedirs("data", exist_ok=True)

OUTPUT = "data/plasmids.fasta"
TARGET = 200   # max records to collect

# ── Approach 1: Entrez esearch for complete plasmid sequences ─────────────────
QUERIES = [
    # Classic cloning / expression vectors — tight size range, complete seqs
    '(pUC19 OR pBR322 OR pACYC OR pET OR pBAD OR pGEM OR pBluescript) '
    'AND "complete sequence"[Title] AND 1000:20000[SLEN]',
    # Broad plasmid search for diversity
    '"complete sequence"[Title] AND plasmid[Title] AND 2000:15000[SLEN] '
    'AND bacteria[filter]',
    # Shuttle / yeast vectors
    '(YEp OR YCp OR pRS OR pYES OR pGAL) AND "complete sequence"[Title] '
    'AND 2000:20000[SLEN]',
    # Agrobacterium / Ti plasmids (large, distinctive k-mer signature)
    '(Ti plasmid OR Ri plasmid OR pTi OR pRi) AND "complete"[Title]',
]

def search_and_fetch(query, max_n=80):
    """Search NCBI nucleotide and fetch up to max_n sequences."""
    records = []
    try:
        print(f"  [SEARCH] {query[:80]}...")
        handle = Entrez.esearch(db="nucleotide", term=query,
                                retmax=max_n, usehistory="y")
        result = Entrez.read(handle); handle.close()
        ids = result.get("IdList", [])
        if not ids:
            print("    No results.")
            return records
        print(f"    Found {len(ids)} IDs — fetching...")
        for attempt in range(3):
            try:
                h = Entrez.efetch(db="nucleotide", id=",".join(ids),
                                  rettype="fasta", retmode="text")
                text = h.read(); h.close()
                recs = list(SeqIO.parse(io.StringIO(text), "fasta"))
                # Filter: keep sequences 1 kb–50 kb (typical plasmid range)
                recs = [r for r in recs
                        if 1_000 <= len(r.seq) <= 50_000]
                records.extend(recs)
                print(f"    Got {len(recs)} valid records.")
                time.sleep(0.5)
                break
            except Exception as e:
                print(f"    Retry {attempt+1}/3 — {e}")
                time.sleep(3)
    except Exception as e:
        print(f"  [ERROR] Search failed: {e}")
    return records


# ── Approach 2: Direct UniVec-subset download from NCBI FTP ──────────────────
UNIVEC_SUBSETS = [
    # Specific well-known vector entries from UniVec (reliable FTP)
    "https://ftp.ncbi.nlm.nih.gov/pub/UniVec/UniVec_Core",
]

def fetch_univec_core(output_tmp="data/_univec_core.fasta"):
    """Download UniVec Core (smaller, curated) as a plasmid supplement."""
    records = []
    for url in UNIVEC_SUBSETS:
        try:
            print(f"  [UNIVEC] Downloading UniVec_Core...")
            r = requests.get(url, timeout=60, stream=True)
            with open(output_tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            all_recs = list(SeqIO.parse(output_tmp, "fasta"))
            # UniVec_Core contains primers, adaptors, vectors — keep 500bp+
            recs = [r for r in all_recs if len(r.seq) >= 500]
            print(f"    UniVec_Core: {len(recs)} sequences ≥500 bp")
            records.extend(recs)
            os.remove(output_tmp)
        except Exception as e:
            print(f"  [WARN] UniVec_Core download failed: {e}")
    return records


# ── Approach 3: Synthetic fallback motifs ────────────────────────────────────
def synthetic_fallback():
    """Write known lab vector motifs as last resort."""
    print("  [FALLBACK] Generating synthetic vector motifs...")
    motifs = {
        "T7_promoter_region":   "TAATACGACTCACTATAGGG" * 50,
        "CMV_promoter":         ("TGACATTGATTATTGACTAGTTATTAATAGTAATCAATTA"
                                 "CGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGCGT"
                                 "TACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCC") * 6,
        "Kanamycin_resistance": ("ATGAGCCATATTCAACGGGAAACGTCTTGCTCGAGGCCG"
                                 "CGATTAAATTCCAACATGGATGCTGATTTATATGGGTATA"
                                 "AATGGGCTCGCGATAATGTCGGGCAATCAGGTGCGACAA") * 8,
        "Ampicillin_resistance":("ATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTT"
                                 "TTTGCGGCATTTTGCTTTCCCTTTATTTGCGGCATTTTG"
                                 "CTTTCCCTTTATGAGTATTCAACATTTCCGTGTCGCCCT") * 8,
        "ColE1_origin":         ("TTTTCAGGGCAAGGGCATGACAAAAACGCGTAACAAAAG"
                                 "TGTCTATAATCAGGGCTTTTTTGCCCTCAGAACCCCGGGT"
                                 "TTTCAGGGCAAGGGCATGACAAAAACGCGTAACAAAAGTG") * 7,
        "pUC_lacZ_MCS":         ("ATGACCATGATTACGCCAAGCTTGCATGCCTGCAGGTCG"
                                 "ACGGATCCCCCGGGAATTCGAGCTCCGTCGACAAGCTTGC"
                                 "ATGCCTGCAGGTCGACGGATCCCCCGGGAATTCGAGCTC") * 8,
        "pET_T7_terminator":    ("GCTAGTTATTGCTCAGCGGTGGCAGCAGCCAACTCAGCT"
                                 "TGGTTTTTTCATGACCAAAATCCTTAACAATGTGTTGGCA"
                                 "GCTAGTTATTGCTCAGCGGTGGCAGCAGCCAACTCAGCTT") * 7,
        "SV40_poly_A":          ("AACTTGTTTATTGCAGCTTATAATGGTTACAAATAAAGCA"
                                 "ATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCACTG"
                                 "CATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTA") * 6,
        "f1_ori":               ("ACGCGTTTTGAATGAGTTTTTTCTGAGTTTATTGATTTT"
                                 "CGTTCAAATATGTATCTGCATCAGATCCCCATGTGCAGCTG"
                                 "ACGCGTTTTGAATGAGTTTTTTCTGAGTTTATTGATTT") * 8,
        "pBR_tetracycline":     ("ATGAAATCTAACAATGCAGACTTACGCTTTTTCTTTCTG"
                                 "TGGCTATCTTTTTCATTTATCGCAGCTATCAATCCAGGTG"
                                 "ATGAAATCTAACAATGCAGACTTACGCTTTTTCTTTCTGT") * 7,
    }
    records = []
    for name, seq in motifs.items():
        rec = SeqRecord(Seq(seq), id=name,
                        description=f"synthetic_plasmid_vector|motif={name}")
        records.append(rec)
    return records


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PLASMID FASTA GENERATOR")
    print("=" * 60)

    all_records = []
    seen_ids    = set()

    # 1. Entrez search
    for q in QUERIES:
        if len(all_records) >= TARGET:
            break
        recs = search_and_fetch(q, max_n=60)
        for r in recs:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                all_records.append(r)
        print(f"  Running total: {len(all_records)} records")

    # 2. UniVec Core supplement
    if len(all_records) < TARGET // 2:
        recs = fetch_univec_core()
        for r in recs:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                all_records.append(r)
        print(f"  Running total after UniVec_Core: {len(all_records)} records")

    # 3. Synthetic fallback if still empty
    if not all_records:
        print("[WARN] NCBI unreachable — using synthetic motif fallback.")
        all_records = synthetic_fallback()

    # Cap at TARGET
    all_records = all_records[:TARGET]

    SeqIO.write(all_records, OUTPUT, "fasta")
    sizes = [len(r.seq) for r in all_records]
    print(f"\n[+] {len(all_records)} plasmid sequences → {OUTPUT}")
    if sizes:
        print(f"    Length range: {min(sizes):,} – {max(sizes):,} bp  "
              f"| mean: {sum(sizes)//len(sizes):,} bp")
    print("=" * 60)
