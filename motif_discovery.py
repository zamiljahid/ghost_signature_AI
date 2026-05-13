from collections import Counter
import numpy as np


def compute_kmer_freq(sequence, k=6):
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    total = len(kmers)
    return {km: count / total for km, count in Counter(kmers).items()}


def find_ghost_motifs(query_seq, background_seqs, k=6, top_n=10):
    """
    Returns kmers enriched in query vs background.
    background_seqs: list of natural virus sequences (strings).
    """
    q_freq = compute_kmer_freq(query_seq, k)

    # Pool background frequencies
    bg_counts = Counter()
    bg_total = 0
    for seq in background_seqs:
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        bg_counts.update(kmers)
        bg_total += len(kmers)

    enrichment = {}
    for kmer, qf in q_freq.items():
        bg_f = (bg_counts.get(kmer, 0) + 1) / (bg_total + 1)  # +1 Laplace smooth
        enrichment[kmer] = qf / bg_f  # fold enrichment

    ranked = sorted(enrichment.items(), key=lambda x: -x[1])
    return ranked[:top_n]  # [(kmer, fold_enrichment), ...]
