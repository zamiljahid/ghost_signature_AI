"""
kmer_utils.py — Shared multi-scale k-mer feature extraction.

All files that previously defined their own seq_to_kmers() now import from here.
Multi-scale (k=4, k=6, k=8) with prefixed tokens prevents vocabulary collision
and provides 54 features for a 23 bp sequence (vs 16 with k=8 only).
"""
from ghost_config import KMER_SIZES

MIN_SEQ_LENGTH = min(KMER_SIZES)   # = 4


def seq_to_kmers_multiscale(seq, k_sizes=None):
    """Return space-joined prefixed k-mer tokens across all scales.

    Example (23 bp input):
        'k4_GCGC k4_CGCA ... k6_GCGCAT ... k8_GCGCATTA ...'
    Produces 54 tokens for a 23 bp sequence; 1,485 for a 500 bp sequence.
    """
    if k_sizes is None:
        k_sizes = KMER_SIZES
    words = []
    for k in k_sizes:
        if len(seq) >= k:
            words.extend(f"k{k}_{seq[i:i+k]}" for i in range(len(seq) - k + 1))
    return " ".join(words)
