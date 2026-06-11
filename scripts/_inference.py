"""
scripts/_inference.py — Shared inference helper for the Phase-3 runner scripts.
Loads the trained Ghost Signature model + vectorizer and produces the
3-class probability matrix for the independent test set. No metric value is
computed or hardcoded here — this only turns FASTA → (y_true, y_pred, y_proba).
"""
from __future__ import annotations

import os
import sys
import numpy as np
from Bio import SeqIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from ghost_config import (
    MODEL_PATH, VECTORIZER_PATH, CHUNK_SIZE,
    LABEL_NATURAL, LABEL_VECTOR, LABEL_GHOST,
    INDEPENDENT_TEST_NATURAL, INDEPENDENT_TEST_VECTOR, INDEPENDENT_TEST_GHOST,
)
from kmer_utils import seq_to_kmers_multiscale


def _clean(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in "ATGCN")


def load_independent_test_set() -> tuple[list[str], np.ndarray, list[str]]:
    """Returns (sequences, y_true, seq_ids) for the 360-sequence Phase-3 set."""
    sequences, y_true, seq_ids = [], [], []
    for fasta, label in [
        (INDEPENDENT_TEST_NATURAL, LABEL_NATURAL),
        (INDEPENDENT_TEST_VECTOR,  LABEL_VECTOR),
        (INDEPENDENT_TEST_GHOST,   LABEL_GHOST),
    ]:
        if not (fasta and os.path.exists(fasta)):
            raise FileNotFoundError(f"Independent test FASTA missing: {fasta}")
        for rec in SeqIO.parse(fasta, "fasta"):
            seq = _clean(str(rec.seq))
            if len(seq) >= 4:
                sequences.append(seq)
                y_true.append(label)
                seq_ids.append(rec.id)
    return sequences, np.array(y_true, dtype=int), seq_ids


def predict_proba_matrix(sequences: list[str]) -> np.ndarray:
    """3-class probability matrix (mean over 500-bp windows) of shape (n, 3)."""
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    classes = [int(c) for c in model.classes_]

    out = np.zeros((len(sequences), 3), dtype=float)
    for i, seq in enumerate(sequences):
        window_probas = []
        for j in range(0, max(1, len(seq)), CHUNK_SIZE):
            frag = seq[j:j + CHUNK_SIZE]
            if len(frag) >= 4:
                text = seq_to_kmers_multiscale(frag)
                if text:
                    p = model.predict_proba(vec.transform([text]))[0]
                    window_probas.append(p)
        if window_probas:
            mean_p = np.mean(window_probas, axis=0)
            for cls, val in zip(classes, mean_p):
                out[i, cls] = val
        else:
            out[i, LABEL_NATURAL] = 1.0
    # Renormalise so each row is a proper distribution over the 3 classes.
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, out=np.full_like(out, 1 / 3), where=row_sums > 0)
    return out


def get_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convenience: returns (y_true, y_pred, y_proba, seq_ids)."""
    sequences, y_true, seq_ids = load_independent_test_set()
    print(f"[INFER] Loaded {len(sequences)} independent test sequences "
          f"({np.bincount(y_true).tolist()} per class). Scoring with trained model...")
    y_proba = predict_proba_matrix(sequences)
    y_pred = np.argmax(y_proba, axis=1)
    return y_true, y_pred, y_proba, seq_ids
