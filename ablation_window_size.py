from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from ghost_config import (
    NATURAL_FASTA, VECTORS_FASTA, GHOST_FASTA,
    LABEL_NATURAL, LABEL_VECTOR, LABEL_GHOST, TFIDF_MAX_FEATURES,
)
from kmer_utils import seq_to_kmers_multiscale, MIN_SEQ_LENGTH
from metrics_utils import delong_ci

WINDOW_SIZES = [250, 500, 1000, 2000]
MAX_GENOMES_PER_CLASS = 60      # cap for tractability; sweep measures separability
OUTPUT_DIR = Path("outputs") / "ablation_window"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["Natural", "Vector", "Ghost"]


def _clean(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in "ATGCN")


def load_sequences() -> tuple[list[str], list[int], list[str]]:
    """Load raw (un-windowed) sequences + labels + genome ids from training FASTAs."""
    seqs, labels, gids = [], [], []
    for fasta, label in [
        (NATURAL_FASTA, LABEL_NATURAL),
        (VECTORS_FASTA, LABEL_VECTOR),
        (GHOST_FASTA,   LABEL_GHOST),
    ]:
        if not os.path.exists(fasta):
            print(f"  [!] Missing {fasta} — skipping")
            continue
        n = 0
        for rec in SeqIO.parse(fasta, "fasta"):
            if n >= MAX_GENOMES_PER_CLASS:
                break
            s = _clean(str(rec.seq))
            if len(s) >= MIN_SEQ_LENGTH:
                seqs.append(s)
                labels.append(label)
                gids.append(rec.id)
                n += 1
    return seqs, labels, gids


def chunk_sequences(sequences, labels, genome_ids, window_size, step=None):
    """Window every sequence; each window inherits its source genome id (group key)."""
    step = step or window_size
    X, y, g = [], [], []
    for seq, lab, gid in zip(sequences, labels, genome_ids):
        for i in range(0, max(1, len(seq) - window_size + 1), step):
            frag = seq[i:i + window_size]
            if len(frag) >= MIN_SEQ_LENGTH:
                X.append(seq_to_kmers_multiscale(frag))
                y.append(lab)
                g.append(gid)
    return X, np.array(y, dtype=int), np.array(g)


def run_ablation(sequences, labels, genome_ids, window_sizes=WINDOW_SIZES):
    records = []
    for ws in window_sizes:
        print(f"\n── Window size: {ws} bp ──")
        t0 = time.time()

        X_text, y_windows, g_ids = chunk_sequences(sequences, labels, genome_ids,
                                                   window_size=ws, step=ws)
        if len(set(y_windows)) < 3:
            print(f"  [SKIP] ws={ws}: not all 3 classes produced windows.")
            continue
        print(f"  Generated {len(X_text)} windows from {len(sequences)} genomes")

        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1,
                              max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
        X_feat = vec.fit_transform(X_text)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        fold_recalls = {c: [] for c in CLASS_NAMES}
        # Accumulate out-of-fold predictions for an honest DeLong CI on Natural-vs-rest.
        oof_true, oof_score = [], []

        for fold_idx, (tr, va) in enumerate(sgkf.split(X_feat, y_windows, groups=g_ids)):
            assert len(set(g_ids[tr]) & set(g_ids[va])) == 0, \
                f"DATA LEAKAGE in fold {fold_idx} for ws={ws}!"

            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=42)
            clf.fit(X_feat[tr], y_windows[tr])
            y_proba = clf.predict_proba(X_feat[va])
            y_pred = clf.predict(X_feat[va])

            # Align proba columns to [0,1,2] even if a fold misses a class.
            proba_full = np.zeros((len(va), 3))
            for col, cls in enumerate(clf.classes_):
                proba_full[:, int(cls)] = y_proba[:, col]

            fold_aucs.append(roc_auc_score(y_windows[va], proba_full,
                                           multi_class="ovr", average="macro"))
            for i, cls in enumerate(CLASS_NAMES):
                mask = (y_windows[va] == i)
                fold_recalls[cls].append(
                    float(np.mean(y_pred[mask] == i)) if mask.any() else np.nan)

            oof_true.append((y_windows[va] == 0).astype(int))   # Natural = positive
            oof_score.append(proba_full[:, 0])

        mean_macro_auc = float(np.mean(fold_aucs))   # macro OvR, CV mean
        # DeLong CI is binary, so report it on the Natural-vs-rest OOF scores and
        # use THAT same quantity for the `auroc` column so the CI brackets it.
        y_oof = np.concatenate(oof_true)
        s_oof = np.concatenate(oof_score)
        auc_nat, ci_lo, ci_hi = delong_ci(y_oof, s_oof)   # DeLong CI, not Wilson

        records.append({
            "window_bp":          ws,
            "n_windows":          len(X_text),
            "auroc_natural_ovr":  auc_nat,      # quantity the DeLong CI describes
            "auroc_ci_lo":        ci_lo,
            "auroc_ci_hi":        ci_hi,
            "auroc_macro_cv":     mean_macro_auc,
            "recall_Natural":     float(np.nanmean(fold_recalls["Natural"])),
            "recall_Vector":      float(np.nanmean(fold_recalls["Vector"])),
            "recall_Ghost":       float(np.nanmean(fold_recalls["Ghost"])),
            "train_time_s":       round(time.time() - t0, 1),
        })
        print(f"  AUROC(macro CV)={mean_macro_auc:.4f} | "
              f"AUROC(Nat-OvR)={auc_nat:.4f} [{ci_lo:.4f},{ci_hi:.4f}] | "
              f"R_Nat={records[-1]['recall_Natural']:.4f} | "
              f"R_Vec={records[-1]['recall_Vector']:.4f} | "
              f"R_Gho={records[-1]['recall_Ghost']:.4f}")

    df = pd.DataFrame(records)
    out = OUTPUT_DIR / "ablation_window_size.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nAblation table written to {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("=" * 70)
    print(" Window-size sensitivity ablation (group-aware CV + DeLong CI)")
    print("=" * 70)
    seqs, labels, gids = load_sequences()
    print(f"[DATA] Loaded {len(seqs)} genomes "
          f"({np.bincount(labels).tolist()} per class).")
    run_ablation(seqs, labels, gids)
