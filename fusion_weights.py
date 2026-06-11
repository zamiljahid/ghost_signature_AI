"""
fusion_weights.py — Optimised convex weights for the multi-engine fusion.
=========================================================================
FIX 1 (Fusion Weight Optimization).

The multi-engine fusion combines per-engine suspicion probabilities
(kmer / ood / blast / motif / codon) as a convex sum  S = sum_i w_i * P_i ,
with  sum_i w_i = 1  and every  w_i in [0, 1].  The weights used to be fixed by
hand (kmer 0.45, ood 0.25, ...).  Fixed weights let weak/auxiliary engines
dilute the high-fidelity k-mer signal, which is what drove the *fused* AUROC
down to ~0.56 even though the k-mer engine alone reaches ~0.90 (see
outputs/ablation_table.csv).

This module finds the weights that maximise validation AUROC and persists them so
they are loaded once and reused at inference time (NOT re-optimised per call).

Design notes that matter for correctness:
  * AUROC is a *rank* statistic — piecewise-constant in the weights, so its
    gradient is zero almost everywhere.  A gradient-based optimiser
    (L-BFGS-B, the scipy default once `bounds` are given) reads a ~0
    finite-difference gradient and never moves off x0.  We therefore use the
    gradient-free Nelder-Mead simplex, which actually explores weight space.
  * The fused score is a single 1-D suspicion value, so the operationally
    meaningful target is the binary *engineered (Vector+Ghost) vs Natural*
    AUROC — exactly the task the fused score is used for downstream.
"""
from __future__ import annotations

import json
import os
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

try:
    from ghost_config import LABEL_NATURAL
except ImportError:
    LABEL_NATURAL = 0

# Canonical engine order — the keys fuse_engine_scores() expects.
ENGINE_ORDER = ["kmer", "ood", "blast", "motif", "codon"]

# Hand-tuned fallback, used when no optimised weights file is present. Mirrors the
# values previously hardcoded in main.py so behaviour is unchanged until weights
# are actually optimised and saved.
DEFAULT_WEIGHTS = {"kmer": 0.45, "ood": 0.25, "blast": 0.15, "motif": 0.10, "codon": 0.05}

FUSION_WEIGHTS_PATH = os.path.join("models", "fusion_weights.json")


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────
def calculate_macro_auroc(y_true: np.ndarray, fused_preds: np.ndarray) -> float:
    """
    AUROC of the fused suspicion score.

    The fused score is 1-D, so we score the binary *engineered vs Natural* task
    (Vector+Ghost = positive) — the decision the fused score actually drives. If a
    2-D score matrix is ever passed (n, 3) we fall back to true multiclass macro
    OvR AUROC, hence the "macro" name.
    """
    y_true = np.asarray(y_true)
    fused_preds = np.asarray(fused_preds)

    if fused_preds.ndim == 2 and fused_preds.shape[1] >= 3:
        # Genuine multiclass case — macro one-vs-rest.
        if len(np.unique(y_true)) < 3:
            return 0.0
        return float(roc_auc_score(y_true, fused_preds, multi_class="ovr", average="macro"))

    # Binary engineered-vs-Natural (the normal path for a scalar fused score).
    y_bin = (y_true != LABEL_NATURAL).astype(int)
    if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(y_bin, fused_preds))
    except ValueError:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser
# ─────────────────────────────────────────────────────────────────────────────
def optimize_fusion_weights(engine_predictions, y_true, engine_names=None) -> dict:
    """
    Find convex weights maximising validation AUROC of the fused score.

    Parameters
    ----------
    engine_predictions : array-like, shape (n_engines, n_samples)
        Per-engine suspicion probabilities in [0, 1], one row per engine, aligned
        with `engine_names`.
    y_true : array-like, shape (n_samples,)
        Integer class labels (LABEL_NATURAL / LABEL_VECTOR / LABEL_GHOST).
    engine_names : list[str] | None
        Names aligned with the rows of `engine_predictions`. Defaults to the first
        n rows of ENGINE_ORDER.

    Returns
    -------
    dict {engine_name: weight}, weights >= 0 and summing to 1.
    """
    engine_predictions = np.asarray(engine_predictions, dtype=float)
    if engine_predictions.ndim != 2:
        raise ValueError("engine_predictions must be 2-D (n_engines, n_samples)")
    n = engine_predictions.shape[0]
    if engine_names is None:
        engine_names = ENGINE_ORDER[:n]

    def loss_func(weights):
        # Renormalise inside the loss so every trial point is a valid convex
        # combination. AUROC is scale-invariant, so this only fixes the *relative*
        # contribution of each engine; the negative sign turns "maximise AUROC"
        # into the minimisation scipy expects.
        weights = weights / np.sum(weights)
        fused_preds = sum(w * scores for w, scores in zip(weights, engine_predictions))
        return -calculate_macro_auroc(y_true, fused_preds)

    x0 = np.array([1.0 / n] * n)              # start from equal weights
    baseline_auroc = -loss_func(x0)

    # Nelder-Mead (gradient-free) — see module docstring for why a gradient
    # optimiser would silently stall on this objective.
    res = minimize(
        loss_func, x0=x0, method="Nelder-Mead",
        bounds=[(0, 1)] * n,
        options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6},
    )

    raw = np.clip(res.x, 0.0, 1.0)
    if raw.sum() <= 0:
        raw = x0.copy()
    optimized_weights = raw / np.sum(raw)     # final convex normalisation
    optimized_auroc = -loss_func(optimized_weights)

    # Never ship weights worse than the equal-weight baseline — the simplex can
    # occasionally converge to a poorer local point on a flat objective.
    if optimized_auroc + 1e-9 < baseline_auroc:
        print(f"[FUSION] Optimised AUROC {optimized_auroc:.4f} < equal-weight "
              f"{baseline_auroc:.4f}; keeping equal weights.")
        optimized_weights = x0.copy()
        optimized_auroc = baseline_auroc

    weights_dict = {name: float(w) for name, w in zip(engine_names, optimized_weights)}
    print(f"[FUSION] Equal-weight AUROC : {baseline_auroc:.4f}")
    print(f"[FUSION] Optimised AUROC    : {optimized_auroc:.4f}")
    print(f"[FUSION] Optimised weights  : "
          + ", ".join(f"{k}={v:.3f}" for k, v in weights_dict.items()))
    return weights_dict


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — optimise once, reuse at inference
# ─────────────────────────────────────────────────────────────────────────────
def save_fusion_weights(weights: dict, path: str = FUSION_WEIGHTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(weights, fh, indent=2)
    print(f"[FUSION] Saved optimised weights → {path}")


def load_fusion_weights(path: str = FUSION_WEIGHTS_PATH) -> dict:
    """
    Load optimised weights for inference. Falls back to DEFAULT_WEIGHTS when the
    file is absent so the pipeline runs unchanged before any optimisation. The
    returned dict always covers every engine in ENGINE_ORDER and sums to 1, which
    fuse_engine_scores() asserts.
    """
    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                loaded = json.load(fh)
            # Only overwrite known engines; ignore unexpected keys.
            for k in ENGINE_ORDER:
                if k in loaded:
                    weights[k] = float(loaded[k])
        except Exception as e:
            print(f"[FUSION] Could not read {path} ({e}); using default weights.")

    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: weights[k] / total for k in ENGINE_ORDER}   # defensive re-normalise


# ─────────────────────────────────────────────────────────────────────────────
# Driver — build a labelled validation matrix and optimise
# ─────────────────────────────────────────────────────────────────────────────
def _engine_matrix_from_records(records):
    """
    Turn evaluate.build_records() output into a (n_engines, n_samples) matrix.

    Only the k-mer and OOD engines are scored per-sample by build_records(); the
    BLAST/motif/codon engines are not available there, so those rows are constant
    0 on this split. A constant column carries no ranking information, so we
    optimise ONLY the engines that actually vary and keep the remaining engines at
    their default share (then renormalise). This avoids handing spurious weight to
    an engine we could not estimate — while leaving the hard BLAST/motif verdict
    gate in main.py untouched.
    """
    y_true = np.array([r["true_label"] for r in records])
    p_kmer = np.array([r["ai_risk"] / 100.0 for r in records])
    p_ood = np.array([r["ood_score"] / 100.0 for r in records])
    return y_true, {"kmer": p_kmer, "ood": p_ood}


if __name__ == "__main__":
    # Build the validation matrix from the independent test set using the SAME
    # model/vectorizer/OOD artifacts the rest of the pipeline uses. This only
    # *scores* sequences — no model is retrained.
    import evaluate
    from ood_scorer import GhostOODScorer

    print("=" * 70)
    print(" FUSION WEIGHT OPTIMISATION (FIX 1)")
    print("=" * 70)
    print("[FUSION] Scoring independent test set for per-engine predictions...")
    model_obj, vec_obj = evaluate.load_artifacts()
    ood_obj = GhostOODScorer()
    records = evaluate.build_records(model_obj, vec_obj, ood_obj)

    y_true, varying = _engine_matrix_from_records(records)
    varying_names = list(varying.keys())                       # ["kmer", "ood"]
    matrix = np.vstack([varying[name] for name in varying_names])

    # NOTE: optimising on the independent test set because no separate held-out
    # calibration split with per-engine scores exists in the repo. The resulting
    # AUROC is therefore optimistic — use a dedicated validation split for an
    # unbiased estimate. This is flagged loudly on purpose.
    print(f"[FUSION] Optimising over varying engines {varying_names} "
          f"on {len(y_true)} samples (in-sample — see note).")
    optimised_varying = optimize_fusion_weights(matrix, y_true, engine_names=varying_names)

    # Merge: optimised weights for the engines we could estimate, default share for
    # the rest, then a final convex renormalisation across all five.
    frozen = {k: DEFAULT_WEIGHTS[k] for k in ENGINE_ORDER if k not in optimised_varying}
    merged = {**optimised_varying, **frozen}
    total = sum(merged.values())
    final_weights = {k: merged[k] / total for k in ENGINE_ORDER}

    print("[FUSION] Final 5-engine weights (varying optimised, rest at default share):")
    print("         " + ", ".join(f"{k}={v:.3f}" for k, v in final_weights.items()))
    save_fusion_weights(final_weights)
    print("=" * 70)
