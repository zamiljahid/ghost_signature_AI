"""
metrics_utils.py — Single source of truth for all reported metrics.
===================================================================
Every metric here is computed from prediction arrays with sklearn/scipy.
NOTHING in this module is hardcoded.

Fixes applied:
  * Issue 2 — export_metrics() recomputes Table IV from scratch and asserts
    self-consistency (mean per-class recall == accuracy on a balanced set).
  * Issue 5 — delong_ci() replaces the (methodologically wrong) Wilson interval
    for AUROC. Wilson is for proportions; AUROC is a rank statistic. delong_ci()
    is now the TRUE DeLong (1988) estimator via the Sun & Xu (2014) fast midrank
    algorithm (non-parametric placement-value variance), not the Hanley & McNeil
    (1982) closed-form approximation it used previously.
  * AUPRC intervals use bootstrap_ap_ci() (stratified percentile bootstrap), since
    DeLong does not apply to average precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
)

CLASSES = ["Natural", "Vector", "Ghost"]


# ─────────────────────────────────────────────────────────────────────────────
# Issue 5 — TRUE DeLong confidence interval for AUROC (replaces Wilson)
# ─────────────────────────────────────────────────────────────────────────────
# This is the genuine DeLong (1988) estimator via the Sun & Xu (2014) fast
# midrank algorithm — variance is computed non-parametrically from the empirical
# placement values, NOT from the Hanley & McNeil (1982) closed-form approximation
# (which assumes a binormal/exponential AUC–variance relationship and is biased
# near AUC≈0.5 or 1.0). Wilson is a proportion interval and is wrong for a rank
# statistic like AUROC; see statistical_tests.wilson_ci for its correct use on
# accuracy/TPR/FPR.
# ─────────────────────────────────────────────────────────────────────────────
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x (ties get the average rank). 1-based, per Sun & Xu (2014)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)        # average 0-based index over the tie block
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1                          # restore original order, make 1-based
    return T2


def _fast_delong(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    Fast DeLong: returns (auc, variance) for a single binary classifier using
    placement (midrank) values. Distribution-free — this is the real DeLong
    structural-component variance, not a parametric approximation.
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("Both classes must have at least one sample.")

    tz = _compute_midrank(np.concatenate([pos, neg]))   # midranks over all, pos first
    tx = _compute_midrank(pos)                           # midranks within positives
    ty = _compute_midrank(neg)                           # midranks within negatives

    auc = (tz[:m].sum() / (m * n)) - (m + 1.0) / (2.0 * n)
    # Structural components (placement values).
    v01 = (tz[:m] - tx) / n          # one per positive: P(score_pos > random neg)
    v10 = 1.0 - (tz[m:] - ty) / m    # one per negative: P(score_pos > this neg)
    # Single-reader DeLong variance = S01/m + S10/n (sample variances of placements).
    s01 = np.var(v01, ddof=1) if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) if n > 1 else 0.0
    var = s01 / m + s10 / n
    return float(auc), float(var)


def delong_ci(y_true: np.ndarray, y_score: np.ndarray,
              alpha: float = 0.05) -> tuple[float, float, float]:
    """
    True DeLong confidence interval for binary AUROC (normal CI on the DeLong
    variance). For multi-class: call once per class with one-vs-rest binarization.

    Returns: (auc, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc, var = _fast_delong(y_true, y_score)
    se = float(np.sqrt(max(var, 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = float(np.clip(auc - z * se, 0.0, 1.0))
    ci_upper = float(np.clip(auc + z * se, 0.0, 1.0))
    return float(auc), ci_lower, ci_upper


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence interval for AUPRC (average precision)
# ─────────────────────────────────────────────────────────────────────────────
# AUPRC has no closed-form variance estimator (DeLong does not apply — it is not
# a simple rank statistic), so its interval is obtained by a stratified bootstrap:
# resample positives and negatives independently (keeping prevalence fixed) and
# recompute average_precision_score, then take percentile bounds.
def bootstrap_ap_ci(y_true: np.ndarray, y_score: np.ndarray,
                    alpha: float = 0.05, n_boot: int = 2000,
                    random_state: int = 42) -> tuple[float, float, float]:
    """Stratified percentile-bootstrap CI for AUPRC. Returns (ap, lo, hi)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    ap = float(average_precision_score(y_true, y_score))
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return ap, float("nan"), float("nan")

    rng = np.random.RandomState(random_state)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        bi = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        boots[b] = average_precision_score(y_true[bi], y_score[bi])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return ap, lo, hi


def report_multiclass_auroc_ci(y_true: np.ndarray, y_proba: np.ndarray,
                               class_names: list[str] = CLASSES,
                               alpha: float = 0.05) -> dict:
    """Per-class DeLong CI for a multi-class classifier (one-vs-rest)."""
    y_true = np.asarray(y_true)
    results = {}
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        auc, lo, hi = delong_ci(y_bin, y_proba[:, i], alpha=alpha)
        results[cls] = {"auroc": auc, "ci_lo": lo, "ci_hi": hi}
        print(f"AUROC {cls}: {auc:.4f} (95% CI [{lo:.4f}, {hi:.4f}]) — DeLong")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Issue 2 — programmatic metric export (Table IV) with self-consistency check
# ─────────────────────────────────────────────────────────────────────────────
def export_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray, output_dir: str = "outputs/statistical_tests") -> pd.DataFrame:
    """
    Computes ALL metrics from scratch using sklearn. No metric value here is
    hardcoded. Outputs a CSV that is the single source of truth for Table IV–V.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    classes = CLASSES

    # --- Core metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=classes, output_dict=True,
                                   zero_division=0)

    per_class_recall = [report[c]["recall"] for c in classes]

    # --- Self-consistency check (IEEE requirement) ---
    # On a balanced set, overall accuracy must equal the mean of per-class recall.
    # The paper's 0.9600 accuracy alongside recalls 0.992/0.633/0.533 (mean 0.719)
    # is arithmetically impossible — proof the value was hardcoded, not computed.
    class_counts = np.bincount(y_true, minlength=3)
    is_balanced = len(set(class_counts[class_counts > 0])) == 1
    mean_recall = float(np.mean(per_class_recall))
    if is_balanced:
        assert abs(mean_recall - accuracy) < 0.01, (
            f"METRIC INCONSISTENCY DETECTED:\n"
            f"  mean per-class recall = {mean_recall:.4f}\n"
            f"  overall accuracy      = {accuracy:.4f}\n"
            f"  These must match on a balanced dataset "
            f"(n per class = {class_counts.tolist()})."
        )
    else:
        print(f"[WARN] Test set is NOT balanced (counts={class_counts.tolist()}); "
              f"accuracy != mean recall by construction. Skipping equality assertion.")

    # --- AUROC (OvR, macro) + per-class DeLong CI ---
    auroc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    auroc_per_class, auroc_ci = {}, {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        auroc_per_class[cls] = roc_auc_score(y_bin, y_proba[:, i])
        _, lo, hi = delong_ci(y_bin, y_proba[:, i])   # Issue 5: DeLong, not Wilson
        auroc_ci[cls] = (lo, hi)

    # --- AUPRC per class + bootstrap CI (DeLong does not apply to AUPRC) ---
    auprc_per_class, auprc_ci = {}, {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        ap, ap_lo, ap_hi = bootstrap_ap_ci(y_bin, y_proba[:, i])
        auprc_per_class[cls] = ap
        auprc_ci[cls] = (ap_lo, ap_hi)

    # --- Build results DataFrame ---
    rows = []
    for i, cls in enumerate(classes):
        lo, hi = auroc_ci[cls]
        ap_lo, ap_hi = auprc_ci[cls]
        rows.append({
            "class":        cls,
            "precision":    report[cls]["precision"],
            "recall":       report[cls]["recall"],
            "f1":           report[cls]["f1-score"],
            "auroc":        auroc_per_class[cls],
            "auroc_ci_lo":  lo,
            "auroc_ci_hi":  hi,
            "auprc":        auprc_per_class[cls],
            "auprc_ci_lo":  ap_lo,
            "auprc_ci_hi":  ap_hi,
            "support":      int(report[cls]["support"]),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "class":       "MACRO_AVG",
        "precision":   report["macro avg"]["precision"],
        "recall":      report["macro avg"]["recall"],
        "f1":          report["macro avg"]["f1-score"],
        "auroc":       auroc_macro,
        "auroc_ci_lo": float("nan"),
        "auroc_ci_hi": float("nan"),
        "auprc":       float("nan"),
        "auprc_ci_lo": float("nan"),
        "auprc_ci_hi": float("nan"),
        "support":     int(report["macro avg"]["support"]),
    }
    df.loc[len(df)] = {
        "class":       "OVERALL_ACCURACY",
        "precision":   accuracy,        # accuracy stored in precision column slot
        "recall":      float("nan"),
        "f1":          float("nan"),
        "auroc":       float("nan"),
        "auroc_ci_lo": float("nan"),
        "auroc_ci_hi": float("nan"),
        "auprc":       float("nan"),
        "auprc_ci_lo": float("nan"),
        "auprc_ci_hi": float("nan"),
        "support":     len(y_true),
    }

    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])

    df.to_csv(f"{output_dir}/metrics_table_IV.csv", index=False, float_format="%.4f")
    cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")

    print("\n=== METRICS (source of truth for Table IV) ===")
    print(df.to_string(index=False))
    print(f"\nOverall Accuracy : {accuracy:.4f}")
    print(f"Mean per-class recall (self-consistency): {mean_recall:.4f}")
    print(f"Macro AUROC      : {auroc_macro:.4f}")
    print(f"\nConfusion Matrix:\n{cm_df}")
    print(f"\nCSV written to {output_dir}/metrics_table_IV.csv")
    return df
