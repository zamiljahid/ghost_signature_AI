"""
scripts/export_metrics.py — Regenerate Table IV/V from the trained model.
=========================================================================
Single source of truth for the paper's performance numbers. Loads the trained
Ghost Signature model, scores the 360-sequence independent test set, and writes
results/metrics_table_IV.csv + results/confusion_matrix.csv.

Every value is recomputed with sklearn — nothing is hardcoded. The
self-consistency assertion in export_metrics() will fail loudly if accuracy and
mean per-class recall ever diverge on the balanced set (the bug behind the
paper's impossible 0.9600 accuracy).

Run: python scripts/export_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_utils import export_metrics, report_multiclass_auroc_ci
from scripts._inference import get_predictions

if __name__ == "__main__":
    print("=" * 70)
    print(" GHOST SIGNATURE — Table IV/V metric export (code-generated)")
    print("=" * 70)

    y_true, y_pred, y_proba, _ = get_predictions()

    export_metrics(y_true, y_pred, y_proba, output_dir="results")

    print("\n--- Per-class AUROC with DeLong 95% CI (Issue 5) ---")
    report_multiclass_auroc_ci(y_true, y_proba)

    print("\n[DONE] results/metrics_table_IV.csv is now the source of truth.")
