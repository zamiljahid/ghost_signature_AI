"""
explainability.py — Per-engine human-readable evidence (Issue 6).
==================================================================
The original forensic report emitted only numeric fusion weights with no
human-readable justification. Reviewers require interpretability per engine.

This module adds:
  * SHAP-based top-k-mer evidence for the AI classifier
  * SHAP-based anomalous-dimension evidence for the OOD IsolationForest
  * motif evidence formatting
  * build_forensic_report() — one engine-by-engine text block per sequence

SHAP is imported lazily and every helper degrades gracefully to a plain-text
fallback if SHAP is unavailable or the model type is unsupported, so the main
pipeline never crashes for lack of an explanation.
"""
from __future__ import annotations

import numpy as np


def _try_import_shap():
    try:
        import shap  # noqa: F401
        return shap
    except Exception:
        return None


# ── A. SHAP for k-mer TF-IDF ─────────────────────────────────────────────────
def kmer_shap_evidence(classifier, X_train_tfidf, X_sample,
                       feature_names, top_n: int = 5) -> str:
    """Top-N k-mers driving the prediction as a human-readable string.

    Works on a linear classifier (e.g. LogisticRegression / LinearSVC). Falls
    back to the largest-magnitude TF-IDF features if SHAP cannot explain the
    given estimator (e.g. an opaque StackingClassifier).
    """
    shap = _try_import_shap()
    try:
        if shap is None:
            raise RuntimeError("shap unavailable")
        explainer = shap.LinearExplainer(classifier, X_train_tfidf,
                                         feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_sample)
        pred_class = int(np.argmax(classifier.predict_proba(X_sample), axis=1)[0])
        sv = shap_vals[pred_class][0] if isinstance(shap_vals, list) else np.asarray(shap_vals)[0]
        top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]
        pairs = [(feature_names[i], float(sv[i])) for i in top_idx]
        return ", ".join(f"{k}({v:+.3f})" for k, v in pairs)
    except Exception:
        # Fallback: rank the active TF-IDF features in this sample by weight.
        x = np.asarray(X_sample.todense()).ravel() if hasattr(X_sample, "todense") \
            else np.asarray(X_sample).ravel()
        top_idx = np.argsort(np.abs(x))[-top_n:][::-1]
        pairs = [(feature_names[i], float(x[i])) for i in top_idx if x[i] != 0]
        if not pairs:
            return "no strongly weighted k-mers"
        return ", ".join(f"{k}({v:+.3f})" for k, v in pairs) + " [tfidf-weight fallback]"


# ── B. OOD SHAP evidence ─────────────────────────────────────────────────────
def ood_shap_evidence(iso_model, X_sample, feature_names, top_n: int = 5) -> str:
    """Top-N anomalous feature dimensions from an IsolationForest via SHAP."""
    shap = _try_import_shap()
    try:
        if shap is None:
            raise RuntimeError("shap unavailable")
        explainer = shap.TreeExplainer(iso_model)
        sv = np.asarray(explainer.shap_values(X_sample))[0]
        top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]
        pairs = [(feature_names[i], float(sv[i])) for i in top_idx]
        return ", ".join(f"{f}({v:+.3f})" for f, v in pairs)
    except Exception:
        # Fallback: most extreme 4-mer frequencies in this sample.
        x = np.asarray(X_sample).ravel()
        top_idx = np.argsort(np.abs(x - x.mean()))[-top_n:][::-1]
        pairs = [(feature_names[i], float(x[i])) for i in top_idx]
        return ", ".join(f"{f}({v:.3f})" for f, v in pairs) + " [freq-deviation fallback]"


# ── C. Motif evidence ────────────────────────────────────────────────────────
def motif_evidence(matched_motifs) -> str:
    """matched_motifs: list of {'name','start','end','score'} dicts, or list[str]."""
    if not matched_motifs:
        return "No known synthetic motifs detected"
    if isinstance(matched_motifs[0], dict):
        return "; ".join(
            f"{m['name']}@{m.get('start', '?')}-{m.get('end', '?')}"
            f"(score={m.get('score', float('nan')):.3f})"
            for m in matched_motifs
        )
    return "; ".join(str(m) for m in matched_motifs)


# ── D. Full forensic report builder ──────────────────────────────────────────
def build_forensic_report(sample_id: str, engines: dict) -> str:
    """
    Engine-by-engine human-readable report.

    Required engines dict keys:
      kmer_prob, kmer_evidence, ood_prob, ood_evidence,
      blast_score, blast_top_hit, motif_score, motif_evidence,
      codon_score, cai_delta, S_fused
    """
    verdict = "⚠ SYNTHETIC" if engines["S_fused"] > 0.5 else "✓ NATURAL"
    lines = [
        f"=== Forensic Report: {sample_id} ===",
        f"[kmer-TFIDF]  P={engines['kmer_prob']:.3f} | Top k-mers: {engines['kmer_evidence']}",
        f"[OOD Engine]  P={engines['ood_prob']:.3f} | Anomalous dims: {engines['ood_evidence']}",
        f"[BLAST]       P={engines['blast_score']:.3f} | Top hit: {engines['blast_top_hit']}",
        f"[Motif]       P={engines['motif_score']:.3f} | {engines['motif_evidence']}",
        f"[Codon Bias]  P={engines['codon_score']:.3f} | CAI delta: {engines['cai_delta']:.4f}",
        f"─────────────────────────────────────────────",
        f"FUSED SCORE:  {engines['S_fused']:.4f}  →  {verdict}",
    ]
    return "\n".join(lines)
