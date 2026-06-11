import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from Bio import SeqIO

warnings.filterwarnings("ignore")

# ── Import configuration defaults ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    from ghost_config import (
        MODEL_PATH, VECTORIZER_PATH, OOD_PATH, KMER_SIZE, CHUNK_SIZE,
        LABEL_GHOST, LABEL_NATURAL, LABEL_VECTOR, OUTPUT_DIR, PLOT_DPI,
        INDEPENDENT_TEST_NATURAL, INDEPENDENT_TEST_VECTOR, INDEPENDENT_TEST_GHOST
    )
    from ood_scorer import GhostOODScorer
except ImportError:
    # Fallback placeholders if ghost_config is missing or running in an isolated folder
    MODEL_PATH, VECTORIZER_PATH, OOD_PATH = "models/rf_model.pkl", "models/vectorizer.pkl", "models/ood_envelope.pkl"
    KMER_SIZE, CHUNK_SIZE = 6, 500
    LABEL_GHOST, LABEL_NATURAL, LABEL_VECTOR = 1, 0, 2
    OUTPUT_DIR, PLOT_DPI = "outputs", 200
    class GhostOODScorer:
        def __init__(self, envelope_path): self.ready = False
        def ghost_anomaly_score(self, seq): return {"score": 0.0}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(seq):
    return "".join(c for c in str(seq).upper() if c in "ATGCN")

def to_kmers(seq, k=None):
    from kmer_utils import seq_to_kmers_multiscale
    return seq_to_kmers_multiscale(seq)

def predict_proba(model, vec, seq):
    frags = [seq[i:i+CHUNK_SIZE] for i in range(0, max(1, len(seq)-CHUNK_SIZE+1), CHUNK_SIZE)]
    probas = []
    for f in frags:
        if len(f) >= 4:
            probas.append(model.predict_proba(vec.transform([to_kmers(f)]))[0])
    return np.mean(probas, axis=0) if probas else np.full(3, 1/3)

def _load_blast_hit_ids():
    """Load sequence IDs with confirmed BLAST hits from the comparison blast results CSV, if present."""
    blast_csv_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "ghost_comparison", "blast_results", "blast_hits.csv"),
        os.path.join(os.path.dirname(__file__), "..", "ghost_comparison", "final_comparison", "detection_matrix.csv"),
    ]
    hit_ids = set()
    for path in blast_csv_candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                id_col = next((c for c in df.columns if c.lower() in ("query_id", "id", "sequence_id")), None)
                if id_col:
                    hit_ids.update(df[id_col].astype(str).tolist())
                    print(f"  [BLAST] Loaded {len(hit_ids)} hit IDs from {path}")
                    break
            except Exception as e:
                print(f"  [WARN] Could not read blast results from {path}: {e}")
    return hit_ids

_BLAST_HIT_IDS = None  # loaded once on first call

def load_records_from_fasta(fasta_path, model, vec, ood, known_label=None):
    """
    Load sequences from a FASTA file with their correct ground-truth labels.

    known_label: pass LABEL_NATURAL / LABEL_VECTOR / LABEL_GHOST explicitly when
    the FASTA file contains sequences of a single known class (e.g., each
    independent test FASTA). This avoids the header-parsing pitfall where
    GenBank accession IDs (NC_038294.1 etc.) contain no class keyword.
    """
    global _BLAST_HIT_IDS
    if _BLAST_HIT_IDS is None:
        _BLAST_HIT_IDS = _load_blast_hit_ids()

    if not os.path.exists(fasta_path):
        print(f"[ERROR] FASTA file not found at: {fasta_path}")
        return []

    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = clean(str(rec.seq))
        if len(seq) < 4:
            continue

        # Priority 1: caller-supplied label (most reliable — label comes from filename choice)
        if known_label is not None:
            lbl = known_label
        else:
            # Priority 2: GHOST_SRC tag injected by collect_ghost_results.py
            desc_upper = rec.description.upper()
            if "GHOST_SRC=LABEL_NATURAL" in desc_upper:
                lbl = LABEL_NATURAL
            elif "GHOST_SRC=LABEL_VECTOR" in desc_upper:
                lbl = LABEL_VECTOR
            elif "GHOST_SRC=LABEL_GHOST" in desc_upper:
                lbl = LABEL_GHOST
            # Priority 3: keyword in ID
            elif "NATURAL" in rec.id.upper():
                lbl = LABEL_NATURAL
            elif "VECTOR" in rec.id.upper():
                lbl = LABEL_VECTOR
            elif "GHOST" in rec.id.upper():
                lbl = LABEL_GHOST
            else:
                # Cannot determine label — skip rather than guess wrong
                print(f"  [SKIP] Cannot determine label for {rec.id} — no keyword or GHOST_SRC tag found")
                continue

        proba = predict_proba(model, vec, seq) if model else np.full(3, 1/3)
        ood_s = ood.ghost_anomaly_score(seq)["score"] if ood.ready else 0.0
        ai_risk = float((proba[LABEL_GHOST] + proba[LABEL_VECTOR]) * 100)

        # Engine 3 (BLAST): use real hit IDs if available; default conservatively to False
        has_blast_hit = rec.id in _BLAST_HIT_IDS if _BLAST_HIT_IDS else False
        # Engine 4 (Motif): no reliable circular proxy — default to False (conservative)
        has_motif_hit = False

        records.append({
            "id": rec.id,
            "true_label": lbl,
            "proba": proba,
            "ood_score": ood_s,
            "ai_risk": ai_risk,
            "blast_hit": has_blast_hit,
            "motif_hit": has_motif_hit,
            "seq": seq,
        })

    ng = sum(1 for r in records if r["true_label"] in (LABEL_GHOST, LABEL_VECTOR))
    nn = sum(1 for r in records if r["true_label"] == LABEL_NATURAL)
    print(f"  Loaded {len(records)} sequences | {ng} Engineered/Ghost | {nn} Natural")
    return records

# ── Scoring Functions (Multi-Engine Implementations) ─────────────────────────
def score_full(r):
    """Production 'Full System': the AI classifier drives the score, with BLAST and
    motif hits as hard deterministic overrides.

    The OOD engine has ZERO decision power — it is a passive diagnostic tag only.
    Ablation proved the unsupervised OOD score (AUROC 0.248) treats natural
    evolutionary variance as synthetic, inverting the ranking and dragging the old
    fused system to 0.564. OOD is therefore EXCLUDED from this score. By design this
    is now identical to the 'No OOD' configuration — which is the whole point: the
    deployed system gives OOD no vote, and the ablation makes that explicit.
    """
    ai = r["ai_risk"] / 100.0
    if r.get("blast_hit", False):
        ai = max(ai, 0.95)
    if r.get("motif_hit", False):
        ai = max(ai, 0.85)
    return ai

def score_no_ai(r):
    """Disable Engine 1: Base score relies entirely on OOD anomalies + heuristics."""
    ood = r["ood_score"] / 100.0
    if r.get("blast_hit", False) or r.get("motif_hit", False):
        return max(ood, 0.85)
    return ood

def score_no_ood(r):
    """Disable Engine 2: Relies strictly on AI classification and discrete hits."""
    ai = r["ai_risk"] / 100.0
    if r.get("blast_hit", False):
        return max(ai, 0.95)
    return ai

def score_no_blast_no_motif(r):
    """Disable Engines 3+4: Evaluate pure statistical/machine learning signals."""
    ai = r["ai_risk"] / 100.0
    ood = r["ood_score"] / 100.0
    return (ai + ood) / 2.0

def score_ai_only(r):
    return r["ai_risk"] / 100.0

def score_ood_only(r):
    return r["ood_score"] / 100.0

def score_random(_r):
    return np.random.uniform(0.0, 1.0)

# ── Metrics Computations ──────────────────────────────────────────────────────
def compute_metrics(records, score_fn, threshold=0.60):
    # threshold = 0.60 is the calibrated peak-F1 operating point (evaluate.py:
    # Engineered recall 82.9% @ Natural FPR 13.3%). Binary TPR/FPR/F1 columns are
    # reported at this production operating point, not an arbitrary 0.5; AUROC is
    # threshold-independent so the headline ranking metric is unaffected.
    from sklearn.metrics import roc_auc_score
    y_true = np.array([1 if r["true_label"] in (LABEL_GHOST, LABEL_VECTOR) else 0 for r in records])
    y_score = np.array([score_fn(r) for r in records])
    y_pred = (y_score >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    ghost_tpr = tp / max(y_true.sum(), 1)
    nat_fpr = fp / max((1 - y_true).sum(), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * ghost_tpr / max(prec + ghost_tpr, 1e-9)

    return {
        "AUROC": round(auroc, 4),
        "Ghost TPR": round(ghost_tpr, 4),
        "Natural FPR": round(nat_fpr, 4),
        "Precision": round(prec, 4),
        "F1": round(f1, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

def run_ablation(records):
    configs = [
        ("Full System (AI + BLAST/Motif; OOD passive)", score_full, True),
        ("No AI Classifier (Engine 1 removed)", score_no_ai, False),
        ("No OOD Scorer (Engine 2 removed)", score_no_ood, False),
        ("No BLAST+Motif (Engines 3+4 removed)", score_no_blast_no_motif, False),
        ("AI Classifier Only", score_ai_only, False),
        ("OOD Scorer Only", score_ood_only, False),
        ("Random Baseline", score_random, False),
    ]

    rows = []
    for name, fn, is_full in configs:
        m = compute_metrics(records, fn)
        rows.append({"Configuration": name, **m, "_is_full": is_full})

    full_auroc = next((r["AUROC"] for r in rows if r["_is_full"]), 1.0)
    for r in rows:
        if not r["_is_full"] and not np.isnan(full_auroc) and not np.isnan(r["AUROC"]):
            r["AUROC Drop"] = round(full_auroc - r["AUROC"], 4)
        else:
            r["AUROC Drop"] = 0.0
    return rows, full_auroc

def plot_ablation(rows, output_dir):
    display_rows = [r for r in rows if r["Configuration"] != "Random Baseline"]
    names = [r["Configuration"].split("(")[0].strip() for r in display_rows]
    aurocs = [0.0 if np.isnan(r["AUROC"]) else r["AUROC"] for r in display_rows]
    tprs = [r["Ghost TPR"] for r in display_rows]
    fprs = [r["Natural FPR"] for r in display_rows]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w, aurocs, w, label="AUROC", color="#6D28D9", alpha=0.85)
    b2 = ax.bar(x, tprs, w, label="Ghost TPR (Recall)", color="#1D9E75", alpha=0.85)
    b3 = ax.bar(x + w, fprs, w, label="Natural FPR", color="#D85A30", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                        f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.axhline(y=0.5, color="#B4B2A9", ls=":", lw=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set(ylabel="Performance Score", ylim=[0, 1.2], title="Engine Ablation Study Breakdown")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ablation_chart.png")
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    print(f"  Chart generated cleanly → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an engine ablation study on independent test sets.")
    parser.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "ablation"),
                        help="Directory to save generated artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 65)
    print("Ghost Signature — Engine Ablation Suite (Independent Test Sets)")
    print("=" * 65)

    try:
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VECTORIZER_PATH)
        print("[OK] Serialized model and vectorizer parsed successfully.")
    except FileNotFoundError:
        print("[WARN] Model artifacts missing. Proceeding with uninitialized placeholders.")
        model, vec = None, None

    ood = GhostOODScorer(envelope_path=OOD_PATH)

    print("[DATA] Loading all independent test sets...")
    records = []
    test_sets = [
        (INDEPENDENT_TEST_NATURAL, LABEL_NATURAL),
        (INDEPENDENT_TEST_VECTOR, LABEL_VECTOR),
        (INDEPENDENT_TEST_GHOST, LABEL_GHOST),
    ]
    for fasta_path, label in test_sets:
        recs = load_records_from_fasta(fasta_path, model, vec, ood, known_label=label)
        records.extend(recs)

    if not records:
        print("[ERROR] Valid sequences could not be collected. Exiting.")
        sys.exit(1)

    print("\n[PROCESS] Running ablation configuration tests...")
    rows, full_auroc = run_ablation(records)

    df_cols = ["Configuration", "AUROC", "AUROC Drop", "Ghost TPR", "Natural FPR", "Precision", "F1", "TP", "FP", "FN", "TN"]
    df = pd.DataFrame(rows)[df_cols]
    csv_out = os.path.join(args.outdir, "ablation_table.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Data matrix recorded → {csv_out}")

    plot_ablation(rows, args.outdir)
    print(f"\n[DONE] Finished execution. Outputs located in: {args.outdir}/")