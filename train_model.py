import os, time, warnings
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer
import joblib

warnings.filterwarnings("ignore")
from ghost_config import (
    KMER_SIZE, CHUNK_SIZE, CV_FOLDS, MODEL_DIR, OUTPUT_DIR, DATA_DIR,
    MODEL_PATH, VECTORIZER_PATH, OOD_PATH, OOD_PROKARYOTE_PATH,
    NATURAL_FASTA, VECTORS_FASTA, EVE_FASTA, GHOST_FASTA,
    PLASMID_FASTA, PROKARYOTE_FASTA,
    LABEL_NATURAL, LABEL_VECTOR, LABEL_GHOST,
    OOD_CONTAMINATION,
    SHORT_SEQ_THRESHOLD, SHORT_SEQ_AI_FLOOR, SHORT_SEQ_OOD_FLOOR,
    PLASMID_GC_THRESHOLD, PLASMID_AI_FLOOR,
)
from ood_scorer import GhostOODScorer

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def seq_to_kmers(seq, k=KMER_SIZE):
    seq = "".join(c for c in seq.upper() if c in "ATGCN")
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))


def classify_sequence_type(seq: str) -> str:
    """
    Heuristically classify a DNA sequence to flag edge cases before AI scoring.

    Returns
    -------
    'short'           — sequence < SHORT_SEQ_THRESHOLD bp after cleaning.
    'plasmid_suspect' — sequence has high GC content (> PLASMID_GC_THRESHOLD),
                        suggesting a prokaryotic plasmid outside the viral
                        training distribution.
    'standard'        — no flags triggered; k-mer model applies normally.
    """
    seq = "".join(c for c in seq.upper() if c in "ATGCN")
    if len(seq) < SHORT_SEQ_THRESHOLD:
        return "short"
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    if gc > PLASMID_GC_THRESHOLD:
        return "plasmid_suspect"
    return "standard"


def get_ai_score_narrative(seq: str, raw_proba: list) -> dict:
    seq_type      = classify_sequence_type(seq)
    ghost_prob    = raw_proba[2] if len(raw_proba) > 2 else 0.0
    vector_prob   = raw_proba[1] if len(raw_proba) > 1 else 0.0
    synthetic_pct = (ghost_prob + vector_prob) * 100.0

    result = {
        "seq_type":         seq_type,
        "raw_ai_prob":      round(synthetic_pct, 2),
        "adjusted_ai_prob": round(synthetic_pct, 2),
        "flag":             None,
        "narrative":        "",
    }

    seq_clean = "".join(c for c in seq.upper() if c in "ATGCN")

    if seq_type == "short":
        adjusted = max(synthetic_pct, SHORT_SEQ_AI_FLOOR)
        result.update({
            "adjusted_ai_prob": round(adjusted, 2),
            "flag":             "SHORT_FRAGMENT",
            "narrative": (
                f"⚠️ Sequence is {len(seq_clean)}bp (< {SHORT_SEQ_THRESHOLD}bp). "
                f"K-mer model requires longer fragments for reliable scoring. "
                f"AI risk floored at {SHORT_SEQ_AI_FLOOR}%. "
                f"Prioritise BLAST hit count and OOD signal for this input."
            ),
        })

    elif seq_type == "plasmid_suspect":
        gc = (seq_clean.count("G") + seq_clean.count("C")) / max(len(seq_clean), 1)
        adjusted = max(synthetic_pct, PLASMID_AI_FLOOR)
        result.update({
            "adjusted_ai_prob": round(adjusted, 2),
            "flag":             "PLASMID_SUSPECT",
            "narrative": (
                f"⚠️ High GC content detected ({gc:.1%}) — possible prokaryotic "
                f"plasmid or expression vector. The AI model was trained primarily "
                f"on viral sequences and may under-score plasmid-derived synthetics. "
                f"AI risk floored at {PLASMID_AI_FLOOR}%. "
                f"BLAST enrichment score is the primary signal for this input."
            ),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_fasta(filepath, label, max_records=None):
    rows = []
    if not os.path.exists(filepath):
        print(f"  [!] Missing: {filepath} — skipping")
        return rows
    for idx, rec in enumerate(SeqIO.parse(filepath, "fasta")):
        if max_records and idx >= max_records:
            break
        seq = "".join(c for c in str(rec.seq).upper() if c in "ATGCN")
        if len(seq) < KMER_SIZE:
            continue
        for i in range(0, len(seq), CHUNK_SIZE):
            frag = seq[i:i + CHUNK_SIZE]
            if len(frag) >= KMER_SIZE:
                rows.append({"text": seq_to_kmers(frag), "label": label,
                             "seq_id": rec.id})
    return rows


def build_dataset():
    print("[DATA] Loading training data...")
    data = (
        load_fasta(NATURAL_FASTA, LABEL_NATURAL) +
        load_fasta(VECTORS_FASTA, LABEL_VECTOR, max_records=300) +
        # Plasmids (pUC19, pBR322 etc.) trained as synthetic vectors so the
        # classifier learns prokaryotic synthetic k-mer signatures.
        load_fasta(PLASMID_FASTA, LABEL_VECTOR, max_records=200) +
        load_fasta(EVE_FASTA,     LABEL_GHOST) +
        load_fasta(GHOST_FASTA,   LABEL_GHOST)
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    counts = df["label"].value_counts().to_dict()
    print(f"  Class 0 natural: {counts.get(0, 0):,}")
    print(f"  Class 1 vector:  {counts.get(1, 0):,}  (includes plasmids)")
    print(f"  Class 2 ghost:   {counts.get(2, 0):,}")
    print(f"  Total:           {len(df):,}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model():
    """Stacked ensemble: RF + GradientBoosting + calibrated SVC → LogisticRegression."""
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", oob_score=True,
        n_jobs=-1, random_state=42,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=5,
        learning_rate=0.08, subsample=0.8, random_state=42,
    )
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced",
                  max_iter=2000, random_state=42),
        cv=3, method="sigmoid",
    )
    return StackingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("svc", svc)],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        cv=3, n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def multiclass_auroc(model, X, y):
    try:
        proba = model.predict_proba(X)
        lb    = LabelBinarizer()
        yb    = lb.fit_transform(y)
        if yb.shape[1] == 1:
            yb = np.hstack([1 - yb, yb])
        return roc_auc_score(yb, proba, multi_class="ovr", average="macro")
    except Exception:
        return 0.0


def train_and_evaluate(df):
    X_text = df["text"].values
    y      = df["label"].values

    print("[VECTORIZER] Fitting TF-IDF (8-mer, max 50k features)...")
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2,
                          max_features=50_000, sublinear_tf=True)
    X = vec.fit_transform(X_text)
    joblib.dump(vec, VECTORIZER_PATH)
    print(f"  Vocabulary: {len(vec.vocabulary_):,} features\n")

    model = build_model()

    print("[CV] 5-fold stratified cross-validation...")
    skf        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    cv_rows     = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model.fit(X[tr_idx], y[tr_idx])
        score = multiclass_auroc(model, X[val_idx], y[val_idx])
        fold_scores.append(score)
        cv_rows.append({"Fold": f"Fold {fold}", "AUROC": f"{score:.4f}"})
        print(f"  Fold {fold}: AUROC = {score:.4f}")

    mean_a, std_a = np.mean(fold_scores), np.std(fold_scores)
    cv_rows += [{"Fold": "Mean", "AUROC": f"{mean_a:.4f}"},
                {"Fold": "Std",  "AUROC": f"{std_a:.4f}"}]
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR + "/cv_results.csv", index=False)
    print(f"\n  Mean AUROC: {mean_a:.4f} ± {std_a:.4f}")

    print("\n[FINAL] Training on full dataset...")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X)
    print("\n[TRAIN REPORT]")
    print(classification_report(y, y_pred,
          target_names=["Natural", "Vector", "Ghost"], digits=3))

    return model, vec, fold_scores


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("GHOST SIGNATURE DETECTOR — Trainer v2 (Stacked Ensemble)")
    print("=" * 60)

    df = build_dataset()
    if df.empty:
        print("[ERROR] No training data. Run dataset_builder.py first.")
        raise SystemExit(1)

    model, vec, cv_scores = train_and_evaluate(df)

    # ── OOD envelope fitting ───────────────────────────────────────────
    print("\n[OOD] Loading viral natural sequences for envelope fitting...")
    nat_seqs = [
        str(r.seq) for r in SeqIO.parse(NATURAL_FASTA, "fasta")
        if len(str(r.seq)) > 200
    ]
    print(f"[OOD] Loaded {len(nat_seqs)} viral sequences.")

    # Load prokaryotic references for the second envelope (optional but
    # strongly recommended — prevents pUC19-style false-low OOD scores).
    prok_seqs = []
    if os.path.exists(PROKARYOTE_FASTA):
        prok_seqs = [
            str(r.seq) for r in SeqIO.parse(PROKARYOTE_FASTA, "fasta")
            if len(str(r.seq)) > 200
        ]
        print(f"[OOD] Loaded {len(prok_seqs)} prokaryotic reference sequences.")
    else:
        print(f"[OOD] {PROKARYOTE_FASTA} not found — single-envelope mode.")
        print(      "      Download E. coli K-12 (NC_000913.3) to enable dual-envelope.")

    ood = GhostOODScorer(
        vectorizer_path=VECTORIZER_PATH,
        envelope_path=OOD_PATH,
        prokaryote_envelope_path=OOD_PROKARYOTE_PATH,
        svd_n_components=128,
    )
    ood.vectorizer = vec   # reuse the already-fitted vectorizer in memory
    ood.fit_on_sequences(
        nat_sequences=nat_seqs,
        prokaryote_sequences=prok_seqs or None,
        contamination=OOD_CONTAMINATION,
        random_state=42,
    )

    print(f"\n[DONE] {(time.time() - t0) / 60:.1f} min")
    print(f"  {MODEL_PATH}")
    print(f"  {VECTORIZER_PATH}")
    print(f"  {OOD_PATH}")
    if prok_seqs:
        print(f"  {OOD_PROKARYOTE_PATH}")
    print(f"  {OUTPUT_DIR}/cv_results.csv")
    print("\nRun: python evaluate.py")
    print("=" * 60)