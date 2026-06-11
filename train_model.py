import os, time, warnings
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer
import joblib

warnings.filterwarnings("ignore")
from ghost_config import (
    KMER_SIZE, KMER_SIZES, CHUNK_SIZE, MODEL_DIR, OUTPUT_DIR, DATA_DIR,
    MODEL_PATH, VECTORIZER_PATH, OOD_PATH, OOD_PROKARYOTE_PATH,
    NATURAL_FASTA, VECTORS_FASTA, EVE_FASTA, GHOST_FASTA,
    PLASMID_FASTA, PROKARYOTE_FASTA,
    LABEL_NATURAL, LABEL_VECTOR, LABEL_GHOST,
    OOD_CONTAMINATION, TFIDF_MAX_FEATURES,
    SHORT_SEQ_THRESHOLD, SHORT_SEQ_AI_FLOOR, SHORT_SEQ_OOD_FLOOR,
    PLASMID_GC_THRESHOLD, PLASMID_AI_FLOOR, CV_FOLDS,
)
from kmer_utils import seq_to_kmers_multiscale, MIN_SEQ_LENGTH
from ood_scorer import GhostOODScorer, CalibratedOODEngine

# ── Fix T5/O1/M1 — dense biological features stacked onto TF-IDF ──────────────
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from motif_discovery import find_ghost_motifs

# ── Fix T2 — in-fold SMOTE oversampling. imbalanced-learn is optional; if it is
# not installed we fall back silently (no oversampling) rather than crashing.
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False

# ── Fix T4 — calibrate an already-fitted RF (Fix T4 specifies cv="prefit").
# sklearn >= 1.6 replaced the cv="prefit" path with FrozenEstimator; we prefer it
# when available and fall back to cv="prefit" on older sklearn. Both have the
# same semantics: calibrate WITHOUT refitting the base estimator.
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Per-stage output subfolder (keeps outputs/ organized by producing script).
TRAINING_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training")
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def seq_to_kmers(seq, k=None):
    seq = "".join(c for c in seq.upper() if c in "ATGCN")
    return seq_to_kmers_multiscale(seq)


def classify_sequence_type(seq: str) -> str:
    seq = "".join(c for c in seq.upper() if c in "ATGCN")
    if len(seq) < SHORT_SEQ_THRESHOLD:
        return "short"
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    if gc > PLASMID_GC_THRESHOLD:
        return "plasmid_suspect"
    return "standard"


def get_ai_score_narrative(seq: str, raw_proba: list) -> dict:
    seq_type = classify_sequence_type(seq)
    ghost_prob = raw_proba[2] if len(raw_proba) > 2 else 0.0
    vector_prob = raw_proba[1] if len(raw_proba) > 1 else 0.0
    synthetic_pct = (ghost_prob + vector_prob) * 100.0

    result = {
        "seq_type": seq_type,
        "raw_ai_prob": round(synthetic_pct, 2),
        "adjusted_ai_prob": round(synthetic_pct, 2),
        "flag": None,
        "narrative": "",
    }

    seq_clean = "".join(c for c in seq.upper() if c in "ATGCN")

    if seq_type == "short":
        adjusted = max(synthetic_pct, SHORT_SEQ_AI_FLOOR)
        result.update({
            "adjusted_ai_prob": round(adjusted, 2),
            "flag": "SHORT_FRAGMENT",
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
            "flag": "PLASMID_SUSPECT",
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

def load_fasta(filepath, label, max_records=None, min_len=MIN_SEQ_LENGTH):
    rows = []
    if not os.path.exists(filepath):
        print(f"  [!] Missing: {filepath} — skipping")
        return rows

    for idx, rec in enumerate(SeqIO.parse(filepath, "fasta")):
        if max_records and idx >= max_records:
            break
        seq = "".join(c for c in str(rec.seq).upper() if c in "ATGCN")
        if len(seq) < min_len:
            continue
        for i in range(0, len(seq), CHUNK_SIZE):
            frag = seq[i:i + CHUNK_SIZE]
            if len(frag) >= min_len:
                rows.append({"text": seq_to_kmers(frag), "label": label,
                             "seq_id": rec.id, "raw_seq": frag})
    return rows


def build_dataset():
    print("[DATA] Loading training datasets into memory...")

    data = (
            load_fasta(NATURAL_FASTA, LABEL_NATURAL) +
            load_fasta(VECTORS_FASTA, LABEL_VECTOR, max_records=300) +
            load_fasta(PLASMID_FASTA, LABEL_VECTOR, max_records=200) +
            load_fasta(EVE_FASTA, LABEL_GHOST) +
            load_fasta(GHOST_FASTA, LABEL_GHOST)
    )

    synth_path = os.path.join(DATA_DIR, "synthetic_constructs.fasta")
    if os.path.exists(synth_path):
        synth_data = load_fasta(synth_path, LABEL_GHOST)
        print(f"  [+] Wired {len(synth_data)} real synthetic construct fragments into Ghost class")
        data = data + synth_data
    else:
        print(f"  [!] {synth_path} not found — run dataset_builder.py to download synthetic constructs")

    # Short synthetic fragments (20-150 bp) — match the length distribution of the
    # independent Ghost test set (MVA patent fragments, 20-113 bp). Without these,
    # training Ghost sequences are all 500-2000 bp and the model learns Ghost k-mer
    # density at the wrong length scale, causing Ghost class collapse at test time.
    short_frag_path = os.path.join(DATA_DIR, "synthetic_short_fragments.fasta")
    if os.path.exists(short_frag_path):
        short_data = load_fasta(short_frag_path, LABEL_GHOST, min_len=4)
        print(f"  [+] Wired {len(short_data)} short synthetic fragments (20-150 bp) into Ghost class")
        data = data + short_data
    else:
        print(f"  [!] {short_frag_path} not found — run dataset_builder.py first")

    df = pd.DataFrame(data)
    if df.empty:
        return df
    counts = df["label"].value_counts().to_dict()
    print(f"  Class 0 natural: {counts.get(0, 0):,}")
    print(f"  Class 1 vector:  {counts.get(1, 0):,}  (includes plasmids)")
    print(f"  Class 2 ghost:   {counts.get(2, 0):,}  (includes real synthetic constructs)")

    # NOTE (Fix T2): Ghost-class oversampling used to happen HERE, before CV.
    # Duplicating rows pre-split leaked near-identical fragments (same seq_id)
    # into both train and validation folds, inflating Ghost recall. Oversampling
    # is now done with SMOTE *inside* each CV fold, on the training split only.

    print(f"  Total fragments available for training: {len(df):,}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model():
    """
    Fix T4: a single calibrated RandomForest replaces the old StackingClassifier.

    StackingClassifier(cv=3) ran an inner 3-fold CV on genome-grouped data where
    each outer training fold may contain only 15–30 genomes — far too few for
    stable tree ensembles, and it tripled training time. A RandomForest with
    out-of-bag scoring and balanced subsampling is both faster and more stable;
    probability calibration is applied per-fold via CalibratedClassifierCV.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    )
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# Dense biological features (Fix T5 / M1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dense_features(sequences: np.ndarray) -> np.ndarray:
    """GC content, CpG rate and 3rd-codon-position GC bias per sequence."""
    feats = []
    for seq in sequences:
        seq = "".join(c for c in seq.upper() if c in "ATGCN")
        n = max(len(seq), 1)
        gc = (seq.count("G") + seq.count("C")) / n
        cpg = seq.count("CG") / max(n - 1, 1)
        codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3) if len(seq[i:i + 3]) == 3]
        gc3 = np.mean([1 if c[2] in "GC" else 0 for c in codons]) if codons else gc
        feats.append([gc, cpg, gc3])
    return np.array(feats, dtype=np.float32)


def motif_max_enrichment(raw_seq: str, background_seqs: list) -> float:
    """Max Ghost-vs-background k-mer fold-enrichment for one sequence (Fix M1)."""
    if len(raw_seq) < 6:
        return 0.0
    motifs = find_ghost_motifs(raw_seq, background_seqs, k=6, top_n=5)
    return float(motifs[0][1]) if motifs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def multiclass_auroc(model, X, y):
    try:
        proba = model.predict_proba(X)
        lb = LabelBinarizer()
        yb = lb.fit_transform(y)
        if yb.shape[1] == 1:
            yb = np.hstack([1 - yb, yb])
        return roc_auc_score(yb, proba, multi_class="ovr", average="macro")
    except Exception:
        return 0.0


def train_and_evaluate(df):
    X_text = df["text"].values
    y = df["label"].values
    # Genome-of-origin for every window. Each long sequence is windowed into many
    # 500-bp fragments that all share rec.id; that id is the grouping key for CV.
    genome_ids = df["seq_id"].values
    # Raw nucleotide strings — needed for dense biological features (Fix T5),
    # OOD scoring (Fix O1) and motif enrichment (Fix M1).
    raw_seqs = df["raw_seq"].values

    # ── Fix O1: fit a global OOD scorer on NATURAL-class sequences only, BEFORE
    # the CV loop. The same fitted scorer is reused inside every fold; it is never
    # refit on fold data, so no validation information leaks into the OOD feature.
    natural_mask = (y == LABEL_NATURAL)
    print(f"[OOD] Fitting global OOD scorer on {int(natural_mask.sum()):,} natural fragments...")
    ood_scorer_global = GhostOODScorer()
    ood_scorer_global.fit(raw_seqs[natural_mask])

    # WHY THE CHANGE: the old StratifiedKFold split on window-level rows, so windows
    # cut from the SAME genome landed in both train and validation folds. That is
    # catastrophic data leakage (the validation "AUROC" was memorisation, ~0.9776).
    # StratifiedGroupKFold keeps every window of a genome inside a single fold.
    #
    # Fix T1/T3: the TF-IDF vectorizer is now fitted INSIDE each fold on the
    # training split only (fit_transform on train, transform on val). Fitting it
    # once on the whole dataset leaked validation IDF statistics and inflated AUROC.
    print(f"[CV] Launching {CV_FOLDS}-fold genome-GROUPED stratified cross-validation...")
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    cv_rows = []

    # The outer folds iterate sequentially so you can track progress clearly
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_text, y, groups=genome_ids), 1):
        fold_start = time.time()
        train_genome_ids = set(genome_ids[tr_idx])
        val_genome_ids   = set(genome_ids[val_idx])

        # MANDATORY LEAKAGE ASSERTION — never remove this.
        overlap = train_genome_ids & val_genome_ids
        assert len(overlap) == 0, (
            f"DATA LEAKAGE in fold {fold}: "
            f"{len(overlap)} genome(s) appear in both splits! e.g. {list(overlap)[:3]}"
        )

        print(f"\n  ================== [ FOLD {fold} / {CV_FOLDS} ] ==================")
        print(f"  [→] {len(train_genome_ids):,} train genomes / {len(val_genome_ids):,} val genomes | "
              f"{len(tr_idx):,} train windows / {len(val_idx):,} val windows")

        # ── Fix T1/T3: fresh TF-IDF fitted on the TRAIN fold only ──────────────
        vec_fold = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            analyzer="word",
        )
        X_train_fold = vec_fold.fit_transform(X_text[tr_idx])
        X_val_fold   = vec_fold.transform(X_text[val_idx])
        y_train_fold = y[tr_idx]

        # ── Fix T5: GC / CpG / GC3 dense features (scaler fit on TRAIN only) ────
        dense_train = compute_dense_features(X_text[tr_idx])
        dense_val   = compute_dense_features(X_text[val_idx])
        scaler = StandardScaler()
        dense_train = scaler.fit_transform(dense_train)
        dense_val   = scaler.transform(dense_val)

        # ── Fix O1: OOD anomaly score as a [0,1] feature column ────────────────
        ood_train = np.array(ood_scorer_global.batch_scores(raw_seqs[tr_idx])).reshape(-1, 1) / 100.0
        ood_val   = np.array(ood_scorer_global.batch_scores(raw_seqs[val_idx])).reshape(-1, 1) / 100.0

        # ── Fix M1: motif max fold-enrichment (30-seq background, scaler train) ─
        background = raw_seqs[tr_idx][:30].tolist()   # 30-sequence sample as background
        motif_train = np.array([motif_max_enrichment(s, background) for s in raw_seqs[tr_idx]]).reshape(-1, 1)
        motif_val   = np.array([motif_max_enrichment(s, background) for s in raw_seqs[val_idx]]).reshape(-1, 1)
        scaler_motif = StandardScaler()
        motif_train = scaler_motif.fit_transform(motif_train)
        motif_val   = scaler_motif.transform(motif_val)

        # ── Combine TF-IDF + dense + OOD + motif features ──────────────────────
        X_train_combined = sp.hstack([
            X_train_fold, sp.csr_matrix(dense_train),
            sp.csr_matrix(ood_train), sp.csr_matrix(motif_train),
        ]).tocsr()
        X_val_combined = sp.hstack([
            X_val_fold, sp.csr_matrix(dense_val),
            sp.csr_matrix(ood_val), sp.csr_matrix(motif_val),
        ]).tocsr()

        # ── Fix T2: SMOTE oversampling on the TRAIN fold only ──────────────────
        # Applied to the fully-combined feature matrix so the synthetic minority
        # rows stay aligned with the dense/OOD/motif columns. Validation and the
        # production fit are never oversampled.
        if _HAS_SMOTE:
            smote = SMOTE(random_state=42, k_neighbors=3)
            try:
                X_train_combined, y_train_fold = smote.fit_resample(X_train_combined, y_train_fold)
            except ValueError as e:
                print(f"  [!] SMOTE skipped for fold {fold}: {e}")
        else:
            print(f"  [!] SMOTE skipped for fold {fold}: imbalanced-learn not installed")

        # ── Fix T4: RandomForest + isotonic calibration on the validation fold ─
        print(f"  [→] Fitting RandomForest and calibrating on the validation fold...")
        rf_base = build_model()
        rf_base.fit(X_train_combined, y_train_fold)   # X_train_combined includes dense features
        if _HAS_FROZEN:
            model_fold = CalibratedClassifierCV(FrozenEstimator(rf_base), method="isotonic")
        else:
            model_fold = CalibratedClassifierCV(rf_base, method="isotonic", cv="prefit")
        model_fold.fit(X_val_combined, y[val_idx])    # calibrate on val fold

        print(f"  [→] Running Out-Of-Fold Validation Performance Matrix...")
        score = multiclass_auroc(model_fold, X_val_combined, y[val_idx])
        fold_scores.append(score)
        cv_rows.append({"Fold": f"Fold {fold}", "AUROC": f"{score:.4f}"})
        print(
            f"  [✓] Fold {fold} Processing Complete! AUROC: {score:.4f} | Slice Duration: {time.time() - fold_start:.1f}s")

    print("\n  ==================================================")
    mean_a, std_a = np.mean(fold_scores), np.std(fold_scores)
    cv_rows += [{"Fold": "Mean", "AUROC": f"{mean_a:.4f}"},
                {"Fold": "Std", "AUROC": f"{std_a:.4f}"}]
    pd.DataFrame(cv_rows).to_csv(os.path.join(TRAINING_OUTPUT_DIR, "cv_results.csv"), index=False)
    print(f"  [METRICS SUMMARY] Mean Validation AUROC: {mean_a:.4f} ± {std_a:.4f}")

    # ── Fix O2: build and save a bounded CalibratedOODEngine artifact ──────────
    print("\n[OOD] Building bounded CalibratedOODEngine artifact (Platt scaling)...")
    nat_seqs   = raw_seqs[y == LABEL_NATURAL]
    ghost_seqs = raw_seqs[y == LABEL_GHOST]
    cal_size = min(100, len(nat_seqs), len(ghost_seqs))
    if cal_size > 0 and len(nat_seqs) > 0:
        X_ood_train = np.vstack([GhostOODScorer._kmer_freq_vector(s) for s in nat_seqs[:500]])
        X_ood_calib = np.vstack([GhostOODScorer._kmer_freq_vector(s)
                                 for seqs in [nat_seqs[:cal_size], ghost_seqs[:cal_size]]
                                 for s in seqs])
        y_calib = np.array([0] * cal_size + [1] * cal_size)
        calibrated_ood = CalibratedOODEngine(n_estimators=300, contamination=0.1)
        calibrated_ood.fit(X_ood_train, X_ood_calib, y_calib)
        joblib.dump(calibrated_ood, "models/calibrated_ood.pkl")
        print("  [✓] CalibratedOODEngine saved → models/calibrated_ood.pkl")
    else:
        print("  [!] Not enough natural/ghost sequences to fit CalibratedOODEngine — skipped.")

    # ── Production artifacts ───────────────────────────────────────────────────
    # The deployed inference path (main.py / evaluate.py) scores raw TF-IDF
    # vectors, so the production vectorizer + model are fit on the full TF-IDF
    # matrix. Fix T3: the production vectorizer uses ngram_range=(1,2), min_df=3
    # for consistency with the per-fold vectorizer. SMOTE is NOT applied here.
    print("\n[VECTORIZER] Refitting global TF-IDF on the full dataset for production...")
    t_vec = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                          max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
                          analyzer="word")
    X = vec.fit_transform(X_text)
    joblib.dump(vec, VECTORIZER_PATH)
    print(f"  Vocabulary: {len(vec.vocabulary_):,} unique features built in {time.time() - t_vec:.1f}s")

    print("\n[FINAL] Fitting production RandomForest on the entire consolidated dataset...")
    t_final = time.time()
    model = build_model()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"  [✓] Full production model saved in {time.time() - t_final:.1f}s")

    print("  [→] Generating final internal calibration report...")
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
    print("GHOST SIGNATURE DETECTOR — Trainer v2 (M4 Multi-Core Optimized)")
    print("=" * 60)

    df = build_dataset()
    if df.empty:
        print("[ERROR] No training data found. Run dataset_builder.py first.")
        raise SystemExit(1)

    model, vec, cv_scores = train_and_evaluate(df)

    # ── OOD envelope fitting ───────────────────────────────────────────
    print("\n[OOD] Accessing viral natural data points for boundary calibration...")
    t_ood = time.time()
    nat_seqs = [
        str(r.seq) for r in SeqIO.parse(NATURAL_FASTA, "fasta")
        if len(str(r.seq)) > 20
    ]
    print(f"  Loaded {len(nat_seqs)} reference natural sequences.")

    prok_seqs = []
    if os.path.exists(PROKARYOTE_FASTA):
        prok_seqs = [
            str(r.seq) for r in SeqIO.parse(PROKARYOTE_FASTA, "fasta")
            if len(str(r.seq)) > 20
        ]
        print(f"  Loaded {len(prok_seqs)} prokaryotic reference signatures (Dual Envelope Enabled).")
    else:
        print(f"  [!] {PROKARYOTE_FASTA} missing — falling back to Single-Envelope strategy.")

    ood = GhostOODScorer(
        envelope_path=OOD_PATH,
        prokaryote_envelope_path=OOD_PROKARYOTE_PATH,
    )

    ghost_seqs_for_cal = []
    if os.path.exists(GHOST_FASTA):
        ghost_seqs_for_cal = [
            str(r.seq) for r in SeqIO.parse(GHOST_FASTA, "fasta")
            if len(str(r.seq)) > 20
        ]
        print(f"  Loaded {len(ghost_seqs_for_cal)} anomaly sequences for calibration reference.")

    print("  [→] Fitting Isolation Forest envelopes to establish genetic boundaries...")
    ood.fit_on_sequences(
        nat_sequences=nat_seqs,
        prokaryote_sequences=prok_seqs or None,
        anomaly_sequences=ghost_seqs_for_cal or None,
        contamination=OOD_CONTAMINATION,
        random_state=42,
    )
    print(f"  [✓] OOD matrices calculated and exported in {time.time() - t_ood:.1f}s")

    # ── OOD sanity check ─────────────────────────────────────────────────────
    print("\n[OOD SANITY CHECK] Verifying scoring distributions...")
    ghost_seqs_check = [
        str(r.seq) for r in SeqIO.parse(GHOST_FASTA, "fasta")
        if len(str(r.seq)) > 20
    ][:20]
    nat_seqs_check = nat_seqs[:20]

    if ghost_seqs_check and nat_seqs_check:
        ghost_scores = ood.batch_scores(ghost_seqs_check)
        nat_scores = ood.batch_scores(nat_seqs_check)
        gm = np.mean(ghost_scores)
        nm = np.mean(nat_scores)
        print(f"  Ghost Mean Score: {gm:.1f} | Natural Mean Score: {nm:.1f}")
        if gm > nm:
            print("  [OK] Distribution confirmed: Engineered variants register structurally higher OOD values.")
        else:
            print("  [WARN] Boundary inversion detected. Review sequence mutation variables.")

    print(f"\n[DONE] Execution Complete. Total Workflow Runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"  ↳ Production Model Artifact: {MODEL_PATH}")
    print(f"  ↳ Vectorizer Matrix:        {VECTORIZER_PATH}")
    print(f"  ↳ Out-of-Distribution File:  {OOD_PATH}")
    if prok_seqs:
        print(f"  ↳ Prokaryote Vector File:    {OOD_PROKARYOTE_PATH}")
    print(f"  ↳ Cross-Validation Log:      {TRAINING_OUTPUT_DIR}/cv_results.csv")
    print("\nRun: python evaluate.py to verify results.")
    print("=" * 60)
