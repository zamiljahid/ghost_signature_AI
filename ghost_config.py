from pathlib import Path

OWNER_NAME   = "Zamil Jahid"
OWNER_EMAIL  = "zamiljahid2002@gmail.com"
OWNERSHIP_TEXT = (
    "This software and all associated reports are the exclusive intellectual "
    "property of Zamil Jahid. Developed as part of an original research thesis. "
    "Unauthorized reproduction or redistribution is prohibited."
)

KMER_SIZE        = 8          # kept for backward compat with any direct callers
KMER_SIZES       = (4, 6, 8)  # multi-scale k-mers — gives 54 features at 23 bp
CHUNK_SIZE       = 500        # base-pair window per training fragment
TFIDF_NGRAM      = (1, 1)     # TF-IDF n-gram range
TFIDF_MAX_FEATURES = 100_000  # up from 50,000 — multi-scale vocab is larger

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_PATH              = "models/ghost_model.pkl"
VECTORIZER_PATH         = "models/dna_vectorizer.pkl"
OOD_PATH                = "models/ood_envelope.pkl"
OOD_PROKARYOTE_PATH     = "models/ood_envelope_prokaryote.pkl"
SCALER_PATH             = "models/feature_scaler.pkl"

# ── Training labels ────────────────────────────────────────────────────────────
LABEL_NATURAL  = 0
LABEL_VECTOR   = 1
LABEL_GHOST    = 2

# ── OOD scorer ─────────────────────────────────────────────────────────────────
OOD_CONTAMINATION   = 0.05    # expected outlier fraction in training set
OOD_THRESHOLD       = 50.0    # plot annotation only — NOT the operational verdict threshold

# ── Operational verdict thresholds (empirically calibrated on 200-seq benchmark) ──
# Natural sequences: mean OOD ≈ 3.72  |  Ghost sequences: mean OOD ≈ 15.53
# At OOD_SUSPECT_THRESHOLD=5.0: FPR ≈ 4 %, Ghost TPR ≈ 45 % — dual-gate validated.
AI_SUSPECT_THRESHOLD     = 60.0   # ai_risk % above which AI gate fires
OOD_SUSPECT_THRESHOLD    =  5.0   # OOD score above which structural gate fires
AI_BORDERLINE_THRESHOLD  = 35.0   # ai_risk % lower bound for BORDERLINE verdict
OOD_BORDERLINE_THRESHOLD =  2.0   # OOD score lower bound for BORDERLINE verdict

# ── Short-sequence policy ───────────────────────────────────────────────────────
SHORT_SEQ_GHOST_POLICY   = 20     # bp below which AI/OOD gates are bypassed (was 100 — 4-mers now provide signal at 20+ bp)

# ── Ghost motif discovery ──────────────────────────────────────────────────────
MOTIF_K             = 6       # k-mer length for motif enrichment
MOTIF_TOP_N         = 10      # top N enriched motifs to report

# ── Ghost benchmark generation ─────────────────────────────────────────────────
GHOST_N_SAMPLES     = 100     # synthetic ghost sequences to generate
GHOST_MUTATION_RATE = 0.15    # realistic synthetic construct mutation level (was 0.45 which created near-random noise)
GHOST_N_BREAKPOINTS = 4       # recombination breakpoints per chimera (raised from 2)

# ── Evaluation thresholds ──────────────────────────────────────────────────────
EVAL_THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]   # for threshold table
CV_FOLDS        = 5  # 5-fold cross-validation

# ── Short-fragment detection ───────────────────────────────────────────────────
SHORT_SEQ_THRESHOLD     = 50    # bp below which to flag as very short (was 300 — multi-scale features work down to 20 bp)
SHORT_SEQ_AI_FLOOR      = 0.0   # floor removed — multi-scale classifier runs properly at any length above MIN_SEQ_LENGTH
SHORT_SEQ_OOD_FLOOR     = 0.0   # floor removed — 4-mer IsolationForest is valid at 23 bp
PLASMID_GC_THRESHOLD    = 0.55  # GC fraction above which plasmid flag triggers
PLASMID_AI_FLOOR        = 25.0  # minimum AI risk % reported for plasmid-suspects

# ── Plotting ───────────────────────────────────────────────────────────────────
PLOT_DPI    = 200
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
FIG_SIZE_SM = (7, 5)
FIG_SIZE_LG = (14, 5)

# ── NCBI / Dataset sources ─────────────────────────────────────────────────────
NCBI_EMAIL = "zamiljahid2002@gmail.com"

# Natural virus accessions (diverse families — expanded)
NATURAL_IDS = [
    # ASFV
    "NC_001659.2", "NC_044487.1", "AM712239.1",
    # Respiratory
    "NC_007374.1", "NC_012532.1", "NC_002549.1", "NC_001405.1", "NC_045512.2",
    # Large DNA viruses
    "NC_000852.5", "NC_006273.2", "NC_003310.1", "NC_001359.1",
    # Phages
    "NC_001416.1", "NC_001422.1", "NC_001604.1", "NC_001330.1",
    # Diverse families
    "NC_003977.2", "NC_001477.1", "NC_001489.1", "NC_001563.2",
    "NC_001436.1", "NC_001716.2", "NC_001722.1", "NC_001802.1",
    "NC_001806.2", "NC_001927.1", "NC_002645.1", "NC_003551.1",
    "NC_004102.1", "NC_004148.2", "NC_004355.1", "NC_005084.2",
    # Additional diversity (herpesviruses, poxviruses, retroviruses)
    "NC_001798.2", "NC_001664.4", "NC_007605.1", "NC_009334.1",
    "NC_001348.1", "NC_004829.1", "NC_006998.1", "NC_003391.1",
]

# EVE / ancient virus accessions for ghost evaluation
EVE_IDS = [
    "NC_022518.1",   # HERV-K113 (human endogenous retrovirus)
    "AY037928.1",    # HERV-H
    "M14123.1",      # HERV-E
    "X89714.1",      # HERV-W (syncytin ancestor)
    "AF152573.1",    # HERV-FRD
]
PROKARYOTE_IDS = [
    "NC_000913.3",   # E. coli K-12 MG1655
    "NC_000964.3",   # B. subtilis 168
    "NC_002695.2",   # E. coli O157:H7
    "NC_003197.2",   # Salmonella typhimurium LT2
]

# Data paths
DATA_DIR            = "data"
NATURAL_FASTA       = f"{DATA_DIR}/natural_viruses.fasta"
VECTORS_FASTA       = f"{DATA_DIR}/vectors.fasta"
GHOST_FASTA         = f"{DATA_DIR}/ghost_benchmark.fasta"
EVE_FASTA           = f"{DATA_DIR}/eve_sequences.fasta"
PLASMID_FASTA       = f"{DATA_DIR}/plasmids.fasta"
PROKARYOTE_FASTA    = f"{DATA_DIR}/prokaryotes.fasta"

# Independent test sets — paths relative to this config file so the project is portable
_PROJECT_ROOT = Path(__file__).resolve().parent
INDEPENDENT_TEST_NATURAL = str(_PROJECT_ROOT / "independent_test_set_Natural.fasta")
INDEPENDENT_TEST_VECTOR  = str(_PROJECT_ROOT / "independent_test_set_Vector.fasta")
INDEPENDENT_TEST_GHOST   = str(_PROJECT_ROOT / "independent_test_set_Ghost.fasta")

OUTPUT_DIR          = "outputs"
MODEL_DIR           = "models"