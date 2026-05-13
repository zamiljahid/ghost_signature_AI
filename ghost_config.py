OWNER_NAME   = "Zamil Jahid"
OWNER_EMAIL  = "zamiljahid2002@gmail.com"
OWNERSHIP_TEXT = (
    "This software and all associated reports are the exclusive intellectual "
    "property of Zamil Jahid. Developed as part of an original research thesis. "
    "Unauthorized reproduction or redistribution is prohibited."
)

KMER_SIZE        = 8          # k-mer length for AI feature extraction
CHUNK_SIZE       = 500        # base-pair window per training fragment
TFIDF_NGRAM      = (1, 1)     # TF-IDF n-gram range

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
OOD_THRESHOLD       = 50.0    # score above this = ghost anomaly
OOD_SCALE_FACTOR    = 10.0    # raw score → 0-100 scale

# ── Ghost motif discovery ──────────────────────────────────────────────────────
MOTIF_K             = 6       # k-mer length for motif enrichment
MOTIF_TOP_N         = 10      # top N enriched motifs to report

# ── Ghost benchmark generation ─────────────────────────────────────────────────
GHOST_N_SAMPLES     = 100     # synthetic ghost sequences to generate
GHOST_MUTATION_RATE = 0.20    # random substitution rate
GHOST_N_BREAKPOINTS = 2       # recombination breakpoints per chimera

# ── Evaluation thresholds ──────────────────────────────────────────────────────
EVAL_THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]   # for threshold table
CV_FOLDS        = 5

# ── Short-fragment detection ───────────────────────────────────────────────────
SHORT_SEQ_THRESHOLD     = 300   # bp below which k-mer model confidence degrades
SHORT_SEQ_AI_FLOOR      = 15.0  # minimum AI risk % reported for short sequences
SHORT_SEQ_OOD_FLOOR     = 20.0  # minimum OOD score reported for short sequences
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
OUTPUT_DIR          = "outputs"
MODEL_DIR           = "models"