import itertools

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from scipy.stats import percentileofscore

# ── Pre-compute 4-mer lookup (256 entries, vocabulary-independent) ─────────
_BASES    = "ATGC"
_KMER4    = ["".join(p) for p in itertools.product(_BASES, repeat=4)]
_KMER4_IDX = {km: i for i, km in enumerate(_KMER4)}   # 256 entries


class GhostOODScorer:

    SHORT_SEQ_THRESHOLD = 50    # bp; 4-mer IsolationForest is valid at 20+ bp
    SHORT_SEQ_OOD_FLOOR = 0.0  # floor removed — 4-mer scoring works at 23 bp

    def __init__(self,
                 envelope_path="models/ood_envelope.pkl",
                 prokaryote_envelope_path="models/ood_envelope_prokaryote.pkl"):

        self.envelope_path            = envelope_path
        self.prokaryote_envelope_path = prokaryote_envelope_path

        # ── Viral envelope ─────────────────────────────────────────────
        self.detector    = None
        self._cal_mean   = None
        self._cal_std    = None

        # ── Prokaryotic envelope ───────────────────────────────────────
        self.detector_prok   = None
        self._cal_mean_prok  = None
        self._cal_std_prok   = None

        # Two-anchor calibration — high anchor from ghost/anomaly sequences
        self._cal_high      = None   # mean raw score on anomaly sequences (viral)
        self._cal_high_prok = None   # mean raw score on anomaly sequences (prok)

        # Direction multiplier: +1 normal, -1 when detector direction is inverted
        self._cal_direction      = 1
        self._cal_direction_prok = 1

        self._load()

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #
    def _load(self):
        """Load both envelope bundles from disk if present."""
        # Viral envelope
        try:
            bundle = joblib.load(self.envelope_path)
            if isinstance(bundle, dict):
                self.detector        = bundle.get("detector")
                self._cal_mean       = bundle.get("cal_mean")
                self._cal_std        = bundle.get("cal_std")
                self._cal_high       = bundle.get("cal_high")
                self._cal_direction  = bundle.get("cal_direction", 1)
            else:
                self.detector = bundle
        except FileNotFoundError:
            pass

        # Prokaryotic envelope
        try:
            bundle_prok = joblib.load(self.prokaryote_envelope_path)
            if isinstance(bundle_prok, dict):
                self.detector_prok       = bundle_prok.get("detector")
                self._cal_mean_prok      = bundle_prok.get("cal_mean")
                self._cal_std_prok       = bundle_prok.get("cal_std")
                self._cal_high_prok      = bundle_prok.get("cal_high")
                self._cal_direction_prok = bundle_prok.get("cal_direction", 1)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #
    @property
    def ready(self):
        return self.detector is not None

    @property
    def prokaryote_ready(self):
        return self.detector_prok is not None

    @property
    def _calibrated(self):
        return self._cal_mean is not None and self._cal_std is not None

    @property
    def _calibrated_prok(self):
        return self._cal_mean_prok is not None and self._cal_std_prok is not None

    # ------------------------------------------------------------------ #
    # Feature extraction — k-mer frequency (vocabulary-independent)       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kmer_freq_vector(seq: str) -> np.ndarray:
        """
        Compute normalised 4-mer frequency vector (256 dims).

        All 256 possible ATGC 4-mers are counted directly from the sequence.
        There is no external vocabulary — every nucleotide combination,
        including those absent from the training set, is represented.
        This prevents ghost sequences with novel k-mers from collapsing
        to a near-zero vector as they do with TF-IDF.
        """
        counts = np.zeros(256, dtype=np.float32)
        n = len(seq) - 3
        for i in range(n):
            idx = _KMER4_IDX.get(seq[i:i + 4])
            if idx is not None:
                counts[idx] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts

    def _embed(self, sequence: str, chunk: int = 500) -> np.ndarray:
        """
        Embed a sequence as a matrix of 4-mer frequency vectors
        (one row per 500-bp chunk).  Shape: (n_chunks, 256).

        Replaces the old TF-IDF + SVD pipeline.  SVD is no longer needed
        because the feature space is already small (256 dims) and fully
        captures composition including novel / mutated k-mers.
        """
        seq = "".join(c for c in sequence.upper() if c in "ATGC")
        rows = []
        for i in range(0, max(1, len(seq) - chunk + 1), chunk):
            frag = seq[i:i + chunk]
            if len(frag) >= 4:
                rows.append(self._kmer_freq_vector(frag))
        return np.array(rows, dtype=np.float32) if rows else np.zeros((1, 256), dtype=np.float32)

    def _clean_len(self, sequence: str) -> int:
        return len("".join(c for c in sequence.upper() if c in "ATGCN"))

    def _is_short(self, sequence: str) -> bool:
        return self._clean_len(sequence) < self.SHORT_SEQ_THRESHOLD

    # ------------------------------------------------------------------ #
    # Scoring helpers                                                      #
    # ------------------------------------------------------------------ #
    def _raw_score(self, X: np.ndarray, detector) -> float:
        """Mean anomaly score across chunks (higher = more anomalous)."""
        return float(np.mean(-detector.decision_function(X)))

    def _calibrated_score(self, raw: float, cal_mean, cal_std,
                          cal_high=None, cal_direction: int = 1) -> float:
        """
        Map raw IsolationForest score to 0-100.

        cal_direction: +1 (normal) or -1 (inverted detector — raw scores are
            negated before applying the calibration formula so that anomalies
            always map toward 100 and natural sequences toward 0).

        Two-anchor mode (preferred): uses natural mean as low anchor (→0)
        and ghost/anomaly mean as high anchor (→100).

        Fallback: single-anchor sigmoid when no high anchor is available.
        """
        if cal_mean is None:
            return self._fallback_score(raw)

        # Apply direction flip (negation of raw and anchors is pre-applied at
        # fit time; here we just negate raw so the formula stays the same).
        effective_raw = raw * cal_direction

        # Two-anchor linear calibration (paper Eq.1).
        # NOTE (Issue 3): this raw linear two-anchor map is unbounded by
        # construction — values below cal_mean go negative and values above
        # cal_high exceed 100. It is only kept in-range by the np.clip() below,
        # which masks miscalibration rather than fixing it. For a calibrated
        # probability in [0,1], use CalibratedOODEngine (Platt scaling) defined
        # at the bottom of this module instead. Fusion must consume bounded
        # probabilities via fuse_engine_scores().
        if cal_high is not None and cal_high > cal_mean:
            span  = cal_high - cal_mean
            score = (effective_raw - cal_mean) / span * 100.0
            return round(float(np.clip(score, 0.0, 100.0)), 2)

        # Single-anchor sigmoid fallback
        std     = max(cal_std if cal_std is not None else 1e-6, 1e-6)
        z       = (effective_raw - cal_mean) / std
        sigmoid = 1.0 / (1.0 + np.exp(-z * 3.0))
        return round(float(sigmoid * 100.0), 2)

    @staticmethod
    def _fallback_score(raw: float) -> float:
        shifted = raw + 0.1
        return round(min(100.0, max(0.0, shifted * 400.0)), 2)

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #
    def _fit_one_envelope(self, sequences, contamination, random_state, label,
                          anomaly_sequences=None):
        """
        Fit a single IsolationForest envelope using 4-mer frequency vectors.

        anomaly_sequences: optional list of known anomalous (ghost/synthetic)
            sequences used to anchor the high end of the calibration scale.
            When provided, scoring becomes direction-invariant.

        Returns (detector, cal_mean, cal_std, cal_high).
        """
        print(f"[OOD] Building 4-mer frequency vectors for {label} sequences...")
        rows = []
        for s in sequences:
            seq = "".join(c for c in s.upper() if c in "ATGC")
            for i in range(0, max(1, len(seq) - 500 + 1), 500):
                frag = seq[i:i + 500]
                if len(frag) >= 4:
                    rows.append(self._kmer_freq_vector(frag))

        if not rows:
            print(f"[OOD] No data for {label} — skipping.")
            return None, None, None, None, 1

        X = np.array(rows, dtype=np.float32)
        print(f"[OOD]   [{label}] {len(X)} chunks × 256 features")

        print(f"[OOD] Fitting IsolationForest (contamination={contamination}) [{label}]...")
        detector = IsolationForest(
            n_estimators=600,
            contamination=contamination,
            max_samples="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        detector.fit(X)

        nat_raw  = -detector.decision_function(X)
        cal_mean = float(np.mean(nat_raw))
        cal_std  = float(np.std(nat_raw))
        print(f"[OOD]   [{label}] Natural anchor: mean={cal_mean:.4f}, std={cal_std:.4f}")

        # Compute high anchor from anomaly sequences
        cal_high      = None
        cal_direction = 1
        if anomaly_sequences:
            anom_rows = []
            for s in anomaly_sequences:
                seq = "".join(c for c in s.upper() if c in "ATGC")
                for i in range(0, max(1, len(seq) - 500 + 1), 500):
                    frag = seq[i:i + 500]
                    if len(frag) >= 4:
                        anom_rows.append(self._kmer_freq_vector(frag))
            if anom_rows:
                X_anom   = np.array(anom_rows, dtype=np.float32)
                anom_raw = -detector.decision_function(X_anom)
                cal_high = float(np.mean(anom_raw))
                print(f"[OOD]   [{label}] Anomaly anchor: mean={cal_high:.4f}")
                if cal_high > cal_mean:
                    print(f"[OOD]   [{label}] Two-anchor calibration ACTIVE — "
                          f"direction confirmed correct.")
                else:
                    print(f"[OOD]   [{label}] WARNING: anomaly anchor <= natural anchor "
                          f"({cal_high:.4f} <= {cal_mean:.4f}). "
                          f"Forcing direction flip via negation.")
                    # The detector's raw scores are inverted: anomalies score LOWER
                    # than natural sequences.  Negate both anchors so the formula
                    # (raw*direction - cal_mean) / (cal_high - cal_mean) still maps
                    # natural→0 and anomaly→100.
                    cal_direction = -1
                    cal_mean = -cal_mean    # negated natural anchor (now the low end)
                    cal_high = -cal_high    # negated anomaly anchor (now the high end)

        return detector, cal_mean, cal_std, cal_high, cal_direction

    def fit(self, nat_sequences, **kwargs):
        """
        Convenience wrapper around fit_on_sequences().

        Added for Fix O1 so train_model.py can fit the scorer on the
        natural-class sequences with a single positional argument:
        ``GhostOODScorer().fit(natural_sequences)``. All keyword arguments
        accepted by fit_on_sequences() (prokaryote_sequences,
        anomaly_sequences, contamination, random_state) are forwarded.
        """
        return self.fit_on_sequences(list(nat_sequences), **kwargs)

    def fit_on_sequences(self, nat_sequences,
                         prokaryote_sequences=None,
                         anomaly_sequences=None,
                         contamination=0.05,
                         random_state=42):
        """
        Fit viral envelope (required) and optionally prokaryotic envelope.

        Parameters
        ----------
        nat_sequences        : list[str]  Natural viral sequences (low anchor).
        prokaryote_sequences : list[str]  Prokaryotic sequences (optional).
        anomaly_sequences    : list[str]  Known synthetic/ghost sequences used
                                          to anchor the high end of the score
                                          scale (strongly recommended).
        contamination        : float      IsolationForest contamination rate.
        random_state         : int        RNG seed.
        """
        # ── Viral envelope ─────────────────────────────────────────────
        (self.detector, self._cal_mean, self._cal_std,
         self._cal_high, self._cal_direction) = \
            self._fit_one_envelope(nat_sequences, contamination,
                                   random_state, label="viral",
                                   anomaly_sequences=anomaly_sequences)

        joblib.dump({
            "detector":      self.detector,
            "cal_mean":      self._cal_mean,
            "cal_std":       self._cal_std,
            "cal_high":      self._cal_high,
            "cal_direction": self._cal_direction,
        }, self.envelope_path)
        print(f"[OOD] Viral bundle saved → {self.envelope_path}")

        # ── Prokaryotic envelope (optional) ────────────────────────────
        if prokaryote_sequences:
            (self.detector_prok, self._cal_mean_prok,
             self._cal_std_prok, self._cal_high_prok,
             self._cal_direction_prok) = \
                self._fit_one_envelope(prokaryote_sequences, contamination,
                                       random_state, label="prokaryotic",
                                       anomaly_sequences=anomaly_sequences)

            joblib.dump({
                "detector":      self.detector_prok,
                "cal_mean":      self._cal_mean_prok,
                "cal_std":       self._cal_std_prok,
                "cal_high":      self._cal_high_prok,
                "cal_direction": self._cal_direction_prok,
            }, self.prokaryote_envelope_path)
            print(f"[OOD] Prokaryotic bundle saved → {self.prokaryote_envelope_path}")
        else:
            print("[OOD] No prokaryotic sequences provided — single-envelope mode.")

    # ------------------------------------------------------------------ #
    # Scoring — public API                                                 #
    # ------------------------------------------------------------------ #
    def ghost_anomaly_score(self, sequence) -> dict:
        """
        Score a single sequence for OOD anomaly.

        Returns dict with keys: score, viral_score, prok_score, flag, narrative.
        """
        if not self.ready:
            return {
                "score": 0.0, "viral_score": 0.0, "prok_score": None,
                "flag": "NOT_READY",
                "narrative": "OOD scorer not fitted. Run train.py first.",
            }

        short   = self._is_short(sequence)
        seq_len = self._clean_len(sequence)

        try:
            # Embed once — both envelopes share the same 4-mer feature space.
            X = self._embed(sequence)

            # ── Viral score ─────────────────────────────────────────────
            raw_viral   = self._raw_score(X, self.detector)
            viral_score = self._calibrated_score(
                raw_viral, self._cal_mean, self._cal_std,
                self._cal_high, self._cal_direction)

            # ── Prokaryotic score (if envelope fitted) ───────────────────
            prok_score = None
            if self.prokaryote_ready:
                raw_prok   = self._raw_score(X, self.detector_prok)
                prok_score = self._calibrated_score(
                    raw_prok, self._cal_mean_prok, self._cal_std_prok,
                    self._cal_high_prok, self._cal_direction_prok)

            # ── Use viral envelope as primary OOD score ───────────────────
            # The viral envelope measures anomaly relative to natural viral
            # sequences — directly answering "is this ghost?".
            # The prokaryotic envelope is NOT combined via MAX because natural
            # viral sequences are anomalous to the prokaryotic detector by
            # design (they're not bacteria), which would artificially inflate
            # natural sequence scores and invert the ghost detection signal.
            # Prokaryotic score is reported in the narrative as secondary info.
            final_score = viral_score
            if prok_score is not None:
                flag      = "DUAL_ENVELOPE"
                narrative = (
                    f"Viral envelope (primary): {viral_score:.1f}/100. "
                    f"Prokaryotic envelope (reference only): {prok_score:.1f}/100. "
                    f"Final OOD score: {final_score:.1f}/100."
                )
            else:
                flag      = None
                narrative = f"Viral envelope only. Score: {viral_score:.1f}/100."

            # ── Short-fragment floor ─────────────────────────────────────
            if short:
                final_score = max(final_score, self.SHORT_SEQ_OOD_FLOOR)
                flag        = "SHORT_FRAGMENT"
                narrative  += (
                    f" ⚠️ Short sequence ({seq_len}bp < "
                    f"{self.SHORT_SEQ_THRESHOLD}bp): k-mer statistics unreliable. "
                    f"OOD score floored at {self.SHORT_SEQ_OOD_FLOOR}."
                )

            return {
                "score":       round(final_score, 2),
                "viral_score": viral_score,
                "prok_score":  prok_score,
                "flag":        flag,
                "narrative":   narrative,
            }

        except Exception as e:
            return {
                "score": 0.0, "viral_score": 0.0, "prok_score": None,
                "flag": "ERROR", "narrative": str(e),
            }

    # ── Convenience wrappers ─────────────────────────────────────────────
    def score_per_window(self, sequence, window=500, step=None):
        """Return list of (position, score) tuples across sliding windows."""
        if not self.ready:
            return []
        seq  = "".join(c for c in sequence.upper() if c in "ATGC")
        step = step or max(50, window // 5)
        return [
            (i + window // 2,
             self.ghost_anomaly_score(seq[i:i + window])["score"])
            for i in range(0, len(seq) - window + 1, step)
        ]

    def batch_scores(self, sequences):
        """Return list of final scores (floats) for a list of sequences."""
        return [self.ghost_anomaly_score(s)["score"] for s in sequences]

    # ── Diagnostics ─────────────────────────────────────────────────────
    def calibration_summary(self) -> dict:
        return {
            "ready":              self.ready,
            "prokaryote_ready":   self.prokaryote_ready,
            "calibrated":         self._calibrated,
            "cal_mean":           self._cal_mean,
            "cal_std":            self._cal_std,
            "cal_high":           self._cal_high,
            "cal_direction":      self._cal_direction,
            "cal_mean_prok":      self._cal_mean_prok,
            "cal_std_prok":       self._cal_std_prok,
            "cal_high_prok":      self._cal_high_prok,
            "cal_direction_prok": self._cal_direction_prok,
        }

    def score_distribution(self, sequences, label="sequences"):
        scores = self.batch_scores(sequences)
        if not scores:
            print(f"[OOD] No scores for {label}.")
            return scores
        arr = np.array(scores)
        print(f"[OOD] {label} (n={len(arr)}): "
              f"mean={arr.mean():.1f}, std={arr.std():.1f}, "
              f"min={arr.min():.1f}, max={arr.max():.1f}")
        return scores


# ═════════════════════════════════════════════════════════════════════════════
# Issue 3 — Bounded OOD calibration & score fusion
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the unbounded linear two-anchor map (paper Eq.1) and the unbounded
# additive fusion (paper Eq.2). Every engine output is forced into [0,1] before
# fusion, and the fused score is a convex combination (weights sum to 1), so it
# is provably in [0,1] without relying on a post-hoc clip.
# ═════════════════════════════════════════════════════════════════════════════
class CalibratedOODEngine:
    """
    Wraps IsolationForest with Platt scaling so all outputs are strictly in
    [0, 1]. Replaces the unbounded linear-map formula from the original
    Equation 1 that produced impossible FPR values (negative / >100%).

    Platt scaling here is a 1-D logistic regression fitted on the IsolationForest
    raw anomaly score. We fit the logistic directly (rather than via
    CalibratedClassifierCV) because IsolationForest is an anomaly detector, not a
    scikit-learn classifier — CalibratedClassifierCV cannot wrap it, and its
    cv='prefit' path was removed in recent sklearn. The logistic sigmoid
    guarantees the output is bounded in (0, 1).
    """

    def __init__(self, n_estimators: int = 100, contamination: float = 0.1,
                 random_state: int = 42):
        self.iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.platt = None   # logistic regression on the raw anomaly score

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Raw anomaly score, higher = more anomalous (outlier)."""
        return -self.iso.decision_function(X)

    def fit(self, X_train: np.ndarray,
            X_calib: np.ndarray, y_calib: np.ndarray) -> "CalibratedOODEngine":
        """
        X_train : inlier-only training features
        X_calib : calibration set features (mix of inliers and outliers)
        y_calib : binary labels (0 = inlier/natural, 1 = outlier/synthetic)
        """
        self.iso.fit(X_train)
        f_calib = self._anomaly_score(X_calib).reshape(-1, 1)
        self.platt = LogisticRegression()
        self.platt.fit(f_calib, np.asarray(y_calib).astype(int))
        return self

    def predict_proba_outlier(self, X: np.ndarray) -> np.ndarray:
        """Returns P(outlier) in [0, 1] for every sample."""
        f = self._anomaly_score(X).reshape(-1, 1)
        P = self.platt.predict_proba(f)[:, 1]
        assert P.min() >= 0.0 and P.max() <= 1.0, (
            f"OOD score out of [0,1]: min={P.min():.4f}, max={P.max():.4f}"
        )
        return P


def empirical_risk_score(train_scores: np.ndarray,
                         test_scores: np.ndarray) -> np.ndarray:
    """
    Converts raw integer engine counts (BLAST hits, motif counts, codon scores)
    into normalized [0,1] risk scores via the empirical CDF on the training set.
    Used for BLAST, Motif and Codon engines so all inputs to fusion are bounded.
    """
    return np.array([
        percentileofscore(train_scores, s, kind="rank") / 100.0
        for s in test_scores
    ])


def fuse_engine_scores(P_kmer:  np.ndarray,
                       P_ood:   np.ndarray,
                       P_blast: np.ndarray,
                       P_motif: np.ndarray,
                       P_codon: np.ndarray,
                       weights: dict) -> np.ndarray:
    """
    Fused suspicion score — guaranteed in [0, 1].
    Formula: S_fused(x) = sum_i w_i * P_i(x),  sum(w_i) = 1.
    All P_i must independently be in [0, 1] before fusion.
    """
    assert abs(sum(weights.values()) - 1.0) < 1e-6, \
        f"Weights must sum to 1, got {sum(weights.values()):.6f}"

    engines = {
        "kmer":  np.asarray(P_kmer,  dtype=float),
        "ood":   np.asarray(P_ood,   dtype=float),
        "blast": np.asarray(P_blast, dtype=float),
        "motif": np.asarray(P_motif, dtype=float),
        "codon": np.asarray(P_codon, dtype=float),
    }
    for name, P in engines.items():
        assert np.all(P >= 0) and np.all(P <= 1), (
            f"Engine '{name}' score out of [0,1]: "
            f"min={P.min():.4f}, max={P.max():.4f}"
        )

    S = sum(weights[k] * engines[k] for k in engines)
    assert S.min() >= 0.0 and S.max() <= 1.0, \
        f"Fused score out of [0,1]: min={S.min():.4f}, max={S.max():.4f}"
    return S
