import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD


class GhostOODScorer:

    SHORT_SEQ_THRESHOLD = 300   # bp; overridden by ghost_config if imported
    SHORT_SEQ_OOD_FLOOR = 20.0

    def __init__(self,
                 vectorizer_path="models/dna_vectorizer.pkl",
                 envelope_path="models/ood_envelope.pkl",
                 prokaryote_envelope_path="models/ood_envelope_prokaryote.pkl",
                 svd_n_components=128):

        self.vectorizer_path          = vectorizer_path
        self.envelope_path            = envelope_path
        self.prokaryote_envelope_path = prokaryote_envelope_path
        self.svd_n_components         = svd_n_components

        # ── Viral envelope ─────────────────────────────────────────────
        self.vectorizer  = None
        self.svd         = None
        self.detector    = None
        self._cal_mean   = None
        self._cal_std    = None

        # ── Prokaryotic envelope ───────────────────────────────────────
        self.svd_prok        = None
        self.detector_prok   = None
        self._cal_mean_prok  = None
        self._cal_std_prok   = None

        self._load()

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #
    def _load(self):
        """Load vectorizer and both envelope bundles from disk if present."""
        # Viral envelope
        try:
            bundle = joblib.load(self.envelope_path)
            if isinstance(bundle, dict):
                self.svd       = bundle.get("svd")
                self.detector  = bundle.get("detector")
                self._cal_mean = bundle.get("cal_mean")
                self._cal_std  = bundle.get("cal_std")
            else:
                # Legacy: bare detector object
                self.detector = bundle
        except FileNotFoundError:
            pass

        # Prokaryotic envelope (optional — absent on first run before fitting)
        try:
            bundle_prok = joblib.load(self.prokaryote_envelope_path)
            if isinstance(bundle_prok, dict):
                self.svd_prok       = bundle_prok.get("svd")
                self.detector_prok  = bundle_prok.get("detector")
                self._cal_mean_prok = bundle_prok.get("cal_mean")
                self._cal_std_prok  = bundle_prok.get("cal_std")
        except FileNotFoundError:
            pass

        # Vectorizer
        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #
    @property
    def ready(self):
        return (self.vectorizer is not None
                and self.svd is not None
                and self.detector is not None)

    @property
    def prokaryote_ready(self):
        return (self.svd_prok is not None and self.detector_prok is not None)

    @property
    def _calibrated(self):
        return self._cal_mean is not None and self._cal_std is not None

    @property
    def _calibrated_prok(self):
        return self._cal_mean_prok is not None and self._cal_std_prok is not None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _to_texts(self, sequence, k=8, chunk=500):
        seq = "".join(c for c in sequence.upper() if c in "ATGCN")
        texts = []
        for i in range(0, max(1, len(seq) - chunk + 1), chunk):
            frag = seq[i:i + chunk]
            if len(frag) >= k:
                texts.append(" ".join(frag[j:j+k] for j in range(len(frag) - k + 1)))
        return texts or (["AAAAAAAA"] if not seq else [seq[:chunk]])

    def _clean_len(self, sequence: str) -> int:
        return len("".join(c for c in sequence.upper() if c in "ATGCN"))

    def _is_short(self, sequence: str) -> bool:
        return self._clean_len(sequence) < self.SHORT_SEQ_THRESHOLD

    def _embed(self, sequence, svd):
        """Vectorize sequence and project into the given SVD space."""
        texts   = self._to_texts(sequence)
        X_tfidf = self.vectorizer.transform(texts)   # sparse (n_chunks, vocab)
        return svd.transform(X_tfidf)                # dense  (n_chunks, n_components)

    def _raw_score(self, X_svd: np.ndarray, detector) -> float:
        """Mean anomaly score across all chunks (higher = more anomalous)."""
        return float(np.mean(-detector.decision_function(X_svd)))

    def _calibrated_score(self, raw: float, cal_mean, cal_std) -> float:
        """Map raw IsolationForest score to 0–100 via sigmoid calibration."""
        if cal_mean is None or cal_std is None:
            return self._fallback_score(raw)
        std     = max(cal_std, 1e-6)
        z       = (raw - cal_mean) / std
        sigmoid = 1.0 / (1.0 + np.exp(-z * 3.0))
        return round(float(sigmoid * 100.0), 2)

    @staticmethod
    def _fallback_score(raw: float) -> float:
        shifted = raw + 0.1
        return round(min(100.0, max(0.0, shifted * 400.0)), 2)

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #
    def _fit_one_envelope(self, sequences, contamination, random_state, label):
        """
        Fit a single IsolationForest envelope on the provided sequences.
        Returns (svd, detector, cal_mean, cal_std) or (None,)*4 if empty.
        """
        print(f"[OOD] Building TF-IDF chunks for {label} sequences...")
        texts = []
        for s in sequences:
            texts.extend(self._to_texts(s))

        if not texts:
            print(f"[OOD] No texts for {label} — skipping.")
            return None, None, None, None

        X_tfidf = self.vectorizer.transform(texts)   # sparse

        print(f"[OOD] Fitting TruncatedSVD ({self.svd_n_components} components) [{label}]...")
        svd = TruncatedSVD(n_components=self.svd_n_components,
                           random_state=random_state)
        X_svd     = svd.fit_transform(X_tfidf)
        explained = svd.explained_variance_ratio_.sum()
        print(f"[OOD]   [{label}] Explained variance: {explained:.3f}")

        print(f"[OOD] Fitting IsolationForest (contamination={contamination}) [{label}]...")
        detector = IsolationForest(
            n_estimators=600,
            contamination=contamination,
            max_samples="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        detector.fit(X_svd)

        train_raw = -detector.decision_function(X_svd)
        cal_mean  = float(np.mean(train_raw))
        cal_std   = float(np.std(train_raw))
        print(f"[OOD]   [{label}] Calibration: mean={cal_mean:.4f}, std={cal_std:.4f}")

        if cal_mean > 0.3:
            print(f"[OOD] WARNING [{label}]: cal_mean is high — "
                  "natural sequences may look anomalous. Check input data.")

        return svd, detector, cal_mean, cal_std

    def fit_on_sequences(self, nat_sequences,
                         prokaryote_sequences=None,
                         contamination=0.05,
                         random_state=42):
        """
        Fit viral envelope (required) and optionally prokaryotic envelope.

        Parameters
        ----------
        nat_sequences        : list[str]  Natural viral / eukaryotic sequences.
        prokaryote_sequences : list[str]  Natural prokaryotic sequences (optional).
                                          If None, only viral envelope is built.
        contamination        : float      IsolationForest contamination rate.
        random_state         : int        RNG seed.
        """
        # ── Viral envelope ─────────────────────────────────────────────
        self.svd, self.detector, self._cal_mean, self._cal_std = \
            self._fit_one_envelope(nat_sequences, contamination,
                                   random_state, label="viral")

        joblib.dump({
            "svd":      self.svd,
            "detector": self.detector,
            "cal_mean": self._cal_mean,
            "cal_std":  self._cal_std,
        }, self.envelope_path)
        print(f"[OOD] Viral bundle saved → {self.envelope_path}")

        # ── Prokaryotic envelope (optional) ────────────────────────────
        if prokaryote_sequences:
            self.svd_prok, self.detector_prok, \
            self._cal_mean_prok, self._cal_std_prok = \
                self._fit_one_envelope(prokaryote_sequences, contamination,
                                       random_state, label="prokaryotic")

            joblib.dump({
                "svd":      self.svd_prok,
                "detector": self.detector_prok,
                "cal_mean": self._cal_mean_prok,
                "cal_std":  self._cal_std_prok,
            }, self.prokaryote_envelope_path)
            print(f"[OOD] Prokaryotic bundle saved → {self.prokaryote_envelope_path}")
        else:
            print("[OOD] No prokaryotic sequences provided — "
                  "single-envelope mode only.")

    # ------------------------------------------------------------------ #
    # Scoring — public API                                                 #
    # ------------------------------------------------------------------ #
    def ghost_anomaly_score(self, sequence) -> dict:
        """
        Score a single sequence for OOD anomaly.

        Returns
        -------
        dict with keys:
            score        float   Final OOD score 0–100 (higher = more anomalous).
            viral_score  float   Score against viral envelope.
            prok_score   float|None  Score against prokaryotic envelope, or None.
            flag         str|None  'SHORT_FRAGMENT' | 'DUAL_ENVELOPE' | None.
            narrative    str     Human-readable explanation for reports.
        """
        if not self.ready:
            return {
                "score": 0.0, "viral_score": 0.0, "prok_score": None,
                "flag": "NOT_READY",
                "narrative": "OOD scorer not fitted. Run train.py first.",
            }

        short    = self._is_short(sequence)
        seq_len  = self._clean_len(sequence)

        try:
            # ── Viral score ─────────────────────────────────────────────
            X_svd_viral = self._embed(sequence, self.svd)
            raw_viral   = self._raw_score(X_svd_viral, self.detector)
            viral_score = self._calibrated_score(
                raw_viral, self._cal_mean, self._cal_std)

            # ── Prokaryotic score (if envelope fitted) ───────────────────
            prok_score = None
            if self.prokaryote_ready:
                X_svd_prok = self._embed(sequence, self.svd_prok)
                raw_prok   = self._raw_score(X_svd_prok, self.detector_prok)
                prok_score = self._calibrated_score(
                    raw_prok, self._cal_mean_prok, self._cal_std_prok)

            # ── Combine: MAX across envelopes ────────────────────────────
            # Logic: a sequence that is anomalous in ANY natural context
            # is a candidate ghost/synthetic. Taking the MAX ensures
            # prokaryotic plasmids cannot hide inside the prokaryotic envelope.
            if prok_score is not None:
                final_score = max(viral_score, prok_score)
                flag        = "DUAL_ENVELOPE"
                narrative   = (
                    f"Dual-envelope scoring — "
                    f"viral envelope: {viral_score:.1f}/100, "
                    f"prokaryotic envelope: {prok_score:.1f}/100. "
                    f"Final score (max): {final_score:.1f}/100."
                )
            else:
                final_score = viral_score
                flag        = None
                narrative   = (
                    f"Viral envelope only (no prokaryotic reference loaded). "
                    f"Score: {viral_score:.1f}/100."
                )

            # ── Short-fragment floor ─────────────────────────────────────
            if short:
                final_score = max(final_score, self.SHORT_SEQ_OOD_FLOOR)
                flag        = "SHORT_FRAGMENT"
                narrative  += (
                    f" ⚠️ Short sequence ({seq_len}bp < "
                    f"{self.SHORT_SEQ_THRESHOLD}bp): k-mer statistics are "
                    f"unreliable. OOD score floored at "
                    f"{self.SHORT_SEQ_OOD_FLOOR}. "
                    f"Use BLAST enrichment as primary signal."
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

    # ── Convenience wrappers (preserve existing call-sites) ─────────────
    def score_per_window(self, sequence, window=500, step=None):
        """Return list of (position, score) tuples across sliding windows."""
        if not self.ready:
            return []
        seq  = "".join(c for c in sequence.upper() if c in "ATGCN")
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
            "ready":            self.ready,
            "prokaryote_ready": self.prokaryote_ready,
            "calibrated":       self._calibrated,
            "cal_mean":         self._cal_mean,
            "cal_std":          self._cal_std,
            "cal_mean_prok":    self._cal_mean_prok,
            "cal_std_prok":     self._cal_std_prok,
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