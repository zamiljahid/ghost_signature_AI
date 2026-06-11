from __future__ import annotations
import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

from ghost_forensics import GhostForensics
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from fpdf import FPDF
from matplotlib.patches import Rectangle

from ghost_config import *
from motif_discovery import find_ghost_motifs
from ood_scorer import GhostOODScorer
# NOTE: fuse_engine_scores / fusion_weights are intentionally NOT imported anymore.
# The headline score is AI-classifier-driven with hard BLAST/motif overrides
# (cascade), not a convex blend — the OOD engine has zero voting power (see
# build_engine_evidence). OOD remains a passive diagnostic tag only.
from explainability import (
    kmer_shap_evidence, ood_shap_evidence, motif_evidence, build_forensic_report,
)
from Codon_optimisation_analyser import CodonOptimisationAnalyser
from narrative_engine import ForensicNarrativeEngine

PROJECT_DIR = Path(__file__).resolve().parent
QUERY_FASTA = PROJECT_DIR / DATA_DIR / "mystery_virus.fasta"
BLAST_DB = PROJECT_DIR / "database" / "univec_db"
PDF_PATH = PROJECT_DIR / OUTPUT_DIR / "Forensic_Analysis_Summary.pdf"
BLAST_CSV = PROJECT_DIR / OUTPUT_DIR / "blast_hits.csv"

LABEL_NAMES = {LABEL_NATURAL: "Natural", LABEL_VECTOR: "Vector", LABEL_GHOST: "Ghost"}

KNOWN_LAB_MOTIFS = {
    "T7 promoter": "TAATACGACTCACTATAGGG",
    "CMV promoter core": "TGACATTGATTATTGACTAG",
    "SV40 polyA signal": "AATAAAATATCTTTATTTTC",
    "Kanamycin resistance motif": "ATGAGCCATATTCAACGGGA",
    "Ampicillin resistance motif": "ATGAATTCACTGGCCGTCGT",
    "Lac operator (pUC/lacZ)": "AATTGTGAGCGGATAACAATT",
    "pUC19/M13 reverse primer site": "CAGGAAACAGCTATGAC",
    "pUC19/M13 forward primer site": "GTAAAACGACGGCCAGT",
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def safe_text(value) -> str:
    text = "" if value is None else str(value)
    return text.encode("latin-1", "replace").decode("latin-1")


def safe_id(value: str, max_len: int = 64) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return cleaned[:max_len] or "sequence"


def clean_sequence(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in "ATGCN")


def to_kmers(seq: str, k: int = None) -> str:
    from kmer_utils import seq_to_kmers_multiscale
    return seq_to_kmers_multiscale(seq)


def chunk_sequence(seq: str, chunk: int = CHUNK_SIZE) -> list[str]:
    chunks = []
    for i in range(0, max(1, len(seq)), chunk):
        frag = seq[i: i + chunk]
        if len(frag) >= 4: chunks.append(frag)
    return chunks


def fasta_stats(path: Path) -> dict:
    lengths = [len(str(r.seq)) for r in SeqIO.parse(path, "fasta")] if path.exists() else []
    if not lengths: return {"file": path.name, "records": 0, "total_bp": 0, "min_bp": 0, "mean_bp": 0, "max_bp": 0}
    return {"file": path.name, "records": len(lengths), "total_bp": int(sum(lengths)), "min_bp": int(min(lengths)),
            "mean_bp": float(np.mean(lengths)), "max_bp": int(max(lengths))}


def load_background_sequences(limit: int = 20) -> list[str]:
    path = project_path(NATURAL_FASTA)
    if not path.exists(): return []
    seqs = []
    for rec in SeqIO.parse(path, "fasta"):
        seq = clean_sequence(str(rec.seq))
        if len(seq) >= 4: seqs.append(seq[:50000])
        if len(seqs) >= limit: break
    return seqs


def _format_evalue(evalue) -> str:
    try:
        ev = float(evalue)
        return "0" if ev == 0 else f"{ev:.2e}"
    except:
        return str(evalue)


def _gc_interpretation(gc: float) -> str:
    if gc < 30:
        return f"The GC content of {gc:.1f}% is unusually low for most known viral genomes. This may suggest codon optimization for a low-GC host."
    elif gc > 65:
        return f"The GC content of {gc:.1f}% is notably high. Elevated GC can indicate codon optimization toward mammalian or bacterial expression systems."
    elif 40 <= gc <= 60:
        return f"The GC content of {gc:.1f}% falls within the typical range for natural viral genomes (40-60%). This alone does not indicate manipulation."
    else:
        return f"The GC content of {gc:.1f}% is slightly outside the canonical 40-60% window."


def _length_interpretation(length: int) -> str:
    if length < 2000:
        return f"At {length:,} bp this is a short sequence. Short fragments are common in engineered constructs. Interpretation of AI and OOD scores should account for limited context."
    elif length < 10000:
        return f"At {length:,} bp this sequence is within the range of small RNA viruses and many cloning vectors. Length is sufficient for meaningful k-mer analysis."
    else:
        return f"At {length:,} bp this is a full-length or near-complete genome. Longer sequences provide high confidence in k-mer frequency profiles and OOD scoring."


def _generate_codon_plot(rec_id: str, codon_result: dict, output_dir: Path) -> Path:
    rscu = codon_result["rscu"]
    if not rscu: return None
    codons = sorted(rscu.keys())
    values = [rscu[c] for c in codons]
    colors = ["#C81E1E" if v > 1.5 else "#1F77B4" if v < 0.5 else "#888888" for v in values]
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.bar(range(len(codons)), values, color=colors, width=0.8, edgecolor="none")
    ax.axhline(1.0, color="#333333", lw=1.2, ls="--", label="Neutral (RSCU=1.0)")
    ax.axhline(1.5, color="#C81E1E", lw=0.8, ls=":", alpha=0.6, label="Preferred threshold")
    ax.axhline(0.5, color="#1F77B4", lw=0.8, ls=":", alpha=0.6, label="Avoided threshold")
    ax.set_xticks(range(len(codons)));
    ax.set_xticklabels(codons, rotation=90, fontsize=6)
    ax.set_ylabel("RSCU");
    ax.set_xlabel("Codon")
    ax.set_title(
        f"Codon Optimisation (RSCU): {rec_id}  |  CAI={codon_result['cai_score']:.3f}  |  RSCU bias={codon_result['rscu_bias']:.3f}")
    ax.legend(fontsize=7, loc="upper right");
    plt.tight_layout()
    out = output_dir / f"codon_{safe_id(rec_id)}.png"
    plt.savefig(out, dpi=PLOT_DPI);
    plt.close(fig)
    return out


class GhostSignatureReport:
    def __init__(self, query_fasta: Path = QUERY_FASTA, blast_db: Path = BLAST_DB):
        self.query_fasta = query_fasta;
        self.blast_db = blast_db
        self.output_dir = project_path(OUTPUT_DIR);
        self.model_path = project_path(MODEL_PATH)
        self.vectorizer_path = project_path(VECTORIZER_PATH);
        self.ood_path = project_path(OOD_PATH)
        self.logs: list[str] = [];
        self.model = None;
        self.vectorizer = None
        self.model_classes: list[int] = [];
        self.model_note = "Model not loaded"
        self.background_sequences = load_background_sequences()
        self.ood = GhostOODScorer(envelope_path=str(self.ood_path))
        self.output_dir.mkdir(exist_ok=True);
        self.load_artifacts()

    def log(self, message: str) -> None:
        print(message);
        self.logs.append(message)

    def load_artifacts(self) -> None:
        try:
            self.model = joblib.load(self.model_path);
            self.vectorizer = joblib.load(self.vectorizer_path)
            classes = getattr(self.model, "classes_", []);
            self.model_classes = [int(c) for c in classes]
            if LABEL_GHOST in self.model_classes:
                self.model_note = "v2 3-class stacked ensemble model loaded"
            elif LABEL_VECTOR in self.model_classes:
                self.model_note = "legacy 2-class model loaded"
            else:
                self.model_note = "model loaded, but class labels are unexpected"
            self.log(f"[OK] AI engine online: {self.model_note}")
        except Exception as exc:
            self.log(f"[WARN] AI engine offline: {exc}")
        self.log("[OK] OOD envelope online" if self.ood.ready else "[WARN] OOD envelope missing.")

    def predict_probabilities(self, seq: str) -> dict[int, float]:
        if self.model is None or self.vectorizer is None: return {LABEL_NATURAL: 1.0, LABEL_VECTOR: 0.0,
                                                                  LABEL_GHOST: 0.0}
        probas = []
        for frag in chunk_sequence(seq):
            text = to_kmers(frag)
            if text: probas.append(self.model.predict_proba(self.vectorizer.transform([text]))[0])
        if not probas: return {LABEL_NATURAL: 1.0, LABEL_VECTOR: 0.0, LABEL_GHOST: 0.0}
        mean_proba = np.mean(probas, axis=0);
        result = {label: 0.0 for label in LABEL_NAMES}
        for cls, prob in zip(self.model_classes, mean_proba): result[int(cls)] = float(prob)
        return result

    def ai_risk_score(self, probas: dict[int, float], seq_len: int = 0) -> float:
        if LABEL_GHOST in self.model_classes:
            ai_risk = (probas.get(LABEL_GHOST, 0.0) + probas.get(LABEL_VECTOR, 0.0)) * 100.0
        else:
            ai_risk = probas.get(LABEL_VECTOR, 0.0) * 100.0
        if seq_len > 0 and seq_len < SHORT_SEQ_THRESHOLD: ai_risk = max(ai_risk, SHORT_SEQ_AI_FLOOR)
        return ai_risk

    def run_blast(self) -> pd.DataFrame:
        columns = ["query_id", "subject_id", "percent_identity", "alignment_length", "mismatches", "gap_opens",
                   "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score"]
        if not shutil.which("blastn"): self.log("[WARN] blastn not found."); return pd.DataFrame(columns=columns)
        cmd = ["blastn", "-query", str(self.query_fasta), "-db", str(self.blast_db), "-evalue", "1e-5", "-outfmt", "10",
               "-out", str(BLAST_CSV)]
        self.log("[BLAST] Searching query against UniVec database...")
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, check=False)
        if result.returncode != 0: return pd.DataFrame(columns=columns)
        if not BLAST_CSV.exists() or BLAST_CSV.stat().st_size == 0: return pd.DataFrame(columns=columns)
        df = pd.read_csv(BLAST_CSV, names=columns);
        self.log(f"[BLAST] {len(df)} homology hit(s) found.");
        return df

    def find_known_motifs(self, seq: str) -> list[str]:
        return [name for name, motif in KNOWN_LAB_MOTIFS.items() if motif in seq]

    def enriched_motifs(self, seq: str) -> list[tuple[str, float]]:
        if not self.background_sequences or len(seq) < MOTIF_K: return []
        try:
            return find_ghost_motifs(seq, self.background_sequences, k=MOTIF_K, top_n=MOTIF_TOP_N)
        except:
            return []

    def verdict(self, hits: pd.DataFrame, ai_risk: float, ood_score: float, known_motifs: list[str],
                seq_len: int = 0) -> tuple[str, str, tuple[int, int, int]]:
        if not hits.empty or known_motifs: return ("CONFIRMED LAB-ENGINEERED", "Direct evidence detected.",
                                                   (190, 25, 25))
        # Short sequences (<100 bp) contain too few k-mers for the AI gate to be reliable.
        # Route to BLAST/motif evidence only; return BORDERLINE as the floor when no direct evidence found.
        if seq_len > 0 and seq_len < SHORT_SEQ_GHOST_POLICY:
            return ("BORDERLINE / REVIEW",
                    f"Sequence is {seq_len} bp — too short for AI classification. "
                    "No direct BLAST or motif evidence found.",
                    (190, 155, 25))
        # AI-DRIVEN VERDICT. The OOD engine has ZERO voting power here: ablation showed
        # its anomaly score (independent-test AUROC 0.248) treats natural evolutionary
        # variance as synthetic, inverting the ranking and dragging the fused system to
        # 0.564. The verdict is now gated on the regularized AI classifier alone (plus
        # the hard BLAST/motif evidence override above). `ood_score` is accepted only so
        # the call site is unchanged; it is logged as passive diagnostic metadata, never
        # used in the decision. (The previous AI-AND-OOD dual gate is removed.)
        if ai_risk >= AI_SUSPECT_THRESHOLD:
            return ("SUSPECTED ENGINEERED", "High AI engineered-risk score.", (235, 125, 20))
        if ai_risk >= AI_BORDERLINE_THRESHOLD:
            return ("BORDERLINE / REVIEW", "Moderate AI engineered-risk score.", (190, 155, 25))
        return "LIKELY NATURAL", "No strong synthetic markers.", (25, 135, 85)

    def generate_plot(self, rec_id: str, seq: str, hits: pd.DataFrame, ai_risk: float, ood_score: float) -> tuple[
        Path, dict]:
        seq_len = len(seq);
        safe = safe_id(rec_id);
        img_path = self.output_dir / f"report_{safe}.png"
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.0]})
        ax_map, ax_gc, ax_ai, ax_entropy = axes
        ax_map.add_patch(Rectangle((0, 0), max(seq_len, 1), 1, color="#F2F2F2", ec="#333333"))
        covered_positions: set[int] = set()
        for _, hit in hits.iterrows():
            qs, qe = int(hit.get("q_start", 0)), int(hit.get("q_end", 0))
            start, width = min(qs, qe), abs(qe - qs) + 1
            ax_map.add_patch(Rectangle((start, 0.08), width, 0.84, color="#C81E1E", alpha=0.45));
            covered_positions.update(range(start, start + width))
        coverage_pct = len(covered_positions) / max(seq_len, 1) * 100
        top_hit = hits.loc[hits["percent_identity"].idxmax()] if not hits.empty else None
        ax_map.set_xlim(0, max(seq_len, 1));
        ax_map.set_ylim(0, 1);
        ax_map.set_yticks([]);
        ax_map.set_title(f"Genome Evidence Map: {rec_id}");
        ax_map.set_ylabel("BLAST")
        window = min(100, max(20, seq_len // 20)) if seq_len else 20
        positions = list(range(0, max(1, seq_len - window + 1), max(1, window // 2))) or [0]
        gc_values = [gc_fraction(seq[i: i + window]) * 100 for i in positions];
        x_gc = [i + window / 2 for i in positions]
        ax_gc.plot(x_gc, gc_values, color="#1F77B4", lw=1.8);
        ax_gc.fill_between(x_gc, gc_values, alpha=0.12, color="#1F77B4");
        ax_gc.set_xlim(0, max(seq_len, 1));
        ax_gc.set_ylabel("GC %")
        ai_positions, ai_values = [], [];
        step = max(50, seq_len // 40) if seq_len else 50;
        span = max(CHUNK_SIZE, step)
        for i in range(0, max(1, seq_len - 4 + 1), step):
            frag = seq[i: i + span]
            if len(frag) >= 4: ai_positions.append(i + len(frag) / 2); ai_values.append(
                self.ai_risk_score(self.predict_probabilities(frag)))
        if not ai_values: ai_positions, ai_values = [seq_len / 2], [ai_risk]
        ax_ai.plot(ai_positions, ai_values, color="#E53935", lw=2.2)
        ax_ai.axhline(40, color="#777777", lw=0.9, ls="--");
        ax_ai.set_xlim(0, max(seq_len, 1));
        ax_ai.set_ylim(0, 100);
        ax_ai.set_ylabel("AI risk")
        entropy_values, entropy_positions = [], [];
        entropy_window = min(80, max(32, seq_len // 15)) if seq_len else 32;
        entropy_step = max(10, entropy_window // 2)
        for i in range(0, max(1, seq_len - entropy_window + 1), entropy_step):
            frag = seq[i: i + entropy_window]
            if frag:
                counts = np.array([frag.count(b) for b in "ATGCN"], dtype=float);
                probs = counts[counts > 0] / counts.sum()
                entropy_values.append(float(-np.sum(probs * np.log2(probs))));
                entropy_positions.append(i + len(frag) / 2)
        if not entropy_values: entropy_positions, entropy_values = [seq_len / 2], [0]
        ax_entropy.plot(entropy_positions, entropy_values, color="#222222", lw=1.5);
        ax_entropy.fill_between(entropy_positions, entropy_values, alpha=0.14, color="#222222")
        ax_entropy.set_xlim(0, max(seq_len, 1));
        ax_entropy.set_ylabel("Entropy");
        ax_entropy.set_xlabel("Base-pair position")
        fig.suptitle(f"AI risk {ai_risk:.1f}% | OOD anomaly {ood_score:.1f}", y=0.995, fontsize=12);
        plt.tight_layout();
        plt.savefig(img_path, dpi=PLOT_DPI);
        plt.close(fig)
        return img_path, {}

    def build_engine_evidence(self, rec_id: str, seq: str, hits: pd.DataFrame,
                              ai_risk: float, ood_details: dict,
                              known_motifs: list[str], codon_result: dict) -> tuple[dict, str]:
        """Issue 6: assemble bounded per-engine probabilities + human-readable
        evidence lines, then fuse them into a single [0,1] suspicion score.

        Returns (engines_dict, forensic_text)."""
        # ── k-mer TF-IDF evidence ───────────────────────────────────────────
        kmer_prob = float(np.clip(ai_risk / 100.0, 0.0, 1.0))
        kmer_evidence = "n/a"
        if self.model is not None and self.vectorizer is not None:
            try:
                X_sample = self.vectorizer.transform([to_kmers(seq)])
                feats = self.vectorizer.get_feature_names_out()
                kmer_evidence = kmer_shap_evidence(self.model, X_sample, X_sample, feats)
            except Exception as e:
                kmer_evidence = f"unavailable ({e})"

        # ── OOD evidence ────────────────────────────────────────────────────
        ood_prob = float(np.clip(ood_details.get("score", 0.0) / 100.0, 0.0, 1.0))
        ood_evidence = "n/a"
        if getattr(self.ood, "ready", False):
            try:
                from ood_scorer import _KMER4
                x4 = self.ood._kmer_freq_vector(
                    "".join(c for c in seq.upper() if c in "ATGC"))
                ood_evidence = ood_shap_evidence(self.ood.detector,
                                                 x4.reshape(1, -1), _KMER4)
            except Exception as e:
                ood_evidence = f"unavailable ({e})"

        # ── BLAST / Motif / Codon (bounded) ─────────────────────────────────
        blast_score = 1.0 if (hits is not None and not hits.empty) else 0.0
        if blast_score and "percent_identity" in hits:
            top = hits.loc[hits["percent_identity"].idxmax()]
            blast_top_hit = f"{top.get('subject_id', '?')} ({top['percent_identity']:.1f}% id)"
        else:
            blast_top_hit = "no UniVec hit"

        motif_score = float(min(1.0, len(known_motifs) / 3.0))
        motif_ev = motif_evidence(known_motifs)

        codon_score = 1.0 if codon_result.get("optimisation_flag") else 0.0
        cai_delta = float(codon_result.get("rscu_bias", 0.0))

        # ── Headline suspicion score: AI classifier + hard DB/motif overrides ──
        # The OOD engine is STRIPPED of all voting power. The previous convex
        # fusion (w_kmer*ai + w_ood*ood + ...) inverted rankings because the
        # unsupervised OOD score (independent-test AUROC 0.248) flags natural
        # evolutionary variance as synthetic — fusing it dragged the system from
        # AI-alone 0.903 down to 0.564 (see outputs/ablation/ablation_table.csv).
        #
        # This is now a cascade, not a weighted blend: direct database/motif
        # evidence is a deterministic override (→1.0); otherwise the regularized
        # AI classifier probability IS the score. OOD and the soft codon signal
        # contribute NOTHING to S_fused — they are emitted below as passive
        # diagnostic metadata only.
        if blast_score >= 1.0 or motif_score > 0.0:
            s_fused = 1.0                      # hard BLAST/motif override
        else:
            s_fused = float(np.clip(kmer_prob, 0.0, 1.0))   # AI classifier drives the score

        engines = {
            "kmer_prob": kmer_prob, "kmer_evidence": kmer_evidence,
            # ── Passive diagnostic tags (NOT used in any score/verdict) ──
            "ood_prob": ood_prob, "ood_evidence": ood_evidence,
            "raw_ood_score": float(ood_details.get("score", 0.0)),
            "is_novel_sequence_warning": bool(ood_details.get("score", 0.0) >= OOD_SUSPECT_THRESHOLD),
            "blast_score": blast_score, "blast_top_hit": blast_top_hit,
            "motif_score": motif_score, "motif_evidence": motif_ev,
            "codon_score": codon_score, "cai_delta": cai_delta,
            "S_fused": s_fused,
        }
        return engines, build_forensic_report(rec_id, engines)

    def analyze(self) -> tuple[list[dict], list[dict]]:
        self.log("=" * 72);
        self.log("Ghost Signature Detector - Final Forensic PDF Generator");
        self.log("=" * 72)
        if not self.query_fasta.exists(): raise FileNotFoundError(f"Missing query FASTA: {self.query_fasta}")
        blast_df = self.run_blast();
        dataset_stats = [fasta_stats(p) for p in sorted((PROJECT_DIR / DATA_DIR).glob("*.fasta"))];
        results = []
        for rec in SeqIO.parse(self.query_fasta, "fasta"):
            seq = clean_sequence(str(rec.seq));
            hits = blast_df[blast_df["query_id"] == rec.id] if not blast_df.empty else pd.DataFrame()
            probas = self.predict_probabilities(seq);
            ai_risk = self.ai_risk_score(probas, seq_len=len(seq))
            if self.ood.ready:
                ood_raw = self.ood.ghost_anomaly_score(seq)
                if isinstance(ood_raw, dict):
                    ood_score = float(ood_raw["score"]);
                    ood_details = ood_raw
                else:
                    ood_score = float(ood_raw);
                    ood_details = {"score": ood_score, "viral_score": ood_score,
                                   "prok_score": None, "flag": None}
            else:
                ood_score = 0.0;
                ood_details = {"score": 0.0, "viral_score": 0.0, "prok_score": None,
                               "flag": "NOT_READY"}
            known_motifs = self.find_known_motifs(seq);
            enriched = self.enriched_motifs(seq)
            verdict, verdict_reason, color = self.verdict(hits, ai_risk, ood_score, known_motifs, seq_len=len(seq))
            img, plot_data = self.generate_plot(rec.id, seq, hits, ai_risk, ood_score)
            self.log(f"[COD] Running codon optimisation analysis for {rec.id}...")
            codon_analyser = CodonOptimisationAnalyser(seq);
            codon_result = codon_analyser.analyse();
            codon_img = _generate_codon_plot(rec.id, codon_result, self.output_dir)
            # Issue 6: build per-engine human-readable evidence + bounded fused score.
            engine_evidence, forensic_text = self.build_engine_evidence(
                rec.id, seq, hits, ai_risk, ood_details, known_motifs, codon_result)
            self.log("\n" + forensic_text + "\n")
            self.log(
                f"[SEQ] {rec.id}: {len(seq)} bp | AI {ai_risk:.1f}% | OOD {ood_score:.1f} | hits {len(hits)} | CAI {codon_result['cai_score']:.3f} | {verdict}")
            results.append({"id": rec.id, "description": rec.description, "seq": seq, "length": len(seq),
                            "gc": gc_fraction(seq) * 100 if seq else 0.0, "hits": hits, "probas": probas,
                            "ai_risk": ai_risk,
                            # ── OOD: passive diagnostic metadata only (no voting power) ──
                            "ood_score": ood_score, "ood_details": ood_details,
                            "raw_ood_score": ood_score,
                            "is_novel_sequence_warning": bool(ood_score >= OOD_SUSPECT_THRESHOLD),
                            "known_motifs": known_motifs, "enriched_motifs": enriched, "verdict": verdict,
                            "verdict_reason": verdict_reason, "color": color, "image": img,
                            "codon_result": codon_result, "codon_image": codon_img,
                            "engine_evidence": engine_evidence, "forensic_text": forensic_text})

        # Issue 6: persist the engine-by-engine evidence so analysts can audit the
        # reasoning without re-running the pipeline.
        ev_path = self.output_dir / "forensic_evidence.txt"
        with open(ev_path, "w") as fh:
            for r in results:
                fh.write(r["forensic_text"] + "\n\n")
        self.log(f"[EVIDENCE] Per-engine forensic evidence written → {ev_path}")
        return results, dataset_stats

    def write_pdf(self, results: list[dict], dataset_stats: list[dict]) -> None:
        pdf = FPDF();
        pdf.set_auto_page_break(auto=True, margin=14);
        self._cover_page(pdf, results)
        for result in results:
            self._sequence_page(pdf, result)
            if not result["hits"].empty:
                self._blast_evidence_page(pdf, result)
        pdf.output(str(PDF_PATH));
        self.log(f"[DONE] PDF generated: {PDF_PATH}")

    def _brand(self, pdf: FPDF) -> None:
        pdf.set_font("Arial", "I", 7);
        pdf.set_text_color(125, 125, 125)
        pdf.cell(0, 5, safe_text(f"Developed by {OWNER_NAME} | {OWNER_EMAIL}"), ln=True, align="R");
        pdf.set_text_color(0, 0, 0)

    def _section_heading(self, pdf: FPDF, title: str) -> None:
        pdf.set_font("Arial", "B", 11);
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, safe_text(title), ln=True, fill=True);
        pdf.set_font("Arial", "", 9);
        pdf.ln(1)


    def _render_blast_section(self, pdf: FPDF, result: dict) -> None:
        s = ForensicNarrativeEngine.from_result_dict(result)
        hits = result.get("hits", None)
        pdf.set_font("Arial", "", 9)

        if s.blast_hit_count == 0:
            pdf.multi_cell(0, 5.5, safe_text(
                "No alignments were found against the UniVec database. The absence of a hit does not "
                "exclude engineering, but removes the strongest class of direct evidence."))
            pdf.ln(2)
            return

        pdf.multi_cell(0, 5.5, safe_text(
            f"{s.blast_hit_count} alignment(s) were detected against the UniVec database."))
        pdf.ln(1)

        headers = ["Metric", "Value"]
        data = [
            ["Total hits", str(s.blast_hit_count)],
            ["Top identity", f"{s.blast_top_identity:.1f}%"],
            ["Coverage", f"{s.blast_coverage_pct:.1f}% of query"],
        ]
        self._add_table(pdf, headers, data, col_widths=[95, 95])
        pdf.ln(2)

        pdf.set_font("Arial", "", 9)
        if s.blast_top_identity >= 95:
            pdf.multi_cell(0, 5.5, safe_text(
                f"The top hit shows {s.blast_top_identity:.1f}% identity, indicating near-exact match "
                f"to a known vector sequence. This is strong direct evidence of laboratory origin."))
        elif s.blast_top_identity >= 85:
            pdf.multi_cell(0, 5.5, safe_text(
                f"The top hit shows {s.blast_top_identity:.1f}% identity, indicating significant homology to a known vector."))

        if s.known_motifs:
            pdf.multi_cell(0, 5.5, safe_text(
                f"\nAdditionally, {len(s.known_motifs)} known laboratory motif(s) were detected: "
                f"{', '.join(s.known_motifs)}."))
        pdf.ln(2)

    def _add_table(self, pdf: FPDF, headers: list[str], data: list[list[str]], col_widths: list[int] = None) -> None:
        if not col_widths:
            total_w = 190
            col_widths = [int(total_w / len(headers))] * len(headers)

        row_h = 6
        header_h = 7

        # --- PAGE-BREAK GUARD: check if entire table fits, if not start a new page ---
        needed = header_h + len(data) * row_h + 4
        usable = pdf.h - pdf.b_margin - pdf.get_y()
        if usable < needed:
            pdf.add_page()
            self._brand(pdf)

        # --- HEADER DRAW HELPER (reused when table continues on new page) ---
        def _draw_header():
            pdf.set_draw_color(0, 0, 0)
            pdf.set_text_color(255, 255, 255)
            pdf.set_fill_color(70, 90, 130)
            pdf.set_font('Arial', 'B', 8)
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], header_h, safe_text(h), border=1, fill=True, align='C')
            pdf.ln()

        _draw_header()

        # --- DATA ROWS ---
        pdf.set_font('Arial', '', 8)

        for row_idx, row in enumerate(data):
            # Per-row page-break guard: if this row won't fit, start new page and re-draw header
            if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
                pdf.add_page()
                self._brand(pdf)
                _draw_header()
                pdf.set_font('Arial', '', 8)

            x_start = pdf.get_x()
            y_start = pdf.get_y()
            row_bg = (255, 255, 255) if row_idx % 2 == 0 else (242, 242, 242)

            # Draw filled background rect for the full row width
            pdf.set_fill_color(*row_bg)
            pdf.rect(x_start, y_start, sum(col_widths), row_h, style='F')

            # Draw cells with fill=False so blue never bleeds in
            pdf.set_text_color(0, 0, 0)
            pdf.set_draw_color(0, 0, 0)
            x = x_start
            for i, cell in enumerate(row):
                pdf.set_xy(x, y_start)
                pdf.cell(col_widths[i], row_h, safe_text(cell), border=1, fill=False, align='C')
                x += col_widths[i]
            pdf.set_xy(x_start, y_start + row_h)

        # Hard reset after table
        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(0, 0, 0)

    def _render_ood_section(self, pdf: FPDF, result: dict) -> None:
        s = ForensicNarrativeEngine.from_result_dict(result)

        pdf.set_font("Arial", "B", 9)
        pdf.multi_cell(0, 5.5, safe_text(f"Overall OOD Score: {s.ood_score:.1f}/100"))
        pdf.ln(1)

        if s.ood_viral is not None or s.ood_prokaryote is not None:
            pdf.set_font("Arial", "B", 9)
            pdf.multi_cell(0, 5.5, "Dual-Envelope Architecture Breakdown:")
            pdf.ln(1)

            headers = ["Envelope", "Training Data", "Score", "Interpretation"]
            data = []
            if s.ood_viral is not None:
                lvl = "Anomalous" if s.ood_viral >= 75 else "Elevated" if s.ood_viral >= 50 else "Borderline" if s.ood_viral >= 35 else "Normal"
                data.append(["Viral", "Natural viral genomes", f"{s.ood_viral:.1f}", lvl])
            if s.ood_prokaryote is not None:
                lvl = "Anomalous" if s.ood_prokaryote >= 75 else "Elevated" if s.ood_prokaryote >= 50 else "Borderline" if s.ood_prokaryote >= 35 else "Normal"
                data.append(["Prokaryotic", "Bacterial genomes (E. coli, etc.)", f"{s.ood_prokaryote:.1f}", lvl])
            data.append(["Final (Viral)", "-", f"{s.ood_score:.1f}", "-"])

            # --- PAGE-BREAK GUARD: keep label + table + rationale together if they fit ---
            # Estimate: header(7) + up to 3 rows(18) + rationale text(~18) + margins(8) = ~51 pt
            estimated_block = 7 + len(data) * 6 + 26
            if pdf.h - pdf.b_margin - pdf.get_y() < estimated_block:
                pdf.add_page()
                self._brand(pdf)
                # Re-print the section label so context isn't lost after page break
                pdf.set_font("Arial", "B", 9)
                pdf.multi_cell(0, 5.5, "Dual-Envelope Architecture Breakdown (continued):")
                pdf.ln(1)

            self._add_table(pdf, headers, data, col_widths=[35, 75, 25, 55])
            pdf.ln(2)

            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5.5, safe_text(
                "Design Rationale: The final OOD score is derived from the viral-envelope IsolationForest, "
                "trained on natural viral genomes. A prokaryotic reference envelope is computed in parallel "
                "and shown above for interpretability, but the operational verdict gates use the viral score. "
                "This approach ensures that sequences with atypical viral structural signatures are flagged "
                "while remaining robust against prokaryotic contamination artefacts."))
            pdf.ln(1)
        else:
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5.5, "Single-Envelope Mode: Only the viral envelope was available.")
            pdf.ln(1)

        if s.ood_flag == "SHORT_FRAGMENT":
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5.5, safe_text(
                f"[!] SHORT FRAGMENT FLAG: Sequence is {s.sequence_length} bp, below the 300 bp reliability threshold."))
            pdf.ln(1)
        elif s.ood_flag == "NOT_READY":
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5.5, "[!] OOD SCORER UNAVAILABLE.")
            pdf.ln(1)

        ood_text = {
            "CRITICAL": "The OOD score indicates the sequence is far outside the natural distribution envelope.",
            "HIGH": "The OOD score is elevated, placing this sequence where natural isolates are sparse.",
            "MODERATE": "The OOD score is mildly elevated, near the boundary of the natural distribution.",
            "LOW": "The OOD score is low. For known vectors or short fragments, this is expected.",
            "NEGATIVE": "The OOD score is very low, well within the natural distribution envelope.",
        }
        score = s.ood_score
        level = "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MODERATE" if score >= 35 else "LOW" if score >= 20 else "NEGATIVE"
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5.5, safe_text(ood_text[level]))
        pdf.ln(2)

    def _body(self, pdf: FPDF, text: str) -> None:
        text = text.replace("—", " - ").replace("–", "-").replace("•", "*").replace("⚠️", "[!]").replace("✅",
                                                                                                         "[+]").replace(
            "❌", "[-]")
        lines = text.split('\n')
        blocks = []  # Holds both text and tables in EXACT original order
        current_table = {"headers": [], "data": []}
        in_table = False

        for line in lines:
            stripped = line.strip()
            is_table_row = stripped.startswith('|') and '---' not in stripped
            is_table_separator = stripped.startswith('|') and '---' in stripped

            if is_table_row:
                in_table = True
                cells = [c.strip().replace("**", "") for c in line.split('|') if c.strip()]
                if not current_table["headers"]:
                    current_table["headers"] = cells  # First row is always header
                else:
                    current_table["data"].append(cells)  # Subsequent rows are data
            elif is_table_separator:
                # Markdown separator row (e.g. |---|---|---|) - skip it but stay in table mode
                in_table = True
            else:
                if in_table:
                    # End of table
                    if current_table["headers"] or current_table["data"]:
                        blocks.append(
                            {"type": "table", "headers": current_table["headers"], "data": current_table["data"]})
                    current_table = {"headers": [], "data": []}
                    in_table = False

                # Only add text block if it's not purely whitespace/empty
                if stripped:
                    blocks.append({"type": "text", "content": line})

        # Catch any table that goes right to the end of the text
        if in_table and (current_table["headers"] or current_table["data"]):
            blocks.append({"type": "table", "headers": current_table["headers"], "data": current_table["data"]})

        # Render blocks strictly in order they appeared
        for block in blocks:
            if block["type"] == "table":
                self._add_table(pdf, block["headers"], block["data"])
                pdf.ln(2)
            else:
                stripped = block["content"].strip()
                if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                    pdf.set_font("Arial", "B", 9)
                    pdf.multi_cell(0, 5.5, safe_text(stripped.replace("**", "")))
                    pdf.set_font("Arial", "", 9)
                else:
                    pdf.multi_cell(0, 5.5, safe_text(block["content"].replace("**", "")))
        pdf.ln(2)

    def _cover_page(self, pdf: FPDF, results: list[dict]) -> None:
        pdf.add_page();
        self._brand(pdf);
        pdf.ln(6);
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 12, "GHOST SIGNATURE FORENSIC GENOMIC REPORT", ln=True, align="C");
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, safe_text(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True, align="C")
        pdf.cell(0, 6, safe_text(f"Query file: {self.query_fasta.name}"), ln=True, align="C");
        pdf.ln(6)

    def _sequence_page(self, pdf: FPDF, result: dict) -> None:
        pdf.set_font("Arial", "B", 16);
        pdf.cell(0, 9, safe_text(f"Forensic Report: {result['id']}"), ln=True)
        if result["description"] and result["description"] != result["id"]:
            pdf.set_font("Arial", "I", 9);
            pdf.cell(0, 6, safe_text(result["description"]), ln=True)
        pdf.ln(2);
        pdf.set_fill_color(*result["color"]);
        pdf.set_text_color(255, 255, 255);
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 11, safe_text(f"VERDICT: {result['verdict']}"), ln=True, align="C", fill=True);
        pdf.set_text_color(0, 0, 0);
        pdf.ln(4)

        snapshot = ForensicNarrativeEngine.from_result_dict(result);
        engine = ForensicNarrativeEngine();
        narratives = engine.generate(snapshot)

        self._section_heading(pdf, "Forensic Assessment");
        self._body(pdf, narratives["executive_summary"])

        self._section_heading(pdf, "Multi-Signal Evidence Analysis");
        self._body(pdf, narratives["signal_interplay"])

        self._section_heading(pdf, "Sequence Properties");
        self._body(pdf, _length_interpretation(result["length"]))
        self._body(pdf, _gc_interpretation(result["gc"]))

        self._section_heading(pdf, "AI Classifier Analysis");
        self._body(pdf, narratives["ai_analysis"])

        self._section_heading(pdf, "Out-of-Distribution (OOD) Anomaly Analysis")
        self._render_ood_section(pdf, result)

        self._section_heading(pdf, "Homology Screening (UniVec BLAST)")
        self._render_blast_section(pdf, result)

        self._section_heading(pdf, "Regulatory Motif Analysis")
        if result["known_motifs"]:
            self._body(pdf, "The following laboratory regulatory elements were detected:\n" + "\n".join(
                f"  - {m}" for m in result["known_motifs"]))
        else:
            self._body(pdf,
                       "No exact matches to the curated library of known laboratory regulatory elements were detected.")

        if result["enriched_motifs"]:
            self._section_heading(pdf, "Statistically Enriched k-mer Motifs")
            pdf.set_font("Courier", "", 8);
            pdf.multi_cell(0, 5, safe_text("  ".join(f"{m}: {f:.1f}x" for m, f in result["enriched_motifs"][:10])));
            pdf.ln(1)
            self._body(pdf, "Fold-enrichment above 10x is unusual in natural sequences.")

        pdf.ln(2);
        pdf.image(str(result["image"]), x=10, w=190)

        # --- New Page for Codon, Saliency, Signal Gap, and Conclusion ---
        pdf.add_page();
        self._brand(pdf);

        self._section_heading(pdf, "Codon Optimisation Analysis")
        cr = result["codon_result"];
        pdf.set_font("Courier", "B", 9)
        pdf.cell(0, 6, safe_text(
            f"CAI: {cr['cai_score']:.4f}  |  RSCU Bias: {cr['rscu_bias']:.4f}  |  Codons: {cr['total_codons']}  |  Flag: {cr['optimisation_flag']}"),
                 ln=True);
        pdf.ln(1)
        self._body(pdf, narratives["codon_analysis"])

        if result["codon_image"] and result["codon_image"].exists():
            pdf.ln(2);
            pdf.image(str(result["codon_image"]), x=10, w=190)

        # --- K-mer Saliency Mapping (Moved here) ---
        pdf.ln(4)
        self._section_heading(pdf, "K-mer Saliency Mapping")
        self._body(pdf,
                   "The saliency track maps each k-mer to its contribution weight for the synthetic classification. Peaks indicate regions where k-mer composition most strongly diverges from natural viral genomes, enabling pinpointing of synthetic signatures at single-position resolution.")
        saliency_path = self.output_dir / "forensics" / f"{safe_id(result['id'])}_saliency.png"
        if saliency_path.exists():
            pdf.image(str(saliency_path), x=10, w=190)
        else:
            self._body(pdf, "[Saliency plot not found]")

        # --- Signal Gap Analysis (Moved here) ---
        pdf.ln(4)
        self._section_heading(pdf, "Signal Gap Analysis")
        self._body(pdf,
                   "The Signal Gap analysis maps sequences based on two independent metrics: homology to known vectors (BLAST) and structural deviation from natural genomes (OOD). This visualizes the detection space, highlighting sequences that are simultaneously dissimilar to known vectors and anomalous relative to natural genomes.")
        signal_gap_path = self.output_dir / "forensics" / "novelty_signal_gap.png"
        if signal_gap_path.exists():
            pdf.image(str(signal_gap_path), x=10, w=190)
        else:
            self._body(pdf, "[Signal Gap plot requires analyzing multiple sequences simultaneously]")

        # --- Conclusion (Strictly at the end) ---
        pdf.ln(4)
        self._section_heading(pdf, "Conclusion")
        self._body(pdf, narratives["conclusion"])

    def _blast_evidence_page(self, pdf: FPDF, result: dict) -> None:
        pdf.add_page();
        self._brand(pdf);
        pdf.set_font("Arial", "B", 14);
        pdf.cell(0, 9, safe_text(f"BLAST Evidence Log: {result['id']}"), ln=True);
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5.5,
                       "Each entry records an alignment between the query and a UniVec entry. Lower e-values and higher bit scores indicate stronger alignments.");
        pdf.ln(3);
        pdf.set_font("Courier", "", 8)
        for i, (_, hit) in enumerate(result["hits"].head(35).iterrows(), 1):
            qs, qe = int(hit["q_start"]), int(hit["q_end"]);
            start, end = min(qs, qe), max(qs, qe);
            snippet = result["seq"][start - 1: end][:70]
            line = (
                f"[{i}] Subject: {hit['subject_id']}\n    Identity: {hit['percent_identity']}%  |  Length: {hit['alignment_length']} bp  |  Range: {start}-{end} bp\n    E-value: {_format_evalue(hit['evalue'])}  |  Bit score: {hit['bit_score']}\n    Snippet: {snippet}...\n")
            pdf.multi_cell(0, 4.8, safe_text(line), border="B")


def main() -> None:
    os.chdir(PROJECT_DIR)
    report = GhostSignatureReport();
    forensics = GhostForensics(output_dir="outputs/forensics")
    results, dataset_stats = report.analyze()
    all_blast_identities, all_ood_scores, all_labels = [], [], []
    print("[*] Generating Forensic Evidence...")
    for res in results:
        blast_val = float(res["hits"]["percent_identity"].max()) if not res["hits"].empty else 0.0
        ood_val = res.get("ood_score", 0.0)
        all_blast_identities.append(blast_val);
        all_ood_scores.append(ood_val)
        all_labels.append(1 if "ghost" in res["id"].lower() or ood_val > OOD_THRESHOLD else 0)
        saliency_scores = forensics.map_kmer_saliency(res["seq"], report.vectorizer, report.model)
        forensics.plot_saliency_track(res["id"], saliency_scores)
    forensics.plot_signal_gap(all_blast_identities, all_ood_scores, all_labels)
    if not results: raise SystemExit("[ERROR] No query records found.")
    report.write_pdf(results, dataset_stats)
    print(f"[DONE] Analysis complete. Plots saved to outputs/forensics/")
if __name__ == "__main__": main()