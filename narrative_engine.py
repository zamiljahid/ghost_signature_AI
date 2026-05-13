from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SignalStrength(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NEGATIVE = "negative"
    UNAVAILABLE = "unavailable"


class AIClass(Enum):
    NATURAL = "natural"
    VECTOR = "vector"
    GHOST = "ghost"
    UNCERTAIN = "uncertain"


class VerdictType(Enum):
    CONFIRMED_ENGINEERED = "confirmed_engineered"
    SUSPECTED_ENGINEERED = "suspected_engineered"
    BORDERLINE = "borderline"
    LIKELY_NATURAL = "likely_natural"


@dataclass
class SignalSnapshot:
    ai_risk: float
    ai_class: AIClass
    prob_natural: float
    prob_vector: float
    prob_ghost: float
    ood_score: float
    ood_viral: Optional[float] = None
    ood_prokaryote: Optional[float] = None
    ood_flag: Optional[str] = None
    blast_hit_count: int = 0
    blast_top_identity: float = 0.0
    blast_coverage_pct: float = 0.0
    known_motifs: list[str] = field(default_factory=list)
    enriched_motifs: list[tuple[str, float]] = field(default_factory=list)
    cai_score: float = 0.0
    rscu_bias: float = 0.0
    codon_flagged: bool = False
    gc_content: float = 0.0
    sequence_length: int = 0
    verdict: VerdictType = VerdictType.LIKELY_NATURAL


class ForensicNarrativeEngine:
    def __init__(self):
        self.snapshot: Optional[SignalSnapshot] = None
        self._signal_strengths: dict[str, SignalStrength] = {}
        self._conflicts: list[str] = []
        self._supporting: list[str] = []

    def generate(self, snapshot: SignalSnapshot) -> dict[str, str]:
        self.snapshot = snapshot
        self._signal_strengths = self._classify_all_signals()
        self._conflicts = self._detect_conflicts()
        self._supporting = self._detect_supporting()

        return {
            "executive_summary": self._build_executive_summary(),
            "signal_interplay": self._build_signal_interplay(),
            "ai_analysis": self._build_ai_analysis(),
            "ood_analysis": self._build_ood_analysis(),
            "homology_analysis": self._build_homology_analysis(),
            "codon_analysis": self._build_codon_analysis(),
            "conclusion": self._build_conclusion(),
        }

    def _classify_all_signals(self) -> dict[str, SignalStrength]:
        s = self.snapshot
        return {
            "ai_risk": self._classify_ai_risk(s.ai_risk),
            "ood": self._classify_ood(s.ood_score),
            "blast": self._classify_blast(s.blast_hit_count, s.blast_top_identity),
            "motifs": self._classify_motifs(s.known_motifs, s.enriched_motifs),
            "codon": self._classify_codon(s.cai_score, s.rscu_bias),
            "gc": self._classify_gc(s.gc_content),
        }

    def _classify_ai_risk(self, score: float) -> SignalStrength:
        if score >= 80: return SignalStrength.CRITICAL
        if score >= 60: return SignalStrength.HIGH
        if score >= 35: return SignalStrength.MODERATE
        if score >= 15: return SignalStrength.LOW
        return SignalStrength.NEGATIVE

    def _classify_ood(self, score: float) -> SignalStrength:
        if score >= 75: return SignalStrength.CRITICAL
        if score >= 50: return SignalStrength.HIGH
        if score >= 35: return SignalStrength.MODERATE
        if score >= 20: return SignalStrength.LOW
        return SignalStrength.NEGATIVE

    def _classify_blast(self, hit_count: int, top_identity: float) -> SignalStrength:
        if hit_count == 0: return SignalStrength.NEGATIVE
        if hit_count >= 10 and top_identity >= 95: return SignalStrength.CRITICAL
        if hit_count >= 5 and top_identity >= 90: return SignalStrength.HIGH
        if hit_count >= 2 and top_identity >= 85: return SignalStrength.MODERATE
        if hit_count >= 1: return SignalStrength.LOW
        return SignalStrength.NEGATIVE

    def _classify_motifs(self, known: list, enriched: list) -> SignalStrength:
        if known: return SignalStrength.CRITICAL
        if enriched and enriched[0][1] >= 20: return SignalStrength.HIGH
        if enriched and enriched[0][1] >= 10: return SignalStrength.MODERATE
        if enriched: return SignalStrength.LOW
        return SignalStrength.NEGATIVE

    def _classify_codon(self, cai: float, rscu_bias: float) -> SignalStrength:
        if cai > 0.75 or rscu_bias > 0.5: return SignalStrength.HIGH
        if cai > 0.65 or rscu_bias > 0.45: return SignalStrength.MODERATE
        if cai > 0.55 or rscu_bias > 0.3: return SignalStrength.LOW
        return SignalStrength.NEGATIVE

    def _classify_gc(self, gc: float) -> SignalStrength:
        if gc < 30 or gc > 65: return SignalStrength.MODERATE
        if gc < 35 or gc > 60: return SignalStrength.LOW
        return SignalStrength.NEGATIVE

    def _detect_conflicts(self) -> list[str]:
        conflicts = []
        s = self.snapshot
        strengths = self._signal_strengths

        if strengths["blast"] in (SignalStrength.CRITICAL, SignalStrength.HIGH):
            if strengths["ai_risk"] == SignalStrength.NEGATIVE:
                conflicts.append(("blast_ai_conflict",
                                  "High BLAST homology to known vectors was detected, but the AI classifier assigned low synthetic probability. This typically occurs when the model's training set included these vectors. BLAST homology is considered the ground truth in this scenario."))

        if strengths["ai_risk"] in (SignalStrength.CRITICAL, SignalStrength.HIGH):
            if strengths["ood"] in (SignalStrength.NEGATIVE, SignalStrength.LOW):
                conflicts.append(("ai_ood_conflict",
                                  "The AI classifier flags this sequence as synthetic, but the OOD anomaly detector places it within the natural distribution. This divergence may indicate: (1) the sequence is a known vector type well-represented in OOD training data, or (2) the k-mer composition triggers AI patterns without global structural anomaly."))

        if strengths["ood"] in (SignalStrength.CRITICAL, SignalStrength.HIGH):
            if strengths["ai_risk"] in (SignalStrength.NEGATIVE, SignalStrength.LOW):
                conflicts.append(("ood_ai_conflict",
                                  "The OOD detector identifies strong structural anomaly, but the AI classifier does not flag synthetic origin. This pattern is characteristic of sequences engineered to evade k-mer-based detection while maintaining non-natural global composition."))

        if strengths["codon"] in (SignalStrength.HIGH, SignalStrength.MODERATE):
            if strengths["ai_risk"] == SignalStrength.NEGATIVE:
                conflicts.append(("codon_ai_conflict",
                                  "Codon usage analysis suggests optimisation for heterologous expression, but the AI classifier does not flag synthetic origin. Codon optimisation alone does not constitute proof of engineering."))

        if s.sequence_length < 300:
            conflicts.append(("short_seq_limitation",
                              f"Sequence length ({s.sequence_length} bp) is below the 300 bp confidence threshold. K-mer statistics become unreliable on short fragments. BLAST homology and motif detection are the primary evidence sources for this input."))

        return conflicts

    def _detect_supporting(self) -> list[str]:
        supporting = []
        strengths = self._signal_strengths
        high_signals = [k for k, v in strengths.items() if v in (SignalStrength.CRITICAL, SignalStrength.HIGH)]

        if len(high_signals) >= 3:
            supporting.append(("strong_convergence",
                               f"Multiple independent signals ({', '.join(high_signals)}) show elevated anomaly levels, providing strong convergent evidence for synthetic origin."))

        if strengths["ai_risk"] in (SignalStrength.HIGH, SignalStrength.CRITICAL):
            if strengths["ood"] in (SignalStrength.HIGH, SignalStrength.CRITICAL):
                supporting.append(("ai_ood_convergence",
                                   "AI classification and OOD anomaly detection independently converge on synthetic origin, substantially increasing confidence in the result."))

        if strengths["blast"] in (SignalStrength.HIGH, SignalStrength.CRITICAL):
            if strengths["motifs"] in (SignalStrength.HIGH, SignalStrength.CRITICAL):
                supporting.append(("blast_motif_convergence",
                                   "BLAST homology and exact motif detection both identify laboratory elements, providing orthogonal confirmation of vector origin."))

        return supporting

    def _build_executive_summary(self) -> str:
        s = self.snapshot
        v = s.verdict
        if v == VerdictType.CONFIRMED_ENGINEERED:
            if s.blast_hit_count > 0:
                evidence = f"({s.blast_hit_count} vector homology hits"
                if s.known_motifs: evidence += f", {len(s.known_motifs)} known lab motif(s)"
                evidence += ")"
            else:
                evidence = f"({len(s.known_motifs)} known lab motif(s) detected)"
            return f"This sequence has been classified as **CONFIRMED LAB-ENGINEERED** based on direct physical evidence {evidence}. These are not statistical inferences - they represent sequence-level detections of laboratory-origin DNA that can be independently verified."
        elif v == VerdictType.SUSPECTED_ENGINEERED:
            return f"This sequence has been classified as **SUSPECTED ENGINEERED**. While no direct BLAST match to known vectors was found, multiple independent computational signals (AI risk: {s.ai_risk:.1f}%, OOD anomaly: {s.ood_score:.1f}/100) indicate compositional inconsistency with natural viral genomes."
        elif v == VerdictType.BORDERLINE:
            return f"This sequence produced **mixed signals** across the detection pipeline (AI risk: {s.ai_risk:.1f}%, OOD: {s.ood_score:.1f}/100). Expert manual review is recommended."
        else:
            return "This sequence was classified as **LIKELY NATURAL**. Across all detection layers, no significant evidence of synthetic origin was identified."

    def _build_signal_interplay(self) -> str:
        parts = []
        parts.append(
            "**Evidence Hierarchy**\n\nThis system employs a layered evidence model where detection methods are ordered by specificity and resistance to evasion:\n\n1. **Direct Evidence** (highest confidence): BLAST homology to UniVec, exact motif matches - these are binary, verifiable detections.\n2. **Statistical Convergence**: Agreement between independent AI and OOD models trained on different feature spaces.\n3. **Compositional Signals**: Codon optimisation, k-mer enrichment - suggestive but not conclusive alone.\n4. **Negative Evidence**: Absence of signals - reduces probability but does not exclude engineering.\n\n")
        if self._conflicts:
            parts.append("**Signal Conflicts Detected**\n\n")
            for conflict_id, explanation in self._conflicts:
                parts.append(f"**{conflict_id.replace('_', ' ').title()}**: {explanation}\n\n")
        if self._supporting:
            parts.append("**Convergent Evidence**\n\n")
            for support_id, explanation in self._supporting:
                parts.append(f"**{support_id.replace('_', ' ').title()}**: {explanation}\n\n")
        if not self._conflicts and not self._supporting:
            parts.append(
                "No signal conflicts or convergences were detected. All detection layers produced consistent results.\n\n")
        return "".join(parts)

    def _build_ai_analysis(self) -> str:
        s = self.snapshot
        parts = [
            f"**Model Output:** Natural {s.prob_natural * 100:.1f}% | Vector {s.prob_vector * 100:.1f}% | Ghost {s.prob_ghost * 100:.1f}% | Combined AI Risk: {s.ai_risk:.1f}%\n\n"]
        class_explanations = {
            AIClass.NATURAL: "The classifier assigned this sequence to the **Natural** class, indicating its k-mer composition is consistent with wild-type viral genomes.",
            AIClass.VECTOR: "The classifier assigned this sequence to the **Vector** class, indicating strong k-mer similarity to laboratory cloning vectors, plasmids, or expression constructs.",
            AIClass.GHOST: "The classifier assigned this sequence to the **Ghost** class, indicating k-mer composition consistent with synthetic constructs that do not match known vector signatures.",
            AIClass.UNCERTAIN: "The classifier produced uncertain probabilities across all classes.",
        }
        parts.append(class_explanations[s.ai_class] + "\n\n")
        parts.append(
            "**Model Architecture**: Stacked ensemble (Random Forest + Gradient Boosting + Calibrated Linear SVC), meta-learned by Logistic Regression. Features: 8-mer TF-IDF vectors computed over 500 bp sliding windows.\n\n")
        strength = self._signal_strengths["ai_risk"]
        risk_interpretations = {
            SignalStrength.CRITICAL: f"AI risk at {s.ai_risk:.1f}% is in the **critical zone**. This strongly suggests deliberate sequence design.",
            SignalStrength.HIGH: f"AI risk at {s.ai_risk:.1f}% is **elevated**. This level is uncommon in natural isolates.",
            SignalStrength.MODERATE: f"AI risk at {s.ai_risk:.1f}% is **borderline**.",
            SignalStrength.LOW: f"AI risk at {s.ai_risk:.1f}% is **low**.",
            SignalStrength.NEGATIVE: f"AI risk at {s.ai_risk:.1f}% is **negligible**.",
        }
        parts.append(risk_interpretations[strength])
        return "\n".join(parts)

    def _build_ood_analysis(self) -> str:
        s = self.snapshot
        parts = [f"**Overall OOD Score: {s.ood_score:.1f}/100**\n\n"]
        if s.ood_viral is not None or s.ood_prokaryote is not None:
            parts.append("**Dual-Envelope Architecture Breakdown:**\n\n")
            parts.append("| Envelope | Training Data | Score | Interpretation |\n")
            parts.append("|---|---|---|---|\n")
            if s.ood_viral is not None:
                parts.append(
                    f"| Viral | Natural viral genomes | {s.ood_viral:.1f} | {self._ood_level_text(s.ood_viral)} |\n")
            if s.ood_prokaryote is not None:
                parts.append(
                    f"| Prokaryotic | Bacterial genomes (E. coli, etc.) | {s.ood_prokaryote:.1f} | {self._ood_level_text(s.ood_prokaryote)} |\n")
            parts.append(f"| **Final (MAX)** | - | **{s.ood_score:.1f}** | - |\n\n")
            parts.append(
                "**Design Rationale**: Taking the maximum score across envelopes ensures that sequences anomalous in ANY natural context are flagged. The dual-envelope approach prevents synthetic sequences from 'hiding' in the envelope most similar to their source organism.\n\n")
        else:
            parts.append("**Single-Envelope Mode**: Only the viral envelope was available.\n\n")
        if s.ood_flag:
            flag_explanations = {
                "SHORT_FRAGMENT": f"**[!] SHORT FRAGMENT FLAG**: Sequence is {s.sequence_length} bp, below the 300 bp reliability threshold.",
                "DUAL_ENVELOPE": "**Dual-Envelope Mode**: Both envelopes were used.",
                "NOT_READY": "**[!] OOD SCORER UNAVAILABLE**.",
            }
            if s.ood_flag in flag_explanations:
                parts.append(flag_explanations[s.ood_flag] + "\n\n")
        strength = self._signal_strengths["ood"]
        ood_interpretations = {
            SignalStrength.CRITICAL: f"The OOD score indicates the sequence is far outside the natural distribution envelope.",
            SignalStrength.HIGH: f"The OOD score is elevated, placing this sequence in a region where natural isolates are sparse.",
            SignalStrength.MODERATE: f"The OOD score is mildly elevated, near the boundary of the natural distribution.",
            SignalStrength.LOW: f"The OOD score is low. For known vectors or short fragments, this is expected.",
            SignalStrength.NEGATIVE: f"The OOD score is very low, well within the natural distribution envelope.",
        }
        parts.append(ood_interpretations[strength])
        return "\n".join(parts)

    def _ood_level_text(self, score: float) -> str:
        if score >= 75: return "Anomalous"
        if score >= 50: return "Elevated"
        if score >= 35: return "Borderline"
        return "Normal"

    def _build_homology_analysis(self) -> str:
        s = self.snapshot
        parts = []
        if s.blast_hit_count == 0:
            parts.append(
                "No alignments were found against the UniVec database. The absence of a hit does not exclude engineering, but removes the strongest class of direct evidence.")
        else:
            parts.append(f"**{s.blast_hit_count} alignment(s)** were detected against the UniVec database.\n\n")
            parts.append("| Metric | Value |\n")
            parts.append("|---|---|\n")
            parts.append(f"| Total hits | {s.blast_hit_count} |\n")
            parts.append(f"| Top identity | {s.blast_top_identity:.1f}% |\n")
            parts.append(f"| Coverage | {s.blast_coverage_pct:.1f}% of query |\n\n")
            if s.blast_top_identity >= 95:
                parts.append(
                    f"The top hit shows **{s.blast_top_identity:.1f}% identity**, indicating near-exact match to a known vector sequence. This is strong direct evidence of laboratory origin.")
            elif s.blast_top_identity >= 85:
                parts.append(
                    f"The top hit shows **{s.blast_top_identity:.1f}% identity**, indicating significant homology to a known vector.")
            if s.known_motifs:
                parts.append(
                    f"\n\nAdditionally, **{len(s.known_motifs)} known laboratory motif(s)** were detected: {', '.join(s.known_motifs)}.")
        return "\n".join(parts)

    def _build_codon_analysis(self) -> str:
        s = self.snapshot
        parts = [
            f"| Metric | Value | Threshold | Flagged |\n|---|---|---|---|\n| CAI Score | {s.cai_score:.4f} | > 0.65 | {'[!] Yes' if s.cai_score > 0.65 else 'No'} |\n| RSCU Bias | {s.rscu_bias:.4f} | > 0.45 | {'[!] Yes' if s.rscu_bias > 0.45 else 'No'} |\n\n"]
        if s.codon_flagged:
            parts.append(
                "**Codon optimisation detected.** Synonymous codon usage shows bias consistent with optimisation for heterologous expression. **Caveat**: This finding should be weighted alongside AI and BLAST evidence.")
        else:
            parts.append(
                "**No significant codon optimisation detected.** Synonymous codon usage is within the range expected for natural viral genomes.")
        return "\n".join(parts)

    def _build_conclusion(self) -> str:
        s = self.snapshot
        v = s.verdict
        parts = ["**Final Synthesis**\n\n"]
        if v == VerdictType.CONFIRMED_ENGINEERED:
            parts.append("This sequence is **confirmed laboratory-engineered** based on direct evidence:\n\n")
            if s.blast_hit_count > 0: parts.append(
                f"- [+] UniVec homology: {s.blast_hit_count} hits ({s.blast_top_identity:.1f}% identity)\n")
            if s.known_motifs: parts.append(f"- [+] Known lab motifs detected\n")
            if s.ai_risk >= 60: parts.append(f"- [+] AI classification: {s.ai_class.value} ({s.ai_risk:.1f}%)\n")
            if self._conflicts: parts.append(
                f"\n**Note on conflicting signals**: {self._conflicts[0][1].split('.')[0]}.")
        elif v == VerdictType.SUSPECTED_ENGINEERED:
            parts.append("This sequence is **suspected engineered** based on statistical convergence:\n\n")
            if s.ai_risk >= 60: parts.append(f"- [!] AI risk elevated: {s.ai_risk:.1f}%\n")
            if s.ood_score >= 50: parts.append(f"- [!] OOD anomaly elevated: {s.ood_score:.1f}/100\n")
        elif v == VerdictType.BORDERLINE:
            parts.append("This sequence produced **mixed signals**. Expert manual review is recommended.")
        else:
            parts.append(
                "This sequence shows **no significant evidence** of synthetic origin across all detection layers.")
        return "\n".join(parts)

    @staticmethod
    def from_result_dict(result: dict) -> SignalSnapshot:
        probas = result.get("probas", {})
        prob_nat, prob_vec, prob_ghost = probas.get(0, 0), probas.get(1, 0), probas.get(2, 0)
        if prob_ghost > 0.5:
            ai_class = AIClass.GHOST
        elif prob_vec > 0.5:
            ai_class = AIClass.VECTOR
        elif prob_nat > 0.5:
            ai_class = AIClass.NATURAL
        else:
            ai_class = AIClass.UNCERTAIN

        verdict_map = {"CONFIRMED LAB-ENGINEERED": VerdictType.CONFIRMED_ENGINEERED,
                       "SUSPECTED ENGINEERED": VerdictType.SUSPECTED_ENGINEERED,
                       "BORDERLINE / REVIEW": VerdictType.BORDERLINE, "LIKELY NATURAL": VerdictType.LIKELY_NATURAL}

        hits = result.get("hits", {})
        if hasattr(hits, 'empty') and not hits.empty:
            top_identity = float(hits["percent_identity"].max())
            blast_coverage = len(
                set().union(*[set(range(int(h["q_start"]), int(h["q_end"]) + 1)) for _, h in hits.iterrows()])) / max(
                result["length"], 1) * 100
        else:
            top_identity, blast_coverage = 0.0, 0.0

        ood_details = result.get("ood_details", {})
        return SignalSnapshot(
            ai_risk=result.get("ai_risk", 0), ai_class=ai_class, prob_natural=prob_nat, prob_vector=prob_vec,
            prob_ghost=prob_ghost,
            ood_score=result.get("ood_score", 0), ood_viral=ood_details.get("viral_score"),
            ood_prokaryote=ood_details.get("prok_score"), ood_flag=ood_details.get("flag"),
            blast_hit_count=len(hits) if hasattr(hits, '__len__') else 0, blast_top_identity=top_identity,
            blast_coverage_pct=blast_coverage,
            known_motifs=result.get("known_motifs", []), enriched_motifs=result.get("enriched_motifs", []),
            cai_score=result.get("codon_result", {}).get("cai_score", 0),
            rscu_bias=result.get("codon_result", {}).get("rscu_bias", 0),
            codon_flagged=result.get("codon_result", {}).get("optimisation_flag", False),
            gc_content=result.get("gc", 0), sequence_length=result.get("length", 0),
            verdict=verdict_map.get(result.get("verdict", ""), VerdictType.LIKELY_NATURAL),
        )
