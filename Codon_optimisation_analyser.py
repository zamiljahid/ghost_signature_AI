from __future__ import annotations

import math
from collections import Counter

_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_ECOLI_FREQ: dict[str, float] = {
    "TTT": 22.0, "TTC": 16.5, "TTA": 14.0, "TTG": 13.1,
    "CTT": 11.4, "CTC": 10.5, "CTA":  3.8, "CTG": 52.5,
    "ATT": 30.4, "ATC": 25.1, "ATA":  7.5, "ATG": 27.6,
    "GTT": 18.3, "GTC": 15.3, "GTA": 11.1, "GTG": 26.1,
    "TCT": 15.3, "TCC": 15.2, "TCA":  7.5, "TCG":  8.9,
    "CCT": 17.5, "CCC": 12.6, "CCA": 20.0, "CCG": 23.0,
    "ACT": 19.0, "ACC": 23.3, "ACA": 13.8, "ACG": 14.3,
    "GCT": 18.3, "GCC": 25.5, "GCA": 21.4, "GCG": 33.7,
    "TAT": 16.2, "TAC": 12.3, "TAA":  2.0, "TAG":  0.3,
    "CAT": 13.0, "CAC":  9.7, "CAA": 13.7, "CAG": 28.7,
    "AAT": 22.7, "AAC": 22.0, "AAA": 33.6, "AAG": 10.5,
    "GAT": 32.5, "GAC": 19.2, "GAA": 39.6, "GAG": 17.8,
    "TGT":  5.1, "TGC":  6.4, "TGA":  0.8, "TGG": 15.2,
    "CGT": 21.0, "CGC": 22.0, "CGA":  3.6, "CGG": 11.5,
    "AGT":  8.8, "AGC": 16.1, "AGA":  2.1, "AGG":  1.2,
    "GGT": 24.7, "GGC": 29.8, "GGA":  8.0, "GGG": 15.3,
}


class CodonOptimisationAnalyser:

    def __init__(self, sequence: str) -> None:
        self.seq = "".join(c for c in sequence.upper() if c in "ATGCN")

    # ------------------------------------------------------------------
    def _extract_codons(self) -> list[str]:
        """Return all valid, non-stop codons from the sequence."""
        return [
            self.seq[i: i + 3]
            for i in range(0, len(self.seq) - 2, 3)
            if len(self.seq[i: i + 3]) == 3
            and self.seq[i: i + 3] in _CODON_TABLE
            and _CODON_TABLE[self.seq[i: i + 3]] != "*"
        ]

    # ------------------------------------------------------------------
    def compute_rscu(self) -> dict[str, float]:

        codons = self._extract_codons()
        counts: Counter = Counter(codons)

        # Group codons by amino acid
        aa_groups: dict[str, list[str]] = {}
        for codon, aa in _CODON_TABLE.items():
            if aa != "*":
                aa_groups.setdefault(aa, []).append(codon)

        rscu: dict[str, float] = {}
        for aa, synonyms in aa_groups.items():
            total = sum(counts.get(c, 0) for c in synonyms)
            n = len(synonyms)
            if total == 0:
                continue
            for codon in synonyms:
                expected = total / n
                rscu[codon] = round(counts.get(codon, 0) / expected, 4) if expected else 0.0

        return rscu

    # ------------------------------------------------------------------
    def compute_cai(self) -> float:
        """
        CAI = exp( (1/L) * sum_i( ln(f_i / f_max_i) ) )

        Where f_i is the E. coli frequency of the i-th codon and f_max_i
        is the highest frequency among synonyms for that amino acid.

        Returns a value in [0, 1].  > 0.65 suggests optimisation for E. coli.
        """
        codons = self._extract_codons()
        if not codons:
            return 0.0

        # Pre-compute per-amino-acid max frequency in E. coli
        aa_max: dict[str, float] = {}
        for codon, aa in _CODON_TABLE.items():
            if aa == "*":
                continue
            freq = _ECOLI_FREQ.get(codon, 0.1)
            aa_max[aa] = max(aa_max.get(aa, 0.0), freq)

        log_sum = 0.0
        n = 0
        for codon in codons:
            aa = _CODON_TABLE.get(codon)
            if not aa or aa == "*":
                continue
            freq = _ECOLI_FREQ.get(codon, 0.1)
            max_f = aa_max.get(aa, 1.0)
            if max_f > 0:
                log_sum += math.log(freq / max_f)
                n += 1

        return round(math.exp(log_sum / n), 4) if n else 0.0

    # ------------------------------------------------------------------
    def compute_rscu_bias(self, rscu: dict[str, float]) -> float:
        """Mean absolute deviation of RSCU values from 1.0 (uniform use).
        Natural sequences: < 0.3.  Codon-optimised: typically > 0.45."""
        if not rscu:
            return 0.0
        return round(sum(abs(v - 1.0) for v in rscu.values()) / len(rscu), 4)

    # ------------------------------------------------------------------
    def analyse(self) -> dict:
        """
        Run the full codon optimisation analysis on the real sequence.

        Returns a dict suitable for passing directly to
        get_graph_interpretation('codon_optimisation', result).
        """
        rscu = self.compute_rscu()
        cai = self.compute_cai()
        bias = self.compute_rscu_bias(rscu)
        total_codons = len(self._extract_codons())
        optimised = cai > 0.65 or bias > 0.45

        # Top preferred and top avoided codons (excluding absent codons)
        observed_rscu = {c: v for c, v in rscu.items() if v > 0}
        top_codon = max(observed_rscu, key=observed_rscu.get) if observed_rscu else "N/A"
        bottom_codon = min(observed_rscu, key=observed_rscu.get) if observed_rscu else "N/A"

        if cai > 0.75 and bias > 0.5:
            verdict = (
                f"Strong codon optimisation signal detected (CAI={cai:.3f}, RSCU bias={bias:.3f}). "
                "Synonymous codon usage is highly non-uniform and strongly consistent with "
                "synthetic optimisation for E. coli or mammalian expression systems."
            )
        elif optimised:
            verdict = (
                f"Moderate codon optimisation signal (CAI={cai:.3f}, RSCU bias={bias:.3f}). "
                "Codon usage departs from natural viral background, suggesting possible synthetic "
                "origin or heterologous expression optimisation."
            )
        else:
            verdict = (
                f"No significant codon optimisation detected (CAI={cai:.3f}, RSCU bias={bias:.3f}). "
                "Synonymous codon usage is broadly consistent with natural viral sequences and does "
                "not suggest deliberate codon engineering."
            )

        return {
            "cai_score": cai,
            "rscu_bias": bias,
            "rscu": rscu,
            "total_codons": total_codons,
            "top_codon": top_codon,
            "top_rscu": observed_rscu.get(top_codon, 0.0),
            "bottom_codon": bottom_codon,
            "bottom_rscu": observed_rscu.get(bottom_codon, 0.0),
            "optimisation_flag": optimised,
            "verdict": verdict,
        }
