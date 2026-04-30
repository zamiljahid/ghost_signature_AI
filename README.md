<div align="center">

# 🧬 Ghost Signature Detector v2.0 🛡️
**Advanced Biogenomic Threat Surveillance & Forensic Ecosystem**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-FF6F00?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-BLAST%2B%20%7C%20BioPython-4CAF50?style=for-the-badge&logo=dna&logoColor=white)](https://biopython.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=for-the-badge)](#)

*Detecting the undetectable. Bridging the gap between natural evolution and synthetic design.*

</div>

<br>

## 🌐 Overview: The Biosecurity Frontier

As DNA synthesis technology becomes more accessible, the ability to "print" custom genetic code has outpaced our ability to verify its origin. Natural organisms evolve through random mutation and environmental selection, leaving behind a "noisy" genetic trail. In contrast, lab-engineered sequences are designed for efficiency, often following a logical, mathematical optimization known as a **Ghost Signature**.

**Ghost Signature Detector** is a specialized forensic tool built to scan genomic data for these hidden patterns. By analyzing the structural "rhythm" of DNA, the software acts as a digital forensic investigator, identifying whether a sequence was crafted by the hands of evolution or the keys of a bio-engineer. It provides biosecurity researchers with an early-warning system to flag potential genomic threats that standard database checks might miss.

---

## 🚀 The Core Challenge

Modern biosecurity faces a critical **"Detection Gap"**:
1. **The Database Limitation:** Traditional tools only flag known pathogens or laboratory vectors.
2. **The Machine Rhythm:** Engineered DNA displays a **"Ghost Signature"**—a low-entropy, hyper-optimized pattern caused by *in-silico* design (codon optimization) that differs fundamentally from **Wild-Type** evolutionary noise.

**Ghost Signature Detector** bridges this gap by providing a **4-Phase Forensic Verdict**.

---

## ⚙️ Unified Architecture & Methodology

The system integrates high-throughput bioinformatics with machine learning classifiers to map genomic threats in real-time.

<p align="center">
  <img src="https://raw.githubusercontent.com/zamiljahid/ghost_signature_AI/main/architecture.png" width="90%" alt="System Architecture">
</p>

<p align="center">
  <a href="https://zamiljahid.github.io/ghost_signature_AI/architecture.html">
    <img src="https://img.shields.io/badge/View-Interactive%20Logic%20Diagram-FF4B4B?style=for-the-badge&logo=github">
  </a>
</p>

### Phase 1: Data Ingestion & Pre-Processing
*   **Validation:** Standardizes raw `.FASTA` / `.FASTQ` inputs.
*   **Filtering:** Removes sequencing artifacts and normalizes nucleotide data for analysis.

### Phase 2: Bioinformatics Analysis Cores
The sequence is processed through two parallel diagnostic engines:
*   **Track A: k-mer Complexity Analysis:** Calculates informational diversity and **Shannon Entropy**. It identifies "Entropy Dips" where machine-optimization has smoothed out natural genetic noise.
*   **Track B: BLAST+ Database Alignment:** Queries against UniVec and laboratory vector databases to find direct "copy-paste" segments from known industrial sources.

### Phase 3: Synthetic Marker Classification
*   **AI Engine:** A **Random Forest Classifier** trained on k-mer frequency distributions and TF-IDF vectors.
*   **Feature Evaluation:** It differentiates between natural mutations and synthetic "Ghost Signatures" based on the structural rhythm of the sequence.

### Phase 4: Genomic Threat Mapping & Output
*   **Coordinate Mapping:** Flags are mapped back to precise base-pair locations.
*   **Reporting:** Generates the forensic evidence suite (Detailed below).

---

## 📈 Forensic PDF Reporting & Visualization

The system generates a publication-ready **Genomic Forensic Report (PDF)** for every sample, containing:

### 📊 Analytical Graphics
*   **Shannon Entropy Heatmap:** A spatial plot highlighting "Entropy Dips"—the smoking gun of codon optimization.
*   **k-mer Frequency Distribution:** Histograms comparing the sample's nucleotide patterns against a "Wild-Type" baseline.
*   **GC-Content Skew:** Detects unnatural shifts in Guanine-Cytosine ratios, often indicating inserted gene cassettes.

### 🔍 Forensic Evidence Snippets
*   **Database Match Snippets:** Raw nucleotide-level views of segments matching known lab vectors.
*   **Anomaly Snippets:** Segments flagged by the Random Forest model as "Machine Designed" despite having no database matches.
*   **Coordinate Reference Table:** Precise location mapping (e.g., `BP 1,024 - 1,580`) for wet-lab validation.

---

## 📊 3-Tier Classification System

| Classification | Homology | AI Risk | Interpretation |
| :--- | :--- | :--- | :--- |
| **Confirmed Engineered** | ✅ Found | High | Direct match to known lab vectors or plasmids. |
| **Suspected Engineered** | ❌ None | **High/Mid** | **Ghost Signature detected.** Pattern matches machine design rhythms. |
| **Fully Natural** | ❌ None | Low | **Wild-Type.** High-entropy, natural evolutionary noise. |

---

## 💻 Deployment
```bash
# Clone the repository
git clone [https://github.com/zamiljahid/ghost_signature_AI.git](https://github.com/zamiljahid/ghost_signature_AI.git)

# Install Forensic Suite
pip install -r requirements.txt

# Run Forensic Analysis
python main.py --input sample.fasta --generate-pdf
