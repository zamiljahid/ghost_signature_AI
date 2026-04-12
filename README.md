# 🧬 Ghost Signature Detector: AI-Powered Biosecurity 🛡️

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)](https://scikit-learn.org/)
[![BioPython](https://img.shields.io/badge/Bioinformatics-BioPython-green)](https://biopython.org/)

**Ghost Signature Detector** is a dual-engine genomic surveillance tool designed to identify synthetic DNA backbones and lab-manipulated sequences. By combining **BLAST+ database alignment** with a **Random Forest Machine Learning model**, it detects both known vector matches and novel "Ghost Signatures"—engineered patterns that don't exist in public databases.

---

## 🚀 The Core Challenge
In modern biosecurity, identifying engineered sequences is difficult because:
1. **Known Vectors:** Easily caught via BLAST (e.g., pUC19, pBR322).
2. **De Novo Synthesis:** Newly engineered sequences may have **zero** database matches but still contain "Ghost Signatures"—statistical anomalies in k-mer frequency that differ from natural evolution.

This tool solves both by providing a **Consensus Risk Score**.

---

## 🛠️ System Architecture

The detector utilizes a two-tier verification pipeline:

1.  **Tier 1: Homology Search (BLASTn)**
    * Performs high-speed local alignment against the **UniVec** database.
    * Identifies known laboratory contaminants and vector backbones.
2.  **Tier 2: AI Pattern Recognition (Random Forest)**
    * Analyzes **k-mer frequency (k=8)** using **TF-IDF Vectorization**.
    * Trained on a balanced dataset of 30+ viral families (including ASFV, Zika, and Phages) vs. thousands of synthetic vectors.
    * Uses **Class Weighting** and **Fragmentation** to identify synthetic signatures in sequences as short as 150bp.



---

## 📊 Sample Analysis Output

### Case A: Known Synthetic Vector (pUC19)
| Engine | Result | Status |
| :--- | :--- | :--- |
| **BLAST** | 390 Hits Found | ✅ Identified |
| **AI Score** | 100% Synthetic Risk | 🚩 Critical |

### Case B: Natural Pathogen (ASFV)
| Engine | Result | Status |
| :--- | :--- | :--- |
| **BLAST** | 0 Hits Found | ✅ Natural |
| **AI Score** | 43% (Low/Moderate) | 🟢 Clear |

---

## 💻 Installation & Usage

### Prerequisites
* [NCBI BLAST+ Executables](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
* Python 3.10+

### Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/ghost-signature-detector.git](https://github.com/yourusername/ghost-signature-detector.git)
cd ghost-signature-detector

# Install dependencies
pip install -r requirements.txt
