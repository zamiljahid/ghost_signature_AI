# 🧬 Ghost Signature Detector: AI-Powered Biosecurity 🛡️

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/AI-Random%20Forest-orange)](https://scikit-learn.org/)
[![Forensics](https://img.shields.io/badge/Analysis-Genomic%20Forensics-red)](https://biopython.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)](https://scikit-learn.org/)
[![BioPython](https://img.shields.io/badge/Bioinformatics-BioPython-green)](https://biopython.org/)

**Ghost Signature Detector** is a forensic genomic surveillance suite designed to identify synthetic DNA backbones and "silent" lab-manipulated sequences. It utilizes a multi-layered detection strategy to catch both **Known Threats** (Database matches) and **Emerging Threats** (AI-detected structural anomalies).

---

## 🚀 The Core Challenge
Modern biosecurity faces a significant "Detection Gap":
1.  **Database Reliance:** Traditional tools only find what they already know. 
2.  **Machine Rhythm:** Engineered DNA displays a **"Ghost Signature"**—a low-entropy, hyper-optimized pattern caused by *in-silico* design (codon optimization) that differs fundamentally from **Wild-Type** evolutionary noise.

This tool bridges the gap by providing a **3-Tier Forensic Verdict**.

---

## 🛠️ Forensic Methodology

The system generates a **Genomic Forensic Map** by analyzing three distinct layers of evidence:

### 1. Database Homology (Tier 1)
* **Engine:** BLASTn + UniVec/Laboratory Vector Database.
* **Evidence:** Identifies direct "copy-paste" segments from known industrial plasmids and vectors.

### 2. Structural AI Analysis (Tier 2)
* **Engine:** Random Forest Classifier + TF-IDF Vectorization (k=8).
* **Evidence:** Detects the **"Machine Rhythm"** of the sequence. It flags sequences that lack stochastic evolutionary noise, identifying synthetic designs even when database matches are zero.

### 3. K-mer Complexity/Shannon Entropy (Tier 3)
* **Engine:** Real-time Shannon Entropy Calculation.
* **Evidence:** Quantifies informational diversity. Synthetic sequences often show **"Entropy Dips"**—localized drops in complexity where a machine has optimized the genetic code for industrial expression.

---
## 🧬 System Architecture

<p align="center">
  <img src="architecture.png" width="100%" alt="Architecture Diagram Preview">
</p>

<p align="center">
  <a href="https://zamiljahid.github.io/ghost_signature_AI/architecture.html">
    <img src="https://img.shields.io/badge/View-Interactive%20Diagram-FF4B4B?style=for-the-badge&logo=github">
  </a>
</p>

## 📊 3-Tier Classification System

| Classification | Homology | AI Risk | Interpretation |
| :--- | :--- | :--- | :--- |
| **Confirmed Engineered** | ✅ Found | High | Direct match to known lab vectors. |
| **Suspected Engineered** | ❌ None | **High/Mid** | **Ghost Signature detected.** Pattern matches machine design. |
| **Fully Natural** | ❌ None | Low | **Wild-Type.** High-entropy, natural evolutionary noise. |

---

## 📈 Visualizing the Evidence

The tool generates a high-resolution **Genomic Forensic Map** for every sample, featuring:
* **The Backbone:** A spatial reference frame of the total sequence length.
* **Risk Heatmap:** A color-coded intensity map of synthetic risk across the genome.
* **Complexity Graph:** A Shannon Entropy plot to identify machine-optimized segments.

---
