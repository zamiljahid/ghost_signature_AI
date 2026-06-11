import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- New Novelty Analysis Class ---

class GhostForensics:
    """
    Module to prove novelty by mapping the 'Signal Gap'
    between homology (BLAST) and statistical anomaly (OOD).
    """
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_signal_gap(self, blast_scores, ood_scores, labels):
        """
        Visualizes the 'Novelty Zone': High OOD score but Low BLAST homology.
        """
        plt.figure(figsize=(10, 7))

        # Normalize scores for comparison
        # X-axis: Homology (Similarity to known vectors)
        # Y-axis: Anomaly (Structural difference from nature)

        plt.scatter(blast_scores, ood_scores, c=labels, cmap='viridis', alpha=0.6, s=100)

        # Highlight the 'Ghost Zone' (High Anomaly, Low Homology)
        plt.gca().add_patch(plt.Rectangle((0, 50), 30, 50,
                                        color='red', alpha=0.1, label='Ghost Novelty Zone'))

        plt.axhline(50, color='black', linestyle='--', alpha=0.5)
        plt.axvline(30, color='black', linestyle='--', alpha=0.5)

        plt.xlabel("Homology Score (BLAST Similarity %)", fontweight='bold')
        plt.ylabel("Ghost Anomaly Score (OOD)", fontweight='bold')
        plt.title("The Signal Gap: Proving Detection of Novel Synthetic DNA", fontsize=14)
        plt.legend()

        plt.savefig(self.output_dir / "novelty_signal_gap.png")
        plt.close()

    def map_kmer_saliency(self, sequence, vectorizer, model, window_size=500):
        """
        Identifies specific k-mers contributing to the classification.
        Handles standard classifiers and StackingClassifiers.
        """
        # 1. Extract Feature Importance/Weights based on model type
        if hasattr(model, 'feature_importances_'):
            # Standard Random Forest / XGBoost
            weights = model.feature_importances_
        elif hasattr(model, 'final_estimator_'):
            # StackingClassifier: named_estimators_ gives a dict of fitted base estimators.
            # estimators_[0] is a (name, estimator) tuple — calling .feature_importances_
            # on a tuple raises AttributeError. Use named_estimators_ instead.
            try:
                weights = model.named_estimators_['rf'].feature_importances_
            except (KeyError, AttributeError):
                try:
                    # Fallback: first estimator if 'rf' key absent
                    weights = model.estimators_[0][1].feature_importances_
                except (IndexError, AttributeError):
                    return np.zeros(len(sequence))
        else:
            # Fallback: weights can't be extracted
            return np.zeros(len(sequence))

        # 2. Map k-mers in the sequence to their 'Ghost Weight'
        kmers = [sequence[i:i + 8] for i in range(len(sequence) - 7)]
        saliency_scores = []

        # Get vocabulary mapping
        vocab = vectorizer.vocabulary_

        for k in kmers:
            if k in vocab:
                idx = vocab[k]
                # Handle cases where weights might be shorter than vocab
                if idx < len(weights):
                    saliency_scores.append(abs(weights[idx]))
                else:
                    saliency_scores.append(0)
            else:
                saliency_scores.append(0)

        return np.array(saliency_scores)

    def plot_saliency_track(self, sequence_id, saliency_scores):
        """
        Generates a forensic 'seismic' track showing where the
        Ghost signatures are strongest in the DNA.
        """
        if len(saliency_scores) == 0:
            return

        plt.figure(figsize=(12, 3))

        # Create the X-axis (base pair positions)
        x = np.arange(len(saliency_scores))

        # Plot the saliency line
        plt.plot(x, saliency_scores, color='#6D28D9', lw=1.5, label='K-mer Saliency')

        # Fill the area underneath for a professional 'seismic' look
        plt.fill_between(x, saliency_scores, color='#6D28D9', alpha=0.2)

        # Formatting for thesis quality
        plt.title(f"Forensic Saliency Map: {sequence_id}", fontsize=12, fontweight='bold')
        plt.xlabel("Base Pair Position (bp)", fontsize=10)
        plt.ylabel("Anomaly Weight", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Remove top/right spines for a cleaner academic look
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()

        # Save to the forensics folder
        save_path = self.output_dir / f"{sequence_id}_saliency.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

# --- Integration into Main Loop ---

def run_forensic_extension(detector, records):
    # DEPRECATED: This function uses np.random.uniform for simulated BLAST scores
    # and is not called from the main pipeline. It is kept for reference only.
    import warnings
    warnings.warn(
        "run_forensic_extension() is deprecated and uses simulated data. "
        "Do not call this in production.",
        DeprecationWarning, stacklevel=2
    )
    forensics = GhostForensics()

    all_blast = []
    all_ood = []

    for rec in records:
        analysis = detector.analyze_sequence(rec)

        # Simulate a BLAST score for demonstration
        # (In reality, fetch the bit-score/identity from your blast_hits.csv)
        simulated_blast = np.random.uniform(0, 100) if "ghost" in rec.id else np.random.uniform(0, 20)

        all_blast.append(simulated_blast)
        all_ood.append(analysis['ood_score'])

        # Generate Saliency Track for Thesis
        saliency = forensics.map_kmer_saliency(str(rec.seq), detector.vectorizer, detector.model)

        plt.figure(figsize=(12, 2))
        plt.plot(saliency, color='purple')
        plt.title(f"K-mer Saliency Map: {rec.id} (Feature Novelty)")
        plt.fill_between(range(len(saliency)), saliency, color='purple', alpha=0.3)
        plt.savefig(f"outputs/{rec.id}_saliency.png")
        plt.close()

    forensics.plot_signal_gap(all_blast, all_ood, labels=[1 if "ghost" in r.id else 0 for r in records])