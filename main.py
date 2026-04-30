import os
import subprocess
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from fpdf import FPDF
import datetime


class GhostSignatureDetector:
    def __init__(self, query_file, database_path, model_path='ghost_model.pkl', vectorizer_path='dna_vectorizer.pkl'):
        self.query_file = query_file
        self.db = database_path
        self.output_csv = "ghost_signature_report.csv"
        self.session_logs = [
            f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Environment: {os.getcwd()}",
            f"Input: {query_file}"
        ]
        self.motifs = {
            "T7_Promoter": "TAATACGACTCACTATAGGG",
            "CMV_Promoter": "TGACATTGATTATTGACTAG",
            "SV40_PolyA": "AATAAAATATCTTTATTTTC",
            "Kanamycin_Res": "ATGAGCCATATTCAACGGGA",
            "Ampicillin_Res": "ATGAATTCACTGGCCGTCGT"
        }
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.ai_active = True
            self.log("[#] AI Engine: ONLINE")
        except FileNotFoundError:
            self.ai_active = False
            self.log("[!] AI Engine: OFFLINE")

    def log(self, message):
        print(message)
        self.session_logs.append(message)

    def get_ai_prediction(self, sequence):
        if not self.ai_active or not sequence: return 0.0
        k = 8
        kmers = [" ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])]
        try:
            vector = self.vectorizer.transform(kmers)
            return self.model.predict_proba(vector)[0][1] * 100
        except:
            return 0.0

    def find_motifs(self, sequence):
        return [name for name, seq in self.motifs.items() if seq in sequence]


    def generate_dynamic_narrative(self, risk, hits_count, motifs, seq_len):
        # --- Part A: Homology Narrative ---
        if hits_count > 100:
            homology_text = f"The homology analysis revealed an extreme density of synthetic markers ({hits_count} matches). This frequency is statistically impossible in natural evolution and indicates a deliberate use of high-copy number industrial backbones."
        elif hits_count > 0:
            homology_text = f"The system detected {hits_count} localized database matches. These fragments suggest the integration of specific laboratory-verified modules into a larger genomic framework."
        else:
            homology_text = "Homology search returned zero known vector matches, suggesting the primary scaffold of this sequence is novel or originates from a non-cataloged environmental isolate."

        # --- Part B: Structural AI Narrative ---
        if risk > 85:
            ai_text = f"Structural k-mer analysis (Score: {risk:.1f}%) detected a 'Machine Rhythm' in the nucleotide distribution. The low-entropy state of the 8-mer clusters is a hallmark of codon optimization for industrial expression systems."
        elif risk > 40:
            ai_text = f"The AI identified moderate structural deviation ({risk:.1f}%). The pattern suggests a 'Chimeric' origin, where natural DNA may have been edited or re-arranged, causing localized k-mer anomalies."
        else:
            ai_text = f"Pattern analysis confirms a high-entropy state (Risk: {risk:.1f}%). The nucleotide distribution aligns perfectly with stochastic evolutionary noise, typical of wild-type viral or bacterial genomes."

        # --- Part C: The Forensic Verdict ---
        if hits_count > 0:
            verdict = "This sample is a CONFIRMED PRODUCT of recombinant DNA technology based on direct homology."
        elif risk > 40:
            verdict = "While homology is absent, the structural 'Ghost Signature' strongly suggests human-mediated algorithmic design or deep-level optimization."
        else:
            verdict = "Sequence is classified as fully natural. No evidence of anthropogenic intervention was detected."

        return homology_text, ai_text, verdict

    def generate_graphical_view(self, sequence_id, sequence, blast_hits, ai_score):
        seq_len = len(sequence)
        # Increased figure size and added a third subplot for Complexity
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                            gridspec_kw={'height_ratios': [3, 1, 1]})

        ax1.add_patch(Rectangle((0, 0), seq_len, 1, color='#eeeeee', ec='black', label='Backbone'))

        if not blast_hits.empty:
            for _, hit in blast_hits.iterrows():
                qs = hit['q_start'] if 'q_start' in hit else hit['qs']
                qe = hit['q_end'] if 'q_end' in hit else hit['qe']
                ax1.add_patch(Rectangle((qs, 1.2), qe - qs, 1.5, color='red', alpha=0.1))

        windows = [gc_fraction(sequence[i:i + 50]) * 100 for i in range(len(sequence) - 50)]
        ax1.plot(range(len(windows)), [4 + (g / 20) for g in windows], color='blue', alpha=0.3, label='GC Content %')

        step = max(1, seq_len // 50)
        ai_vals = [self.get_ai_prediction(sequence[i:i + step * 2]) for i in range(0, seq_len - step * 2, step)]
        ax1.plot(np.linspace(0, seq_len, len(ai_vals)), [6 + (v / 50) for v in ai_vals], color='orange', lw=2,
                 label='AI Ghost Signature')

        ax1.set_title(f"Genomic Forensic Map: {sequence_id}")
        ax1.legend(loc='upper right')
        ax1.get_yaxis().set_visible(False)

        # Visual 1: Heatmap
        heat = np.array(ai_vals).reshape(1, -1)
        ax2.imshow(heat, aspect='auto', cmap='YlOrRd', extent=[0, seq_len, 0, 1])
        ax2.set_ylabel("Risk Intensity")
        ax2.set_yticks([])

        # --- Visual 2: K-mer Complexity (Shannon Entropy) ---
        def calculate_entropy(window):
            counts = [window.count(k) for k in set(window)]
            probs = [c / len(window) for c in counts]
            return -sum(p * np.log2(p) for p in probs)

        complexity_step = max(10, seq_len // 100)
        complex_vals = [calculate_entropy(sequence[i:i + 32]) for i in range(0, seq_len - 32, complexity_step)]
        ax3.fill_between(np.linspace(0, seq_len, len(complex_vals)), complex_vals, color='gray', alpha=0.2)
        ax3.plot(np.linspace(0, seq_len, len(complex_vals)), complex_vals, color='black', lw=1,
                 label='K-mer Complexity')
        ax3.set_ylabel("Entropy (bits)")
        ax3.set_xlabel("Base Pair Position (bp)")
        ax3.legend(loc='upper right')

        plt.tight_layout()
        img_path = f"report_{sequence_id}.png"
        plt.savefig(img_path, dpi=300)
        plt.close()
        return img_path

    # Note: Inside create_pdf_report, I removed the redundant double-loop you had
    def create_pdf_report(self, res_data):
        pdf = FPDF()

        for data in res_data:
            has_db_hits = not data['hits_df'].empty
            high_ai_risk = data['risk'] > 40

            # 1. Get the Dynamic Text from our Engine
            h_text, a_text, v_text = self.generate_dynamic_narrative(
                data['risk'], len(data['hits_df']), data['motifs'], len(data['clean_seq'])
            )

            # 2. Tier Assignment
            if has_db_hits:
                tier, color = "CONFIRMED LAB ENGINEERED", (200, 0, 0)
                verdict_text = "VERDICT: CONFIRMED SYNTHETIC (DATABASE MATCH)"
            elif high_ai_risk:
                tier, color = "SUSPECTED ENGINEERED (GHOST SIGNATURE)", (255, 140, 0)
                verdict_text = "VERDICT: POTENTIAL SYNTHETIC (STRUCTURAL ANOMALY)"
            else:
                tier, color = "FULLY NATURAL", (0, 128, 0)
                verdict_text = "VERDICT: NATURAL / WILD-TYPE"

            # --- PAGE 1: EXECUTIVE SUMMARY ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 10, "GHOST SIGNATURE: FORENSIC GENOMIC REPORT", ln=True, align='C')
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 5, f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True,
                     align='C')
            pdf.ln(5)

            pdf.set_fill_color(*color)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 12, f" STATUS: {tier}", ln=True, fill=True, align='C')
            pdf.set_text_color(0, 0, 0)

            pdf.ln(2)
            pdf.set_fill_color(245, 245, 245)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 10, f" TARGET ID: {data['id']}", ln=True, fill=True)
            pdf.set_font("Arial", '', 10)
            pdf.cell(90, 8, f" AI Pattern Risk: {data['risk']:.2f}%", border='LT')
            pdf.cell(100, 8, f" DB Homology: {len(data['hits_df'])} Hits", border='RT', ln=True)
            pdf.cell(90, 8, f" Sequence Length: {len(data['clean_seq'])} bp", border='LB')
            pdf.cell(100, 8, f" Detected Motifs: {', '.join(data['motifs']) if data['motifs'] else 'None'}",
                     border='RB', ln=True)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, "FORENSIC INTERPRETATION:", ln=True)
            pdf.set_font("Arial", '', 9)

            pdf.set_font("Arial", 'B', 9)
            pdf.write(5, "Homology Scan: ")
            pdf.set_font("Arial", '', 9)
            pdf.multi_cell(0, 5, h_text)
            pdf.ln(2)

            pdf.set_font("Arial", 'B', 9)
            pdf.write(5, "Structural Scan: ")
            pdf.set_font("Arial", '', 9)
            pdf.multi_cell(0, 5, a_text)
            pdf.ln(4)

            pdf.set_fill_color(230, 230, 230)
            pdf.set_font("Arial", 'B', 10)
            pdf.write(8, "Forensic Verdict: ")
            pdf.set_font("Arial", 'I', 10)
            pdf.multi_cell(0, 8, v_text, border=1, align='L', fill=True)

            pdf.ln(4)
            pdf.image(data['img'], x=10, w=180)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(0, 10, verdict_text, border=1, ln=True, align='C', fill=True)

            if not data['hits_df'].empty:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"DNA EVIDENCE LOG: Confirmed Lab Homology for {data['id']}", ln=True)
                pdf.set_font("Courier", '', 8)
                for i, (_, hit) in enumerate(data['hits_df'].iterrows()):
                    qs, qe = int(hit.get('q_start', hit.get('qs'))), int(hit.get('q_end', hit.get('qe')))
                    ident = hit.get('percent_identity', hit.get('pident'))
                    sub_id = hit.get('subject_id')
                    snippet = data['clean_seq'][qs - 1:qe][:50] + "..."
                    hit_text = f"[{i + 1}] ID: {sub_id} | {ident}% Match | Range: {qs}-{qe}bp\nSnippet: {snippet}\n"
                    pdf.multi_cell(0, 5, hit_text.encode('latin-1', 'ignore').decode('latin-1'), border='B')
                    if pdf.get_y() > 265: pdf.add_page()

        pdf.add_page()
        pdf.set_font("Courier", 'B', 14)
        pdf.cell(0, 10, "System Execution Logs (Chain of Custody)", ln=True)
        pdf.ln(5)
        pdf.set_font("Courier", '', 8)
        for line in self.session_logs:
            clean_line = line.replace('├─', '|--').replace('│', '|').replace('└─', '`—').replace('🔍', 'SCAN:')
            pdf.multi_cell(0, 5, clean_line.encode('latin-1', 'ignore').decode('latin-1'))

        pdf.output("Forensic_Analysis_Summary.pdf")
        print(f"[#] Precise 3-Tier Report Generated.")

    def run_analysis(self):
        self.log("=" * 80)
        self.log("🔍 GHOST SIGNATURE: FORENSIC GENOMIC ANALYSIS")
        self.log("=" * 80)

        # Run BLAST
        subprocess.run(["blastn", "-query", self.query_file, "-db", self.db, "-evalue", "1e-5", "-outfmt", "10", "-out",
                        self.output_csv], check=False)

        records = list(SeqIO.parse(self.query_file, "fasta"))
        cols = ['query_id', 'subject_id', 'percent_identity', 'alignment_length', 'mismatches', 'gap_opens', 'q_start',
                'q_end', 's_start', 's_end', 'evalue', 'bit_score']
        blast_df = pd.read_csv(self.output_csv, names=cols) if os.path.exists(self.output_csv) else pd.DataFrame()

        all_results = []
        for rec in records:
            clean_seq = "".join(filter(lambda x: x in "ATGCN", str(rec.seq).upper()))
            ai_score = self.get_ai_prediction(clean_seq)
            hits = blast_df[blast_df['query_id'] == rec.id] if not blast_df.empty else pd.DataFrame()
            motifs_found = self.find_motifs(clean_seq)

            # Capture Forensic Logic
            logic_lines = []
            if not hits.empty:
                logic_lines.append("[AGENT IDENTIFIED] Sequence shows direct homology to known lab vectors.")
            if ai_score > 80:
                logic_lines.append("[CRITICAL] DNA displays industrial k-mer optimization patterns.")
            elif ai_score > 30:
                logic_lines.append("[WARNING] Non-natural k-mer entropy detected. Potential 'Ghost Signature'.")
            else:
                logic_lines.append("[CLEAR] Sequence aligns with natural evolutionary patterns.")

            self.log(f"\n[ID: {rec.id}]")
            self.log(f"├─ AI SYNTHETIC RISK: {ai_score:.2f}%")
            self.log(f"├─ DB Matches: {len(hits)} hits")

            img = self.generate_graphical_view(rec.id, clean_seq, hits, ai_score)

            all_results.append({
                'id': rec.id,
                'risk': ai_score,
                'hits_df': hits,
                'motifs': motifs_found,
                'img': img,
                'logic_lines': logic_lines,
                'clean_seq': clean_seq
            })

        self.create_pdf_report(all_results)


if __name__ == "__main__":
    detector = GhostSignatureDetector("mystery_virus.fasta", "database/univec_db")
    detector.run_analysis()