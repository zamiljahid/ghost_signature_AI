import os
import subprocess
import pandas as pd
import joblib
from Bio import SeqIO


class GhostSignatureDetector:
    def __init__(self, query_file, database_path, model_path='ghost_model.pkl', vectorizer_path='dna_vectorizer.pkl'):
        self.query_file = query_file
        self.db = database_path
        self.output_csv = "ghost_signature_report.csv"
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.ai_active = True
            print("[#] AI Engine: ONLINE (Model and Vectorizer loaded)")
        except FileNotFoundError:
            print("[!] AI Engine: OFFLINE (Run trainer.py to generate .pkl files)")
            self.ai_active = False

    def get_ai_prediction(self, sequence):
        if not self.ai_active:
            return 0.0
        if len(sequence) < 150:
            raw_score = self._compute_raw_score(sequence)
            return round(raw_score * 0.5, 2)

        return self._compute_raw_score(sequence)

    def _compute_raw_score(self, sequence):
        k = 8
        kmers = [" ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])]
        try:
            vector = self.vectorizer.transform(kmers)
            return self.model.predict_proba(vector)[0][1] * 100
        except:
            return 0.0

    def run_blast(self):
        if not os.path.exists(self.query_file):
            print(f"Error: {self.query_file} not found.")
            return False

        print(f"\n--- Running BLAST Database Search ---")
        command = [
            "blastn",
            "-query", self.query_file,
            "-db", self.db,
            "-evalue", "1e-5",
            "-outfmt", "10",
            "-out", self.output_csv,
            "-num_threads", "8"
        ]

        try:
            subprocess.run(command, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: BLAST execution failed. Check if BLAST+ is in your system PATH.")
            return False

    def generate_combined_report(self):
        print(f"\n" + "=" * 50)
        print("FINAL GHOST SIGNATURE ANALYSIS REPORT")
        print("=" * 50)
        records = list(SeqIO.parse(self.query_file, "fasta-pearson"))

        if not records:
            print("[!] Warning: No sequences were detected in the input file.")

        for record in records:
            clean_seq = "".join(filter(lambda x: x in "ATGCN", str(record.seq).upper()))

            ai_score = self.get_ai_prediction(clean_seq)
            print(f"\nSEQUENCE ID: {record.id}")
            print(f"AI SYNTHETIC RISK SCORE: {ai_score}%")

            if ai_score > 80:
                print("STATUS: [!] CRITICAL - High probability of laboratory origin.")
            elif ai_score > 40:
                print("STATUS: [?] WARNING - Potential synthetic patterns detected.")
            else:
                print("STATUS: [+] CLEAR - Pattern appears naturally evolved.")
        columns = [
            'query_id', 'subject_id', 'percent_identity', 'alignment_length',
            'mismatches', 'gap_opens', 'q_start', 'q_end', 's_start', 's_end', 'evalue', 'bit_score'
        ]

        if os.path.exists(self.output_csv) and os.path.getsize(self.output_csv) > 0:
            df = pd.read_csv(self.output_csv, names=columns)
            df['risk_index'] = (df['percent_identity'] * (df['alignment_length'] / 50)).round(1)

            print(f"\nDATABASE HITS FOUND: {len(df)}")
            significant = df.sort_values(by='risk_index', ascending=False).head(5)
            print(significant[['subject_id', 'percent_identity', 'alignment_length', 'risk_index']].to_string(
                index=False))
        else:
            print("\nDATABASE HITS FOUND: 0 (No known vector matches)")

        print("=" * 50)


if __name__ == "__main__":
    QUERY = "mystery_virus.fasta"
    DATABASE = "database/univec_db"

    detector = GhostSignatureDetector(QUERY, DATABASE)

    if detector.run_blast():
        detector.generate_combined_report()
    else:
        print("Analysis stopped due to BLAST error.")