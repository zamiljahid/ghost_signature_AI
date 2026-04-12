import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def load_fasta_to_kmers(file_path, label, k=8, chunk_size=500):
    data = []
    print(f"Reading and Fragmenting {file_path}...")
    for record in SeqIO.parse(file_path, "fasta-pearson"):
        full_seq = "".join(filter(lambda x: x in "ATGCN", str(record.seq).upper()))
        for i in range(0, len(full_seq), chunk_size):
            chunk = full_seq[i: i + chunk_size]
            if len(chunk) < k:
                continue
            kmers = [chunk[j:j + k] for j in range(len(chunk) - k + 1)]
            data.append({"text": " ".join(kmers), "label": label})
    return data
print("Loading sequences...")
synthetic_data = load_fasta_to_kmers("vectors.fasta", 1)
natural_data = load_fasta_to_kmers("natural_viruses.fasta", 0)

if not synthetic_data or not natural_data:
    print("Error: One of your input files is empty or not found!")
    exit()
df = pd.DataFrame(synthetic_data + natural_data)
print(f"Vectorizing DNA (Synthetic: {len(synthetic_data)}, Natural: {len(natural_data)})...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
print("Teaching the model (Applying balanced weights)...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X, y)
joblib.dump(model, 'ghost_model.pkl')
joblib.dump(vectorizer, 'dna_vectorizer.pkl')

print("--- TRAINING COMPLETE ---")
print("Files 'ghost_model.pkl' and 'dna_vectorizer.pkl' are ready.")