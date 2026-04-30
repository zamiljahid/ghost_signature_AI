import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(file, label, k=8, chunk=500):
    dataset = []
    for rec in SeqIO.parse(file, "fasta"):
        s = "".join(filter(lambda x: x in "ATGCN", str(rec.seq).upper()))
        for i in range(0, len(s), chunk):
            frag = s[i:i+chunk]
            if len(frag) < k: continue
            kmers = [frag[j:j+k] for j in range(len(frag)-k+1)]
            dataset.append({"text": " ".join(kmers), "label": label})
    return dataset

print("[#] Training AI Engine...")
data = load_data("vectors.fasta", 1) + load_data("natural_viruses.fasta", 0)
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(ngram_range=(1,1))
X = vectorizer.fit_transform(df['text'])
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, df['label'])

joblib.dump(model, 'ghost_model.pkl')
joblib.dump(vectorizer, 'dna_vectorizer.pkl')
print("--- Training Ready ---")