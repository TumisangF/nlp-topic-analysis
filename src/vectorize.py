"""
vectorize.py

Text vectorization module for NLP topic modelling pipeline.
Implements two complementary approaches:

1. Bag-of-Words (CountVectorizer)
   - Represents each document as raw term counts
   - Interpretable and directly compatible with LDA
   - Does not account for how common a term is across the corpus

2. TF-IDF (TfidfVectorizer)
   - Weights term frequency against document frequency
   - Downweights common terms, upweights distinctive ones
   - Better suited for NMF and similarity-based tasks

Both vectorizers use the same vocabulary parameters for a fair comparison.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz

# -------------------------------
# Configuration
# -------------------------------
DATA_PATH = "../data/complaints_clean.csv"
MODELS_DIR = "../models"

# Shared vocabulary parameters — applied to both vectorizers for fair comparison
MAX_FEATURES = 10000   # Retain only the top N terms by frequency
MIN_DF = 5             # Ignore terms appearing in fewer than 5 documents (noise filter)
MAX_DF = 0.95          # Ignore terms appearing in >95% of documents (too generic)

os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------
# Load and Deduplicate Data
# -------------------------------
df = pd.read_csv(DATA_PATH)
df = df[df["processed_text"].notna()]
df["processed_text"] = df["processed_text"].astype(str).str.strip()
df = df[df["processed_text"] != ""]

before = len(df)
df = df.drop_duplicates(subset=["processed_text"])
print(f"Removed {before - len(df)} duplicate documents. Remaining: {len(df)}")

texts = df["processed_text"].tolist()

# -------------------------------
# 1. Bag-of-Words Vectorization
# -------------------------------
bow_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF
)

bow_matrix = bow_vectorizer.fit_transform(texts)

# Save BoW artifacts
save_npz(f"{MODELS_DIR}/bow_matrix.npz", bow_matrix)

with open(f"{MODELS_DIR}/bow_vectorizer.pkl", "wb") as f:
    pickle.dump(bow_vectorizer, f)

# Save feature names for interpretability in topic modelling
bow_features = bow_vectorizer.get_feature_names_out()
pd.Series(bow_features).to_csv(f"{MODELS_DIR}/bow_feature_names.csv", index=False)

print("\n--- Bag-of-Words ---")
print(f"Matrix shape : {bow_matrix.shape}")
print(f"Stored values: {bow_matrix.nnz:,}  (non-zero entries)")
print(f"Sparsity     : {1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]):.4%}")
print(f"Sample terms : {list(bow_features[:10])}")

# -------------------------------
# 2. TF-IDF Vectorization
# -------------------------------
tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Save TF-IDF artifacts
save_npz(f"{MODELS_DIR}/tfidf_matrix.npz", tfidf_matrix)

with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

tfidf_features = tfidf_vectorizer.get_feature_names_out()
pd.Series(tfidf_features).to_csv(f"{MODELS_DIR}/tfidf_feature_names.csv", index=False)

print("\n--- TF-IDF ---")
print(f"Matrix shape : {tfidf_matrix.shape}")
print(f"Stored values: {tfidf_matrix.nnz:,}  (non-zero entries)")
print(f"Sparsity     : {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4%}")
print(f"Sample terms : {list(tfidf_features[:10])}")

# -------------------------------
# Comparison Summary
# -------------------------------
print("\n--- Comparison ---")
print(f"{'Metric':<30} {'BoW':>15} {'TF-IDF':>15}")
print("-" * 62)
print(f"{'Documents':<30} {bow_matrix.shape[0]:>15,} {tfidf_matrix.shape[0]:>15,}")
print(f"{'Vocabulary size':<30} {bow_matrix.shape[1]:>15,} {tfidf_matrix.shape[1]:>15,}")
print(f"{'Non-zero entries':<30} {bow_matrix.nnz:>15,} {tfidf_matrix.nnz:>15,}")
print(f"{'Value type':<30} {'Integer counts':>15} {'Float weights':>15}")
print(f"{'Downstream use':<30} {'LDA':>15} {'NMF':>15}")
print("\nNote: BoW preserves raw term counts, making it the standard")
print("input for LDA's probabilistic model. TF-IDF weights distinctive")
print("terms more heavily, which benefits NMF's matrix factorisation.")
