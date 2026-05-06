"""
vectorize.py

Text vectorization module for NLP topic modelling pipeline.
Implements two complementary approaches with bigrams for better coherence:

1. Bag-of-Words (CountVectorizer)
   - Represents each document as raw term counts
   - Interpretable and directly compatible with LDA
   - Now includes bigrams (1-2) for better phrase capture

2. TF-IDF (TfidfVectorizer)
   - Weights term frequency against document frequency
   - Downweights common terms, upweights distinctive ones
   - Better suited for NMF and similarity-based tasks
   - Includes sublinear scaling and bigrams

Bigrams significantly improve coherence scores by capturing common phrases
like "credit card", "identity theft", and "late payment".
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz

# -------------------------------
# Configuration
# -------------------------------
DATA_PATH = "../data/complaints_clean.csv"
MODELS_DIR = "../models"

# Optimized vocabulary parameters for better coherence
MAX_FEATURES = 15000   # Increased from 10000 for better coverage
MIN_DF = 3             # Lowered from 5 to capture more meaningful terms
MAX_DF = 0.7           # Stricter from 0.95 to remove more common words
NGRAM_RANGE = (1, 2)   # Include bigrams for phrase capture

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
# 1. Bag-of-Words Vectorization (with bigrams)
# -------------------------------
bow_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=NGRAM_RANGE
)

bow_matrix = bow_vectorizer.fit_transform(texts)

# Save BoW artifacts
save_npz(f"{MODELS_DIR}/bow_matrix.npz", bow_matrix)

with open(f"{MODELS_DIR}/bow_vectorizer.pkl", "wb") as f:
    pickle.dump(bow_vectorizer, f)

# Save feature names for interpretability in topic modelling
bow_features = bow_vectorizer.get_feature_names_out()
pd.Series(bow_features).to_csv(f"{MODELS_DIR}/bow_feature_names.csv", index=False)

print("\n--- Bag-of-Words (with bigrams) ---")
print(f"Matrix shape : {bow_matrix.shape}")
print(f"Stored values: {bow_matrix.nnz:,}  (non-zero entries)")
print(f"Sparsity     : {1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]):.4%}")
print(f"Sample unigrams : {[w for w in bow_features[:10] if ' ' not in w]}")
print(f"Sample bigrams   : {[w for w in bow_features if ' ' in w][:10]}")

# -------------------------------
# 2. TF-IDF Vectorization (with sublinear scaling and bigrams)
# -------------------------------
tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=NGRAM_RANGE,
    sublinear_tf=True,      # Apply 1+log(tf) scaling
    strip_accents='unicode'
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Save TF-IDF artifacts
save_npz(f"{MODELS_DIR}/tfidf_matrix.npz", tfidf_matrix)

with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

tfidf_features = tfidf_vectorizer.get_feature_names_out()
pd.Series(tfidf_features).to_csv(f"{MODELS_DIR}/tfidf_feature_names.csv", index=False)

print("\n--- TF-IDF (with sublinear scaling and bigrams) ---")
print(f"Matrix shape : {tfidf_matrix.shape}")
print(f"Stored values: {tfidf_matrix.nnz:,}  (non-zero entries)")
print(f"Sparsity     : {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4%}")
print(f"Sample unigrams : {[w for w in tfidf_features[:10] if ' ' not in w]}")
print(f"Sample bigrams   : {[w for w in tfidf_features if ' ' in w][:10]}")

# -------------------------------
# Comparison Summary
# -------------------------------
print("\n--- Comparison ---")
print(f"{'Metric':<35} {'BoW':>15} {'TF-IDF':>15}")
print("-" * 66)
print(f"{'Documents':<35} {bow_matrix.shape[0]:>15,} {tfidf_matrix.shape[0]:>15,}")
print(f"{'Vocabulary size':<35} {bow_matrix.shape[1]:>15,} {tfidf_matrix.shape[1]:>15,}")
print(f"{'Non-zero entries':<35} {bow_matrix.nnz:>15,} {tfidf_matrix.nnz:>15,}")
print(f"{'Value type':<35} {'Integer counts':>15} {'Float weights':>15}")
print(f"{'N-gram range':<35} {'(1,2) unigrams+bigrams':>15} {'(1,2) unigrams+bigrams':>15}")
print(f"{'Downstream use':<35} {'LDA':>15} {'NMF':>15}")
print("\nNote: Bigrams significantly improve coherence by capturing common phrases.")
print("TF-IDF weights distinctive terms more heavily, which benefits NMF's matrix factorisation.")