"""
nmf.py

Topic extraction using Non-negative Matrix Factorization (NMF).

This module:
- Loads the TF-IDF matrix and vocabulary from vectorize.py artifacts
- Searches over a range of topic numbers using coherence scores (c_v)
- Fits a final NMF model using the optimal number of topics
- Displays and saves the top terms per topic (12 words for better context)
- Persists the model and results for downstream use

Optimizations from notebook:
- Uses TF-IDF with bigrams (1-2)
- Uses 15 topics (optimal found)
- Achieves coherence score: 0.6467 C_v, 0.1575 C_npmi

Why TF-IDF for NMF:
    NMF factorizes the document-term matrix into two non-negative matrices
    (W: document-topic, H: topic-term). TF-IDF weights are preferred over
    raw counts because NMF benefits from the discrimination that TF-IDF
    provides — downweighting common terms makes topic boundaries sharper.

NMF vs LDA:
    LDA is probabilistic and uses raw counts (BoW). NMF is algebraic and
    works better with TF-IDF. NMF topics tend to be more focused and
    less noisy, but LDA provides a richer probabilistic interpretation.

Windows note:
    All execution code is inside if __name__ == "__main__" and
    processes=1 is set on CoherenceModel. Both are required because
    gensim internally spawns worker processes, which crashes on Windows
    unless the main guard is present.
"""

import os
import pickle
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for scripts on Windows
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.decomposition import NMF
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore")

# -------------------------------
# Configuration
# -------------------------------
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH         = os.path.join(BASE_DIR, "data", "complaints_clean.csv")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "models", "tfidf_matrix.npz")
TFIDF_VOCAB_PATH  = os.path.join(BASE_DIR, "models", "tfidf_feature_names.csv")
MODEL_DIR         = os.path.join(BASE_DIR, "models")
RESULTS_DIR       = os.path.join(BASE_DIR, "results")

TOPIC_RANGE  = [5, 10, 15, 20, 25]  # Tested range
N_TOP_WORDS  = 12  # Increased from 10 for better topic representation
RANDOM_STATE = 42
MAX_ITER     = 1000  # Increased for better convergence

# -------------------------------
# Helper Functions
# -------------------------------

def get_top_words(model, feature_names, n_top_words=12):
    """Return list of top-word lists for each topic."""
    topics = []
    for topic in model.components_:
        top_indices = topic.argsort()[-n_top_words:][::-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


def compute_coherence(topics, tokenized_texts, gensim_dict):
    """
    Compute c_v coherence score for a set of topics.

    c_v coherence measures semantic similarity between high-scoring words
    in each topic. Scores range from 0 to 1; higher is more coherent.
    processes=1 is required on Windows to avoid multiprocessing errors.
    """
    # Filter out any topics that might be too short
    valid_topics = [topic for topic in topics if len(topic) >= 5]
    
    if len(valid_topics) < len(topics):
        print(f"  Warning: Filtered out {len(topics) - len(valid_topics)} short topics")
    
    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_texts,
        dictionary=gensim_dict,
        coherence="c_v",
        processes=1,
    )
    return coherence_model.get_coherence()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------
    # Load Artifacts
    # ----------------------------
    print("Loading vectorization artifacts...")

    tfidf_matrix  = load_npz(TFIDF_MATRIX_PATH)
    feature_names = pd.read_csv(TFIDF_VOCAB_PATH).iloc[:, 0].tolist()

    df = pd.read_csv(DATA_PATH)
    df = df[df["processed_text"].notna()]
    df = df.drop_duplicates(subset=["processed_text"])
    tokenized_texts = [text.split() for text in df["processed_text"].astype(str)]

    gensim_dict = corpora.Dictionary(tokenized_texts)

    print(f"Loaded TF-IDF matrix  : {tfidf_matrix.shape}")
    print(f"Vocabulary size       : {len(feature_names):,}")
    print(f"Documents for scoring : {len(tokenized_texts):,}")

    # ----------------------------
    # Coherence Search Over Topic Range
    # ----------------------------
    print(f"\nSearching over topic range: {TOPIC_RANGE}")
    print("This may take several minutes...\n")

    coherence_scores = {}

    for n_topics in TOPIC_RANGE:
        print(f"  Fitting NMF with n_topics={n_topics}...", end=" ", flush=True)

        nmf = NMF(
            n_components=n_topics,
            random_state=RANDOM_STATE,
            init='random',      # Better for topic modeling
            max_iter=MAX_ITER,
            tol=1e-4
        )
        nmf.fit(tfidf_matrix)

        topics = get_top_words(nmf, feature_names, N_TOP_WORDS)
        score  = compute_coherence(topics, tokenized_texts, gensim_dict)
        coherence_scores[n_topics] = (score, nmf)

        print(f"coherence (c_v) = {score:.4f}")

    # ----------------------------
    # Select Optimal Number of Topics
    # ----------------------------
    optimal_n            = max(coherence_scores, key=lambda k: coherence_scores[k][0])
    best_score, best_nmf = coherence_scores[optimal_n]

    print(f"\nOptimal number of topics : {optimal_n}")
    print(f"Best coherence score     : {best_score:.4f}")

    # ----------------------------
    # Plot Coherence Scores
    # ----------------------------
    ns     = list(coherence_scores.keys())
    scores = [coherence_scores[n][0] for n in ns]

    plt.figure(figsize=(8, 4))
    plt.plot(ns, scores, marker="o", linewidth=2, markersize=8, color='#2E86AB')
    plt.axvline(x=optimal_n, color='red', linestyle='--', linewidth=2, label=f"Optimal n={optimal_n}")
    plt.title("NMF Coherence Score (c_v) by Number of Topics", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Topics", fontsize=11)
    plt.ylabel("Coherence Score (c_v)", fontsize=11)
    plt.xticks(ns)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nmf_coherence_plot.png"), dpi=150)
    plt.close()
    print(f"Coherence plot saved to {RESULTS_DIR}/nmf_coherence_plot.png")

    # ----------------------------
    # Display and Save Final Topics
    # ----------------------------
    final_topics = get_top_words(best_nmf, feature_names, N_TOP_WORDS)

    print(f"\n--- NMF Topics (n={optimal_n}) ---")
    lines = []
    for idx, words in enumerate(final_topics):
        line = f"Topic {idx + 1:>2}: {', '.join(words)}"
        print(line)
        lines.append(line)

    topics_path = os.path.join(RESULTS_DIR, "nmf_topics.txt")
    with open(topics_path, "w") as f:
        f.write("NMF Topic Modeling Results (Optimized)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Optimal n_topics   : {optimal_n}\n")
        f.write(f"Coherence (c_v)    : {best_score:.4f}\n")
        f.write(f"Top words per topic: {N_TOP_WORDS}\n")
        f.write(f"Features           : TF-IDF with bigrams (1-2)\n\n")
        f.write("--- Final Topics ---\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n--- Coherence Scores by Topic Count ---\n")
        for n, (s, _) in coherence_scores.items():
            marker = " <-- optimal" if n == optimal_n else ""
            f.write(f"  n={n:>2}: {s:.4f}{marker}\n")

    print(f"\nTopics saved to {topics_path}")

    # ----------------------------
    # Save Final Model
    # ----------------------------
    with open(os.path.join(MODEL_DIR, "nmf_model.pkl"), "wb") as f:
        pickle.dump(best_nmf, f)

    print(f"NMF model saved to {MODEL_DIR}/nmf_model.pkl")
    print("\nNMF pipeline complete.")
    print(f"Expected coherence: ~0.6467 (matching optimized notebook)")