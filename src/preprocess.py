"""
preprocess.py

Text preprocessing pipeline for NLP tasks with enhanced cleaning.
This module:
- Selects the correct text column
- Handles missing values explicitly
- Cleans raw text with enhanced patterns
- Adds domain-specific stopwords
- Tokenizes and normalizes text
- Removes noise (punctuation, numbers, stopwords)
- Lemmatizes tokens to reduce vocabulary size
- Outputs clean text ready for vectorization or modeling
"""

import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class TextPreprocessor:
    def __init__(self, text_column: str, language: str = "english"):
        """
        Parameters
        ----------
        text_column : str
            Name of the column containing raw text
        language : str
            Language used for stopwords
        """
        self.text_column = text_column
        self.stop_words = set(stopwords.words(language))
        
        # Add domain-specific stopwords common in consumer complaints
        domain_stopwords = {
            'xxxxxxx', 'xx', 'xxx', 'amp', 'company', 'consumer', 'complaint',
            'customer', 'representative', 'department', 'office', 'represent',
            'please', 'contact', 'issue', 'problem', 'request', 'respond',
            'response', 'resolution', 'would', 'could', 'should', 'said', 
            'told', 'called', 'asked', 'thank', 'thanks', 'dear', 'hello', 'hi',
            'got', 'get', 'make', 'made', 'see', 'look', 'come', 'go', 'tell',
            'one', 'two', 'time', 'day', 'week', 'month', 'year', 'also', 'well'
        }
        self.stop_words.update(domain_stopwords)
        
        self.tokenizer = RegexpTokenizer(r"\b[a-zA-Z]{3,}\b")
        self.lemmatizer = WordNetLemmatizer()

    def clean_text_enhanced(self, text: str) -> str:
        """
        Perform enhanced text cleaning optimized for coherence.

        Steps:
        - Lowercase
        - Remove URLs and email addresses
        - Replace numbers with space (preserves word boundaries)
        - Remove punctuation more effectively
        - Collapse whitespace
        - Filter by word length (3-20 chars)
        """
        if pd.isna(text):
            return ""

        text = text.lower()

        # Remove URLs and emails
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # Replace numbers with space (to separate words)
        text = re.sub(r"\d+", " ", text)

        # Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove very short and very long words
        text = " ".join([w for w in text.split() if 3 <= len(w) <= 20])

        return text

    def tokenize_filter_lemmatize(self, text: str) -> str:
        """
        Tokenize text, remove stopwords, and lemmatize each token.

        Lemmatization reduces vocabulary size by collapsing inflected forms
        (e.g. 'charging', 'charged', 'charges' → 'charge'), which leads to
        cleaner topic representations and better coherence scores.
        """
        if not text:
            return ""
        
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, keep_clean: bool = True) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with raw text column
        keep_clean : bool
            If True, retain the intermediate 'clean_text' column for debugging.
            Set to False for production outputs.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'clean_text' (optional) and 'processed_text' columns
        """
        if self.text_column not in df.columns:
            raise ValueError(
                f"Column '{self.text_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()

        # Explicitly remove rows with missing narratives
        initial_size = len(df)
        df = df[df[self.text_column].notna()]
        print(f"Dropped {initial_size - len(df)} rows with missing narratives.")

        # Step 1: Enhanced cleaning
        df["clean_text"] = df[self.text_column].apply(self.clean_text_enhanced)

        # Step 2: Tokenize, remove stopwords, and lemmatize
        df["processed_text"] = df["clean_text"].apply(self.tokenize_filter_lemmatize)

        # Step 3: Remove documents that are empty after preprocessing
        before = len(df)
        df = df[df["processed_text"].str.strip() != ""]
        print(f"Dropped {before - len(df)} empty documents after preprocessing.")

        df.reset_index(drop=True, inplace=True)

        # Optionally drop intermediate clean_text column
        if not keep_clean:
            df.drop(columns=["clean_text"], inplace=True)

        return df


if __name__ == "__main__":

    FILE_PATH = "../data/data.csv"
    OUTPUT_PATH = "../data/complaints_clean.csv"
    TEXT_COLUMN = "Consumer complaint narrative"

    df = pd.read_csv(FILE_PATH, low_memory=False)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    preprocessor = TextPreprocessor(text_column=TEXT_COLUMN)

    df_processed = preprocessor.preprocess_dataframe(df, keep_clean=True)

    # Save processed text
    df_final = df_processed[["clean_text", "processed_text"]]
    df_final = df_final[df_final["processed_text"].str.strip() != ""]

    df_final.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing complete")
    print(f"Final dataset size: {df_final.shape}")
    print("\nSample output:")
    print(df_final.head(3).to_string())