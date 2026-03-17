"""
preprocess.py

Text preprocessing pipeline for NLP tasks.
This module:
- Selects the correct text column
- Handles missing values explicitly
- Cleans raw text (including CFPB-style redaction tokens like XXXX)
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
        self.tokenizer = RegexpTokenizer(r"\b[a-zA-Z]{3,}\b")
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Perform basic text cleaning.

        Steps:
        - Lowercase
        - Remove URLs and email addresses
        - Remove CFPB-style redaction tokens (e.g. XXXX, XX/XX/XXXX)
        - Remove digits
        - Remove punctuation
        - Collapse whitespace
        """
        if pd.isna(text):
            return ""

        text = text.lower()

        # Remove URLs and emails
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # Remove CFPB redaction tokens: sequences of x's, optionally separated by slashes
        # e.g. "XXXX", "XX/XX/XXXX", "xxxx", "xx/xx/xxxx"
        text = re.sub(r"\b[x]+(?:/[x]+)*\b", "", text)

        # Remove digits
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_filter_lemmatize(self, text: str) -> str:
        """
        Tokenize text, remove stopwords, and lemmatize each token.

        Lemmatization reduces vocabulary size by collapsing inflected forms
        (e.g. 'charging', 'charged', 'charges' → 'charge'), which leads to
        cleaner topic representations and better coherence scores.
        """
        tokens = self.tokenizer.tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
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

        # Step 1: Clean raw text
        df["clean_text"] = df[self.text_column].apply(self.clean_text)

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

    # keep_clean=True retains 'clean_text' alongside 'processed_text' for debugging.
    # Switch to keep_clean=False before final model runs.
    df_processed = preprocessor.preprocess_dataframe(df, keep_clean=True)

    # Save both clean_text and processed_text for inspection
    df_final = df_processed[["clean_text", "processed_text"]]
    df_final = df_final[df_final["processed_text"].str.strip() != ""]

    df_final.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing complete")
    print(f"Final dataset size: {df_final.shape}")
    print("\nSample output:")
    print(df_final.head(3).to_string())
