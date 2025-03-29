import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from collections import Counter

class DataPreprocessor:
    def __init__(self, remove_stopwords=True, use_stemming=False):
        """Initialize with preprocessing options, no length or vocab constraints yet."""
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.word_to_idx = {}  # For vectorization
        self.idx_to_word = {}  # For reverse mapping
    
    def clean_text(self, text):
        """Thoroughly clean a single text string."""
        if not isinstance(text, str) or pd.isna(text):  # Handle NaN or non-strings
            return ""
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.[a-zA-Z0-9\-.]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text and apply optional stemming/stopword removal."""
        tokens = word_tokenize(text)
        if self.stop_words:
            tokens = [word for word in tokens if word not in self.stop_words]
        if self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]
        return tokens
    
    def process_text(self, text):
        """Process a single text string up to tokenization."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens
    
    def process_dataframe(self, df):
        """Process a DataFrame with 'text' and 'label' columns."""
        if not isinstance(df, pd.DataFrame) or 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("DataFrame must have 'text' and 'label' columns")
        df_processed = df.copy()
        df_processed['processed_text'] = df_processed['text'].apply(self.process_text)
        return df_processed[['processed_text', 'label']]
    
    def process(self, data):
        """Universal entry point: handle DataFrame or single string."""
        if isinstance(data, pd.DataFrame):
            return self.process_dataframe(data)
        elif isinstance(data, str):
            return self.process_text(data)
        elif isinstance(data, (list, tuple)) and all(isinstance(d, pd.DataFrame) for d in data):
            return tuple(self.process_dataframe(df) for df in data)
        else:
            raise ValueError("Input must be a DataFrame, string, or list/tuple of DataFrames")
    
    def build_vocabulary(self, tokenized_texts, vocab_size):
        """Build vocabulary from tokenized sequences with a specified size."""
        all_tokens = [token for seq in tokenized_texts for token in seq]
        word_counts = Counter(all_tokens).most_common(vocab_size)
        self.word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(word_counts)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def vectorize(self, tokenized_texts, max_length):
        """Convert tokenized sequences to indices with padding, using specified max_length."""
        sequences = []
        for tokens in tokenized_texts:
            seq = [self.word_to_idx.get(token, 0) for token in tokens[:max_length]]
            if len(seq) < max_length:
                seq.extend([0] * (max_length - len(seq)))
            sequences.append(seq)
        return np.array(sequences)
    
    def process_and_vectorize(self, data, max_length, vocab_size):
        """Full pipeline for later: process, build vocab, and vectorize."""
        processed = self.process(data)
        if isinstance(processed, pd.DataFrame):
            self.build_vocabulary(processed['processed_text'], vocab_size)
            vectors = self.vectorize(processed['processed_text'], max_length)
            return vectors, np.array(processed['label'])
        elif isinstance(processed, list):
            if not self.word_to_idx:
                raise ValueError("Vocabulary not built yet; run on training data first")
            vectors = self.vectorize([processed], max_length)
            return vectors
        elif isinstance(processed, tuple):
            self.build_vocabulary(processed[0]['processed_text'], vocab_size)
            return tuple((self.vectorize(df['processed_text'], max_length), np.array(df['label'])) for df in processed)