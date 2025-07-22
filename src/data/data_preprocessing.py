import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in str(text).split() if word not in stop_words]
    return " ".join(filtered)

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame, column: str = "content") -> pd.DataFrame:
    """Set text to NaN if sentence has fewer than 3 words."""
    df[column] = df[column].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
    return df

def normalize_text(df: pd.DataFrame, column: str = "content") -> pd.DataFrame:
    """Apply all preprocessing steps to the specified column of the DataFrame."""
    try:
        df[column] = df[column].astype(str).apply(lower_case)
        df[column] = df[column].apply(remove_stop_words)
        df[column] = df[column].apply(removing_numbers)
        df[column] = df[column].apply(removing_punctuations)
        df[column] = df[column].apply(removing_urls)
        df[column] = df[column].apply(lemmatization)
        df = remove_small_sentences(df, column)
        logging.info(f"Text normalization completed for column '{column}'.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        raise

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Failed to save data to {path}: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data preprocessing."""
    try:
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")

        train_data = normalize_text(train_data, column="content")
        test_data = normalize_text(test_data, column="content")

        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Data preprocessing pipeline failed: {e}")

if __name__ == "__main__":
    main()