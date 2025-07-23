import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Train data loaded from {train_path} with shape {train_data.shape}")
        logging.info(f"Test data loaded from {test_path} with shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def extract_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract Bag of Words features and return feature DataFrames."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logging.info("Feature extraction completed.")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        raise

def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save feature DataFrames to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_tfidf.csv")
        test_path = os.path.join(output_dir, "test_tfidf.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info(f"Features saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving features: {e}")
        raise

def main() -> None:
    """Main function to orchestrate feature extraction."""
    try:
        params = load_params('params.yaml')
        max_features = params['features']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        train_df, test_df = extract_features(train_data, test_data, max_features)
        save_features(train_df, test_df, "data/interim")
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Feature engineering pipeline failed: {e}")

if __name__ == "__main__":
    main()