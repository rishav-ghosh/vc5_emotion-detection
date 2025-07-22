import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
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

def load_data(url: str) -> pd.DataFrame:
    """Load dataset from a URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Data loaded from {url} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame for binary sentiment classification."""
    try:
        # Remove unnecessary columns
        if 'tweet_id' in df.columns:
            df = df.drop(columns=['tweet_id'])
            logging.info("Dropped 'tweet_id' column.")
        # Filter for 'happiness' and 'sadness'
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        logging.info(f"Filtered DataFrame shape: {df.shape}")
        # Encode sentiments
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Encoded 'sentiment' column.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Split data: train shape {train_data.shape}, test shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """Save train and test DataFrames to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train and test data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_data(url)
        processed_df = preprocess_data(df)
        train_data, test_data = split_data(processed_df, test_size)
        save_data(train_data, test_data, 'data/raw')
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()