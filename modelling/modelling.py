import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Any
from sklearn.ensemble import RandomForestClassifier

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

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int
) -> RandomForestClassifier:
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(x_train, y_train)
        logging.info("RandomForest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: Any, path: str) -> None:
    """Save the trained model to a file."""
    try:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model training and saving."""
    try:
        params = load_params('params.yaml')
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']

        train_data = load_data("data/interim/train_bow.csv")
        x_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        model = train_model(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")

if __name__ == "__main__":
    main()