import pandas as pd
import pickle
import json
import logging
from typing import Any, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(path: str) -> Any:
    """Load a trained model from a file."""
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {path}: {e}")
        raise

def load_data(path: str) -> pd.DataFrame:
    """Load test data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Test data loaded from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load test data from {path}: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save metrics to {path}: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_data("data/interim/test_bow.csv")
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()