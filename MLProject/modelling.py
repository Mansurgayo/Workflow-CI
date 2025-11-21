import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import os

def load_data(path: str) -> pd.DataFrame:
    """Load dataset dari CSV"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {path}")
    df = pd.read_csv(path)
    return df

def train_model(df: pd.DataFrame) -> tuple:
    """Latih model RandomForest dan hitung akurasi"""
    # Kolom target pada dataset adalah "Outcome"
    if "Outcome" not in df.columns:
        raise ValueError("Kolom 'Outcome' tidak ditemukan di dataset")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

def main(data_path: str):
    """Main function untuk MLflow run"""
    mlflow.sklearn.autolog()  # Auto-log parameter & metrics

    with mlflow.start_run():
        df = load_data(data_path)
        model, acc = train_model(df)

        print(f"Accuracy: {acc}")

        # Log metric secara manual (optional, karena autolog juga sudah mencatatnya)
        mlflow.log_metric("accuracy", acc)

        # Log model ke folder 'model' untuk artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest model and log to MLflow")
    parser.add_argument("--data_path", type=str, required=True, help="Path ke CSV dataset")
    args = parser.parse_args()

    main(args.data_path)
