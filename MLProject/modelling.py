import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ⚡ LOAD DATASET ASLI ⚡
# Path relatif dari lokasi script
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "cleaned_dataset.csv")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded from {DATA_PATH}")
except FileNotFoundError:
    print(f"❌ Dataset not found at {DATA_PATH}. Workflow failed.")
    sys.exit(1)  # Stop script jika dataset tidak ditemukan

# Pisahkan fitur dan target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment MLflow
mlflow.set_experiment("Workflow_CI_Experiment")

# Training dan logging model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metric dan model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")
    print(f"✅ Model training completed. Accuracy: {acc}")
