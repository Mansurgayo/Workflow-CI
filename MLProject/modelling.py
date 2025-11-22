import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ⚡ LOAD DATASET ASLI ⚡
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "cleaned_dataset.csv")

if not os.path.exists(DATA_PATH):
    print(f"❌ Dataset not found at {DATA_PATH}. Workflow failed.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded from {DATA_PATH}")

# Pisahkan fitur dan target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment MLflow
mlflow.set_experiment("Workflow_CI_Experiment")

# Folder untuk menyimpan artifact (bisa diupload ke Drive nanti)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "drive_upload")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Training dan logging model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Simpan model ke folder artifact
    model_path = os.path.join(ARTIFACT_DIR, "random_forest_model")
    mlflow.sklearn.save_model(model, model_path)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"✅ Model training completed. Accuracy: {acc}")
    print(f"✅ Model saved to: {model_path}")
