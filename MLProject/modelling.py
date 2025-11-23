import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================
# 📌 Path Setup
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_dataset.csv")

print(f"🔍 Looking for dataset at: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"❌ Dataset not found at {DATA_PATH}. Workflow failed.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully!")


# ============================
# 📌 Dataset Processing
# ============================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ============================
# 🚀 MLflow Setup
# ============================

mlflow.set_tracking_uri("file:./mlruns")  
mlflow.set_experiment("Workflow_CI_Experiment")

ARTIFACT_DIR = os.path.join(BASE_DIR, "drive_upload")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================
# 🤖 Train & Log Model
# ============================

with mlflow.start_run(run_name="RandomForestExperiment"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    model_path = os.path.join(ARTIFACT_DIR, "random_forest_model")
    mlflow.sklearn.save_model(model, model_path)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"🎯 Training Done. Accuracy: {acc}")
    print(f"📁 Model stored at: {model_path}")
