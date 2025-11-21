import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("namadataset_preprocessing/data.csv")

# Sesuaikan kolom target
X = df.drop("Outcome", axis=1)  # Fitur
y = df["Outcome"]               # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment MLflow
mlflow.set_experiment("Workflow_CI_Experiment")

with mlflow.start_run():
    # Buat model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log metric dan model ke MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Model accuracy: {acc}")
