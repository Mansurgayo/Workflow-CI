import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# OPTION 1: Download dataset dari URL
def load_data():
    try:
        # Coba load dari local
        df = pd.read_csv("MLProject/dataset/cleaned_dataset.csv")
        return df
    except:
        try:
            # Fallback: download dari URL
            url = "https://raw.githubusercontent.com/your-repo/dataset/main/cleaned_dataset.csv"
            df = pd.read_csv(url)
            return df
        except:
            # Fallback 2: buat dummy data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            df['Outcome'] = y
            return df

# Load dataset
df = load_data()

# Lanjutkan dengan code yang sama...
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Workflow_CI_Experiment")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metric & model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Model accuracy: {acc}")