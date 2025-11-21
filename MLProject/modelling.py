# ==============================
# modelling.py (VERSI MLFLOW PROJECT - KRITERIA 3 - FIXED)
# ==============================

import json
import os
import shutil 
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real

import mlflow
import mlflow.sklearn
import joblib

# --- IMPOR STABIL UNTUK LOGGING MANUAL ---
from mlflow.sklearn import get_default_conda_env, get_default_pip_requirements
from mlflow.models.model import Model
from mlflow.models.signature import infer_signature
# -----------------------------------------------

# ==========================
# DISABLE AUTOLOG
# ==========================
mlflow.sklearn.autolog(disable=True)

# ==========================
# FUNGSI ARTEFAK
# ==========================
def save_estimator_html(model, cols, out_path="estimator.html"):
    try:
        coef_rows = "".join(
            f"<tr><td>{c}</td><td>{float(coef):.6f}</td></tr>"
            for c, coef in zip(cols, model.coef_.ravel())
        )
        intercept = float(model.intercept_[0])
    except:
        coef_rows = "<tr><td colspan='2'>Model tidak punya koefisien</td></tr>"
        intercept = "N/A"

    html = f"""
    <html><body>
    <h1>Estimator Summary</h1>
    <h2>{model.__class__.__name__}</h2>

    <h3>Params</h3>
    <pre>{json.dumps(model.get_params(), indent=2)}</pre>

    <h3>Intercept</h3>
    <p>{intercept}</p>

    <h3>Coefficients</h3>
    <table border="1" cellpadding="4" cellspacing="0">
        <tr><th>Feature</th><th>Coef</th></tr>
        {coef_rows}
    </table>
    </body></html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

def save_metrics_json(data, out_path="metric_info.json"):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path

def save_confusion_matrix_png(y_true, y_pred, out_path="training_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# ==========================
# LOAD DATA
# ==========================
def load_and_preprocess_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if "Outcome" not in df.columns:
        raise ValueError("Dataset harus punya kolom 'Outcome'.")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(X_scaled, columns=X.columns),
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

# ==========================
# TRAINING - MODIFIKASI UTAMA UNTUK GITHUB ACTIONS
# ==========================
def train_and_tune(data_path, experiment_name="Diabetes_Prediction_CI"):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Inisialisasi variabel artefak lokal
    est, met, cm = "estimator.html", "metric_info.json", "training_confusion_matrix.png"
    MODEL_ARTIFACT_PATH = "model"

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:  # CAPTURE RUN OBJECT
            search_space = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "penalty": ["l1", "l2"],
            }
            base_model = LogisticRegression(solver="liblinear", random_state=42)
            opt = BayesSearchCV(
                estimator=base_model,
                search_spaces=search_space,
                n_iter=20,
                cv=5,
                scoring="f1",
                random_state=42,
                n_jobs=1,
            )

            print("\n🔍 Tuning hyperparameter…")
            opt.fit(X_train, y_train)
            best_model = opt.best_estimator_

            # LOG PARAMS & METRICS
            mlflow.log_params({
                "C": float(opt.best_params_["C"]),
                "penalty": str(opt.best_params_["penalty"]),
                "tuning_method": "BayesSearchCV",
                "n_iter": 20
            })
            
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
            mlflow.log_metrics(metrics)

            # ARTEFAK NON-MODEL
            est = save_estimator_html(best_model, X_train.columns.tolist())
            met = save_metrics_json({"metrics": metrics, "best_params": opt.best_params_})
            cm = save_confusion_matrix_png(y_test, y_pred)
            mlflow.log_artifact(est)
            mlflow.log_artifact(met)
            mlflow.log_artifact(cm)

            # LOG MODEL LENGKAP SECARA MANUAL
            print("📦 Logging model lengkap secara manual...")
            os.makedirs(MODEL_ARTIFACT_PATH, exist_ok=True)
            
            # 1. SIMPAN MODEL.PKL
            model_pkl_path = os.path.join(MODEL_ARTIFACT_PATH, "model.pkl")
            joblib.dump(best_model, model_pkl_path)
            
            # 2. BUAT METADATA MLMODEL
            signature = infer_signature(X_train, best_model.predict(X_train))
            
            mlflow_model = Model(
                artifact_path=MODEL_ARTIFACT_PATH, 
                run_id=run.info.run_id,  # GUNAKAN run.info.run_id
                signature=signature
            )
            
            mlflow_model.add_flavor(
                mlflow.sklearn.FLAVOR_NAME, 
                serialization_format='joblib', 
                sklearn_version=sklearn.__version__, 
                pickled_model="model.pkl"
            )
            
            mlflow_model_path = os.path.join(MODEL_ARTIFACT_PATH, "MLmodel")
            mlflow_model.save(mlflow_model_path)
            
            # 3. BUAT FILE LINGKUNGAN
            conda_env_path = os.path.join(MODEL_ARTIFACT_PATH, "conda.yaml")
            with open(conda_env_path, "w") as f:
                conda_dict = get_default_conda_env(include_cloudpickle=False)
                import yaml
                yaml.dump(conda_dict, f, default_flow_style=False)
                 
            req_path = os.path.join(MODEL_ARTIFACT_PATH, "requirements.txt")
            with open(req_path, "w") as f:
                 f.write('\n'.join(get_default_pip_requirements()))

            python_env_path = os.path.join(MODEL_ARTIFACT_PATH, "python_env.yaml")
            with open(python_env_path, "w") as f:
                f.write('{}') 
            
            # 4. LOG SEMUA ARTEFAK
            mlflow.log_artifacts(MODEL_ARTIFACT_PATH, artifact_path="model")
            
            print(f"\n🎉 Training selesai. Run ID: {run.info.run_id}")
            
            # SIMPAN RUN_ID KE FILE UNTUK GITHUB ACTIONS
            with open("run_id.txt", "w") as f:
                f.write(run.info.run_id)
            
            return run.info.run_id
            
    finally:
        # PEMBERSIHAN FILE LOKAL
        for f in [est, met, cm]:
            if os.path.exists(f):
                os.remove(f)
        
        if os.path.exists(MODEL_ARTIFACT_PATH):
            shutil.rmtree(MODEL_ARTIFACT_PATH)

# ==========================
# MAIN FUNCTION UNTUK MLFLOW PROJECT
# ==========================
def main():
    parser = argparse.ArgumentParser(description='Train Diabetes Prediction Model')
    parser.add_argument('--data_path', type=str, default='./dataset/cleaned_dataset.csv', 
                       help='Path to dataset')
    parser.add_argument('--experiment_name', type=str, default='Diabetes_Prediction_CI',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting MLflow Project Training")
    print(f"📁 Data path: {args.data_path}")
    print(f"🔬 Experiment: {args.experiment_name}")
    
    # Validasi path dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at: {args.data_path}")
    
    run_id = train_and_tune(args.data_path, args.experiment_name)
    print(f"✅ Training completed successfully! Run ID: {run_id}")

if __name__ == "__main__":
    main()