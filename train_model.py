"""
train_model.py - End-to-End Machine Learning Pipeline for Loan Prediction System.

This script:
1. Loads historical loan data from data/loan_dataset.csv.
2. Cleans features and maps target column (Loan_Status: 'Y' -> 1, 'N' -> 0).
3. Splits into stratified train and test sets (80/20).
4. Builds a scikit-learn preprocessing ColumnTransformer (handling missing data and scaling).
5. Trains and compares Logistic Regression, Decision Tree, and Random Forest models.
6. Evaluates all models with Accuracy, Precision, Recall, F1, and Confusion Matrix.
7. Selects the best performing model based on test evaluation criteria.
8. Persists the winning fitted pipeline to model/loan_model.joblib and saves genuine metrics to model/model_metrics.json.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


def load_and_preprocess_dataset(data_path=DATA_PATH):
    """Loads CSV, validates columns, cleans target, and returns X and y along with raw stats."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please download train.csv from "
            "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv "
            f"and place it at {data_path}."
        )

    logging.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    total_raw_rows = len(df)
    logging.info(f"Dataset shape: {df.shape}")

    # Ensure required columns exist
    required_cols = [
        "Loan_Status", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Drop identifier column if present
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Clean target: only keep rows with valid target ('Y' or 'N')
    df = df[df["Loan_Status"].isin(["Y", "N"])].copy()
    usable_rows = len(df)

    # Encode target: Y=1, N=0
    y = df["Loan_Status"].map({"Y": 1, "N": 0}).values
    X = df.drop(columns=["Loan_Status"])

    # Dataset statistics
    approved_count = int(np.sum(y == 1))
    rejected_count = int(np.sum(y == 0))
    approval_rate = round((approved_count / usable_rows) * 100, 2)

    stats = {
        "total_records": total_raw_rows,
        "usable_records": usable_rows,
        "features_count": X.shape[1],
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "approval_rate_percent": approval_rate,
        "feature_columns": list(X.columns),
    }

    logging.info(
        f"Dataset summary: Usable={usable_rows}, Approved={approved_count} ({approval_rate}%), "
        f"Rejected={rejected_count}"
    )
    return X, y, stats


def build_preprocessor():
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features."""
    numerical_features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    # Numerical pipeline: impute missing with median, then standardize
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing with most frequent category, then one-hot encode
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )
    return preprocessor, numerical_features, categorical_features


def train_and_evaluate(data_path=DATA_PATH, save_artifacts=True):
    """Trains candidate models, evaluates them, selects best, and saves artifacts."""
    X, y, stats = load_and_preprocess_dataset(data_path)
    preprocessor, num_cols, cat_cols = build_preprocessor()

    # Stratified 80/20 train/test split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logging.info(f"Split sizes: Train={len(X_train)}, Test={len(X_test)}")

    # Define candidate models
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42
        ),
    }

    evaluated_models = {}
    best_model_name = None
    best_pipeline = None
    best_f1 = -1.0
    best_acc = -1.0

    for name, clf in candidates.items():
        logging.info(f"Training candidate model: {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train, y_train)

        # Predictions on test set
        y_pred = pipeline.predict(X_test)
        has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
        y_prob = pipeline.predict_proba(X_test)[:, 1] if has_proba else None

        # Metrics computation
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred).tolist()

        evaluated_models[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm,
            "supports_probability": bool(has_proba),
        }

        roc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        logging.info(
            f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_str}"
        )

        # Selection criteria: prioritize test F1-score with secondary tie-breaker on accuracy
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_model_name = name
            best_pipeline = pipeline

    logging.info(f"Selected best model: {best_model_name} (F1: {best_f1:.4f}, Acc: {best_acc:.4f})")

    metrics_payload = {
        "dataset_summary": stats,
        "train_set_size": len(X_train),
        "test_set_size": len(X_test),
        "selected_model": best_model_name,
        "selection_criteria": (
            "Model selected based on highest F1-score on the stratified hold-out test set (20%), "
            "ensuring an optimal balance between precision and recall for lending risk assessment."
        ),
        "models": evaluated_models,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if save_artifacts:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_pipeline, MODEL_PATH)
        logging.info(f"Fitted pipeline successfully saved to: {MODEL_PATH}")

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        logging.info(f"Evaluation metrics successfully saved to: {METRICS_PATH}")

    return best_model_name, best_pipeline, metrics_payload


if __name__ == "__main__":
    logging.info("Starting model training and evaluation...")
    name, _, metrics = train_and_evaluate(save_artifacts=True)
    logging.info(f"Pipeline complete! Best model: {name}")
