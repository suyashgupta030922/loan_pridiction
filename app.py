"""
app.py - Flask Web Backend & REST API for the Loan Prediction System.

Provides:
- Web routes for Landing Page, Prediction Interface, Dashboard, and About Model.
- REST API endpoints for /api/predict, /api/model-metrics, /api/health, and /api/history.
- Strict input validation, error handling, SQLite persistence for history, and model loading.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "model_metrics.json")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DATABASE_PATH = os.path.join(INSTANCE_DIR, "predictions.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-insecure-secret-key-change-in-prod")
app.config["ENABLE_HISTORY"] = os.environ.get("ENABLE_HISTORY", "true").lower() == "true"

# Cache loaded model in memory
_cached_pipeline = None


def get_db_connection():
    """Returns an SQLite connection to instance/predictions.db."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite schema for storing predictions."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                applicant_income REAL NOT NULL,
                coapplicant_income REAL NOT NULL,
                loan_amount REAL NOT NULL,
                loan_term REAL NOT NULL,
                credit_history REAL NOT NULL,
                gender TEXT NOT NULL,
                married TEXT NOT NULL,
                dependents TEXT NOT NULL,
                education TEXT NOT NULL,
                self_employed TEXT NOT NULL,
                property_area TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                approval_score REAL
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to initialize SQLite database: {e}")


def load_model():
    """Loads the fitted model pipeline from disk if available."""
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline

    if os.path.exists(MODEL_PATH):
        try:
            _cached_pipeline = joblib.load(MODEL_PATH)
            logging.info("Model pipeline loaded successfully.")
            return _cached_pipeline
        except Exception as e:
            logging.error(f"Error loading model from {MODEL_PATH}: {e}")
            return None
    return None


def validate_input(data):
    """
    Validates form data or JSON payload.
    Returns (cleaned_dict, None) if valid, or (None, error_message) if invalid.
    """
    if not isinstance(data, dict):
        return None, "Request payload must be a valid JSON object."

    required_fields = [
        "applicant_income",
        "coapplicant_income",
        "loan_amount",
        "loan_term",
        "credit_history",
        "gender",
        "married",
        "dependents",
        "education",
        "self_employed",
        "property_area",
    ]

    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            return None, f"Missing required field: '{field}'."

    # Validate numerical fields
    try:
        app_income = float(data["applicant_income"])
        if app_income < 0:
            return None, "Applicant Income cannot be negative."
        if app_income > 1_000_000:
            return None, "Applicant Income exceeds plausible maximum limit (1,000,000)."
    except (ValueError, TypeError):
        return None, "Applicant Income must be a valid number."

    try:
        coapp_income = float(data["coapplicant_income"])
        if coapp_income < 0:
            return None, "Co-applicant Income cannot be negative."
        if coapp_income > 1_000_000:
            return None, "Co-applicant Income exceeds plausible maximum limit (1,000,000)."
    except (ValueError, TypeError):
        return None, "Co-applicant Income must be a valid number."

    try:
        loan_amt = float(data["loan_amount"])
        if loan_amt <= 0:
            return None, "Loan Amount must be greater than zero."
        if loan_amt > 10_000:
            return None, "Loan Amount exceeds maximum supported limit (10,000 in thousands, i.e., $10M)."
    except (ValueError, TypeError):
        return None, "Loan Amount must be a valid number (in thousands)."

    try:
        loan_term = float(data["loan_term"])
        if loan_term <= 0 or loan_term > 600:
            return None, "Loan Amount Term must be between 12 and 600 months."
    except (ValueError, TypeError):
        return None, "Loan Amount Term must be a valid number (in months)."

    try:
        credit_hist = float(data["credit_history"])
        if credit_hist not in [0.0, 1.0, 0, 1]:
            return None, "Credit History must be either 1.0 (meets guidelines) or 0.0 (does not meet guidelines)."
    except (ValueError, TypeError):
        return None, "Credit History must be 1.0 or 0.0."

    # Validate categorical fields
    gender = str(data["gender"]).strip().capitalize()
    if gender not in ["Male", "Female"]:
        return None, "Gender must be 'Male' or 'Female'."

    married = str(data["married"]).strip().capitalize()
    if married not in ["Yes", "No"]:
        return None, "Marital Status must be 'Yes' or 'No'."

    dependents = str(data["dependents"]).strip()
    if dependents not in ["0", "1", "2", "3+"]:
        return None, "Dependents must be '0', '1', '2', or '3+'."

    education = str(data["education"]).strip()
    # Normalize 'Not graduate' -> 'Not Graduate'
    if education.lower() == "graduate":
        education = "Graduate"
    elif education.lower() in ["not graduate", "notgraduate", "non graduate"]:
        education = "Not Graduate"
    else:
        return None, "Education must be 'Graduate' or 'Not Graduate'."

    self_employed = str(data["self_employed"]).strip().capitalize()
    if self_employed not in ["Yes", "No"]:
        return None, "Self Employed status must be 'Yes' or 'No'."

    property_area = str(data["property_area"]).strip().capitalize()
    if property_area not in ["Urban", "Semiurban", "Rural"]:
        return None, "Property Area must be 'Urban', 'Semiurban', or 'Rural'."

    cleaned = {
        "ApplicantIncome": app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amt,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_hist,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area,
    }
    return cleaned, None


# Web View Routes
@app.route("/")
def index():
    """Serves the Landing Page."""
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    """Serves the Loan Prediction Form interface."""
    return render_template("predict.html")


@app.route("/dashboard")
def dashboard_page():
    """Serves the Model Performance Analytics Dashboard."""
    return render_template("dashboard.html")


@app.route("/about")
def about_page():
    """Serves the Educational About Model page."""
    return render_template("about.html")


# API Endpoints
@app.route("/api/health", methods=["GET"])
def health():
    """Returns basic system health status."""
    model_ready = os.path.exists(MODEL_PATH)
    metrics_ready = os.path.exists(METRICS_PATH)
    return jsonify({
        "status": "healthy" if (model_ready and metrics_ready) else "degraded",
        "model_loaded": model_ready,
        "metrics_available": metrics_ready,
        "timestamp": datetime.now().isoformat(),
        "service": "Loan Prediction System API",
    }), 200


@app.route("/api/model-metrics", methods=["GET"])
def get_metrics():
    """Returns verified evaluation metrics and dataset statistics."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({
            "error": "Model metrics file not found. Please execute train_model.py first."
        }), 404

    try:
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        logging.error(f"Error reading model metrics: {e}")
        return jsonify({"error": "Failed to read model metrics file."}), 500


@app.route("/api/predict", methods=["POST"])
def predict_loan():
    """
    Validates input features, predicts loan eligibility using trained pipeline,
    and returns structured prediction and model approval score.
    """
    try:
        content_type = request.content_type or ""
        if "application/json" in content_type:
            raw_data = request.get_json(silent=True)
        else:
            raw_data = request.form.to_dict()

        if not raw_data:
            return jsonify({"status": "error", "message": "No data provided in request."}), 400

        cleaned_data, error_msg = validate_input(raw_data)
        if error_msg:
            return jsonify({"status": "error", "message": error_msg}), 400

        pipeline = load_model()
        if pipeline is None:
            return jsonify({
                "status": "error",
                "message": "Model pipeline is not trained or found. Please run 'python train_model.py' to generate the model."
            }), 503

        # Prepare single-row DataFrame matching the exact training columns
        input_df = pd.DataFrame([cleaned_data])

        # Run prediction through fitted Scikit-learn Pipeline
        prediction_int = int(pipeline.predict(input_df)[0])
        prediction_label = "Likely Approved" if prediction_int == 1 else "Likely Not Approved"

        # Model-estimated approval score (probability of class 1)
        approval_score = None
        approval_score_percent = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            # proba[1] corresponds to class 1 (Approved)
            approval_score = round(float(proba[1]), 4)
            approval_score_percent = round(approval_score * 100, 1)

        # Assessment narrative
        if prediction_int == 1:
            assessment = (
                "The statistical model indicates favorable eligibility characteristics, "
                "primarily driven by sound credit history and balanced loan-to-income proportions."
            )
        else:
            assessment = (
                "The statistical model indicates higher risk indicators for this application profile, "
                "often associated with credit history status or debt-to-income ratio."
            )

        # Optional persistence to SQLite
        if app.config["ENABLE_HISTORY"]:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        timestamp, applicant_income, coapplicant_income, loan_amount,
                        loan_term, credit_history, gender, married, dependents,
                        education, self_employed, property_area, prediction_label,
                        approval_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        cleaned_data["ApplicantIncome"],
                        cleaned_data["CoapplicantIncome"],
                        cleaned_data["LoanAmount"],
                        cleaned_data["Loan_Amount_Term"],
                        cleaned_data["Credit_History"],
                        cleaned_data["Gender"],
                        cleaned_data["Married"],
                        cleaned_data["Dependents"],
                        cleaned_data["Education"],
                        cleaned_data["Self_Employed"],
                        cleaned_data["Property_Area"],
                        prediction_label,
                        approval_score,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                logging.warning(f"Optional history logging skipped due to DB error: {db_err}")

        # Summary of the application for user review
        total_income = cleaned_data["ApplicantIncome"] + cleaned_data["CoapplicantIncome"]
        app_summary = {
            "applicant_income": cleaned_data["ApplicantIncome"],
            "coapplicant_income": cleaned_data["CoapplicantIncome"],
            "total_income": total_income,
            "loan_amount": cleaned_data["LoanAmount"],
            "loan_term_months": int(cleaned_data["Loan_Amount_Term"]),
            "credit_history": "Meets Guidelines (1.0)" if cleaned_data["Credit_History"] == 1.0 else "Does Not Meet (0.0)",
            "gender": cleaned_data["Gender"],
            "married": cleaned_data["Married"],
            "dependents": cleaned_data["Dependents"],
            "education": cleaned_data["Education"],
            "self_employed": cleaned_data["Self_Employed"],
            "property_area": cleaned_data["Property_Area"],
        }

        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "is_approved": bool(prediction_int == 1),
            "approval_score": approval_score,
            "approval_score_percent": approval_score_percent,
            "assessment": assessment,
            "disclaimer": (
                "Notice: This is a statistical estimate produced by a machine learning model "
                "trained on historical data. It does not constitute a guaranteed lending decision, "
                "legal commitment, or credit underwriting approval."
            ),
            "application_summary": app_summary,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }), 200

    except Exception as e:
        logging.error(f"Unexpected prediction error: {e}", exc_info=False)
        return jsonify({
            "status": "error",
            "message": "An error occurred while evaluating your application. Please check your inputs and try again."
        }), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns recent anonymized predictions for auditing / presentation demonstration."""
    if not app.config["ENABLE_HISTORY"]:
        return jsonify({"history": [], "message": "History persistence is disabled."}), 200

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, loan_amount, loan_term, credit_history,
                   property_area, prediction_label, approval_score
            FROM predictions
            ORDER BY id DESC LIMIT 10
            """
        )
        rows = cursor.fetchall()
        conn.close()

        history = [dict(row) for row in rows]
        return jsonify({"history": history, "count": len(history)}), 200
    except Exception as e:
        logging.error(f"Error fetching prediction history: {e}")
        return jsonify({"error": "Failed to retrieve history."}), 500


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Clears SQLite prediction history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Prediction history cleared."}), 200
    except Exception as e:
        logging.error(f"Error clearing history: {e}")
        return jsonify({"status": "error", "message": "Failed to clear history."}), 500


@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 handler."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found", "status_code": 404}), 404
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    """Custom 500 handler without leaking stack traces."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error", "status_code": 500}), 500
    return render_template("404.html", message="An unexpected server error occurred."), 500


# Initialize DB on module import
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Loan Prediction System Flask app at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
