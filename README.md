# 🏦 Smart Loan Prediction System (Machine Learning & Full-Stack Web Application)

A complete, production-ready, full-stack machine learning application built for academic and college-level engineering demonstration. The system trains a scikit-learn pipeline on historical loan records, serves predictions via a modular Python Flask backend, and presents results on a modern, white-themed SaaS dashboard interface with Chart.js visualizations.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technology Stack](#-technology-stack)
3. [Folder Structure](#-folder-structure)
4. [Dataset Details & Source](#-dataset-details--source)
5. [Prerequisites & Python Requirements](#-prerequisites--python-requirements)
6. [Step-by-Step Setup Guide](#-step-by-step-setup-guide)
7. [Training the Machine Learning Pipeline](#-training-the-machine-learning-pipeline)
8. [Running the Application](#-running-the-application)
9. [REST API Documentation](#-rest-api-documentation)
10. [Automated Testing](#-automated-testing)
11. [Troubleshooting Common Issues](#-troubleshooting-common-issues)
12. [Production Deployment](#-production-deployment)
13. [Academic Limitations & Disclaimer](#-academic-limitations--disclaimer)

---

## 🌟 Project Overview

Evaluating credit eligibility involves balancing credit history, applicant and co-applicant income, requested loan volume, and repayment tenure. This project implements a leak-free scikit-learn preprocessing and classification pipeline comparing:
- **Logistic Regression (Selected Classifier - F1: 90.81%, Accuracy: 86.18%)**
- **Decision Tree Classifier (F1: 87.50%, Accuracy: 82.11%)**
- **Random Forest Classifier (F1: 90.32%, Accuracy: 85.37%)**

All evaluation metrics are calculated on a strictly isolated 20% stratified hold-out test set ($N = 123$), ensuring zero data leakage and authentic model transparency.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Modern CSS3 (Clean White Theme, Responsive Grid), Vanilla JavaScript, Chart.js (v4.4) |
| **Backend** | Python 3.12+, Flask 3.1 |
| **Machine Learning** | Scikit-Learn 1.9+, Pandas 3.1+, NumPy 2.5+, Joblib |
| **Database** | SQLite 3 (`instance/predictions.db` for optional history logging) |
| **Testing** | Pytest 9.1 |
| **WSGI Server** | Waitress (Windows/Linux) / Gunicorn (Linux/macOS) |

---

## 📁 Folder Structure

```
loan-prediction-system/
├── app.py                     # Flask web server & REST API endpoints
├── train_model.py             # ML pipeline: data ingestion, preprocessing, training, evaluation
├── wsgi.py                    # Production WSGI server entry point (Waitress)
├── requirements.txt           # Python dependency declarations
├── README.md                  # Complete documentation and setup manual
├── .gitignore                 # Excludes models, venv, and sqlite db from git
├── run_app.bat                # Windows 1-click launcher for Flask web app
├── train.bat                  # Windows 1-click launcher for ML model training
├── run_tests.bat              # Windows 1-click launcher for automated pytest suite
├── data/
│   └── loan_dataset.csv       # Historical dataset (614 records)
├── model/
│   ├── loan_model.joblib      # Persisted winning Scikit-Learn pipeline
│   └── model_metrics.json     # Genuine hold-out evaluation metrics & confusion matrix
├── templates/
│   ├── base.html              # Base HTML template with responsive navbar & footer
│   ├── index.html             # Landing page with hero, features & workflow
│   ├── predict.html           # Prediction form interface with instant result card
│   ├── dashboard.html         # Analytics dashboard with Chart.js benchmarks & confusion matrix
│   ├── about.html             # Educational explanation of ML, metrics, and limitations
│   └── 404.html               # Custom 404 error template
├── static/
│   ├── css/
│   │   └── style.css          # Unified white-themed SaaS design system
│   └── js/
│       ├── main.js            # Mobile navbar toggle & navigation state
│       ├── predict.js         # Client-side validation, JSON fetch, & results renderer
│       └── dashboard.js       # Dynamic metrics loader & Chart.js renderer
├── instance/
│   └── predictions.db         # SQLite local database for prediction history
└── tests/
    └── test_app.py            # 15 automated pytest unit and integration test cases
```

---

## 📊 Dataset Details & Source

The dataset used is the standard historical **Loan Prediction Dataset**:
- **Source:** [GitHub - shrikant-temburwar/Loan-Prediction-Dataset](https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv)
- **File Location:** `data/loan_dataset.csv`
- **Total Records:** 614 rows (422 Approved [68.73%], 192 Rejected [31.27%])
- **Features Used:**
  1. `Gender`: Male / Female
  2. `Married`: Yes / No
  3. `Dependents`: 0, 1, 2, 3+
  4. `Education`: Graduate / Not Graduate
  5. `Self_Employed`: Yes / No
  6. `ApplicantIncome`: Primary monthly income
  7. `CoapplicantIncome`: Additional household monthly income
  8. `LoanAmount`: Loan requested in thousands ($)
  9. `Loan_Amount_Term`: Tenure in months (e.g. 360)
  10. `Credit_History`: 1.0 (meets guidelines) or 0.0 (defaults/does not meet)
  11. `Property_Area`: Urban / Semiurban / Rural
  12. `Loan_Status`: Target variable ('Y' -> 1, 'N' -> 0)

---

## ⚙️ Prerequisites & Python Requirements

- **Python:** 3.10, 3.11, or 3.12 installed on your machine.
- **Pip:** Version 23.0 or higher.
- **Git** (optional).

---

## 🚀 Step-by-Step Setup Guide

### 1. Navigate to the Project Directory
```bash
cd loan-prediction-system
```

### 2. Create a Virtual Environment
- **Windows (Command Prompt / PowerShell):**
  ```bash
  python -m venv venv
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  ```

### 3. Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧠 Training the Machine Learning Pipeline

Before starting the web application, train the machine learning models and persist the pipeline artifacts:

```bash
python train_model.py
```

*Or on Windows, simply double-click `train.bat`.*

### What happens during training:
1. Ingests `data/loan_dataset.csv`.
2. Encodes target labels (`Y` $\to 1$, `N` $\to 0$).
3. Performs an **80/20 Stratified Split** (`train_size=491`, `test_size=123`).
4. Builds a `ColumnTransformer` with median imputation for numericals and mode imputation + one-hot encoding for categoricals.
5. Trains and tests **Logistic Regression**, **Decision Tree**, and **Random Forest**.
6. Selects the highest F1-Score model (**Logistic Regression: 90.81% F1, 86.18% Accuracy**).
7. Saves the fitted pipeline into `model/loan_model.joblib`.
8. Saves verified performance metrics and confusion matrices into `model/model_metrics.json`.

---

## 💻 Running the Application

### Development Mode:
```bash
python app.py
```
*Or on Windows, double-click `run_app.bat`.*

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Production WSGI Mode:
```bash
python wsgi.py
```
*Runs via Waitress on port 5000 with multi-threaded request processing.*

---

## 🌐 Application Pages & Routes

1. **Home / Landing Page (`/`):**
   - High-level overview of algorithmic risk assessment.
   - Interactive feature cards & 4-step workflow.
   - Direct action buttons to "Predict My Loan" and "Explore Dashboard".

2. **Loan Prediction Form (`/predict`):**
   - 11 validated inputs (numeric ranges, dropdowns, help tooltips).
   - Real-time client and backend validation.
   - Non-blocking JSON fetch submission with loading spinner.
   - Dynamic Results Card showing **Likely Approved / Likely Not Approved**, **Model-Estimated Approval Score**, applicant breakdown, and disclaimer.

3. **Analytics Dashboard (`/dashboard`):**
   - Real KPI metric counters (Total records, Test Accuracy, Precision, Recall, F1-Score).
   - **Chart.js Visualizations:** Class distribution doughnut chart & Multi-model benchmark bar chart.
   - Confusion Matrix (TN, FP, FN, TP) strictly computed on hold-out data.
   - Full model comparison table with selected model highlight.
   - Recent SQLite prediction history audit table with "Clear History" capability.

4. **About Model (`/about`):**
   - In-depth college presentation guide explaining Supervised Classification, pipeline mechanics, leakage prevention, confusion matrix mathematics, and algorithmic fairness considerations.

---

## 📡 REST API Documentation

### 1. `POST /api/predict`
Calculates eligibility for a submitted application profile.

- **Content-Type:** `application/json`
- **Request Body Example:**
  ```json
  {
    "applicant_income": 5400,
    "coapplicant_income": 1800,
    "loan_amount": 140,
    "loan_term": 360,
    "credit_history": 1.0,
    "gender": "Male",
    "married": "Yes",
    "dependents": "1",
    "education": "Graduate",
    "self_employed": "No",
    "property_area": "Semiurban"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "status": "success",
    "prediction": "Likely Approved",
    "is_approved": true,
    "approval_score": 0.8842,
    "approval_score_percent": 88.4,
    "assessment": "The statistical model indicates favorable eligibility characteristics...",
    "disclaimer": "Notice: This is a statistical estimate produced by a machine learning model...",
    "application_summary": { ... },
    "timestamp": "2026-10-22 21:28:53"
  }
  ```

### 2. `GET /api/model-metrics`
Returns genuine hold-out evaluation metrics and dataset statistics.

- **Response (200 OK):**
  ```json
  {
    "dataset_summary": {
      "total_records": 614,
      "usable_records": 614,
      "approved_count": 422,
      "rejected_count": 192,
      "approval_rate_percent": 68.73
    },
    "selected_model": "Logistic Regression",
    "models": { ... }
  }
  ```

### 3. `GET /api/health`
Health check endpoint reporting model availability and API readiness.

### 4. `GET /api/history` & `POST /api/history/clear`
Audits recent prediction entries stored in `instance/predictions.db`.

---

## 🧪 Automated Testing

The project includes 15 automated unit and integration tests using `pytest`.

To run tests:
```bash
pytest tests/test_app.py -v
```
*Or on Windows, double-click `run_tests.bat`.*

### Test Coverage Checklist:
- [x] Landing page (`/`) loads with HTTP 200.
- [x] Prediction page (`/predict`) loads with form controls.
- [x] Dashboard page (`/dashboard`) loads with KPI containers.
- [x] About page (`/about`) loads with educational sections.
- [x] Custom 404 handler returns structured not found response.
- [x] Health API (`/api/health`) verifies model state.
- [x] Metrics API (`/api/model-metrics`) returns unhardcoded dataset stats.
- [x] Valid applicant produces `Likely Approved` prediction.
- [x] Defaulter applicant produces `Likely Not Approved` prediction.
- [x] Missing fields return 400 Bad Request with field name.
- [x] Negative income / loan values are rejected.
- [x] Invalid categorical options are rejected.
- [x] Non-binary credit history values are rejected.
- [x] Missing model file triggers graceful 503 Service Unavailable.
- [x] Prediction history persistence and clear operations succeed.

---

## ⚠️ Troubleshooting Common Issues

1. **`Model pipeline is not trained or found (HTTP 503)`:**
   - Run `python train_model.py` to regenerate `model/loan_model.joblib`.
2. **`Dataset not found`:**
   - Confirm that `data/loan_dataset.csv` exists. If missing, download it from the GitHub link above.
3. **`ModuleNotFoundError: No module named 'flask'`:**
   - Ensure the virtual environment is activated (`venv\Scripts\activate` or `source venv/bin/activate`).
4. **Port 5000 is already in use:**
   - Run on another port:
     ```bash
     PORT=5050 python app.py
     ```

---

## 🚢 Production Deployment

For deploying to production platforms (Render, Railway, Heroku, AWS EC2, or VPS):

1. **Set Environment Variables:**
   - `PORT=5000` (or platform default)
   - `SECRET_KEY=generate-a-strong-random-key`
   - `ENABLE_HISTORY=true` (or `false` to disable SQLite persistence)
2. **Launch with Production WSGI:**
   ```bash
   python wsgi.py
   ```
   *Or with Gunicorn on Linux:*
   ```bash
   gunicorn -w 4 -b 0.0.0.0:$PORT wsgi:app
   ```

---

## ⚖️ Academic Limitations & Disclaimer

This project is developed solely for **academic, educational, and viva demonstration purposes**.
1. **Public Sample Limitation:** The training dataset contains 614 records from a public benchmark. Real banking institutions use millions of transactional records across diverse macroeconomic conditions.
2. **Historical Bias:** Historical loan data can reflect legacy demographic disparities. In production lending, models require strict compliance with the Equal Credit Opportunity Act (ECOA) and rigorous algorithmic fairness auditing.
3. **No Lending Guarantee:** Output scores and predictions represent purely empirical model estimates and do not constitute actual credit approval or formal financial underwriting.
