# ⚙️ MachineGuard: Predictive Maintenance ML Platform

An end-to-end industrial machine learning system for predicting equipment failure risks using real-time operational telemetry (rotational speed, temperatures, torque, and tool wear).

---

## 🏗️ Architecture

The project is structured with a clean separation of **Backend** and **Frontend**:

```
predictive-maintenance-ml/
├── backend/                  # FastAPI Application & Database
│   ├── main.py               # REST API endpoints (/predict, /history, /health, /model-info)
│   └── database.py           # SQLite persistence layer with SQLAlchemy
│
├── frontend/                 # Modern UI Application
│   └── dashboard.py          # Streamlit UI with Dark Mode, Plotly charts, Presets & Diagnostics
│
├── models/                   # Serialized ML Models
│   ├── final_model.joblib    # Tuned XGBoost Pipeline
│   └── xgboost_tuned.joblib
│
├── reports/                  # Model metrics, SHAP analysis, validation thresholds
├── data/                     # Raw and preprocessed CSV datasets & SQLite DB
├── src/                      # ML pipeline training, tuning, and evaluation scripts
└── tests/                    # Pytest test suite
```

---

## 🚀 Quick Start

### 1. Activate Environment & Install Dependencies
```bash
# Activate existing virtual environment
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements-docker.txt
```

### 2. Launch the Backend API
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 3. Launch the Frontend UI
```bash
streamlit run frontend/dashboard.py --server.port 8501
```
- Access the UI in your browser: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker / Docker Compose

To run both services isolated with docker-compose:
```bash
docker compose up --build
```
- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`
