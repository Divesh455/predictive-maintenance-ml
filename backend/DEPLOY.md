## Backend Deployment — Render.com

### What's in this folder

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application (all REST endpoints) |
| `database.py` | SQLAlchemy models + SQLite/PostgreSQL support |
| `requirements.txt` | Python dependencies for Render build |
| `render.yaml` | Render Blueprint (auto-configure the service) |
| `Procfile` | Render start command fallback |

---

### Deploy to Render (Step-by-Step)

#### Option A — Manual (Quickest)

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. **New +** → **Web Service**
3. Connect your GitHub repo
4. Set these fields:

   | Field | Value |
   |-------|-------|
   | Name | `predictive-maintenance-api` |
   | Runtime | `Python 3` |
   | Build Command | `pip install -r backend/requirements.txt` |
   | Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
   | Plan | Free |

5. Click **Create Web Service**

#### Option B — Blueprint (Automated)

1. Copy `backend/render.yaml` → root of your repo as `render.yaml`
2. Render → **New +** → **Blueprint** → Connect repo
3. Render reads the YAML and auto-configures everything

---

### Environment Variables (set on Render dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Optional | PostgreSQL connection URL for persistent history. If not set, uses SQLite (ephemeral). |
| `PYTHONUNBUFFERED` | Yes | Set to `1` (auto-set by Render) |

---

### Add Persistent PostgreSQL (Recommended)

1. Render → **New +** → **PostgreSQL** → Plan: Free → Create
2. Copy the **Internal Database URL**
3. Go to your Web Service → **Environment** tab
4. Add `DATABASE_URL` = *(paste the URL)*
5. Redeploy — history now persists across restarts

---

### Live Endpoints (after deploy)

```
GET  https://your-app.onrender.com/           # API info
GET  https://your-app.onrender.com/health     # health check
GET  https://your-app.onrender.com/docs       # Swagger UI
POST https://your-app.onrender.com/predict    # run inference
GET  https://your-app.onrender.com/history    # prediction log
GET  https://your-app.onrender.com/model-info # model metadata
GET  https://your-app.onrender.com/feature-importance  # SHAP
```
