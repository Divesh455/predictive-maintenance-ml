## Frontend Deployment — Streamlit Community Cloud

### What's in this folder

| File | Purpose |
|------|---------|
| `dashboard.py` | Streamlit UI application |
| `requirements.txt` | Python dependencies for Streamlit Cloud |
| `.streamlit/config.toml` | Dark theme + server settings |

---

### Deploy to Streamlit Cloud (Step-by-Step)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:

   | Field | Value |
   |-------|-------|
   | Repository | `YOUR_USERNAME/predictive-maintenance-ml` |
   | Branch | `main` |
   | **Main file path** | `frontend/dashboard.py` |

5. Click **"Advanced settings"** and add this secret:

   ```toml
   PREDICTIVE_MAINTENANCE_API = "https://your-render-app.onrender.com"
   ```

   *(Replace with your actual Render backend URL)*

6. Click **"Deploy!"**

---

### Environment Variables / Secrets

Set these in **App Settings → Secrets** on Streamlit Cloud:

| Secret Key | Value | Required |
|-----------|-------|----------|
| `PREDICTIVE_MAINTENANCE_API` | Your Render backend URL e.g. `https://predictive-maintenance-api.onrender.com` | ✅ Yes |

The dashboard reads it like this (already configured in `dashboard.py`):
```python
API_BASE_URL = os.getenv("PREDICTIVE_MAINTENANCE_API", "http://127.0.0.1:8000")
```

---

### How Streamlit Cloud Finds Requirements

Streamlit Cloud checks for `requirements.txt` in this priority order:
1. **Same folder as the main app file** ← `frontend/requirements.txt` ✅
2. Root of repo

Because `requirements.txt` is in `frontend/` alongside `dashboard.py`, it is automatically picked up — no extra configuration needed.

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| "Backend Offline" on first load | Render free tier sleeps — wait 30 sec for it to wake |
| App won't build | Check `frontend/requirements.txt` has no ML/training packages |
| Predictions not saving | Check Render backend has `DATABASE_URL` env var set |
| Wrong API URL | Set `PREDICTIVE_MAINTENANCE_API` secret in Streamlit Cloud app settings |
