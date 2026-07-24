# Lung Cancer Detection — Deployed API + Frontend

FastAPI + Streamlit deployment of the replicated & improved Voting Ensemble
model (RF + SVM + LR) from `Ali T.M. et al., 2025` (DOI: 10.1155/bmri/9961773).

## 1. Add your trained model files

Copy your existing joblib artifacts into `model/` — no retraining needed,
`.sav` and `.pkl` are the same joblib format:

```
model/lung_model.sav
model/lung_selector.sav
```

## 2. Run locally without Docker (fastest way to test)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

Open http://127.0.0.1:8000/docs to test the API directly, and
http://localhost:8501 for the UI.

## 3. Run with Docker Compose (recommended before deploying)

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501

This proves your Docker images work exactly like they will in production.

## 4. Deploy for free

### Backend → Render
1. Push this repo to GitHub.
2. On [render.com](https://render.com) → New → Web Service → connect your repo.
3. Environment: **Docker**. Dockerfile path: `Dockerfile.api`.
4. Instance type: Free.
5. Deploy. Copy the resulting URL, e.g. `https://lung-cancer-api.onrender.com`.

> Free Render web services sleep after ~15 min of inactivity and take
> ~30-60s to wake on the next request — fine for a demo/portfolio project.

### Frontend → Hugging Face Spaces
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/new-space).
2. SDK: **Docker**.
3. Upload `frontend/app.py`, `requirements.txt`, and `Dockerfile.frontend`
   (rename to `Dockerfile` in the Space, HF Spaces expects that name).
4. In the Space settings, add a secret/variable: `API_URL` =
   your Render URL from above.
5. The Space builds and serves your Streamlit app automatically, no credit
   card required.

### Optional: Docker Hub
```bash
docker build -f Dockerfile.api -t yourusername/lung-cancer-api .
docker push yourusername/lung-cancer-api
```
Useful if you want Render/Railway to pull a prebuilt image instead of
building from source.

## 5. Extending to Dataset 1 (multi-class Low/Medium/High)

The same pattern applies:
- Add `lung_model_ds1.sav` / `lung_selector_ds1.sav` to `model/`.
- Add a second Pydantic schema with DS1's 23 clinical features.
- Add a `/predict/ds1` route in `api/main.py` mirroring `/predict`.
- Add a second tab or toggle in the Streamlit app.

## API reference

**POST /predict**
```json
{
  "GENDER": "M", "AGE": 62, "SMOKING": 2, "YELLOW_FINGERS": 2,
  "ANXIETY": 1, "PEER_PRESSURE": 1, "CHRONIC DISEASE": 2,
  "FATIGUE": 2, "ALLERGY": 1, "WHEEZING": 2, "ALCOHOL CONSUMING": 2,
  "COUGHING": 2, "SHORTNESS OF BREATH": 2, "SWALLOWING DIFFICULTY": 1,
  "CHEST PAIN": 2
}
```
Response:
```json
{
  "prediction": "YES",
  "probability_cancer": 0.94,
  "probability_no_cancer": 0.06,
  "model_version": "voting-ensemble-v1"
}
```

**GET /health** — confirms the model loaded correctly, useful for
debugging a failed deploy.

## Disclaimer
This is an academic/portfolio project. Predictions are statistical
outputs from a symptom-survey model, not a medical diagnosis.
