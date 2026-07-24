# 🫁 Lung Cancer Risk Prediction — Deployed ML System

A full end-to-end deployment of a replicated & improved Voting Ensemble model
(Random Forest + SVM + Logistic Regression) for lung cancer risk classification,
built on `Ali T.M. et al., 2025` (DOI: [10.1155/bmri/9961773](https://doi.org/10.1155/bmri/9961773)).

**Live API:** https://lung-cancer-api.fastapicloud.dev/docs
**Live frontend:** https://lung-cancer-api.streamlit.app/
**Docker image:** https://hub.docker.com/r/ahmadabdullah3027/lung-cancer-api

---

## What this is

This project takes a trained scikit-learn model (Random Forest + SVM + Logistic
Regression Voting Ensemble, SMOTE-balanced, RF-based feature selection) from an
academic replication study and turns it into a real, publicly usable product:

- **FastAPI backend** — serves predictions via a `/predict` REST endpoint, with
  automatic input validation and interactive Swagger docs
- **Streamlit frontend** — a custom-designed dark glassmorphic UI for entering
  patient data and viewing risk predictions with live gauge charts
- **Dockerized** — both services containerize cleanly for local development and
  are also pushed to Docker Hub as a portable image
- **Cloud-deployed** — API on FastAPI Cloud, frontend on Streamlit Community Cloud,
  both on free tiers

---

## Architecture

```
                    ┌─────────────────────┐
   Patient data →   │   Streamlit UI       │
                    │  (frontend/app.py)   │
                    └──────────┬───────────┘
                               │ POST /predict
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI backend     │
                    │   (api/main.py)       │
                    │                       │
                    │  SelectFromModel  →   │
                    │  (23 → 12 features)   │
                    │         ↓             │
                    │  StandardScaler →     │
                    │  SMOTE →              │
                    │  VotingClassifier     │
                    │  (RF + SVM + LR)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │  model/lung_model.sav │
                    │  model/lung_selector.sav │
                    └───────────────────────┘
```

---

## Model details

- **Dataset**: `LungcancerDs.csv` (Cancer Patient Dataset), 1,000 patients,
  23 clinical/exposure features, 3-class target (Low / Medium / High)
- **Feature selection**: `SelectFromModel` (RandomForestClassifier,
  threshold='median') → reduces 23 raw features to 12
- **Pipeline**: `StandardScaler` → `SMOTE (k_neighbors=4)` → `VotingClassifier`
  (soft voting, weights `[LR=1, RF=2, SVM=1]`)
- **Reported accuracy**: 99% (paper) — see the project's Replication and
  Improvisation reports for a full breakdown, including the discovery that
  84.8% of the raw dataset consisted of duplicate rows, and the honest
  post-deduplication accuracy of 95.65%

---

## Project structure

```
lung-cancer-api/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, /predict and /health endpoints
│   └── schema.py             # Pydantic input/output models
├── frontend/
│   └── app.py                 # Streamlit UI
├── model/
│   ├── lung_model.sav          # trained Voting Ensemble pipeline
│   └── lung_selector.sav       # trained feature selector
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── pyproject.toml              # FastAPI Cloud deploy config
├── requirements.txt            # API dependencies (Docker / local dev)
└── README.md
```

---

## Running locally

### Without Docker

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
pip install streamlit requests plotly
streamlit run app.py
```

- API docs: http://127.0.0.1:8000/docs
- Frontend: http://localhost:8501

### With Docker Compose

```bash
docker compose up --build
```

Same ports as above. This is the closest local approximation to how the
containers actually run in production.

---

## API reference

**POST `/predict`**

```json
{
  "Age": 45, "Gender": 1, "Air Pollution": 6, "Alcohol use": 5,
  "Dust Allergy": 6, "OccuPational Hazards": 5, "Genetic Risk": 4,
  "chronic Lung Disease": 4, "Balanced Diet": 3, "Obesity": 4,
  "Smoking": 6, "Passive Smoker": 5, "Chest Pain": 5,
  "Coughing of Blood": 4, "Fatigue": 5, "Weight Loss": 3,
  "Shortness of Breath": 5, "Wheezing": 4, "Swallowing Difficulty": 3,
  "Clubbing of Finger Nails": 4, "Frequent Cold": 3, "Dry Cough": 4,
  "Snoring": 3
}
```

Response:
```json
{
  "prediction": "Low",
  "probability_low": 0.8499,
  "probability_medium": 0.1012,
  "probability_high": 0.0489,
  "model_version": "voting-ensemble-ds1-v1"
}
```

**GET `/health`** — confirms the model and selector loaded correctly.

---

## Deployment

### Backend — FastAPI Cloud

```bash
pip install "fastapi[standard]" fastapi-cloud-cli
fastapi deploy
```

Key config that makes this work, in `pyproject.toml`:
- `requires-python = "==3.12.*"` — pinned to match the wheels available for
  `numpy==1.26.4` / `pandas==2.2.2`; an open `>=3.12` range let the platform
  pick Python 3.14, which has no prebuilt wheels for those pins and fails to
  compile from source without a C++ toolchain
- `[tool.setuptools.packages.find] include = ["api*"]` — without this,
  `setuptools` sees `api/`, `model/`, and `frontend/` as ambiguous top-level
  packages and refuses to build
- `[tool.fastapi] entrypoint = "api.main:app"` — the app lives at `api/main.py`,
  not one of the auto-detected default locations (`main.py`, `app.py`,
  `app/main.py`)

### Frontend — Streamlit Community Cloud

1. Push to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select this repo
3. Main file path: `frontend/app.py`
4. **Python version: 3.12** (same reasoning as above — avoids wheel-availability
   issues with newer Python versions)
5. Secrets:
   ```toml
   API_URL = "https://lung-cancer-api.fastapicloud.dev"
   ```

### Docker Hub

Pull and run the API image directly, no build needed:

```bash
docker pull ahmadabdullah3027/lung-cancer-api:latest
docker run -p 8000:8000 ahmadabdullah3027/lung-cancer-api:latest
```

Image page: https://hub.docker.com/r/ahmadabdullah3027/lung-cancer-api

The model files (`lung_model.sav`, `lung_selector.sav`) are baked into this
image at build time via `Dockerfile.api`, so it runs standalone with no
extra setup. To swap in a different model without rebuilding, mount a
volume over `/app/model`:

```bash
docker run -p 8000:8000 -v $(pwd)/model:/app/model ahmadabdullah3027/lung-cancer-api:latest
```

---

## Lessons learned during deployment

This project's deployment surfaced a real chain of environment-consistency
bugs worth documenting:

1. **scikit-learn version drift** — the `.sav` model files were pickled with
   `scikit-learn==1.8.0`, but the serving environment had `1.4.2` installed,
   causing `AttributeError: 'LogisticRegression' object has no attribute
   'multi_class'` at inference time. Fix: pin the exact training-time version
   everywhere the model is loaded.
2. **Python version drift** — cloud platforms defaulting to the newest
   available Python (3.14) broke source builds for pinned older package
   versions lacking prebuilt wheels. Fix: pin `requires-python` explicitly.
3. **Ambiguous package discovery** — `setuptools` auto-discovery doesn't know
   which top-level folder is "the package" when a repo also contains data and
   frontend folders. Fix: explicit `packages.find` config.

None of these were bugs in the model or the application code — all three were
environment/tooling mismatches between where the model was trained and where
it was ultimately served.

---

## Disclaimer

This is an academic/portfolio project built on a peer-reviewed research
replication. Predictions are statistical outputs from a symptom/exposure
survey model, not a clinical diagnosis. Consult a physician for medical
concerns.

---

## Author

**Ahmad Abdullah**
