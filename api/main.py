"""
FastAPI backend for the Lung Cancer Detection model (Dataset 1).

Loads:
  - model/lung_selector.sav  -> SelectFromModel, fit on 23 raw features, outputs 12
  - model/lung_model.sav     -> imblearn Pipeline (StandardScaler + SMOTE + VotingClassifier)
                                 classes_ = [0, 1, 2] = [Low, Medium, High]

Run locally with:
    uvicorn api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schema import PatientInputDS1, PredictionOutputDS1

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lung_model.sav")
SELECTOR_PATH = os.path.join(MODEL_DIR, "lung_selector.sav")

app = FastAPI(
    title="Lung Cancer Detection API (Dataset 1)",
    description="Serves predictions from the replicated Voting Ensemble "
                 "(RF + SVM + LR) trained on LungcancerDs.csv. "
                 "Outputs Low / Medium / High risk level.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's exact URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
selector = None

# Exact order confirmed from selector.feature_names_in_ - do not reorder.
RAW_COLUMNS = [
    "Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
    "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
    "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Chest Pain",
    "Coughing of Blood", "Fatigue", "Weight Loss", "Shortness of Breath",
    "Wheezing", "Swallowing Difficulty", "Clubbing of Finger Nails",
    "Frequent Cold", "Dry Cough", "Snoring",
]

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


@app.on_event("startup")
def load_artifacts():
    global model, selector
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SELECTOR_PATH):
        print(f"[WARNING] Model files not found in {MODEL_DIR}. "
              f"Copy lung_model.sav and lung_selector.sav there before predicting.")
        return
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    print("[OK] Model and selector loaded successfully.")
    print(f"[OK] Selector expects {selector.n_features_in_} raw features -> "
          f"{len(selector.get_feature_names_out())} selected features.")


@app.get("/")
def root():
    return {"message": "Lung Cancer Detection API (Dataset 1) is running. See /docs for usage."}


@app.get("/health")
def health():
    return {
        "status": "ok" if (model is not None and selector is not None) else "model_not_loaded",
        "model_loaded": model is not None,
        "selector_loaded": selector is not None,
    }


@app.post("/predict", response_model=PredictionOutputDS1)
def predict(patient: PatientInputDS1):
    if model is None or selector is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Place lung_model.sav and lung_selector.sav in the model/ folder."
        )

    data = patient.dict(by_alias=True)
    row = pd.DataFrame([[data[col] for col in RAW_COLUMNS]], columns=RAW_COLUMNS)

    try:
        selected = selector.transform(row)
        pred = int(model.predict(selected)[0])
        proba = model.predict_proba(selected)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # model.classes_ is [0, 1, 2] in that order per the dump you shared,
    # but we index by position defensively rather than assuming order.
    classes = list(model.classes_)
    prob_by_class = {int(c): float(p) for c, p in zip(classes, proba)}

    return PredictionOutputDS1(
        prediction=LABEL_MAP[pred],
        probability_low=round(prob_by_class.get(0, 0.0), 4),
        probability_medium=round(prob_by_class.get(1, 0.0), 4),
        probability_high=round(prob_by_class.get(2, 0.0), 4),
    )
