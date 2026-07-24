"""
Pydantic input schema for the Lung Cancer (Dataset 1) prediction API.

These are the exact 23 raw features the selector was fit on, in the
exact order confirmed via `selector.feature_names_in_`. Column names
and casing (e.g. "OccuPational Hazards", "chronic Lung Disease") are
kept byte-for-byte identical to the training data - the selector does
NOT do fuzzy matching, a renamed column will silently misalign values.

Value convention (from the original "Cancer Patient Dataset"):
  - Age: patient age in years (min observed in report = 14)
  - Gender: 1 = Male, 2 = Female
  - All other fields: ordinal severity/exposure scale, 1 (lowest) to 9 (highest)
"""

from pydantic import BaseModel, Field
from typing import Literal

Scale = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]


class PatientInputDS1(BaseModel):
    Age: int = Field(..., ge=10, le=100)
    Gender: Literal[1, 2] = Field(..., description="1 = Male, 2 = Female")
    Air_Pollution: Scale = Field(..., alias="Air Pollution")
    Alcohol_use: Scale = Field(..., alias="Alcohol use")
    Dust_Allergy: Scale = Field(..., alias="Dust Allergy")
    OccuPational_Hazards: Scale = Field(..., alias="OccuPational Hazards")
    Genetic_Risk: Scale = Field(..., alias="Genetic Risk")
    chronic_Lung_Disease: Scale = Field(..., alias="chronic Lung Disease")
    Balanced_Diet: Scale = Field(..., alias="Balanced Diet")
    Obesity: Scale
    Smoking: Scale
    Passive_Smoker: Scale = Field(..., alias="Passive Smoker")
    Chest_Pain: Scale = Field(..., alias="Chest Pain")
    Coughing_of_Blood: Scale = Field(..., alias="Coughing of Blood")
    Fatigue: Scale
    Weight_Loss: Scale = Field(..., alias="Weight Loss")
    Shortness_of_Breath: Scale = Field(..., alias="Shortness of Breath")
    Wheezing: Scale
    Swallowing_Difficulty: Scale = Field(..., alias="Swallowing Difficulty")
    Clubbing_of_Finger_Nails: Scale = Field(..., alias="Clubbing of Finger Nails")
    Frequent_Cold: Scale = Field(..., alias="Frequent Cold")
    Dry_Cough: Scale = Field(..., alias="Dry Cough")
    Snoring: Scale

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Gender": 1,
                "Air Pollution": 6,
                "Alcohol use": 5,
                "Dust Allergy": 6,
                "OccuPational Hazards": 5,
                "Genetic Risk": 4,
                "chronic Lung Disease": 4,
                "Balanced Diet": 3,
                "Obesity": 4,
                "Smoking": 6,
                "Passive Smoker": 5,
                "Chest Pain": 5,
                "Coughing of Blood": 4,
                "Fatigue": 5,
                "Weight Loss": 3,
                "Shortness of Breath": 5,
                "Wheezing": 4,
                "Swallowing Difficulty": 3,
                "Clubbing of Finger Nails": 4,
                "Frequent Cold": 3,
                "Dry Cough": 4,
                "Snoring": 3,
            }
        }


class PredictionOutputDS1(BaseModel):
    prediction: str  # "Low", "Medium", or "High"
    probability_low: float
    probability_medium: float
    probability_high: float
    model_version: str = "voting-ensemble-ds1-v1"

    class Config:
        # Pydantic reserves the "model_" prefix for its own internals;
        # this field just happens to collide with that convention.
        # Disabling the check removes the UserWarning seen at API startup.
        protected_namespaces = ()