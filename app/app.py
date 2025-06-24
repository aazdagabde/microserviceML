#!/usr/bin/env python3
from typing import Literal
from pydantic import BaseModel, conlist  # ← import conlist

import joblib, pathlib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Energy ML Predictor • hybrid outputs")

MODELS = {p.stem: joblib.load(p) for p in pathlib.Path("models").glob("*.joblib")}
LAGS   = {"week": 168, "month": 30, "year": 12}
HORIZON= {"week": 168, "month": 30, "year": 1}

# ---------- Schemas ----------
class Record(BaseModel):
    date: str
    consumption: float

class PredictReq(BaseModel):
    horizon: Literal["week", "month", "year"]
    values: conlist(Record, min_items=1)  # ← remplace Field(..., min_length=1)
# -----------------------------

@app.post("/predict")
def predict(req: PredictReq):
    need = LAGS[req.horizon]
    ordered = sorted(req.values, key=lambda r: r.date, reverse=True)
    lags = [r.consumption for r in ordered][:need]
    if len(lags) < need:
        raise HTTPException(400, f"{need} valeurs requises pour '{req.horizon}'")

    X = np.array(lags).reshape(1, -1)
    model = MODELS[req.horizon]
    pred = model.predict(X)[0]

    # ----- réponse suivant l’horizon -----
    if req.horizon == "year":
        return {"total_prediction_wh": float(pred)}

    pred_list = pred.tolist()
    return {
        "predictions": pred_list,
        "total_prediction_wh": float(sum(pred_list))
    }
