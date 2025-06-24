#!/usr/bin/env python3
"""
Entraîne 3 modèles :
  • week  : 168 pas horaires  → liste + total
  • month : 30  pas journaliers → liste + total
  • year  : 1   pas annuel (total année suivante) → total seul

Usage :
    python train_hybrid.py --csv data/energy_measurements_2023-05-01_2025-05-31.csv
"""

import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error as mape

# --------- Config ---------
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True, parents=True)
BASE_ESTIMATOR = HistGradientBoostingRegressor(max_depth=3, random_state=42)  # rapide
MAX_SPLITS = 3  # CV plus courte
# --------------------------

def log(msg: str):
    print(time.strftime("[%H:%M:%S]"), msg)

def make_supervised(series: pd.Series, n_lags: int, horizon: int):
    X = pd.concat([series.shift(i) for i in range(1, n_lags + 1)], axis=1)
    X.columns = [f"lag_{i}" for i in range(1, n_lags + 1)]

    if horizon == 1:  # cible scalaire
        y = series.shift(-1).rename("target")
    else:             # cible multi-pas
        y_cols = [series.shift(-(h + 1)).rename(f"step_{h+1}") for h in range(horizon)]
        y = pd.concat(y_cols, axis=1)

    data = pd.concat([X, y], axis=1).dropna()
    return data.iloc[:, :n_lags], data.iloc[:, n_lags:]

def train_one(series, n_lags, horizon, name):
    X, y = make_supervised(series, n_lags, horizon)
    n = len(X)
    if n < 2:
        log(f"[SKIP] {name}: jeu trop court ({n} lignes)")
        return None

    log(f"{name.upper()} ▸ {n} lignes, {n_lags} lags, horizon={horizon}")

    model = (GradientBoostingRegressor(random_state=42)
             if horizon == 1
             else MultiOutputRegressor(BASE_ESTIMATOR, n_jobs=-1))

    # CV temporelle
    splits = min(MAX_SPLITS, n - 1)
    scores = []
    if splits >= 2:
        for tr, ts in TimeSeriesSplit(n_splits=splits).split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            scores.append(mape(y.iloc[ts], model.predict(X.iloc[ts])))
        log(f"  MAPE moyen = {np.mean(scores):.3%}")

    # entraînement final
    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    log(f"  modèle enregistré → {MODELS_DIR / f'{name}.joblib'}")
    return float(np.mean(scores)) if scores else None

def main(csv: Path):
    t0 = time.perf_counter()
    log(f"Lecture · {csv}")
    df = pd.read_csv(csv, parse_dates=["taken_at"]).set_index("taken_at")

    metrics = {
        "week":  train_one(df['value'].resample('h').sum(), 168, 168, "week"),
        "month": train_one(df['value'].resample('D').sum(),  30,  30, "month"),
        "year":  train_one(df['value'].resample('ME').sum(), 12,   1,  "year"),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"Métriques → {MODELS_DIR / 'metrics.json'}")
    log(f"✅ Terminé en {time.perf_counter()-t0:.1f} s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default=Path("data/energy_measurements_2023-05-01_2025-05-31.csv"),
                    type=Path,
                    help="Chemin du CSV horaire")
    main(ap.parse_args().csv)
