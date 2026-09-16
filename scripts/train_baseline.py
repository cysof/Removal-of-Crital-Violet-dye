"""
train_baseline.py
------------------
Trains and tunes the two "baseline" (non-ANN) models described in
thesis section 3.3:

    - Random Forest Regression (RFR): ensemble of trees, bootstrap
      samples, random feature subsets. n_estimators & max_depth tuned.
    - Support Vector Regression (SVR): RBF kernel. C and gamma tuned
      via exhaustive grid search (as explicitly stated in 3.3.3).

Both use 5-fold cross-validation on the TRAINING set only (test set
stays held out, per 3.1.3 / 3.3.4). Best models (refit on full
training set with best hyper-parameters) are saved to models/.
"""

import time
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
import joblib

RANDOM_SEED = 42
N_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    return X_train, y_train


def tune_random_forest(X_train, y_train, cv):
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    search = GridSearchCV(
        rf, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=0
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[RF]  best params: {search.best_params_}")
    print(f"[RF]  best CV R2:  {search.best_score_:.4f}  ({elapsed:.1f}s)")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_svr(X_train, y_train, cv):
    param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [0.001, 0.01, 0.1, 1, "scale"],
        "epsilon": [0.01, 0.1, 0.5],
    }
    svr = SVR(kernel="rbf")
    search = GridSearchCV(
        svr, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=0
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[SVR] best params: {search.best_params_}")
    print(f"[SVR] best CV R2:  {search.best_score_:.4f}  ({elapsed:.1f}s)")
    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    X_train, y_train = load_data()
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rf_model, rf_params, rf_cv_r2 = tune_random_forest(X_train, y_train, cv)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")

    svr_model, svr_params, svr_cv_r2 = tune_svr(X_train, y_train, cv)
    joblib.dump(svr_model, MODELS_DIR / "svr.joblib")

    summary = pd.DataFrame([
        {"model": "RandomForest", "best_params": rf_params, "cv_r2": rf_cv_r2},
        {"model": "SVR", "best_params": svr_params, "cv_r2": svr_cv_r2},
    ])
    summary.to_csv(OUTPUTS_DIR / "baseline_tuning_summary.csv", index=False)

    print("\nSaved: models/random_forest.joblib, models/svr.joblib")
    print("Saved: outputs/baseline_tuning_summary.csv")


if __name__ == "__main__":
    main()