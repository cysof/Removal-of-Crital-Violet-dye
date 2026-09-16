"""
train_ann.py
------------
Trains and tunes the Artificial Neural Network described in thesis
section 3.3.2: a multilayer perceptron with 2 hidden layers, ReLU
activation, and the Adam optimizer. Hyper-parameters tuned via 5-fold
cross-validation are the number of neurons in each hidden layer and
the L2 regularisation strength (alpha), as explicitly named in 3.3.2.
"""

import time
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
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


def tune_ann(X_train, y_train, cv):
    # Candidate 2-hidden-layer architectures (neurons per layer)
    param_grid = {
        "hidden_layer_sizes": [
            (10, 10), (20, 10), (20, 20),
            (50, 20), (50, 50), (100, 50),
        ],
        "alpha": [0.0001, 0.001, 0.01, 0.1],  # L2 regularization
    }

    ann = MLPRegressor(
        activation="relu",
        solver="adam",
        max_iter=5000,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=RANDOM_SEED,
    )

    search = GridSearchCV(
        ann, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=0
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[ANN] best params: {search.best_params_}")
    print(f"[ANN] best CV R2:  {search.best_score_:.4f}  ({elapsed:.1f}s)")
    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    X_train, y_train = load_data()
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ann_model, ann_params, ann_cv_r2 = tune_ann(X_train, y_train, cv)
    joblib.dump(ann_model, MODELS_DIR / "ann.joblib")

    summary_path = OUTPUTS_DIR / "ann_tuning_summary.csv"
    pd.DataFrame([{"model": "ANN", "best_params": ann_params, "cv_r2": ann_cv_r2}]).to_csv(
        summary_path, index=False
    )

    print(f"\nSaved: models/ann.joblib")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()